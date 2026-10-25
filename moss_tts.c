/* ============================================================================
 *  moss_tts.c -- MOSS-TTS-Nano (0.1B) 纯 C11 单文件推理实现
 * ----------------------------------------------------------------------------
 *  参考实现: https://github.com/mzdk100/moss-tts-nano (Rust)
 *  模型:     OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX
 *            OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX
 *
 *  依赖:
 *    - onnxruntime.dll (运行时 LoadLibrary 动态加载, 不需要 .lib / 导入库)
 *    - onnxruntime_c_api.h / onnxruntime_error_code.h (编译期头文件)
 *    - 静态 CRT (/MT), 因此生成的 exe 不依赖 vcruntime/ucrt DLL
 *
 *  流水线 (与 Rust 实现一致):
 *
 *    文本 --SentencePiece(BPE)--> token ids
 *         |
 *         v
 *    [user_prefix][audio_start][ref audio codes][audio_end][after_ref]
 *    [text tokens][assistant_prefix][audio_start]        <- 每行 17 列 = 1 文本 + 16 码本
 *         |
 *         v
 *    prefill.onnx  ->  global_hidden[1,S,768] + 12 层 KV cache
 *         |
 *         v   (取最后一个位置的 hidden)
 *    local_fixed_sampled_frame.onnx  ->  frame_token_ids[16]  (一次性采样 16 个通道)
 *         |                                    ^
 *         |                                    |  repetition_seen_mask + 随机数 u
 *         v
 *    decode_step.onnx (输入上一步的 16 个 token) -> 新的 hidden + 新 KV
 *         |
 *         v   (循环直到 should_continue==0 / 尾部静音 / 达到帧数上限)
 *    codec decode_step.onnx (流式, 固定 KV 缓存)  ->  48kHz 立体声 PCM
 *         |
 *         v
 *    out.wav (16-bit PCM)
 *
 *  性能设计 (相对朴素实现的实测收益):
 *    - 四个 session 并行加载          加载 13.0s -> 2.7s
 *    - codec_encode 延迟到需要时再加载  省 4s / 44MB (只有 --ref-wav 才用)
 *    - 输入张量复用 + 掩码增量更新     单帧 90ms -> 44ms
 *    - 关闭 ORT 自旋等待              混合架构 CPU 上避免线程被调度到能效核
 *    - 流式 codec 解码 (固定 13MB 缓存) 长音频内存 6.7GB -> 1.3GB
 *    - 边生成边解码                    省掉整体解码那一遍
 *    - --play / --server 流式播放     首字 ~1.1s 出声, 之后生成追着播放跑
 *    - --server 常驻                  模型只加载一次 (省 ~7.6s), 行协议换音色/种子
 * ==========================================================================*/

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <mmsystem.h>      /* waveOut 流式播放 */
#include <wininet.h>       /* 对话模式 HTTPS */
#include <io.h>            /* _isatty (server 模式判断交互终端) */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "onnxruntime_c_api.h"

/* ---------------------------------------------------------------------------
 * 0. 模型常量 (来自 tts_browser_onnx_meta.json / codec_browser_onnx_meta.json)
 * -------------------------------------------------------------------------*/
#define N_VQ              16      /* RVQ 码本数量 */
#define ROW_WIDTH         (N_VQ + 1)  /* 每行 = 1 个文本 token + 16 个音频 token */
#define HIDDEN            768
#define GLOBAL_LAYERS     12
#define VOCAB_SIZE        16384
#define AUDIO_CODEBOOK    1024    /* 每个码本大小 */
#define AUDIO_PAD_TOKEN   1024
#define PAD_TOKEN         3
#define AUDIO_START_TOKEN 6
#define AUDIO_END_TOKEN   7
#define AUDIO_USER_SLOT   8       /* 参考音频帧的行首 token */
#define AUDIO_ASST_SLOT   9       /* 生成音频帧的行首 token */

#define SAMPLE_RATE       48000
#define CHANNELS          2
#define DOWNSAMPLE        3840    /* 每帧 3840 个采样点 -> 12.5 Hz */

#define MAX_FRAMES_DEF    375     /* 上限 30 秒 */
#define EOS_WINDOW        48

/* ---------------------------------------------------------------------------
 * 1. 基础工具
 * -------------------------------------------------------------------------*/
static void die(const char* msg)
{
    fprintf(stderr, "\n[fatal] %s\n", msg);
    fflush(stderr);
    exit(1);
}

static void* xmalloc(size_t n)
{
    void* p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void* xcalloc(size_t n, size_t sz)
{
    void* p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) die("out of memory");
    return p;
}

static char* w2u(const wchar_t* w)
{
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    if (n <= 0) return NULL;
    char* b = (char*)xmalloc((size_t)n);
    WideCharToMultiByte(CP_UTF8, 0, w, -1, b, n, NULL, NULL);
    return b;
}

/* server 模式从 stdin 收到的是 UTF-8 字节行, 命令参数要转回宽字符 */
static wchar_t* u2w(const char* s)
{
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    if (n <= 0) return NULL;
    wchar_t* b = (wchar_t*)xmalloc(sizeof(wchar_t) * (size_t)n);
    MultiByteToWideChar(CP_UTF8, 0, s, -1, b, n);
    return b;
}

static uint64_t now_ms(void)
{
    return (uint64_t)GetTickCount64();
}

/* 读取整个文件 */
static uint8_t* read_whole_file(const wchar_t* path, size_t* out_len)
{
    FILE* f = _wfopen(path, L"rb");
    if (!f) return NULL;
    if (_fseeki64(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long long sz = _ftelli64(f);
    if (sz < 0) { fclose(f); return NULL; }
    if (_fseeki64(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    uint8_t* buf = (uint8_t*)xmalloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = 0;
    if (out_len) *out_len = rd;
    return buf;
}

/* UTF-8 解码: 返回码点, *len 为字节数 */
static uint32_t utf8_dec(const unsigned char* s, int* len)
{
    unsigned char b = s[0];
    if (b < 0x80)            { *len = 1; return b; }
    if ((b & 0xE0) == 0xC0)  { *len = 2; return ((uint32_t)(b & 0x1F) << 6) | (s[1] & 0x3F); }
    if ((b & 0xF0) == 0xE0)  { *len = 3; return ((uint32_t)(b & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F); }
    if ((b & 0xF8) == 0xF0)  { *len = 4; return ((uint32_t)(b & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) | ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F); }
    *len = 1;
    return b;
}

static int utf8_clen(unsigned char b)
{
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 1;
}

static int is_cjk_cp(uint32_t c)
{
    return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) ||
           (c >= 0x3040 && c <= 0x30FF) || (c >= 0xAC00 && c <= 0xD7AF) ||
           (c >= 0x3000 && c <= 0x303F);
}

static int text_has_cjk(const char* s)
{
    const unsigned char* p = (const unsigned char*)s;
    while (*p) {
        int n;
        uint32_t c = utf8_dec(p, &n);
        if (is_cjk_cp(c)) return 1;
        p += n;
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * 2. 随机数 (splitmix64 -> [0,1) float), 只为采样提供 u 值
 * -------------------------------------------------------------------------*/
typedef struct { uint64_t s; } Rng;

static void rng_seed(Rng* r, uint64_t seed) { r->s = seed ? seed : 0x9E3779B97F4A7C15ull; }

static uint64_t rng_next(Rng* r)
{
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* [0, 1) 之间, 用高 24 位保证与 float 精度匹配 */
static float rng_u(Rng* r)
{
    uint32_t v = (uint32_t)(rng_next(r) >> 40);   /* 24 bit */
    return (float)v * (1.0f / 16777216.0f);
}

/* ---------------------------------------------------------------------------
 * 3. ONNX Runtime 动态加载封装
 * -------------------------------------------------------------------------*/
static const OrtApi* ort = NULL;
static OrtEnv* g_env = NULL;
static OrtMemoryInfo* g_meminfo = NULL;

#define OK(expr)                                                              \
    do {                                                                      \
        OrtStatus* _st = (expr);                                              \
        if (_st) {                                                            \
            fprintf(stderr, "\n[onnxruntime] %s\n  (%s:%d)\n",                \
                    ort->GetErrorMessage(_st), __FILE__, __LINE__);           \
            fflush(stderr);                                                   \
            ort->ReleaseStatus(_st);                                          \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

static void ort_load(void)
{
    HMODULE h = LoadLibraryA("onnxruntime.dll");
    if (!h) {
        /* 退回到 exe 所在目录 */
        wchar_t exe[MAX_PATH];
        GetModuleFileNameW(NULL, exe, MAX_PATH);
        wchar_t* slash = wcsrchr(exe, L'\\');
        if (slash) {
            wcscpy(slash + 1, L"onnxruntime.dll");
            h = LoadLibraryW(exe);
        }
    }
    if (!h) die("cannot load onnxruntime.dll (put it next to the exe)");

    typedef const OrtApiBase* (ORT_API_CALL * GetApiBaseFn)(void);
    GetApiBaseFn fn = (GetApiBaseFn)(void*)GetProcAddress(h, "OrtGetApiBase");
    if (!fn) die("onnxruntime.dll has no OrtGetApiBase export");

    const OrtApiBase* base = fn();
    fprintf(stderr, "[onnxruntime] %s (header api %d)\n", base->GetVersionString(), ORT_API_VERSION);
    /* 头文件版本优先, 失败则依次降级 (OrtApi 结构体只追加不改变) */
    static const int versions[] = { ORT_API_VERSION, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    int chosen = 0;
    for (size_t i = 0; i < sizeof(versions) / sizeof(versions[0]) && !ort; i++) {
        ort = base->GetApi(versions[i]);
        chosen = versions[i];
    }
    if (!ort) die("onnxruntime.dll is too old for this header");
    fprintf(stderr, "[onnxruntime] using OrtApi v%d\n", chosen);

    OK(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "moss_tts", &g_env));
    OK(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &g_meminfo));
}

/* 各阶段耗时 / 峰值内存, 供 --verbose 输出 */
typedef struct {
    uint64_t load_ort, load_prefill, load_decode, load_frame, load_dec, load_enc;
    uint64_t prefill, frame, step, probe;
    int steps;
    size_t peak_rss;
} Stats;
static Stats g_stats;

static size_t cur_working_set(size_t* priv_out)
{
    PROCESS_MEMORY_COUNTERS_EX pmc;
    pmc.cb = sizeof(pmc);
    typedef BOOL (WINAPI * Fn)(HANDLE, PROCESS_MEMORY_COUNTERS_EX*, DWORD);
    static Fn fn = NULL;
    if (!fn) fn = (Fn)(void*)GetProcAddress(GetModuleHandleA("kernel32.dll"), "K32GetProcessMemoryInfo");
    if (fn && fn(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        if (priv_out) *priv_out = (size_t)pmc.PrivateUsage;
        return (size_t)pmc.WorkingSetSize;
    }
    return 0;
}

static size_t peak_working_set(void)
{
    PROCESS_MEMORY_COUNTERS pmc;
    pmc.cb = sizeof(pmc);
    typedef BOOL (WINAPI * Fn)(HANDLE, PROCESS_MEMORY_COUNTERS*, DWORD);
    static Fn fn = NULL;
    if (!fn) fn = (Fn)(void*)GetProcAddress(GetModuleHandleA("kernel32.dll"), "K32GetProcessMemoryInfo");
    if (fn && fn(GetCurrentProcess(), &pmc, sizeof(pmc))) return (size_t)pmc.PeakWorkingSetSize;
    return 0;
}

/* 一个 session + 输入输出名字表 */
typedef struct {
    OrtSession* s;
    size_t n_in, n_out;
    char in_name[64][48];
    char out_name[64][48];
    const char* in_ptr[64];
    const char* out_ptr[64];
} Session;

static void sess_open_ex(Session* S, const wchar_t* dir, const wchar_t* file, int threads,
                         uint64_t* t_out, int no_arena)
{
    uint64_t t0 = now_ms();
    wchar_t path[MAX_PATH * 2];
    _snwprintf(path, MAX_PATH * 2, L"%s\\%s", dir, file);

    OrtSessionOptions* so = NULL;
    OK(ort->CreateSessionOptions(&so));
    OK(ort->SetIntraOpNumThreads(so, threads));
    /* 关闭自旋等待: 这个模型每步的计算量很小, 自旋线程在混合架构 CPU 上会被
     * 调度到能效核, 反而拖慢整体 (实测 8 线程: 开启自旋 33s, 关闭后 17.8s)。 */
    OK(ort->AddSessionConfigEntry(so, "session.intra_op.allow_spinning", "0"));
    if (no_arena & 1) OK(ort->DisableCpuMemArena(so));
    if (no_arena & 2) OK(ort->DisableMemPattern(so));
    /* BASIC 而非 ALL: 实测这个规模的图, ALL 级别的额外优化 (布局变换等)
     * 对每步小矩阵是负收益 —— 每帧 52.3 -> 46.5ms (-11%), decode步 -16%。
     * 加载时长与级别无关 (瓶颈在 .data 读取, 试过 .ort 缓存更慢已弃)。
     * 输出与 ALL 逐字节同长, 最大 1 LSB 舍入差, 数值等价。 */
    OK(ort->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_BASIC));
    OrtStatus* st = ort->CreateSession(g_env, path, so, &S->s);
    ort->ReleaseSessionOptions(so);
    if (st) {
        char* p = w2u(path);
        fprintf(stderr, "\n[onnxruntime] %s\n  while loading %s\n",
                ort->GetErrorMessage(st), p ? p : "?");
        ort->ReleaseStatus(st);
        exit(1);
    }
    if (S->n_in == 0) {
        size_t n = 0;
        OK(ort->SessionGetInputCount(S->s, &n));
        S->n_in = n;
    }
    if (S->n_out == 0) {
        size_t n = 0;
        OK(ort->SessionGetOutputCount(S->s, &n));
        S->n_out = n;
    }
    if (t_out) *t_out = now_ms() - t0;
}

static void sess_open(Session* S, const wchar_t* dir, const wchar_t* file, int threads, uint64_t* t_out)
{
    sess_open_ex(S, dir, file, threads, t_out, 0);
}

static void sess_run(Session* S, OrtValue** ins, OrtValue** outs)
{
    OK(ort->Run(S->s, NULL, S->in_ptr, (const OrtValue* const*)ins, S->n_in,
                S->out_ptr, S->n_out, outs));
}

static void* tensor_data(OrtValue* v, size_t* count)
{
    void* p = NULL;
    OK(ort->GetTensorMutableData(v, &p));
    if (count) {
        OrtTensorTypeAndShapeInfo* info = NULL;
        OK(ort->GetTensorTypeAndShape(v, &info));
        OK(ort->GetTensorShapeElementCount(info, count));
        ort->ReleaseTensorTypeAndShapeInfo(info);
    }
    return p;
}

static OrtValue* mk_tensor(void* data, const int64_t* shape, int rank,
                           size_t elem_size, ONNXTensorElementDataType dt)
{
    size_t n = 1;
    for (int i = 0; i < rank; i++) n *= (size_t)shape[i];
    OrtValue* v = NULL;
    OK(ort->CreateTensorWithDataAsOrtValue(g_meminfo, data, n * elem_size,
                                           shape, (size_t)rank, dt, &v));
    return v;
}

#define MK_I32(buf, shape, rank) mk_tensor((buf), (shape), (rank), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
#define MK_F32(buf, shape, rank) mk_tensor((buf), (shape), (rank), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)

/* ---------------------------------------------------------------------------
 * 4. SentencePiece (BPE + precompiled_charsmap 归一化)
 *
 *    tokenizer.model 是 protobuf, 这里手写最小的 wire-format 解析器,
 *    不做任何第三方依赖。
 * -------------------------------------------------------------------------*/
typedef struct {
    const char* p;     /* 指向模型文件内的字节, 不保证 NUL 结尾 */
    int32_t     len;
    float       score;
    int32_t     type;  /* 1=normal 2=unknown 3=control 4=user 5=unused 6=byte */
} Piece;

typedef struct {
    uint8_t*  blob;        /* 整个 tokenizer.model */
    Piece*    pieces;
    int32_t   n_pieces;

    int32_t*  table;       /* 哈希表: piece id, -1 为空 */
    uint32_t  table_mask;

    const uint8_t* charsmap;   /* precompiled_charsmap */
    size_t    charsmap_len;
    const uint32_t* units;     /* darts double-array */
    uint32_t  n_units;
    const char* norm_blob;     /* 归一化替换串池 */
    size_t    norm_blob_len;

    int add_dummy_prefix;
    int remove_extra_ws;
    int escape_ws;
    int unk_id;
    int model_type;        /* 2 = BPE */
} SP;

static uint32_t pb_varint(const uint8_t* b, size_t* i)
{
    uint32_t r = 0; int sh = 0;
    for (;;) {
        uint8_t x = b[(*i)++];
        r |= (uint32_t)(x & 0x7F) << sh;
        if (!(x & 0x80)) return r;
        sh += 7;
        if (sh > 28) return r;
    }
}

/* 遍历 protobuf 字段: 返回 field number, 通过 *wt 给出 wire type */
static int pb_next(const uint8_t* b, size_t len, size_t* i, int* wt, uint32_t* vlen)
{
    if (*i >= len) return -1;
    uint32_t key = pb_varint(b, i);
    int fn = (int)(key >> 3);
    *wt = (int)(key & 7);
    if (*wt == 0) {
        *vlen = pb_varint(b, i);
    } else if (*wt == 2) {
        *vlen = pb_varint(b, i);
    } else if (*wt == 5) {
        *vlen = 4;
    } else if (*wt == 1) {
        *vlen = 8;
    } else {
        *vlen = 0;
    }
    return fn;
}

static uint32_t sp_hash(const char* p, int len)
{
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) { h ^= (uint8_t)p[i]; h *= 16777619u; }
    return h;
}

static int sp_find(const SP* sp, const char* p, int len)
{
    uint32_t h = sp_hash(p, len) & sp->table_mask;
    for (;;) {
        int32_t id = sp->table[h];
        if (id < 0) return -1;
        if (sp->pieces[id].len == len && memcmp(sp->pieces[id].p, p, (size_t)len) == 0)
            return id;
        h = (h + 1) & sp->table_mask;
    }
}

static void sp_parse_piece(SP* sp, const uint8_t* b, size_t len, Piece* out)
{
    size_t i = 0;
    out->p = NULL; out->len = 0; out->score = 0.0f; out->type = 1;
    while (i < len) {
        int wt; uint32_t vl;
        int fn = pb_next(b, len, &i, &wt, &vl);
        if (fn < 0) break;
        if (fn == 1 && wt == 2) {
            out->p = (const char*)(b + i);
            out->len = (int32_t)vl;
            i += vl;
        } else if (fn == 2 && wt == 5) {
            float f;
            memcpy(&f, b + i, 4);
            out->score = f;
            i += 4;
        } else if (fn == 3 && wt == 0) {
            out->type = (int32_t)vl;
        } else {
            i += vl;
        }
    }
}

static void sp_load(SP* sp, const wchar_t* path)
{
    memset(sp, 0, sizeof(*sp));
    sp->add_dummy_prefix = 1;
    sp->remove_extra_ws = 1;
    sp->escape_ws = 1;
    sp->model_type = 1;

    size_t len = 0;
    sp->blob = read_whole_file(path, &len);
    if (!sp->blob) die("cannot read tokenizer.model");

    /* 先数一遍 piece 数量 */
    int32_t cap = 0;
    {
        size_t i = 0;
        while (i < len) {
            int wt; uint32_t vl;
            int fn = pb_next(sp->blob, len, &i, &wt, &vl);
            if (fn < 0) break;
            if (fn == 1 && wt == 2) cap++;
            i += vl;
        }
    }
    sp->pieces = (Piece*)xcalloc((size_t)cap, sizeof(Piece));
    sp->n_pieces = 0;

    size_t i = 0;
    while (i < len) {
        int wt; uint32_t vl;
        int fn = pb_next(sp->blob, len, &i, &wt, &vl);
        if (fn < 0) break;
        const uint8_t* body = sp->blob + i;
        if (fn == 1 && wt == 2) {
            sp_parse_piece(sp, body, vl, &sp->pieces[sp->n_pieces++]);
        } else if (fn == 2 && wt == 2) {           /* TrainerSpec */
            size_t j = 0;
            while (j < vl) {
                int w2; uint32_t v2;
                int f2 = pb_next(body, vl, &j, &w2, &v2);
                if (f2 < 0) break;
                if (f2 == 3 && w2 == 0) sp->model_type = (int)v2;
                else if (f2 == 40 && w2 == 0) sp->unk_id = (int)v2;
                j += (w2 == 0) ? 0 : v2;
            }
        } else if (fn == 3 && wt == 2) {           /* NormalizerSpec */
            size_t j = 0;
            while (j < vl) {
                int w2; uint32_t v2;
                int f2 = pb_next(body, vl, &j, &w2, &v2);
                if (f2 < 0) break;
                if (f2 == 2 && w2 == 2) {
                    sp->charsmap = body + j;
                    sp->charsmap_len = v2;
                } else if (f2 == 3 && w2 == 0) sp->add_dummy_prefix = (int)v2;
                else if (f2 == 4 && w2 == 0) sp->remove_extra_ws = (int)v2;
                else if (f2 == 5 && w2 == 0) sp->escape_ws = (int)v2;
                j += (w2 == 0) ? 0 : v2;
            }
        }
        i += vl;
    }
    if (sp->n_pieces <= 0) die("tokenizer.model: no pieces");

    /* 哈希表 */
    uint32_t sz = 1;
    while (sz < (uint32_t)sp->n_pieces * 2u) sz <<= 1;
    sp->table = (int32_t*)xmalloc(sizeof(int32_t) * sz);
    for (uint32_t k = 0; k < sz; k++) sp->table[k] = -1;
    sp->table_mask = sz - 1;
    for (int32_t k = 0; k < sp->n_pieces; k++) {
        uint32_t h = sp_hash(sp->pieces[k].p, sp->pieces[k].len) & sp->table_mask;
        while (sp->table[h] >= 0) h = (h + 1) & sp->table_mask;
        sp->table[h] = k;
    }

    /* charsmap: uint32 trie_size + trie + 替换串池 */
    if (sp->charsmap && sp->charsmap_len > 8) {
        uint32_t trie_bytes;
        memcpy(&trie_bytes, sp->charsmap, 4);
        if (trie_bytes + 4 <= sp->charsmap_len) {
            sp->units = (const uint32_t*)(sp->charsmap + 4);
            sp->n_units = trie_bytes / 4;
            sp->norm_blob = (const char*)(sp->charsmap + 4 + trie_bytes);
            sp->norm_blob_len = sp->charsmap_len - 4 - trie_bytes;
        }
    }
}

/* darts-clone 双数组 trie 的位布局 */
#define DA_LABEL(u)     ((uint32_t)((u) & 0xFF))
#define DA_HAS_LEAF(u)  ((uint32_t)(((u) >> 8) & 1))
#define DA_VALUE(u)     ((uint32_t)((u) & 0x7FFFFFFFu))
#define DA_OFFSET(u)    ((uint32_t)(((u) >> 10) << (((u) & (1u << 9)) >> 6)))

/* 在 charsmap trie 上做最长前缀匹配.
 * 返回消耗的输入字节数; *rep / *rep_len 给出替换串 (可能是空串 = 删除). */
static int sp_norm_prefix(const SP* sp, const char* s, int avail,
                          const char** rep, int* rep_len)
{
    int best_len = 0;
    uint32_t best_val = 0;

    if (sp->units && sp->n_units > 0) {
        uint32_t node = 0;
        uint32_t u = sp->units[0];
        node ^= DA_OFFSET(u);
        for (int k = 0; k < avail; k++) {
            uint32_t ch = (uint8_t)s[k];
            node ^= ch;
            if (node >= sp->n_units) break;
            u = sp->units[node];
            if (DA_LABEL(u) != ch) break;
            node ^= DA_OFFSET(u);
            if (DA_HAS_LEAF(u)) { best_len = k + 1; best_val = DA_VALUE(u); }
        }
    }

    if (best_len == 0) {                      /* 无匹配: 原样复制一个 UTF-8 字符 */
        int n = utf8_clen((unsigned char)s[0]);
        if (n > avail) n = avail;
        *rep = s;
        *rep_len = n;
        return n;
    }

    /* 替换串在池中以 NUL 结尾 */
    const char* p = sp->norm_blob + best_val;
    size_t remain = sp->norm_blob_len > best_val ? sp->norm_blob_len - best_val : 0;
    const char* z = (const char*)memchr(p, 0, remain);
    *rep = p;
    *rep_len = (int)(z ? (size_t)(z - p) : remain);
    return best_len;
}

/* 归一化: 应用 charsmap + 空格转义 + dummy prefix */
static int sp_normalize(const SP* sp, const char* text, char** out)
{
    size_t n_in = strlen(text);
    size_t cap = n_in * 4 + 64;
    char* buf = (char*)xmalloc(cap);
    size_t o = 0;

    const char* SPACE = "\xE2\x96\x81";   /* U+2581 ▁ */
    int prev_space = sp->add_dummy_prefix;
    if (sp->add_dummy_prefix) { memcpy(buf + o, SPACE, 3); o += 3; }

    size_t i = 0;
    while (i < n_in) {
        const char* rep = NULL;
        int rep_len = 0;
        int consumed = sp_norm_prefix(sp, text + i, (int)(n_in - i), &rep, &rep_len);
        if (consumed <= 0) consumed = 1;
        i += (size_t)consumed;

        if (o + (size_t)rep_len * 4 + 16 > cap) {
            cap = cap * 2 + (size_t)rep_len * 4 + 64;
            buf = (char*)realloc(buf, cap);
            if (!buf) die("out of memory");
        }

        /* 把替换串按字符喂进空白处理逻辑 */
        int j = 0;
        while (j < rep_len) {
            unsigned char c = (unsigned char)rep[j];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                if (sp->escape_ws) {
                    if (!prev_space) {
                        memcpy(buf + o, SPACE, 3); o += 3;
                        prev_space = 1;
                    }
                    if (!sp->remove_extra_ws) { memcpy(buf + o, SPACE, 3); o += 3; }
                } else {
                    if (!sp->remove_extra_ws || !prev_space) { buf[o++] = ' '; prev_space = 1; }
                }
                j++;
            } else {
                int n = utf8_clen(c);
                if (j + n > rep_len) n = rep_len - j;
                memcpy(buf + o, rep + j, (size_t)n);
                o += (size_t)n;
                prev_space = 0;
                j += n;
            }
        }
    }
    buf[o] = 0;
    *out = buf;
    return (int)o;
}

/* 文本级预处理 (对应 Rust normalize_for_tts) */
static char* normalize_for_tts(const char* text)
{
    size_t n = strlen(text);
    char* s = (char*)xmalloc(n + 32);
    size_t o = 0;
    int prev_space = 0;
    /* 去首尾空白 + 折叠空白 + 换行制表转空格 */
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            if (o == 0) continue;          /* 跳过前导空白 */
            if (prev_space) continue;
            s[o++] = ' ';
            prev_space = 1;
        } else {
            s[o++] = (char)c;
            prev_space = 0;
        }
    }
    while (o > 0 && s[o - 1] == ' ') o--;
    s[o] = 0;

    if (o == 0) return s;

    int has_end = 0;
    /* ASCII 标点或常见中英文结束符 */
    if (o >= 1) {
        unsigned char c = (unsigned char)s[o - 1];
        if (c < 0x80 && ((c >= '!' && c <= '/') || (c >= ':' && c <= '@') ||
                         (c >= '[' && c <= '`') || (c >= '{' && c <= '~'))) has_end = 1;
    }
    static const char* ends[] = { "。", "！", "？", "；", "：", "、", "，", "）", "】",
                                  "》", "」", "』", "\"", "'" };
    for (size_t k = 0; k < sizeof(ends) / sizeof(ends[0]); k++) {
        size_t el = strlen(ends[k]);
        if (o >= el && memcmp(s + o - el, ends[k], el) == 0) { has_end = 1; break; }
    }

    if (!has_end) {
        if (text_has_cjk(s)) { s[o] = (char)0xE3; s[o + 1] = (char)0x80; s[o + 2] = (char)0x82; o += 3; }
        else { s[o] = '.'; o += 1; }
        s[o] = 0;
    }
    return s;
}

/* BPE 编码: 从单字符开始, 每次合并 score 最高的相邻对 */
static int sp_encode(const SP* sp, const char* text, int32_t* out, int max_out)
{
    char* norm = NULL;
    int nb = sp_normalize(sp, text, &norm);
    if (nb <= 0) { free(norm); return 0; }

    /* 字符边界 */
    int nch = 0;
    for (int i = 0; i < nb; ) { i += utf8_clen((unsigned char)norm[i]); nch++; }
    if (nch == 0) { free(norm); return 0; }

    int* off = (int*)xmalloc(sizeof(int) * (size_t)(nch + 1));
    {
        int i = 0, k = 0;
        while (i < nb) { off[k++] = i; i += utf8_clen((unsigned char)norm[i]); }
        off[nch] = nb;
    }

    typedef struct { int a, b, id; } Tok;   /* a,b = 字符下标 [a,b) */
    Tok* tk = (Tok*)xmalloc(sizeof(Tok) * (size_t)(nch + 2));
    int nt = 0;
    for (int k = 0; k < nch; k++) {
        int id = sp_find(sp, norm + off[k], off[k + 1] - off[k]);
        if (id < 0) id = sp->unk_id;
        tk[nt].a = k; tk[nt].b = k + 1; tk[nt].id = id;
        nt++;
    }

    char tmp[128];
    while (nt > 1) {
        int best_i = -1;
        float best_s = -1e30f;
        for (int k = 0; k + 1 < nt; k++) {
            int a = off[tk[k].a], b = off[tk[k + 1].b];
            int L = b - a;
            if (L >= (int)sizeof(tmp)) continue;
            memcpy(tmp, norm + a, (size_t)L);
            int id = sp_find(sp, tmp, L);
            if (id < 0) continue;
            if (sp->pieces[id].score > best_s) { best_s = sp->pieces[id].score; best_i = k; }
        }
        if (best_i < 0) break;
        tk[best_i].b = tk[best_i + 1].b;
        {
            int a = off[tk[best_i].a], b = off[tk[best_i].b];
            int L = b - a;
            memcpy(tmp, norm + a, (size_t)L);
            tk[best_i].id = sp_find(sp, tmp, L);
        }
        memmove(tk + best_i + 1, tk + best_i + 2, sizeof(Tok) * (size_t)(nt - best_i - 2));
        nt--;
    }

    int n = 0;
    for (int k = 0; k < nt && n < max_out; k++) out[n++] = tk[k].id;

    free(off); free(tk); free(norm);
    return n;
}

static int sp_count(const SP* sp, const char* text)
{
    static int32_t buf[4096];
    return sp_encode(sp, text, buf, 4096);
}

/* ---------------------------------------------------------------------------
 * 5. WAV I/O
 * -------------------------------------------------------------------------*/
static void wav_write_stereo(const wchar_t* path, const float* planar, size_t frames,
                             int channels, int rate)
{
    uint32_t data_bytes = (uint32_t)(frames * (size_t)channels * 2u);
    uint32_t byte_rate = (uint32_t)(rate * channels * 2);
    uint16_t block_align = (uint16_t)(channels * 2);

    FILE* f = _wfopen(path, L"wb");
    if (!f) die("cannot open output wav");

    uint8_t hdr[44];
    uint32_t riff = 36 + data_bytes;
    memcpy(hdr + 0, "RIFF", 4);
    memcpy(hdr + 4, &riff, 4);
    memcpy(hdr + 8, "WAVEfmt ", 8);
    uint32_t fmtlen = 16;      memcpy(hdr + 16, &fmtlen, 4);
    uint16_t pcm = 1;          memcpy(hdr + 20, &pcm, 2);
    uint16_t ch = (uint16_t)channels; memcpy(hdr + 22, &ch, 2);
    uint32_t sr = (uint32_t)rate;     memcpy(hdr + 24, &sr, 4);
    memcpy(hdr + 28, &byte_rate, 4);
    memcpy(hdr + 32, &block_align, 2);
    uint16_t bits = 16;        memcpy(hdr + 34, &bits, 2);
    memcpy(hdr + 36, "data", 4);
    memcpy(hdr + 40, &data_bytes, 4);
    fwrite(hdr, 1, 44, f);

    int16_t* pcmbuf = (int16_t*)xmalloc((size_t)frames * (size_t)channels * sizeof(int16_t));
    for (size_t i = 0; i < frames * (size_t)channels; i++) {
        float v = planar[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        pcmbuf[i] = (int16_t)(v * 32767.0f);
    }
    fwrite(pcmbuf, sizeof(int16_t), frames * (size_t)channels, f);
    free(pcmbuf);
    fclose(f);
}

/* 读 wav -> 48kHz 立体声交错浮点; 支持 PCM16 / float32, 单声道自动复制 */
static float* wav_read_48k_stereo(const wchar_t* path, size_t* out_frames)
{
    size_t len = 0;
    uint8_t* b = read_whole_file(path, &len);
    if (!b) return NULL;
    if (len < 44 || memcmp(b, "RIFF", 4) != 0 || memcmp(b + 8, "WAVE", 4) != 0) {
        free(b); return NULL;
    }

    int channels = 0, bits = 0, fmt = 0, rate = 0;
    const uint8_t* data = NULL;
    size_t data_len = 0;

    size_t i = 12;
    while (i + 8 <= len) {
        const char* id = (const char*)(b + i);
        uint32_t sz;
        memcpy(&sz, b + i + 4, 4);
        const uint8_t* body = b + i + 8;
        if (i + 8 + sz > len) sz = (uint32_t)(len - i - 8);
        if (memcmp(id, "fmt ", 4) == 0 && sz >= 16) {
            uint16_t v16;
            memcpy(&v16, body + 0, 2); fmt = v16;
            memcpy(&v16, body + 2, 2); channels = v16;
            memcpy(&v16, body + 14, 2); bits = v16;
            uint32_t v32; memcpy(&v32, body + 4, 4); rate = (int)v32;
        } else if (memcmp(id, "data", 4) == 0) {
            data = body; data_len = sz;
        }
        i += 8 + sz + (sz & 1);
    }
    if (!data || channels <= 0 || rate <= 0) { free(b); return NULL; }

    size_t n_in_frames = 0;
    int bytes_per_sample = bits / 8;
    if (bytes_per_sample <= 0) { free(b); return NULL; }
    n_in_frames = data_len / ((size_t)channels * (size_t)bytes_per_sample);

    /* 先转成 float 平面 */
    float* src = (float*)xmalloc(sizeof(float) * n_in_frames * (size_t)channels);
    for (size_t k = 0; k < n_in_frames * (size_t)channels; k++) {
        float v = 0.0f;
        if (fmt == 3 && bits == 32) {
            memcpy(&v, data + k * 4, 4);
        } else if (bits == 16) {
            int16_t s; memcpy(&s, data + k * 2, 2); v = (float)s / 32768.0f;
        } else if (bits == 24) {
            const uint8_t* p = data + k * 3;
            int32_t s = (int32_t)((uint32_t)p[0] << 8 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 24) >> 8;
            v = (float)s / 8388608.0f;
        } else if (bits == 32) {
            int32_t s; memcpy(&s, data + k * 4, 4); v = (float)((double)s / 2147483648.0);
        }
        src[k] = v;
    }
    free(b);

    /* 线性重采样到 48k */
    double ratio = (double)rate / (double)SAMPLE_RATE;
    size_t n_out = (size_t)((double)n_in_frames / ratio);
    if (n_out < 1) { free(src); return NULL; }

    float* out = (float*)xmalloc(sizeof(float) * n_out * (size_t)CHANNELS);
    for (size_t k = 0; k < n_out; k++) {
        double pos = (double)k * ratio;
        size_t i0 = (size_t)pos;
        double frac = pos - (double)i0;
        size_t i1 = i0 + 1 < n_in_frames ? i0 + 1 : i0;
        for (int c = 0; c < CHANNELS; c++) {
            int sc = c < channels ? c : 0;
            float a = src[i0 * (size_t)channels + (size_t)sc];
            float bb = src[i1 * (size_t)channels + (size_t)sc];
            out[k * CHANNELS + c] = (float)(a + (bb - a) * frac);
        }
    }
    free(src);
    *out_frames = n_out;
    return out;
}

/* ---------------------------------------------------------------------------
 * 6. 提示词模板 + 内置音色 (数据由 tools/gen_prompt_c.py 从 manifest 生成)
 * -------------------------------------------------------------------------*/
/* ---- prompt templates (from browser_poc_manifest.json) ---- */
static const int32_t k_user_prefix[12] = {
    600, 289, 10356, 13, 10356, 10425, 1860, 4546, 12907, 10363, 13325, 11492,
};

static const int32_t k_after_ref[56] = {
    10356, 10425, 3965, 7738, 11492, 505, 587, 10356, 10425, 352, 500, 856, 11492, 505, 587, 10356,
    10425, 2128, 1247, 594, 11492, 505, 587, 10356, 10425, 348, 909, 561, 3648, 11492, 505, 587,
    10356, 10425, 2818, 1305, 355, 348, 909, 11492, 505, 587, 10356, 10425, 484, 339, 10367, 783,
    11492, 505, 587, 10356, 10425, 2427, 980, 11492,
};

static const int32_t k_asst_prefix[6] = {
    10356, 14, 5, 4, 8165, 430,
};

/* ---- 内嵌默认音色 "神里绫华" (用户自备参考音频), 104 frames x 16 codebooks ----
 * 这是 manifest 缺失时的兜底, 与 manifest 里同名条目内容一致 */
static const int16_t k_ref_codes_ayaka[1664] = {
    938, 415, 898, 792, 2, 43, 114, 218, 205, 304, 844, 286, 546, 24, 129, 855,
    600, 294, 341, 630, 469, 618, 136, 116, 551, 188, 69, 508, 811, 822, 845, 217,
    853, 68, 936, 545, 365, 504, 932, 1014, 862, 810, 170, 526, 134, 961, 523, 121,
    471, 532, 429, 127, 331, 544, 320, 814, 452, 782, 90, 846, 558, 227, 560, 762,
    263, 285, 220, 323, 922, 534, 538, 469, 919, 100, 340, 251, 318, 572, 358, 350,
    777, 855, 469, 179, 589, 392, 272, 716, 349, 113, 213, 392, 1019, 215, 775, 652,
    118, 548, 228, 361, 373, 312, 46, 358, 83, 1007, 608, 917, 372, 649, 869, 477,
    106, 695, 339, 746, 212, 593, 193, 810, 541, 721, 541, 632, 363, 904, 79, 394,
    666, 17, 819, 25, 337, 3, 101, 243, 462, 143, 248, 123, 388, 444, 996, 933,
    948, 589, 264, 739, 331, 738, 739, 975, 597, 826, 518, 600, 855, 2, 751, 404,
    531, 221, 332, 240, 223, 1007, 642, 987, 764, 521, 309, 749, 480, 534, 1017, 604,
    441, 176, 549, 103, 137, 512, 129, 753, 225, 903, 999, 58, 703, 624, 587, 660,
    4, 652, 26, 171, 405, 370, 116, 677, 341, 718, 143, 310, 967, 390, 639, 673,
    156, 761, 772, 55, 243, 826, 502, 132, 911, 366, 844, 463, 71, 465, 945, 798,
    404, 1013, 873, 577, 217, 303, 874, 346, 520, 113, 41, 740, 144, 277, 920, 656,
    358, 873, 142, 55, 392, 338, 178, 629, 479, 767, 372, 203, 900, 634, 262, 888,
    365, 35, 572, 362, 122, 919, 527, 452, 867, 1007, 734, 106, 912, 254, 63, 502,
    825, 258, 474, 84, 321, 276, 981, 1006, 743, 300, 561, 96, 659, 136, 333, 799,
    233, 841, 672, 302, 334, 180, 124, 2, 170, 675, 634, 447, 244, 85, 888, 367,
    233, 355, 474, 241, 574, 180, 740, 181, 816, 834, 634, 687, 392, 570, 32, 91,
    24, 821, 42, 84, 735, 307, 738, 540, 567, 536, 459, 248, 256, 908, 17, 954,
    988, 87, 663, 631, 177, 355, 309, 321, 978, 663, 244, 849, 461, 794, 635, 503,
    679, 905, 102, 95, 698, 809, 947, 235, 166, 225, 945, 640, 641, 422, 956, 29,
    852, 643, 3, 516, 1015, 626, 692, 815, 787, 965, 641, 213, 1003, 749, 634, 244,
    722, 418, 682, 740, 865, 756, 551, 1017, 391, 983, 156, 533, 283, 448, 415, 681,
    303, 766, 70, 769, 953, 151, 766, 397, 611, 324, 48, 210, 678, 153, 243, 158,
    50, 982, 144, 465, 180, 163, 790, 1004, 790, 577, 95, 1012, 573, 454, 1000, 203,
    276, 76, 20, 514, 661, 148, 993, 453, 100, 139, 390, 799, 281, 762, 268, 960,
    850, 48, 42, 111, 227, 818, 500, 373, 997, 746, 737, 255, 625, 906, 837, 517,
    764, 964, 211, 271, 124, 720, 847, 906, 854, 923, 1023, 853, 913, 558, 293, 388,
    210, 425, 362, 546, 761, 373, 276, 73, 801, 994, 373, 25, 222, 316, 185, 931,
    501, 955, 507, 738, 981, 730, 994, 39, 859, 637, 56, 72, 931, 886, 624, 527,
    660, 180, 945, 758, 287, 617, 180, 74, 211, 663, 836, 620, 514, 1022, 662, 54,
    636, 816, 728, 940, 983, 505, 252, 601, 106, 158, 457, 561, 954, 60, 1002, 255,
    345, 154, 986, 96, 408, 848, 729, 119, 798, 779, 96, 444, 790, 456, 756, 237,
    125, 205, 857, 703, 733, 844, 547, 954, 631, 114, 61, 44, 280, 967, 318, 848,
    460, 773, 455, 177, 627, 82, 261, 251, 763, 368, 343, 1010, 276, 487, 679, 29,
    106, 703, 266, 457, 586, 356, 157, 108, 376, 914, 842, 8, 959, 546, 546, 907,
    641, 473, 406, 480, 374, 40, 178, 213, 346, 216, 635, 722, 525, 732, 872, 77,
    9, 946, 560, 394, 204, 693, 824, 979, 44, 372, 794, 948, 159, 625, 962, 32,
    651, 499, 601, 874, 896, 43, 477, 971, 814, 460, 13, 140, 718, 939, 26, 46,
    636, 342, 515, 500, 622, 752, 647, 236, 341, 915, 423, 194, 268, 351, 83, 681,
    648, 731, 787, 796, 498, 284, 34, 756, 848, 868, 790, 471, 832, 56, 57, 912,
    617, 480, 107, 395, 212, 790, 129, 586, 685, 391, 525, 781, 277, 275, 499, 442,
    156, 416, 360, 250, 667, 119, 358, 844, 324, 91, 135, 961, 383, 1015, 568, 1005,
    305, 870, 122, 331, 159, 790, 16, 371, 965, 667, 438, 874, 960, 439, 442, 733,
    365, 700, 735, 139, 271, 146, 731, 269, 164, 786, 69, 529, 693, 427, 651, 6,
    1002, 666, 474, 989, 159, 30, 834, 881, 304, 135, 495, 385, 108, 251, 744, 223,
    233, 355, 474, 302, 922, 406, 417, 404, 810, 623, 723, 1003, 701, 748, 596, 536,
    233, 304, 569, 148, 574, 406, 740, 431, 561, 334, 68, 324, 542, 1010, 272, 263,
    233, 355, 446, 148, 334, 180, 740, 611, 799, 623, 1019, 533, 629, 347, 0, 49,
    233, 355, 320, 815, 96, 406, 740, 626, 171, 623, 551, 273, 112, 978, 655, 438,
    233, 355, 446, 341, 334, 180, 124, 611, 759, 334, 1019, 828, 112, 347, 1012, 467,
    233, 355, 446, 815, 334, 406, 793, 2, 759, 623, 501, 533, 870, 347, 272, 49,
    233, 355, 446, 815, 334, 406, 740, 2, 171, 623, 957, 273, 826, 446, 272, 438,
    233, 355, 446, 241, 334, 573, 793, 407, 397, 623, 923, 962, 826, 513, 586, 767,
    633, 850, 260, 431, 751, 866, 103, 1020, 52, 308, 648, 984, 831, 277, 104, 93,
    67, 694, 74, 445, 785, 40, 1008, 787, 564, 560, 982, 614, 302, 960, 25, 518,
    9, 62, 546, 423, 0, 568, 443, 399, 987, 823, 368, 619, 768, 244, 157, 589,
    67, 720, 685, 445, 673, 613, 753, 243, 385, 11, 203, 659, 248, 860, 182, 76,
    286, 171, 382, 868, 637, 917, 731, 328, 110, 907, 671, 335, 571, 310, 188, 507,
    429, 598, 648, 233, 130, 49, 443, 447, 306, 336, 363, 106, 372, 732, 493, 306,
    180, 715, 576, 740, 699, 542, 889, 630, 985, 152, 829, 706, 124, 508, 629, 179,
    246, 192, 333, 137, 264, 447, 690, 687, 829, 628, 338, 745, 956, 490, 75, 417,
    404, 943, 743, 794, 332, 451, 835, 846, 211, 106, 986, 684, 795, 463, 362, 966,
    796, 480, 107, 775, 420, 597, 94, 320, 227, 35, 736, 148, 465, 678, 322, 999,
    61, 355, 725, 759, 90, 615, 95, 131, 820, 606, 669, 1003, 206, 602, 407, 799,
    233, 355, 474, 241, 909, 302, 885, 346, 397, 166, 1019, 854, 493, 570, 799, 744,
    233, 355, 219, 815, 909, 721, 124, 123, 799, 388, 923, 687, 533, 570, 1012, 910,
    233, 355, 42, 131, 420, 406, 793, 901, 331, 675, 68, 687, 870, 54, 914, 323,
    10, 281, 736, 39, 472, 659, 175, 625, 643, 678, 155, 526, 771, 748, 495, 651,
    922, 28, 169, 306, 889, 298, 480, 170, 992, 980, 516, 785, 844, 589, 331, 758,
    621, 53, 982, 592, 70, 525, 738, 1011, 551, 356, 480, 294, 877, 292, 69, 702,
    951, 715, 101, 197, 705, 202, 607, 365, 76, 159, 167, 618, 274, 843, 554, 830,
    653, 705, 979, 1011, 619, 569, 43, 709, 382, 643, 17, 342, 155, 451, 41, 934,
    852, 8, 978, 1005, 736, 656, 779, 657, 864, 388, 282, 345, 664, 222, 593, 1011,
    634, 884, 701, 778, 147, 540, 926, 986, 105, 318, 441, 901, 610, 751, 86, 1021,
    68, 705, 23, 1018, 763, 359, 374, 638, 1, 346, 2, 801, 249, 593, 843, 945,
    126, 604, 742, 648, 655, 745, 354, 798, 275, 581, 131, 458, 12, 5, 801, 694,
    782, 973, 47, 261, 794, 947, 697, 367, 546, 423, 165, 504, 883, 898, 270, 79,
    818, 712, 106, 645, 590, 784, 986, 107, 762, 305, 488, 679, 850, 229, 258, 649,
    838, 716, 926, 879, 886, 843, 226, 420, 252, 331, 723, 264, 657, 441, 179, 578,
    38, 776, 727, 484, 933, 400, 206, 130, 126, 686, 814, 505, 877, 282, 778, 645,
    876, 776, 626, 760, 590, 932, 597, 601, 505, 177, 647, 473, 557, 416, 250, 567,
    156, 252, 750, 37, 410, 511, 570, 908, 293, 281, 105, 453, 280, 48, 590, 35,
    404, 191, 268, 872, 437, 573, 560, 447, 51, 201, 30, 834, 216, 904, 540, 547,
    288, 35, 749, 700, 239, 58, 500, 377, 98, 980, 22, 216, 182, 823, 172, 804,
    796, 666, 448, 216, 668, 347, 848, 531, 230, 828, 255, 931, 788, 220, 580, 271,
    233, 258, 474, 815, 686, 721, 303, 1014, 397, 290, 1019, 0, 392, 689, 889, 238,
    233, 666, 474, 302, 574, 406, 124, 2, 561, 649, 68, 324, 650, 85, 988, 603,
    922, 737, 217, 263, 315, 87, 377, 259, 150, 382, 129, 647, 820, 24, 189, 564,
    139, 538, 679, 322, 120, 998, 999, 236, 79, 687, 7, 761, 719, 274, 811, 468,
    102, 784, 3, 151, 925, 78, 1000, 178, 560, 668, 584, 485, 944, 386, 153, 860,
    128, 312, 180, 800, 307, 109, 803, 270, 406, 401, 678, 104, 107, 447, 813, 370,
    550, 357, 680, 54, 161, 664, 465, 866, 450, 360, 348, 674, 248, 563, 983, 108,
    852, 43, 267, 559, 322, 151, 955, 168, 419, 525, 414, 411, 769, 875, 804, 110,
    51, 811, 357, 947, 207, 133, 922, 271, 909, 29, 330, 642, 60, 228, 116, 639,
    843, 915, 729, 241, 928, 206, 883, 105, 703, 537, 787, 1012, 479, 557, 49, 911,
    168, 34, 74, 40, 852, 123, 533, 619, 809, 398, 991, 876, 582, 60, 403, 92,
    201, 221, 47, 682, 438, 375, 276, 875, 187, 449, 1007, 401, 4, 908, 423, 797,
    918, 852, 919, 822, 31, 960, 162, 108, 81, 856, 882, 952, 37, 1013, 936, 214,
    779, 547, 178, 781, 336, 776, 362, 452, 80, 384, 719, 677, 46, 295, 339, 432,
    995, 679, 344, 105, 785, 600, 117, 835, 249, 651, 916, 4, 945, 146, 446, 244,
    288, 870, 382, 956, 178, 826, 101, 430, 949, 868, 898, 694, 95, 202, 736, 717,
};
#define REF_FRAMES_AYAKA 104

/* ---------------------------------------------------------------------------
 * 7. 推理
 * -------------------------------------------------------------------------*/
#ifndef MOSS_SD_BATCH
#define MOSS_SD_BATCH 16
#endif
#define SD_N_ATTN  12
/* 各注意力层的上下文长度, 与 codec_browser_onnx_meta.json 的 streaming_decode 一致 */
static const int k_sd_ctx[SD_N_ATTN] = { 500, 500, 500, 500, 800, 800, 1200, 1200, 1600, 1600, 1600, 1600 };

typedef struct {
    Session prefill;
    Session decode;
    Session frame;
    Session enc;      /* codec encode: 只在 --ref-wav 时才加载 */
    Session dec;      /* codec decode_full */
    wchar_t dir[MAX_PATH];
    int threads;
} Models;

static void models_close(Models* m)
{
    Session* all[5] = { &m->prefill, &m->decode, &m->frame, &m->dec, &m->enc };
    for (int k = 0; k < 5; k++)
        if (all[k]->s) { ort->ReleaseSession(all[k]->s); all[k]->s = NULL; }
}

typedef struct {
    Session* S;
    const wchar_t* dir;
    const wchar_t* file;
    int threads;
    uint64_t* t_out;
    int no_arena;
} LoadJob;

static DWORD WINAPI load_worker(LPVOID p)
{
    LoadJob* j = (LoadJob*)p;
    sess_open_ex(j->S, j->dir, j->file, j->threads, j->t_out, j->no_arena);
    return 0;
}

/* models/ 里放了 <name>.int8.onnx 就优先加载 (--fp32 强制原始精度)。
 * 结果写入调用方给的缓冲 (并行加载时不能共用静态块)。 */
static const wchar_t* pick_model(wchar_t* out, int cap, const wchar_t* dir,
                                 const wchar_t* base, int force_fp32)
{
    if (force_fp32) return base;
    wchar_t path[MAX_PATH * 2];
    const wchar_t* dot = wcsrchr(base, L'.');
    if (dot) _snwprintf(path, MAX_PATH * 2, L"%ls\\%.*ls.int8.onnx", dir, (int)(dot - base), base);
    else     _snwprintf(path, MAX_PATH * 2, L"%ls\\%ls.int8.onnx", dir, base);
    if (GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES) {
        const wchar_t* slash = wcsrchr(path, L'\\');
        wcsncpy(out, slash ? slash + 1 : path, (size_t)cap - 1);
        out[cap - 1] = 0;
        return out;
    }
    return base;
}


static void models_open(Models* m, const wchar_t* dir, int threads, int force_fp32)
{
    memset(m, 0, sizeof(*m));
    ort_load();

    /* --- prefill: input_ids[1,S,17], attention_mask[1,S] -> hidden + 12*(k,v) --- */
    m->prefill.n_in = 2;
    strcpy(m->prefill.in_name[0], "input_ids");
    strcpy(m->prefill.in_name[1], "attention_mask");
    m->prefill.n_out = 1 + GLOBAL_LAYERS * 2;
    strcpy(m->prefill.out_name[0], "global_hidden");
    for (int i = 0; i < GLOBAL_LAYERS; i++) {
        sprintf(m->prefill.out_name[1 + i * 2], "present_key_%d", i);
        sprintf(m->prefill.out_name[2 + i * 2], "present_value_%d", i);
    }

    /* --- decode_step: 同 prefill 的输出, 加上 past_* --- */
    m->decode.n_in = 2 + GLOBAL_LAYERS * 2;
    strcpy(m->decode.in_name[0], "input_ids");
    strcpy(m->decode.in_name[1], "past_valid_lengths");
    for (int i = 0; i < GLOBAL_LAYERS; i++) {
        sprintf(m->decode.in_name[2 + i * 2], "past_key_%d", i);
        sprintf(m->decode.in_name[3 + i * 2], "past_value_%d", i);
    }
    m->decode.n_out = 1 + GLOBAL_LAYERS * 2;
    strcpy(m->decode.out_name[0], "global_hidden");
    for (int i = 0; i < GLOBAL_LAYERS; i++) {
        sprintf(m->decode.out_name[1 + i * 2], "present_key_%d", i);
        sprintf(m->decode.out_name[2 + i * 2], "present_value_%d", i);
    }

    /* --- local_fixed_sampled_frame --- */
    m->frame.n_in = 4;
    strcpy(m->frame.in_name[0], "global_hidden");
    strcpy(m->frame.in_name[1], "repetition_seen_mask");
    strcpy(m->frame.in_name[2], "assistant_random_u");
    strcpy(m->frame.in_name[3], "audio_random_u");
    m->frame.n_out = 2;
    strcpy(m->frame.out_name[0], "should_continue");
    strcpy(m->frame.out_name[1], "frame_token_ids");

    /* --- codec --- */
    m->enc.n_in = 2;
    strcpy(m->enc.in_name[0], "waveform");
    strcpy(m->enc.in_name[1], "input_lengths");
    m->enc.n_out = 2;
    strcpy(m->enc.out_name[0], "audio_codes");
    strcpy(m->enc.out_name[1], "audio_code_lengths");

    /* 流式解码: 2 个业务输入 + 4 个 transformer offset + 12 组注意力缓存 */
    {
        int n = 0;
        strcpy(m->dec.in_name[n++], "audio_codes");
        strcpy(m->dec.in_name[n++], "audio_code_lengths");
        for (int i = 0; i < 4; i++) sprintf(m->dec.in_name[n++], "transformer_offset_%d", i);
        for (int i = 0; i < SD_N_ATTN; i++) {
            sprintf(m->dec.in_name[n++], "attn_offset_%d", i);
            sprintf(m->dec.in_name[n++], "attn_cached_keys_%d", i);
            sprintf(m->dec.in_name[n++], "attn_cached_values_%d", i);
            sprintf(m->dec.in_name[n++], "attn_cached_positions_%d", i);
        }
        m->dec.n_in = (size_t)n;

        n = 0;
        strcpy(m->dec.out_name[n++], "audio");
        strcpy(m->dec.out_name[n++], "audio_lengths");
        for (int i = 0; i < 4; i++) sprintf(m->dec.out_name[n++], "transformer_offset_out_%d", i);
        for (int i = 0; i < SD_N_ATTN; i++) {
            sprintf(m->dec.out_name[n++], "attn_offset_out_%d", i);
            sprintf(m->dec.out_name[n++], "attn_cached_keys_out_%d", i);
            sprintf(m->dec.out_name[n++], "attn_cached_values_out_%d", i);
            sprintf(m->dec.out_name[n++], "attn_cached_positions_out_%d", i);
        }
        m->dec.n_out = (size_t)n;
    }

    for (int k = 0; k < 64; k++) {
        m->prefill.in_ptr[k] = m->prefill.in_name[k];
        m->prefill.out_ptr[k] = m->prefill.out_name[k];
        m->decode.in_ptr[k] = m->decode.in_name[k];
        m->decode.out_ptr[k] = m->decode.out_name[k];
        m->frame.in_ptr[k] = m->frame.in_name[k];
        m->frame.out_ptr[k] = m->frame.out_name[k];
        m->enc.in_ptr[k] = m->enc.in_name[k];
        m->enc.out_ptr[k] = m->enc.out_name[k];
        m->dec.in_ptr[k] = m->dec.in_name[k];
        m->dec.out_ptr[k] = m->dec.out_name[k];
    }

    /* 四个 session 互不依赖, 并行加载 (加载阶段是图优化 + I/O 混合, 单线程很浪费) */
    {
        wchar_t dec_file[MAX_PATH], frame_file[MAX_PATH];
        LoadJob jobs[4] = {
            { &m->prefill, dir, L"moss_tts_prefill.onnx",                    threads, &g_stats.load_prefill, 0 },
            { &m->decode,  dir, pick_model(dec_file, MAX_PATH, dir, L"moss_tts_decode_step.onnx", force_fp32), threads, &g_stats.load_decode, 0 },
            { &m->frame,   dir, pick_model(frame_file, MAX_PATH, dir, L"moss_tts_local_fixed_sampled_frame.onnx", force_fp32), threads, &g_stats.load_frame, 0 },
            { &m->dec,     dir, L"moss_audio_tokenizer_decode_step.onnx",    threads, &g_stats.load_dec, 0 },
        };
        HANDLE th[4];
        int started = 0;
        for (int k = 0; k < 4; k++) {
            th[k] = CreateThread(NULL, 0, load_worker, &jobs[k], 0, NULL);
            if (th[k]) started++;
        }
        if (started == 4) {
            WaitForMultipleObjects(4, th, TRUE, INFINITE);
            for (int k = 0; k < 4; k++) CloseHandle(th[k]);
        } else {
            /* 线程创建失败就退回串行 */
            for (int k = 0; k < started; k++) { WaitForSingleObject(th[k], INFINITE); CloseHandle(th[k]); }
            for (int k = started; k < 4; k++) load_worker(&jobs[k]);
        }
    }

    /* codec_encode 只在做声音克隆时用, 延迟到第一次需要时再加载 (省 4s + 44MB) */
    wcsncpy(m->dir, dir, MAX_PATH - 1);
    m->threads = threads;
}

/* 语音结束检测 (对应 Rust EndOfSpeechDetector, 但窗口从 8 帧放宽到 48 帧)。
 *
 * 原实现用 8 帧 (0.64s) 判断第一通道重复率, 会把句子之间的停顿误判成结束:
 * 实测一句 50 字的中文在 step 31 就被截断 (模型其实还有 100 多帧要说),
 * 英文 "Hello world." 后面也有近 2 秒的自然停顿。
 * 48 帧 = 3.8 秒, 正常停顿远达不到, 而真正说完之后的静音会一直持续下去,
 * 所以既能自动收尾, 又不会把话说一半就掐掉。 */
typedef struct {
    int32_t frames[EOS_WINDOW][N_VQ];
    int n;
} EosDetector;

static int eos_check(EosDetector* d, const int32_t* frame, int step)
{
    (void)step;
    if (d->n < EOS_WINDOW) {
        memcpy(d->frames[d->n], frame, sizeof(int32_t) * N_VQ);
        d->n++;
    } else {
        memmove(d->frames[0], d->frames[1], sizeof(d->frames[0]) * (EOS_WINDOW - 1));
        memcpy(d->frames[EOS_WINDOW - 1], frame, sizeof(int32_t) * N_VQ);
    }
    if (d->n < EOS_WINDOW) return 0;

    /* 1) 整窗完全一致: 模型卡死在同一个 token 上 */
    int same = 1;
    for (int k = 1; k < EOS_WINDOW && same; k++)
        if (memcmp(d->frames[k], d->frames[0], sizeof(int32_t) * N_VQ) != 0) same = 0;
    if (same) return 1;

    /* 2) 第一通道重复率 > 0.75: 静音段的典型特征 */
    int dom = d->frames[0][0], cnt = 0;
    for (int k = 0; k < EOS_WINDOW; k++) if (d->frames[k][0] == dom) cnt++;
    if ((float)cnt / (float)EOS_WINDOW > 0.75f) return 1;

    /* 3) 第一通道只有 <= 2 种取值: 来回抖动, 同样视为静音 */
    int uniq[EOS_WINDOW], nu = 0;
    for (int k = 0; k < EOS_WINDOW; k++) {
        int seen = 0;
        for (int j = 0; j < nu; j++) if (uniq[j] == d->frames[k][0]) { seen = 1; break; }
        if (!seen) uniq[nu++] = d->frames[k][0];
    }
    if (nu <= 2) return 1;

    return 0;
}

/* ---------------------------------------------------------------------------
 * 流式 codec 解码器 (moss_audio_tokenizer_decode_step.onnx)
 *
 * 原先用 decode_full 一次性解码整段, 中间张量随音频长度线性膨胀 —— 实测 27 秒
 * 音频要 ~1.9GB。这里换成带固定 KV 缓存的逐批解码: 缓存总量恒定 12.4MB, 与
 * 音频长度无关; 而且边生成边解码, 省掉了最后那次整体解码。
 * -------------------------------------------------------------------------*/
typedef struct {
    OrtValue* off[4];
    OrtValue* aoff[SD_N_ATTN];
    OrtValue* k[SD_N_ATTN];
    OrtValue* v[SD_N_ATTN];
    OrtValue* p[SD_N_ATTN];
    int32_t   off_buf[4];
    int32_t   aoff_buf[SD_N_ATTN];
    float*    kv_own[SD_N_ATTN * 2];   /* 初始零缓存, 首次调用后即被 ORT 的输出替换 */
    int32_t*  pos_own[SD_N_ATTN];
    float*    pcm;                     /* 累积的交错音频 */
    size_t    n_samples;               /* 每声道采样点数 */
    size_t    cap;
    double    last_energy;             /* 最近一批的均方能量 */
    /* 每批音频解码完成后的回调 (流式播放挂在这里), energy 传给回调自己判 */
    void    (*cb)(void* ctx, const float* interleaved, size_t n, double energy);
    void*     cb_ctx;
} StreamDec;

static void sd_free(StreamDec* sd)
{
    for (int i = 0; i < 4; i++) if (sd->off[i]) ort->ReleaseValue(sd->off[i]);
    for (int i = 0; i < SD_N_ATTN; i++) {
        if (sd->aoff[i]) ort->ReleaseValue(sd->aoff[i]);
        if (sd->k[i])    ort->ReleaseValue(sd->k[i]);
        if (sd->v[i])    ort->ReleaseValue(sd->v[i]);
        if (sd->p[i])    ort->ReleaseValue(sd->p[i]);
        free(sd->kv_own[i * 2]);
        free(sd->kv_own[i * 2 + 1]);
        free(sd->pos_own[i]);
    }
    free(sd->pcm);
    memset(sd, 0, sizeof(*sd));
}

static void sd_init(StreamDec* sd)
{
    memset(sd, 0, sizeof(*sd));
    for (int i = 0; i < 4; i++) {
        int64_t sh[1] = { 1 };
        sd->off_buf[i] = 0;
        sd->off[i] = MK_I32(&sd->off_buf[i], sh, 1);
    }
    for (int i = 0; i < SD_N_ATTN; i++) {
        int ctx = k_sd_ctx[i];
        int64_t sh_off[1] = { 1 };
        int64_t sh_kv[4]  = { 1, 4, ctx, 64 };
        int64_t sh_pos[2] = { 1, ctx };
        sd->aoff_buf[i] = 0;
        sd->aoff[i] = MK_I32(&sd->aoff_buf[i], sh_off, 1);
        float*   kb = (float*)xcalloc((size_t)4 * (size_t)ctx * 64, sizeof(float));
        float*   vb = (float*)xcalloc((size_t)4 * (size_t)ctx * 64, sizeof(float));
        int32_t* pb = (int32_t*)xmalloc(sizeof(int32_t) * (size_t)ctx);
        for (int t = 0; t < ctx; t++) pb[t] = -1;   /* 未使用的位置填 -1 */
        sd->kv_own[i * 2] = kb;
        sd->kv_own[i * 2 + 1] = vb;
        sd->pos_own[i] = pb;
        sd->k[i] = MK_F32(kb, sh_kv, 4);
        sd->v[i] = MK_F32(vb, sh_kv, 4);
        sd->p[i] = MK_I32(pb, sh_pos, 2);
    }
}

static void sd_append(StreamDec* sd, const float* interleaved, size_t n)
{
    if (sd->n_samples + n > sd->cap) {
        size_t nc = sd->cap ? sd->cap : 65536;
        while (nc < sd->n_samples + n) nc *= 2;
        float* p = (float*)realloc(sd->pcm, sizeof(float) * nc * CHANNELS);
        if (!p) die("out of memory");
        sd->pcm = p;
        sd->cap = nc;
    }
    memcpy(sd->pcm + sd->n_samples * CHANNELS, interleaved, sizeof(float) * n * CHANNELS);
    sd->n_samples += n;
}

/* 解码 n 帧 (n*3840 个采样点), 结果追加到 sd->pcm */
static void sd_run(Models* m, StreamDec* sd, const int16_t* codes, size_t n)
{
    if (n == 0) return;

    int32_t* ac = (int32_t*)xmalloc(sizeof(int32_t) * n * N_VQ);
    for (size_t k = 0; k < n * N_VQ; k++) ac[k] = codes[k];
    int32_t len_buf = (int32_t)n;
    int64_t sh_codes[3] = { 1, (int64_t)n, N_VQ };
    int64_t sh_len[1] = { 1 };

    OrtValue* in[64] = {0};
    int p = 0;
    in[p++] = MK_I32(ac, sh_codes, 3);
    in[p++] = MK_I32(&len_buf, sh_len, 1);
    for (int i = 0; i < 4; i++) in[p++] = sd->off[i];
    for (int i = 0; i < SD_N_ATTN; i++) {
        in[p++] = sd->aoff[i];
        in[p++] = sd->k[i];
        in[p++] = sd->v[i];
        in[p++] = sd->p[i];
    }

    OrtValue* out[64] = {0};
    sess_run(&m->dec, in, out);

    size_t cnt = 0, alen = 0;
    float*   au = (float*)tensor_data(out[0], &cnt);
    int32_t* al = (int32_t*)tensor_data(out[1], &alen);
    size_t per = (al && alen > 0 && al[0] > 0) ? (size_t)al[0] : 0;
    if (per == 0 || cnt < per * CHANNELS) die("stream decode returned no audio");

    float* inter = (float*)xmalloc(sizeof(float) * per * CHANNELS);
    /* 只用尾部 8 帧 (0.64s) 的能量判断: 整批平均会被批次中间的短促音拉高,
     * 导致"已经说完、后面只是零星杂音"时迟迟停不下来。 */
    size_t tail = 8 * DOWNSAMPLE;
    if (tail > per) tail = per;
    double acc = 0.0;
    for (size_t t = 0; t < per; t++) {
        for (int c = 0; c < CHANNELS; c++) {
            float v = au[c * per + t];
            inter[t * CHANNELS + c] = v;
            if (t + tail >= per) acc += (double)v * (double)v;
        }
    }
    sd->last_energy = acc / (double)(tail * CHANNELS);
    sd_append(sd, inter, per);
    if (sd->cb) sd->cb(sd->cb_ctx, inter, per, sd->last_energy);
    free(inter);

    /* 状态交替换入: 输出直接作为下一次的输入 */
    ort->ReleaseValue(out[0]);
    ort->ReleaseValue(out[1]);
    ort->ReleaseValue(in[0]);
    ort->ReleaseValue(in[1]);
    for (int i = 0; i < 4; i++) ort->ReleaseValue(sd->off[i]);
    for (int i = 0; i < SD_N_ATTN; i++) {
        ort->ReleaseValue(sd->aoff[i]);
        ort->ReleaseValue(sd->k[i]);
        ort->ReleaseValue(sd->v[i]);
        ort->ReleaseValue(sd->p[i]);
    }
    p = 2;
    for (int i = 0; i < 4; i++) sd->off[i] = out[p++];
    for (int i = 0; i < SD_N_ATTN; i++) {
        sd->aoff[i] = out[p++];
        sd->k[i] = out[p++];
        sd->v[i] = out[p++];
        sd->p[i] = out[p++];
    }
    free(ac);
}

/* ---------------------------------------------------------------------------
 * 流式播放 (waveOut)
 *
 * 合成比实时快 (RTF<1), 边生成边把每批音频推给声卡; 8 块环形缓冲约 4 秒,
 * 播放进度反过来给生成背压。两个保护:
 *   - 开播预缓冲: 请求开始先攒 ~1s 音频再出声。直接首播的话队列没有余量,
 *     OS 调度抖一下/chunk 交界的 prefill 断流就会欠载, 听感是开头吃字、
 *     说半句断一下, 攒一会儿又恢复
 *   - 静音扣留 (hold): 静音批次先扣住, 后续有声批次按原顺序补播 (句中停顿
 *     照常保留), 请求结束整体丢弃。静音阈值取文件路径尾部裁剪同一档
 *     (RMS 0.01), 模型结尾吐的低幅噪声/稀疏脉冲就不会被播出去
 * -------------------------------------------------------------------------*/
#define PL_NBUF     8
#define PL_BUF_SMPL (SAMPLE_RATE / 2)     /* 每块 0.5s */
#define PL_PRE_SMPL (SAMPLE_RATE * 3 / 5) /* 开播前先攒 0.6s */
#define PL_SILENT_E 1.0e-4                /* 静音判据 (与文件尾部裁剪一致) */

typedef struct {
    HWAVEOUT ho;
    WAVEHDR  hdr[PL_NBUF];
    int16_t* buf[PL_NBUF];
    int      used[PL_NBUF];
    float*   hold;                        /* 扣住的静音批次 */
    size_t   hold_n, hold_cap;
    float*   pre;                         /* 开播前的预缓冲 */
    size_t   pre_n, pre_cap;
    int      prebuffering;
    int      live;
    uint64_t t_first;                     /* 首批音频出声时刻 */
    size_t   emitted;                     /* 本请求实际推给声卡的采样数 */
    size_t   emitted_total;               /* 设备生命周期累计 (算队列深度用, 不重置) */
    size_t   last_bs[2];                  /* 最近两批的采样数 (自适应扣留) */
    int      nbs;
} Player;

static void player_emit(Player* p, const float* inter, size_t n);

static void player_open(Player* p, int allow_device)
{
    memset(p, 0, sizeof(*p));
    p->prebuffering = 1;            /* 开播前先攒 ~1s */
    if (!allow_device) return;      /* --no-play: 只走回调计时, 不碰声卡 */
    WAVEFORMATEX wf;
    wf.wFormatTag = WAVE_FORMAT_PCM;
    wf.nChannels = CHANNELS;
    wf.nSamplesPerSec = SAMPLE_RATE;
    wf.wBitsPerSample = 16;
    wf.nBlockAlign = (WORD)(wf.nChannels * wf.wBitsPerSample / 8);
    wf.nAvgBytesPerSec = wf.nSamplesPerSec * wf.nBlockAlign;
    wf.cbSize = 0;
    if (waveOutOpen(&p->ho, WAVE_MAPPER, &wf, 0, 0, CALLBACK_NULL) != MMSYSERR_NOERROR) {
        p->ho = NULL;
        fprintf(stderr, "[warn] 打不开音频设备, 只合成不播放\n");
        return;
    }
    for (int i = 0; i < PL_NBUF; i++)
        p->buf[i] = (int16_t*)xmalloc(sizeof(int16_t) * PL_BUF_SMPL * CHANNELS);
    p->live = 1;
}

static void player_close(Player* p)
{
    if (!p->live) return;
    /* 等排队的音频播完再关 (上限 15s), 避免尾巴被掐掉 */
    for (int t = 0; t < 1500; t++) {
        int busy = 0;
        for (int i = 0; i < PL_NBUF; i++)
            if (p->hdr[i].dwFlags & WHDR_INQUEUE) { busy = 1; break; }
        if (!busy) break;
        Sleep(10);
    }
    waveOutReset(p->ho);
    for (int i = 0; i < PL_NBUF; i++) {
        if (p->used[i]) waveOutUnprepareHeader(p->ho, &p->hdr[i], sizeof(WAVEHDR));
        free(p->buf[i]);
    }
    waveOutClose(p->ho);
    free(p->hold);
    free(p->pre);
    p->live = 0;
}

static void player_begin(Player* p)
{
    p->hold_n = 0;
}

/* 新请求边界: 重置首字时刻与预缓冲 (t_first 在 push 时置位, 一次请求内跨 chunk 保持) */
static size_t player_queue_depth(Player* p);

static void player_request(Player* p)
{
    p->t_first = 0;
    p->pre_n = 0;
    p->emitted = 0;
    /* 队列里已有余量 (对话模式连播下一句) 就不再攒预缓冲, 句间更连贯;
     * 冷启动 (深度不足) 才需要开播保护 */
    p->prebuffering = p->live ? (player_queue_depth(p) < (size_t)(SAMPLE_RATE / 2)) : 1;
    p->nbs = 0;
    p->last_bs[0] = p->last_bs[1] = 0;
}

/* 与文件输出同一套尾部裁剪: 最后一个 20ms 窗 RMS>1e-4 之后保留 0.2s */
static size_t player_trim_tail(const float* pcm, size_t samples)
{
    const size_t win = SAMPLE_RATE / 50;          /* 20ms */
    size_t last = 0;
    for (size_t w = samples; w >= win; w -= win) {
        double acc = 0.0;
        for (size_t i = (w - win) * CHANNELS; i < w * CHANNELS; i++)
            acc += (double)pcm[i] * (double)pcm[i];
        if (acc / (double)(win * CHANNELS) > 1.0e-4) { last = w; break; }
    }
    if (last == 0) return 0;
    size_t keep = last + SAMPLE_RATE / 5;
    return keep < samples ? keep : samples;
}

static void pre_append(Player* p, const float* inter, size_t n)
{
    if (p->pre_n + n > p->pre_cap) {
        size_t nc = p->pre_cap ? p->pre_cap : 65536;
        while (nc < p->pre_n + n) nc *= 2;
        float* b = (float*)realloc(p->pre, sizeof(float) * nc * CHANNELS);
        if (!b) die("out of memory");
        p->pre = b;
        p->pre_cap = nc;
    }
    memcpy(p->pre + p->pre_n * CHANNELS, inter, sizeof(float) * n * CHANNELS);
    p->pre_n += n;
}

/* 声卡里还排着多少音频 (采样数) */
static size_t player_queue_depth(Player* p)
{
    if (!p->live) return 0;
    MMTIME mm;
    memset(&mm, 0, sizeof(mm));
    mm.wType = TIME_SAMPLES;
    if (waveOutGetPosition(p->ho, &mm, sizeof(mm)) != MMSYSERR_NOERROR) return 0;
    return p->emitted_total > (size_t)mm.u.sample
               ? p->emitted_total - (size_t)mm.u.sample : 0;
}

static void player_push(Player* p, const float* inter, size_t n, double energy)
{
    if (n == 0) return;
    int silent = energy < PL_SILENT_E;
    if (silent) {
        if (!p->live) return;
        if (p->hold_n + n > p->hold_cap) {
            size_t nc = p->hold_cap ? p->hold_cap : 65536;
            while (nc < p->hold_n + n) nc *= 2;
            float* b = (float*)realloc(p->hold, sizeof(float) * nc * CHANNELS);
            if (!b) die("out of memory");
            p->hold = b;
            p->hold_cap = nc;
        }
        memcpy(p->hold + p->hold_n * CHANNELS, inter, sizeof(float) * n * CHANNELS);
        p->hold_n += n;
        return;
    }
    /* 首个有声批次就绪的时刻 = 首字延迟 (与有没有音频设备无关) */
    if (!p->t_first) p->t_first = now_ms();
    if (!p->live) return;

    if (p->prebuffering) {
        /* 攒够 PL_PRE_SMPL 才开播; 期间扣住的静音一并按序进缓冲 */
        if (p->hold_n) {
            pre_append(p, p->hold, p->hold_n);
            p->hold_n = 0;
        }
        pre_append(p, inter, n);
        if (p->pre_n >= PL_PRE_SMPL) {
            p->prebuffering = 0;
            player_emit(p, p->pre, p->pre_n);
            p->pre_n = 0;
        }
        return;
    }

    /* 队列深度自适应扣留: 队列深 (生成远快于播放) -> 扣住最近两批,
     * 收尾时对最多 2.5s 的尾巴做窗口级裁剪, 拦住跨两批的结尾垃圾;
     * 队列浅 (机器忙) -> 立即放行, 宁可少裁也不制造断流/脱节。 */
    if (p->hold_n) {
        player_emit(p, p->hold, p->hold_n);
        p->hold_n = 0;
    }
    pre_append(p, inter, n);
    p->last_bs[1] = p->last_bs[0];
    p->last_bs[0] = n;
    if (p->nbs < 2) p->nbs++;

    size_t depth = player_queue_depth(p);
    int keep = depth >= (size_t)(SAMPLE_RATE * 3 / 2) ? 2
             : depth >= (size_t)(SAMPLE_RATE * 4 / 5) ? 1 : 0;
    if (keep > p->nbs) keep = p->nbs;
    size_t keep_smpl = 0;
    for (int k = 0; k < keep; k++) keep_smpl += p->last_bs[k];
    if (p->pre_n > keep_smpl) {
        size_t emit_n = p->pre_n - keep_smpl;
        player_emit(p, p->pre, emit_n);
        memmove(p->pre, p->pre + emit_n * CHANNELS,
                sizeof(float) * keep_smpl * CHANNELS);
        p->pre_n = keep_smpl;
    }
}

static void player_on_audio(void* ctx, const float* inter, size_t n, double energy)
{
    player_push((Player*)ctx, inter, n, energy);
}
static void player_emit(Player* p, const float* inter, size_t n)
{
    size_t off = 0;
    while (off < n) {
        size_t take = n - off;
        if (take > PL_BUF_SMPL) take = PL_BUF_SMPL;
        int i;
        for (;;) {
            for (i = 0; i < PL_NBUF; i++)
                if (!p->used[i] || (p->hdr[i].dwFlags & WHDR_DONE)) break;
            if (i < PL_NBUF) break;
            Sleep(5);
        }
        if (p->used[i])
            waveOutUnprepareHeader(p->ho, &p->hdr[i], sizeof(WAVEHDR));

        int16_t* dst = p->buf[i];
        const float* src = inter + off * CHANNELS;
        for (size_t s = 0; s < take * CHANNELS; s++) {
            float v = src[s] * 32767.0f;
            dst[s] = (int16_t)(v >= 32767.0f ? 32767 : v <= -32768.0f ? -32768 : v);
        }
        memset(&p->hdr[i], 0, sizeof(WAVEHDR));
        p->hdr[i].lpData = (LPSTR)dst;
        p->hdr[i].dwBufferLength = (DWORD)(take * CHANNELS * sizeof(int16_t));
        waveOutPrepareHeader(p->ho, &p->hdr[i], sizeof(WAVEHDR));
        waveOutWrite(p->ho, &p->hdr[i], sizeof(WAVEHDR));
        p->used[i] = 1;
        off += take;
    }
    p->emitted += n;
    p->emitted_total += n;
}

/* ---------------------------------------------------------------------------
 * codec 解码异步化: sd_run 只消费已生成的帧, 与主循环的 frame/decode_step
 * 无数据依赖 -> 挪到工作线程流水并行, 主循环每帧省 ~9ms (RTF 0.85 -> 0.74)。
 * 结束判据用的批能量由工作线程发布, 主线程读到的最多滞后一批 (16 帧静音,
 * 文件路径被尾部裁剪掐掉, 播放路径被 hold 丢弃, 语义不变)。
 * -------------------------------------------------------------------------*/
typedef struct {
    Models*   m;
    StreamDec sd;                        /* 工作线程独占 */
    CRITICAL_SECTION cs;
    HANDLE    wake;                      /* auto-reset: 有作业或要退出 */
    HANDLE    th;
    int16_t   job[MOSS_SD_BATCH * N_VQ]; /* 待取作业 (深度 1) */
    int       job_n;
    int       queued;
    int       stop;
    uint64_t* stat_ms;                   /* 解码耗时记账 */
    volatile LONG published;             /* 发布给主线程的连续静音批数 */
} Probe;

static DWORD WINAPI probe_worker(LPVOID p)
{
    Probe* pr = (Probe*)p;
    int consec = 0;
    for (;;) {
        WaitForSingleObject(pr->wake, INFINITE);
        int16_t buf[MOSS_SD_BATCH * N_VQ];
        int n = 0, stop = 0;
        EnterCriticalSection(&pr->cs);
        if (pr->queued) {
            memcpy(buf, pr->job, sizeof(int16_t) * (size_t)pr->job_n * N_VQ);
            n = pr->job_n;
            pr->queued = 0;
        }
        stop = pr->stop && !pr->queued;
        LeaveCriticalSection(&pr->cs);
        if (n > 0) {
            uint64_t t = now_ms();
            sd_run(pr->m, &pr->sd, buf, (size_t)n);
            if (pr->stat_ms) *pr->stat_ms += now_ms() - t;
            consec = (pr->sd.last_energy < 1.0e-5) ? consec + 1 : 0;
            InterlockedExchange(&pr->published, (LONG)consec);
        }
        if (stop) break;
    }
    return 0;
}

static void probe_start(Probe* pr, Models* m, Player* pl)
{
    memset(pr, 0, sizeof(*pr));
    pr->m = m;
    pr->stat_ms = &g_stats.probe;
    sd_init(&pr->sd);
    if (pl) {
        pr->sd.cb = player_on_audio;
        pr->sd.cb_ctx = pl;
        player_begin(pl);
    }
    InitializeCriticalSection(&pr->cs);
    pr->wake = CreateEvent(NULL, FALSE, FALSE, NULL);
    pr->th = CreateThread(NULL, 0, probe_worker, pr, 0, NULL);
    if (!pr->th || !pr->wake) die("cannot start probe worker");
}

/* 提交一批 (攒满或收尾时); 工作线程没取走上一批时等它取走 (背压) */
static void probe_submit(Probe* pr, const int16_t* codes, int n)
{
    for (;;) {
        EnterCriticalSection(&pr->cs);
        if (!pr->queued) break;
        LeaveCriticalSection(&pr->cs);
        Sleep(1);
    }
    memcpy(pr->job, codes, sizeof(int16_t) * (size_t)n * N_VQ);
    pr->job_n = n;
    pr->queued = 1;
    LeaveCriticalSection(&pr->cs);
    SetEvent(pr->wake);
}

/* 收尾: 等工作线程把队列里的活干完, 交还累积的音频 */
static float* probe_finish(Probe* pr, size_t* out_samples)
{
    EnterCriticalSection(&pr->cs);
    pr->stop = 1;
    LeaveCriticalSection(&pr->cs);
    SetEvent(pr->wake);
    WaitForSingleObject(pr->th, INFINITE);
    CloseHandle(pr->th);
    CloseHandle(pr->wake);
    DeleteCriticalSection(&pr->cs);
    float* pcm = pr->sd.pcm;
    *out_samples = pr->sd.n_samples;
    pr->sd.pcm = NULL;
    sd_free(&pr->sd);
    return pcm;
}

/* 请求收尾: 丢弃扣住的尾部静音 (对应文件输出的尾部裁剪) */
/* 守护模式: 短句完整生成并验证后再一次性播放 (绕过预缓冲/hold) */
static void player_play_all(Player* p, const float* pcm, size_t n)
{
    if (n == 0) return;
    if (!p->t_first) p->t_first = now_ms();
    if (!p->live) return;
    p->prebuffering = 0;
    p->hold_n = 0;
    player_emit(p, pcm, n);
}

static void player_drop_hold(Player* p)
{
    p->hold_n = 0;
    /* 扣住的尾批 (自适应最多 2 批) 按文件同款规则裁剪 (20ms 窗) 后放行;
     * 没攒满预缓冲的短句也走同一裁剪 */
    if (p->pre_n > 0) {
        p->prebuffering = 0;
        size_t keep = player_trim_tail(p->pre, p->pre_n);
        if (keep && p->live) player_emit(p, p->pre, keep);
        p->pre_n = 0;
    }
    p->nbs = 0;
}

/* 生成语音: 帧生成与 codec 解码交错进行, 音频直接累积到 out_pcm
 * 返回生成的帧数 */
/* 一个 chunk 的 prefill 结果: 帧循环的初始状态 (kv 由 gen_run 释放) */
typedef struct {
    OrtValue* kv[GLOBAL_LAYERS * 2];
    float     hidden[HIDDEN];
    int32_t   past_len;
} GenPre;

/* 拼 prompt 行并跑 prefill, 产出初始 hidden + KV */
static void gen_prefill(Models* m, const int32_t* text_ids, size_t n_text,
                        const int16_t* ref, size_t ref_frames,
                        int verbose, GenPre* st)
{
    size_t S = (size_t)12 + 1 + ref_frames + 1 + 56 + n_text + 6 + 1;

    int32_t* ids = (int32_t*)xmalloc(sizeof(int32_t) * S * ROW_WIDTH);
    int32_t* amask = (int32_t*)xmalloc(sizeof(int32_t) * S);
    size_t r = 0;

#define PUSH_TEXT_ROW(tok)                                                    \
    do {                                                                      \
        int32_t* _row = ids + r * ROW_WIDTH;                                   \
        _row[0] = (tok);                                                       \
        for (int _c = 0; _c < N_VQ; _c++) _row[1 + _c] = AUDIO_PAD_TOKEN;      \
        amask[r] = 1;                                                          \
        r++;                                                                   \
    } while (0)

    for (int k = 0; k < 12; k++) PUSH_TEXT_ROW(k_user_prefix[k]);
    PUSH_TEXT_ROW(AUDIO_START_TOKEN);

    for (size_t f = 0; f < ref_frames; f++) {
        int32_t* row = ids + r * ROW_WIDTH;
        row[0] = AUDIO_USER_SLOT;
        for (int c = 0; c < N_VQ; c++) row[1 + c] = ref[f * N_VQ + c];
        amask[r] = 1;
        r++;
    }

    PUSH_TEXT_ROW(AUDIO_END_TOKEN);
    for (int k = 0; k < 56; k++) PUSH_TEXT_ROW(k_after_ref[k]);
    for (size_t k = 0; k < n_text; k++) PUSH_TEXT_ROW(text_ids[k]);
    for (int k = 0; k < 6; k++) PUSH_TEXT_ROW(k_asst_prefix[k]);
    PUSH_TEXT_ROW(AUDIO_START_TOKEN);
#undef PUSH_TEXT_ROW

    S = r;

    /* ---- prefill ---- */
    int64_t sh_ids[3] = { 1, (int64_t)S, ROW_WIDTH };
    int64_t sh_msk[2] = { 1, (int64_t)S };
    OrtValue* pin[2] = {0};
    pin[0] = MK_I32(ids, sh_ids, 3);
    pin[1] = MK_I32(amask, sh_msk, 2);

    /* 重要: outputs 数组必须清零, 否则 ORT 会把栈上的垃圾值当作已有
     * OrtValue 去增减引用计数 -> 访问违例 (输出个数多时必现) */
    OrtValue* pout[32] = {0};
    if (verbose) fprintf(stderr, "  prefill: seq=%zu\n", S);
    uint64_t _t0 = now_ms();
    sess_run(&m->prefill, pin, pout);
    g_stats.prefill += now_ms() - _t0;

    float* hidden_all = (float*)tensor_data(pout[0], NULL);
    memcpy(st->hidden, hidden_all + (S - 1) * HIDDEN, sizeof(float) * HIDDEN);

    ort->ReleaseValue(pout[0]);

    for (int k = 0; k < GLOBAL_LAYERS * 2; k++) st->kv[k] = pout[1 + k];
    st->past_len = (int32_t)S;

    ort->ReleaseValue(pin[0]);
    ort->ReleaseValue(pin[1]);
    free(ids);
    free(amask);
}

/* 自回归帧循环 (边生成边解码): 消费并释放 st 的 KV, 返回生成的帧数 */
static size_t gen_run(Models* m, GenPre* st,
                      uint64_t seed, int max_frames,
                      int verbose, int use_eos, const wchar_t* dump_frames,
                      Player* pl, float** out_pcm, size_t* out_samples)
{
    float hidden[HIDDEN];
    memcpy(hidden, st->hidden, sizeof(float) * HIDDEN);
    OrtValue** kv = st->kv;
    int32_t past_len = st->past_len;
    uint64_t _t0;
    Rng rng;
    rng_seed(&rng, seed);
    EosDetector eos;
    memset(&eos, 0, sizeof(eos));

    uint8_t seen[N_VQ][AUDIO_CODEBOOK];
    memset(seen, 0, sizeof(seen));

    int32_t* mask = (int32_t*)xcalloc((size_t)N_VQ * AUDIO_CODEBOOK, sizeof(int32_t));
    int64_t sh_mask[3] = { 1, N_VQ, AUDIO_CODEBOOK };
    int64_t sh_hid[2] = { 1, HIDDEN };
    int64_t sh_one[1] = { 1 };
    int64_t sh_rand[2] = { 1, N_VQ };

    float* u_asst = (float*)xmalloc(sizeof(float));
    float* u_audio = (float*)xmalloc(sizeof(float) * N_VQ);

    int32_t* step_ids = (int32_t*)xmalloc(sizeof(int32_t) * ROW_WIDTH);
    int64_t sh_step[3] = { 1, 1, ROW_WIDTH };
    int32_t* pvl_buf = (int32_t*)xmalloc(sizeof(int32_t));
    int64_t sh_pvl[1] = { 1 };

    /* 这几个输入张量的形状是固定的, 创建一次反复复用: ORT 不会改写输入,
     * 每帧重新 CreateTensor 只是白白多出几十次分配/释放。 */
    OrtValue* t_hidden = MK_F32(hidden, sh_hid, 2);
    OrtValue* t_mask   = MK_I32(mask, sh_mask, 3);
    OrtValue* t_asst   = MK_F32(u_asst, sh_one, 1);
    OrtValue* t_audio  = MK_F32(u_audio, sh_rand, 2);
    OrtValue* t_step   = MK_I32(step_ids, sh_step, 3);
    OrtValue* t_pvl    = MK_I32(pvl_buf, sh_pvl, 1);

    int16_t* out = (int16_t*)xmalloc(sizeof(int16_t) * (size_t)max_frames * N_VQ);
    size_t n_frames = 0;

    Probe probe;
    probe_start(&probe, m, pl);

    /* 待解码的帧: 攒够 SD_BATCH 帧一起解码 (太小调度开销大, 太大峰值内存高) */
    enum { SD_BATCH = MOSS_SD_BATCH };
    int16_t pend[SD_BATCH * N_VQ];
    int n_pend = 0;
    int batch_goal = 4;             /* 前几批小批量: 4 -> 8 -> 16 */

    for (int step = 0; step < max_frames; step++) {
        u_asst[0] = rng_u(&rng);
        for (int c = 0; c < N_VQ; c++) u_audio[c] = rng_u(&rng);

        OrtValue* fin[4] = { t_hidden, t_mask, t_asst, t_audio };

        OrtValue* fout[2] = { NULL, NULL };
        _t0 = now_ms();
        sess_run(&m->frame, fin, fout);
        g_stats.frame += now_ms() - _t0;

        /* 形状固定 (1,1) / (1,16), 不用再查 shape */
        int32_t* sc = (int32_t*)tensor_data(fout[0], NULL);
        int32_t* fr = (int32_t*)tensor_data(fout[1], NULL);
        int32_t frame[N_VQ];
        int keep = (sc && sc[0] != 0);
        for (int c = 0; c < N_VQ; c++) frame[c] = fr[c];

        ort->ReleaseValue(fout[0]);
        ort->ReleaseValue(fout[1]);

        if (!keep) {
            if (verbose) fprintf(stderr, "  [step %d] should_continue=0, stop\n", step);
            break;
        }

        /* 记录已见 token: seen 只增不减, 所以 mask 也只需增量置位 */
        for (int c = 0; c < N_VQ; c++) {
            int t = frame[c];
            if (t >= 0 && t < AUDIO_CODEBOOK && !seen[c][t]) {
                seen[c][t] = 1;
                mask[(size_t)c * AUDIO_CODEBOOK + t] = 1;
            }
        }

        if (use_eos && eos_check(&eos, frame, step)) {
            if (verbose) fprintf(stderr, "  [step %d] degenerate loop detected\n", step);
            break;
        }

        for (int c = 0; c < N_VQ; c++) {
            out[n_frames * N_VQ + c] = (int16_t)frame[c];
            pend[n_pend * N_VQ + c] = (int16_t)frame[c];
        }
        n_frames++;
        n_pend++;

        /* 攒够一批就丢给工作线程解码; 连续静音判据由它发布 (最多滞后一批) */
        if (n_pend >= batch_goal) {
            uint64_t _tp = now_ms();
            probe_submit(&probe, pend, n_pend);
            g_stats.probe += now_ms() - _tp;
            n_pend = 0;
            batch_goal = batch_goal == 4 ? 8 : SD_BATCH;
            if (probe.published >= 2) {
                if (verbose) fprintf(stderr, "  [step %d] trailing silence, stop\n", step);
                break;
            }
        }

        /* decode_step 推进全局 hidden + KV */
        step_ids[0] = AUDIO_ASST_SLOT;
        for (int c = 0; c < N_VQ; c++) step_ids[1 + c] = frame[c];
        pvl_buf[0] = past_len;

        OrtValue* din[32] = {0};
        din[0] = t_step;
        din[1] = t_pvl;
        for (int k = 0; k < GLOBAL_LAYERS * 2; k++) din[2 + k] = kv[k];

        OrtValue* dout[32] = {0};
        _t0 = now_ms();
        sess_run(&m->decode, din, dout);
        g_stats.step += now_ms() - _t0;
        g_stats.steps++;

        /* global_hidden 恒为 [1,1,768], 不必再查 shape */
        float* dh = (float*)tensor_data(dout[0], NULL);
        memcpy(hidden, dh, sizeof(float) * HIDDEN);

        ort->ReleaseValue(dout[0]);
        for (int k = 0; k < GLOBAL_LAYERS * 2; k++) ort->ReleaseValue(kv[k]);
        for (int k = 0; k < GLOBAL_LAYERS * 2; k++) kv[k] = dout[1 + k];

        past_len++;

        if (verbose && (step % 50 == 0))
            fprintf(stderr, "  frame %d/%d (seq %d)\n", step + 1, max_frames, past_len);
    }

    /* 剩余不足一批的帧提交掉, 等工作线程收尾并交还音频 */
    if (n_pend > 0) {
        uint64_t _tp = now_ms();
        probe_submit(&probe, pend, n_pend);
        g_stats.probe += now_ms() - _tp;
        n_pend = 0;
    }
    size_t sd_n = 0;
    float* sd_pcm = probe_finish(&probe, &sd_n);
    if (pl) player_drop_hold(pl);

    if (dump_frames && dump_frames[0]) {
        FILE* df = _wfopen(dump_frames, L"w");
        if (df) {
            for (size_t k = 0; k < n_frames; k++) {
                for (int c = 0; c < N_VQ; c++) fprintf(df, "%d%s", out[k * N_VQ + c], c + 1 < N_VQ ? " " : "\n");
            }
            fclose(df);
        }
    }

    for (int k = 0; k < GLOBAL_LAYERS * 2; k++) ort->ReleaseValue(kv[k]);
    ort->ReleaseValue(t_hidden); ort->ReleaseValue(t_mask);
    ort->ReleaseValue(t_asst);   ort->ReleaseValue(t_audio);
    ort->ReleaseValue(t_step);   ort->ReleaseValue(t_pvl);
    free(mask); free(u_asst); free(u_audio);
    free(step_ids); free(pvl_buf);
    free(out);

    *out_pcm = sd_pcm;
    *out_samples = sd_n;
    return n_frames;
}
/* 后台 prefill: 下一块的 prompt 预计算与上一块的帧循环并行,
 * chunk 交界的等待从 '整个 prefill' 缩到 '取结果' (通常 0ms) */
typedef struct {
    Models* m;
    int32_t* tids;
    size_t nt;
    const int16_t* ref;
    size_t ref_frames;
    GenPre st;
    HANDLE done;
} PreJob;

static DWORD WINAPI prefill_worker(LPVOID p)
{
    PreJob* j = (PreJob*)p;
    gen_prefill(j->m, j->tids, j->nt, j->ref, j->ref_frames, 0, &j->st);
    SetEvent(j->done);
    return 0;
}

/* codec 编码: 参考音频 -> codes */
static int16_t* codec_encode(Models* m, const float* interleaved, size_t frames, size_t* out_frames)
{
    if (!m->enc.s)
        sess_open(&m->enc, m->dir, L"moss_audio_tokenizer_encode.onnx", m->threads, &g_stats.load_enc);

    float* planar = (float*)xmalloc(sizeof(float) * frames * CHANNELS);
    for (size_t s = 0; s < frames; s++)
        for (int c = 0; c < CHANNELS; c++)
            planar[c * frames + s] = interleaved[s * CHANNELS + c];

    int32_t len_buf = (int32_t)frames;
    int64_t sh_wav[3] = { 1, CHANNELS, (int64_t)frames };
    int64_t sh_len[1] = { 1 };

    OrtValue* in[2] = {0};
    in[0] = MK_F32(planar, sh_wav, 3);
    in[1] = MK_I32(&len_buf, sh_len, 1);

    OrtValue* outv[2] = { NULL, NULL };
    sess_run(&m->enc, in, outv);

    size_t total = 0;
    int32_t* ac = (int32_t*)tensor_data(outv[0], &total);
    size_t ln = 0;
    int32_t* al = (int32_t*)tensor_data(outv[1], &ln);
    size_t nf = (al && ln > 0 && al[0] > 0) ? (size_t)al[0] : (total / N_VQ);
    if (nf * N_VQ > total) nf = total / N_VQ;

    int16_t* codes = (int16_t*)xmalloc(sizeof(int16_t) * nf * N_VQ);
    for (size_t k = 0; k < nf * N_VQ; k++) codes[k] = (int16_t)ac[k];

    ort->ReleaseValue(outv[0]);
    ort->ReleaseValue(outv[1]);
    ort->ReleaseValue(in[0]);
    ort->ReleaseValue(in[1]);
    free(planar);

    *out_frames = nf;
    return codes;
}

/* ---------------------------------------------------------------------------
 * 8. 文本分块 (按句末标点, 控制每次 prefill 的 token 数)
 * -------------------------------------------------------------------------*/
#define MAX_TEXT_TOKENS 100

static int is_sentence_end(uint32_t c)
{
    return c == '.' || c == '!' || c == '?' || c == ';' ||
           c == 0x3002 || c == 0xFF01 || c == 0xFF1F || c == 0xFF1B || c == 0xFF1A;
}

/* 把 text 切成若干块; 返回块数, 结果写入 out (每块 malloc) */
static int split_chunks(const SP* sp, const char* text, char*** out, int max_tokens)
{
    size_t n = strlen(text);
    /* 先按句末标点切段 */
    char** seg = (char**)xmalloc(sizeof(char*) * (n + 2));
    int nseg = 0;
    size_t start = 0, i = 0;
    while (i < n) {
        int cl;
        uint32_t c = utf8_dec((const unsigned char*)text + i, &cl);
        i += (size_t)cl;
        if (is_sentence_end(c)) {
            /* 吸收紧跟的结束引号 */
            while (i < n) {
                int cl2;
                uint32_t c2 = utf8_dec((const unsigned char*)text + i, &cl2);
                if (c2 == '"' || c2 == '\'' || c2 == ')' || c2 == ']' || c2 == '}' ||
                    c2 == 0x300D || c2 == 0x300F || c2 == 0xFF09 || c2 == 0x3011 || c2 == 0x300B)
                    i += (size_t)cl2;
                else break;
            }
            size_t len = i - start;
            char* s = (char*)xmalloc(len + 1);
            memcpy(s, text + start, len);
            s[len] = 0;
            /* trim */
            char* p = s;
            while (*p == ' ') p++;
            size_t pl = strlen(p);
            while (pl > 0 && p[pl - 1] == ' ') p[--pl] = 0;
            if (pl > 0) { if (p != s) memmove(s, p, pl + 1); seg[nseg++] = s; }
            else free(s);
            while (i < n && text[i] == ' ') i++;
            start = i;
        }
    }
    if (start < n) {
        size_t len = n - start;
        char* s = (char*)xmalloc(len + 1);
        memcpy(s, text + start, len);
        s[len] = 0;
        char* p = s;
        while (*p == ' ') p++;
        size_t pl = strlen(p);
        while (pl > 0 && p[pl - 1] == ' ') p[--pl] = 0;
        if (pl > 0) { if (p != s) memmove(s, p, pl + 1); seg[nseg++] = s; } else free(s);
    }

    char** chunks = (char**)xmalloc(sizeof(char*) * (size_t)(nseg + 1));
    int nch = 0;
    char cur[8192];
    cur[0] = 0;

    for (int k = 0; k < nseg; k++) {
        if (cur[0] == 0) {
            _snprintf(cur, sizeof(cur), "%s", seg[k]);
            free(seg[k]);
            continue;
        }
        char joined[8192];
        int cjk = text_has_cjk(cur) || text_has_cjk(seg[k]);
        _snprintf(joined, sizeof(joined), cjk ? "%s%s" : "%s %s", cur, seg[k]);
        if (sp_count(sp, joined) <= max_tokens) {
            strcpy(cur, joined);
            free(seg[k]);
        } else {
            chunks[nch++] = _strdup(cur);
            _snprintf(cur, sizeof(cur), "%s", seg[k]);
            free(seg[k]);
        }
    }
    if (cur[0]) chunks[nch++] = _strdup(cur);
    free(seg);

    *out = chunks;
    return nch;
}


/* ---------------------------------------------------------------------------
 * 8. 内置音色表 (从 browser_poc_manifest.json 读取/写入)
 *
 * manifest 的 builtin_voices 里每个音色带一段参考音频的 RVQ 码。
 * 为了不把两万多个数字塞进源码, 这里用一个极简的 JSON 扫描器在运行时读取;
 * 文件缺失时退回内嵌的神里绫华。
 * 写入方向: --save-voice 把 --ref-wav 编码出的参考码按同样的格式追加回数组,
 * 克隆来的声音从此和官方预置音色一样, 用 --voice <name> 直接调用。
 * -------------------------------------------------------------------------*/
static const char* js_skip_ws(const char* p, const char* end)
{
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) p++;
    return p;
}

/* p 指向双引号, 读出字符串, 返回右引号之后的位置; 失败返回 NULL */
static const char* js_read_str(const char* p, const char* end, char* out, int cap)
{
    if (p >= end || *p != '"') return NULL;
    p++;
    int n = 0;
    while (p < end && *p != '"') {
        if (*p == '\\' && p + 1 < end) {
            p++;
            int esc = 1;                    /* JSON 转义解码 (对话回复里有换行等) */
            if (*p == 'n') { if (n < cap) out[n++] = '\n'; }
            else if (*p == 't') { if (n < cap) out[n++] = '\t'; }
            else if (*p == 'r') { if (n < cap) out[n++] = '\r'; }
            else if (*p == 'u') {           /* \uXXXX: 模型直接出 UTF-8, 这里兜底 */
                p++;
                for (int k = 0; k < 4 && p < end; k++, p++) {}
                if (n < cap) out[n++] = '?';
                esc = 0;
            }
            if (esc && n < cap) out[n++] = *p;
            p++;
            continue;
        }
        if (n + 1 < cap) out[n++] = *p;
        p++;
    }
    out[n] = 0;
    return p + 1;
}

/* 从 p 往后找 key 为 want 的字符串值; 返回该值之后的位置, 值写入 out */
static const char* js_find_str(const char* p, const char* end, const char* want, char* out, int cap)
{
    char key[64];
    while (p < end) {
        if (*p != '"') { p++; continue; }
        const char* q = js_read_str(p, end, key, sizeof(key));
        if (!q) return NULL;
        if (!strcmp(key, want)) {
            q = js_skip_ws(q, end);
            if (q < end && *q == ':') {
                q = js_skip_ws(q + 1, end);
                return js_read_str(q, end, out, cap);
            }
        }
        p = q;
    }
    return NULL;
}

/* 从 p 往后找 key 为 want 的整数数组, 把其中所有整数收集起来 */
static int16_t* js_find_int_array(const char* p, const char* end, const char* want, size_t* out_n)
{
    char key[64];
    for (;;) {
        if (p >= end) return NULL;
        if (*p != '"') { p++; continue; }
        const char* q = js_read_str(p, end, key, sizeof(key));
        if (!q) return NULL;
        if (!strcmp(key, want)) { p = q; break; }
        p = q;
    }
    p = js_skip_ws(p, end);
    if (p >= end || *p != ':') return NULL;
    p = js_skip_ws(p + 1, end);
    if (p >= end || *p != '[') return NULL;

    size_t cap = 4096, n = 0;
    int16_t* v = (int16_t*)xmalloc(sizeof(int16_t) * cap);
    int depth = 0;
    while (p < end) {
        char c = *p;
        if (c == '[') { depth++; p++; continue; }
        if (c == ']') { depth--; p++; if (depth == 0) break; continue; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            char* e2 = NULL;
            long x = strtol(p, &e2, 10);
            if (e2 == p) { p++; continue; }
            p = e2;
            if (n == cap) {
                cap *= 2;
                int16_t* nv = (int16_t*)realloc(v, sizeof(int16_t) * cap);
                if (!nv) die("out of memory");
                v = nv;
            }
            v[n++] = (int16_t)x;
            continue;
        }
        p++;
    }
    *out_n = n;
    return v;
}

/* 取指定音色的参考码; 找不到返回 NULL */
static int16_t* manifest_voice(const wchar_t* path, const char* name, size_t* out_frames)
{
    size_t len = 0;
    char* b = (char*)read_whole_file(path, &len);
    if (!b) return NULL;
    const char* end = b + len;
    const char* p = b;
    char nm[64];
    while ((p = js_find_str(p, end, "voice", nm, sizeof(nm))) != NULL) {
        if (!strcmp(nm, name)) {
            size_t n = 0;
            int16_t* codes = js_find_int_array(p, end, "prompt_audio_codes", &n);
            free(b);
            if (codes && n >= N_VQ && (n % N_VQ) == 0) {
                *out_frames = n / N_VQ;
                return codes;
            }
            free(codes);
            return NULL;
        }
    }
    free(b);
    return NULL;
}

/* ---- 追加音色 (--save-voice) ---- */

/* 把 s 按 JSON 字符串规则转义后写入 out, 返回长度 */
static int json_escape_into(const char* s, char* out, int cap)
{
    int n = 0;
    for (; *s && n < cap - 8; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { out[n++] = '\\'; out[n++] = (char)c; }
        else if (c == '\n') { out[n++] = '\\'; out[n++] = 'n'; }
        else if (c == '\r') { out[n++] = '\\'; out[n++] = 'r'; }
        else if (c == '\t') { out[n++] = '\\'; out[n++] = 't'; }
        else if (c < 0x20) { n += snprintf(out + n, (size_t)(cap - n), "\\u%04x", c); }
        else out[n++] = (char)c;
    }
    out[n] = 0;
    return n;
}

/* 定位顶层 "builtin_voices" 数组: off_lo 指向 '[', off_hi 指向匹配的 ']'。
 * 扫描时跳过字符串内容, 避免被文本里的括号骗到。找到返回 1。 */
static int js_locate_builtin_voices(const char* b, size_t len, size_t* off_lo, size_t* off_hi)
{
    const char* end = b + len;
    const char* p = b;
    char tmp[64];
    while (p < end) {
        if (*p != '"') { p++; continue; }
        char key[64];
        const char* q = js_read_str(p, end, key, sizeof(key));
        if (!q) return 0;
        int hit = !strcmp(key, "builtin_voices");
        p = q;
        if (hit) {
            while (p < end && *p != '[') {
                if (*p == '"') { p = js_read_str(p, end, tmp, sizeof(tmp)); if (!p) return 0; }
                else p++;
            }
            if (p >= end) return 0;
            const char* lo = p;
            p++;
            int depth = 1;
            while (p < end && depth > 0) {
                if (*p == '"') { p = js_read_str(p, end, tmp, sizeof(tmp)); if (!p) return 0; continue; }
                if (*p == '[') depth++;
                else if (*p == ']') depth--;
                p++;
            }
            if (depth != 0) return 0;
            *off_lo = (size_t)(lo - b);
            *off_hi = (size_t)(p - 1 - b);
            return 1;
        }
    }
    return 0;
}

/* builtin_voices 数组 [lo, hi) 里是否已有同名音色 */
static int js_voice_exists(const char* lo, const char* hi, const char* name)
{
    const char* p = lo;
    char nm[64];
    while ((p = js_find_str(p, hi, "voice", nm, sizeof(nm))) != NULL)
        if (!strcmp(nm, name)) return 1;
    return 0;
}

/* 把 text 追加到 e[n], 剩余空间不足时返回 -1 (避免 snprintf 截断后返回值
 * 让 n 越过 ecap, 下一次 ecap-n 下溢导致越界写) */
static int e_append(char* e, size_t ecap, int n, const char* text)
{
    size_t need = strlen(text);
    if (n < 0 || (size_t)n + need + 1 > ecap) return -1;
    memcpy(e + n, text, need);
    n += (int)need;
    e[n] = 0;
    return n;
}

/* 把一个音色条目追加进 manifest 的 builtin_voices (原文件备份成 .bak)。
 * codes 按 帧 x N_VQ 排列, 与 generate_frames 的读法一致。成功返回 0。 */
static int manifest_save_voice(const wchar_t* path, const char* name, const char* desc,
                               const wchar_t* ref_wav, const int16_t* codes, size_t frames)
{
    size_t len = 0;
    char* b = (char*)read_whole_file(path, &len);
    if (!b) { fprintf(stderr, "[warn] 读不到 manifest, 音色没有保存\n"); return -1; }

    size_t lo = 0, hi = 0;
    if (!js_locate_builtin_voices(b, len, &lo, &hi)) {
        fprintf(stderr, "[warn] manifest 里找不到 builtin_voices 数组\n");
        free(b); return -1;
    }
    if (js_voice_exists(b + lo, b + hi, name)) {
        fprintf(stderr, "[warn] 音色 \"%s\" 已存在, 换个名字再存\n", name);
        free(b); return -1;
    }

    /* 新条目文本, 排版与现有条目一致 (CRLF, 数字逐行) */
    size_t ecap = frames * N_VQ * 24 + 1024;
    char* e = (char*)xmalloc(ecap);
    char nm[128], ds[256], af[MAX_PATH * 3];
    json_escape_into(name, nm, sizeof(nm));
    json_escape_into(desc, ds, sizeof(ds));
    const wchar_t* leaf = ref_wav;              /* audio_file 只是备注, 运行时不读 */
    for (const wchar_t* c = ref_wav; *c; c++)
        if (*c == L'\\' || *c == L'/') leaf = c + 1;
    char* af_u8 = w2u(leaf);
    json_escape_into(af_u8 ? af_u8 : "", af, sizeof(af));
    free(af_u8);

    int n = 0;
    n = e_append(e, ecap, n, "    {\r\n");
    n = e_append(e, ecap, n, "      \"voice\": \""); n = e_append(e, ecap, n, nm);
    n = e_append(e, ecap, n, "\",\r\n      \"display_name\": \""); n = e_append(e, ecap, n, ds);
    n = e_append(e, ecap, n, "\",\r\n      \"group\": \"Custom\",\r\n      \"audio_file\": \"");
    n = e_append(e, ecap, n, af);
    n = e_append(e, ecap, n, "\",\r\n      \"prompt_audio_codes\": [\r\n");
    char line[32];
    for (size_t f = 0; f < frames && n >= 0; f++) {
        n = e_append(e, ecap, n, "        [\r\n");
        for (int k = 0; k < N_VQ && n >= 0; k++) {
            snprintf(line, sizeof(line), "          %d%s\r\n",
                     (int)codes[f * N_VQ + k], k + 1 < N_VQ ? "," : "");
            n = e_append(e, ecap, n, line);
        }
        snprintf(line, sizeof(line), "        ]%s\r\n", f + 1 < frames ? "," : "");
        n = e_append(e, ecap, n, line);
    }
    if (n >= 0) n = e_append(e, ecap, n, "      ]\r\n    }");
    if (n < 0) {
        fprintf(stderr, "[warn] 条目文本太长\n");
        free(e); free(b); return -1;
    }

    /* 插到数组最后一个条目后面: 找 ']' 前最后一个非空白字符 */
    size_t ins = hi;
    while (ins > lo + 1) {
        char c = b[ins - 1];
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
        ins--;
    }
    int need_comma = (ins > lo + 1);
    size_t add = (size_t)n + (need_comma ? 3 : 0);

    char* nb = (char*)xmalloc(len + add + 1);
    memcpy(nb, b, ins);
    if (need_comma) memcpy(nb + ins, ",\r\n", 3);
    memcpy(nb + ins + (need_comma ? 3 : 0), e, (size_t)n);
    memcpy(nb + ins + (need_comma ? 3 : 0) + (size_t)n, b + ins, len - ins);

    wchar_t bak[MAX_PATH * 2];
    _snwprintf(bak, MAX_PATH * 2, L"%ls.bak", path);
    FILE* bf = _wfopen(bak, L"wb");
    if (bf) { fwrite(b, 1, len, bf); fclose(bf); }

    FILE* f = _wfopen(path, L"wb");
    if (!f) {
        fprintf(stderr, "[warn] manifest 写不进去, 音色没有保存\n");
        free(nb); free(e); free(b); return -1;
    }
    fwrite(nb, 1, len + add, f);
    fclose(f);
    free(nb); free(e); free(b);
    return 0;
}

/* 列出 manifest 里的所有音色 */
static int manifest_list_voices(const wchar_t* path)
{
    size_t len = 0;
    char* b = (char*)read_whole_file(path, &len);
    if (!b) return 0;
    const char* end = b + len;
    const char* p = b;
    char nm[64], disp[96], grp[48];
    int n = 0;
    while ((p = js_find_str(p, end, "voice", nm, sizeof(nm))) != NULL) {
        const char* q = js_find_str(p, end, "display_name", disp, sizeof(disp));
        const char* r = js_find_str(p, end, "group", grp, sizeof(grp));
        if (q && r) printf("  %-8s %-16s %s\n", nm, grp, disp);
        else           printf("  %s\n", nm);
        if (++n >= 64) break;
    }
    free(b);
    return n;
}

/* 参考音频预处理: 裁掉首尾静音, 峰值归一化到 0.9。
 * 用户自己录的参考音频常常前面有几秒空白、电平也偏小, 直接送进 codec 编码
 * 会浪费 prompt 长度、还会让克隆出来的音量忽大忽小。 */
static size_t prep_reference(float* pcm, size_t frames)
{
    if (frames < (size_t)(SAMPLE_RATE / 10)) return frames;
    const size_t win = SAMPLE_RATE / 50;          /* 20ms */
    const double th = 2.5e-5;                     /* RMS > 0.005 */

    size_t first = 0, last = 0;
    for (size_t w = 0; w + win <= frames; w += win) {
        double acc = 0;
        for (size_t i = w * CHANNELS; i < (w + win) * CHANNELS; i++) acc += (double)pcm[i] * pcm[i];
        if (acc / (double)(win * CHANNELS) > th) { first = w; break; }
    }
    for (size_t w = frames; w >= win; w -= win) {
        double acc = 0;
        for (size_t i = (w - win) * CHANNELS; i < w * CHANNELS; i++) acc += (double)pcm[i] * pcm[i];
        if (acc / (double)(win * CHANNELS) > th) { last = w; break; }
    }
    if (last <= first) return frames;

    size_t pad = (size_t)(SAMPLE_RATE / 20);      /* 前后各留 50ms */
    size_t a = first > pad ? first - pad : 0;
    size_t b = last + pad < frames ? last + pad : frames;

    float peak = 0.0f;
    for (size_t i = a * CHANNELS; i < b * CHANNELS; i++) {
        float v = pcm[i] < 0 ? -pcm[i] : pcm[i];
        if (v > peak) peak = v;
    }
    if (peak > 0.01f) {
        float gain = 0.9f / peak;
        for (size_t i = a * CHANNELS; i < b * CHANNELS; i++) pcm[i] *= gain;
    }
    if (a > 0) memmove(pcm, pcm + a * CHANNELS, sizeof(float) * (b - a) * CHANNELS);
    return b - a;
}

/* ---------------------------------------------------------------------------
 * 8.5 合成主流程 (一次性模式与 --server 共用)
 * -------------------------------------------------------------------------*/

/* 按名字解析音色 -> 参考码。成功返回 0: *ref 指向内嵌静态数据或 *owned 的
 * 堆内存 (调用方负责 free(*owned)); 找不到返回 -1 (已打印原因)。 */
static int resolve_voice(const wchar_t* manifest_path, const wchar_t* name_w,
                         const int16_t** ref, size_t* ref_frames, int16_t** owned)
{
    *owned = NULL;
    if (!name_w || !name_w[0]) return -1;
    char* u8 = w2u(name_w);
    if (!u8) return -1;

    if (wcscmp(name_w, L"神里绫华") == 0) {
        /* 默认音色: 优先用 manifest 里的 (参考可能更新过), 查不到退回内嵌的一份 */
        *owned = manifest_voice(manifest_path, u8, ref_frames);
        if (*owned) {
            *ref = *owned;
        } else {
            *ref = k_ref_codes_ayaka;
            *ref_frames = REF_FRAMES_AYAKA;
            fprintf(stderr, "  [note] manifest 缺失或没有默认音色, 用内嵌的神里绫华\n");
        }
    } else {
        *owned = manifest_voice(manifest_path, u8, ref_frames);
        if (!*owned) {
            fprintf(stderr, "找不到音色 \"%s\", 用 --list-voices 看看有哪些\n", u8);
            free(u8);
            return -1;
        }
        *ref = *owned;
    }
    printf("voice     : %s (%zu 帧)\n", u8, *ref_frames);
    free(u8);
    return 0;
}

/* 分块合成一段已归一化的文本: 逐块 generate (可选流式播放), 拼接后做
 * 尾部裁剪 + 峰值归一化, 返回整段音频 (调用方 free); *out_total 为采样数。
 * 没生成出任何音频返回 NULL。 */
static float* synthesize(Models* m, SP* sp, const char* norm,
                         const int16_t* ref, size_t ref_frames,
                         uint64_t seed, int max_frames,
                         int verbose, int use_eos, const wchar_t* dump_frames,
                         Player* pl, size_t* out_total)
{
    char** chunks = NULL;
    int nch = split_chunks(sp, norm, &chunks, MAX_TEXT_TOKENS);

    size_t total_samples = 0;
    float* all = NULL;

    /* token 编码是毫秒级, 全部提前做; 下一块的 prefill 才能提前到后台跑 */
    int32_t** tok = (int32_t**)xcalloc((size_t)nch, sizeof(int32_t*));
    int* ntok = (int*)xcalloc((size_t)nch, sizeof(int));
    for (int ci = 0; ci < nch; ci++) {
        tok[ci] = (int32_t*)xmalloc(sizeof(int32_t) * (strlen(chunks[ci]) * 2 + 16));
        ntok[ci] = sp_encode(sp, chunks[ci], tok[ci], (int)strlen(chunks[ci]) * 2 + 16);
    }

    PreJob job;
    HANDLE hjob = NULL;

    for (int ci = 0; ci < nch; ci++) {
        int nt = ntok[ci];
        if (nt <= 0) continue;
        if (verbose || nch > 1) printf("chunk %d/%d: %d tokens\n", ci + 1, nch, nt);

        /* 上一块循环期间下一块的 prefill 已在后台跑完, 这里通常瞬间取走 */
        GenPre st;
        if (hjob) {
            WaitForSingleObject(hjob, INFINITE);
            CloseHandle(hjob);
            hjob = NULL;
            st = job.st;
            /* job.tids 即 tok[ci], 所有权在 tok 表, 循环收尾统一 free (免双重释放) */
        } else {
            gen_prefill(m, tok[ci], (size_t)nt, ref, ref_frames, verbose, &st);
        }

        /* 短句守护: 单块短文本先不带播放完整生成, 时长远超预期判定为
         * 模型跑偏 (一句正常话后零碎冒音节, 静音/重复判据都拦不住),
         * 换种子重试; 干净的结果再一次性播放。"输入什么说什么"优先于
         * 短句的流式低延迟。 */
        int guarded = (nch == 1 && nt <= 48);

        uint64_t t1 = now_ms();
        float* pcm = NULL;
        size_t nsamp = 0;
        size_t frames = 0;
        for (int attempt = 0; ; attempt++) {
            if (attempt > 0) gen_prefill(m, tok[ci], (size_t)nt, ref, ref_frames, 0, &st);
            /* 验证矩阵实测: 合法短句 ~2.5 帧/token (你是谁 10~30 帧,
             * 1加1等于几 21 帧), 跑偏形态 60 帧仍带 2.8s 句中静音;
             * 4 帧/token + 20 帧余量能拦住跑偏而不误伤慢读 */
            double cap = (double)nt * 4.0 + 20.0;
            /* 前两次尝试把帧数上限压到预期附近: 跑偏的早停早换, 别等它
             * 完整吐完 147 帧才发现; 最后一次兜底放开全长 */
            int mf = (guarded && attempt < 2) ? (int)(cap * 1.2) : max_frames;
            frames = gen_run(m, &st, seed + (uint64_t)attempt, mf, verbose, use_eos,
                             dump_frames, guarded ? NULL : pl, &pcm, &nsamp);
            if (!guarded || attempt >= 2 || (double)frames <= cap)
                break;
            fprintf(stderr, "  [regen] 短句疑似跑偏 (%zu 帧 > 预期 %.0f), 换种子重试\n",
                    frames, cap);
            free(pcm);
            pcm = NULL;
        }
        if (guarded && pl) player_play_all(pl, pcm, nsamp);
        uint64_t t2 = now_ms();
        if (frames == 0 || !pcm || nsamp == 0) {
            fprintf(stderr, "  [warn] no frames generated for this chunk\n");
            free(pcm);
            continue;
        }
        printf("  %zu frames -> %.2f s audio  (%llu ms)\n",
               frames, (double)nsamp / SAMPLE_RATE, (unsigned long long)(t2 - t1));

        size_t pause = 0;
        if (ci + 1 < nch) pause = (size_t)(SAMPLE_RATE * 0.25);
        float* merged = (float*)xmalloc(sizeof(float) * (total_samples + nsamp + pause) * CHANNELS);
        if (all) {
            memcpy(merged, all, sizeof(float) * total_samples * CHANNELS);
            free(all);
        }
        memcpy(merged + total_samples * CHANNELS, pcm, sizeof(float) * nsamp * CHANNELS);
        total_samples += nsamp;
        if (pause) {
            memset(merged + total_samples * CHANNELS, 0, sizeof(float) * pause * CHANNELS);
            total_samples += pause;
        }
        all = merged;

        free(pcm);

        /* 下一块有货: prefill 丢后台, 与本块循环的收尾/下一轮并行 */
        if (ci + 1 < nch && ntok[ci + 1] > 0) {
            job.m = m;
            job.tids = tok[ci + 1];
            job.nt = (size_t)ntok[ci + 1];
            job.ref = ref;
            job.ref_frames = ref_frames;
            job.done = CreateEvent(NULL, FALSE, FALSE, NULL);
            HANDLE th = CreateThread(NULL, 0, prefill_worker, &job, 0, NULL);
            if (th && job.done) {
                CloseHandle(th);
                hjob = job.done;
            } else {
                if (job.done) CloseHandle(job.done);
                gen_prefill(m, tok[ci + 1], (size_t)ntok[ci + 1], ref, ref_frames, 0, &job.st);
            }
        }
    }

    for (int k = 0; k < nch; k++) free(chunks[k]);
    for (int k = 0; k < nch; k++) free(tok[k]);
    free(tok);
    free(ntok);
    free(chunks);

    if (!all || total_samples == 0) return NULL;

    /* 模型说完之后还会继续吐一段静音 (偶尔夹杂稀疏的脉冲), 把末尾多余的部分裁掉。
     * 用 20ms 窗口的 RMS 判断而不是单点幅值, 免得被孤立脉冲带偏。 */
    {
        const size_t win = SAMPLE_RATE / 50;          /* 20ms */
        size_t last = 0;
        for (size_t w = total_samples; w >= win; w -= win) {
            double acc = 0.0;
            for (size_t i = (w - win) * CHANNELS; i < w * CHANNELS; i++)
                acc += (double)all[i] * (double)all[i];
            if (acc / (double)(win * CHANNELS) > 1.0e-4) { last = w; break; }   /* RMS > 0.01 */
        }
        if (last > 0) {
            size_t keep = last + (size_t)(SAMPLE_RATE * 0.2);
            if (keep < total_samples) total_samples = keep;
        }
    }

    /* 峰值归一化到 0.95: 模型会复刻参考音频的电平, 不同音色输出音量能差 12dB,
     * 归一化之后切换音色音量一致, 顺带消掉峰值超过 1.0 造成的削波。纯增益, 不改音质。 */
    {
        float peak = 0.0f;
        for (size_t i = 0; i < total_samples * CHANNELS; i++) {
            float v = all[i] < 0 ? -all[i] : all[i];
            if (v > peak) peak = v;
        }
        if (peak > 0.05f) {
            float gain = 0.95f / peak;
            for (size_t i = 0; i < total_samples * CHANNELS; i++) all[i] *= gain;
            if (verbose) fprintf(stderr, "  归一化: 峰值 %.2f -> 0.95 (增益 %+.1f dB)\n",
                                 peak, 20.0 * log10((double)gain));
        }
    }

    *out_total = total_samples;
    return all;
}

/* ---------------------------------------------------------------------------
 * 8.7 对话模式: 输入发给大模型, 回复用 TTS 实时说出来 (--chat, 配合 --server)
 *
 * API 为 xfyun MaaS 的 /v1/responses (OpenAI Responses 风格):
 *   请求 {"model", "input": [消息数组], "stream": false}
 *   回复顶层 output_text = 最终答案 (summary 里是推理模型的思维链, 不能读)
 * 多轮上下文保留最近 CHAT_HIST_MAX 条; /reset 清空, /say <文本> 跳过大模型
 * 直接朗读。密钥来自 --api-key 或环境变量 CHAT_API_KEY, 不写进仓库。
 * -------------------------------------------------------------------------*/
#define CHAT_HIST_MAX 16

typedef struct {
    char* role;                  /* "user" / "assistant" */
    char* content;
} ChatMsg;

typedef struct {
    ChatMsg hist[CHAT_HIST_MAX];
    int     n;
    int     drop;                /* 环形队列起点 */
    char    host[160];
    char    path[160];
    char    model[64];
    char    key[160];
    char    effort[16];             /* reasoning effort, "none" = 不带 */
} Chat;

static void chat_init(Chat* c, const wchar_t* api_url, const wchar_t* api_model,
                      const wchar_t* api_key_w, const wchar_t* api_effort)
{
    memset(c, 0, sizeof(*c));
    strcpy(c->host, "maas-api.cn-huabei-1.xf-yun.com");
    strcpy(c->path, "/v1/responses");
    strcpy(c->model, "spark-x2.6");
    strcpy(c->effort, "low");       /* 实测 8.4s -> 6.0s, 答案不受影响 */

    if (api_url && api_url[0]) {
        char* u = w2u(api_url);
        if (u) {
            const char* hp = u;
            if (!strncmp(hp, "https://", 8)) hp += 8;
            const char* slash = strchr(hp, '/');
            if (slash) {
                size_t hl = (size_t)(slash - hp);
                if (hl > 0 && hl < sizeof(c->host)) {
                    memcpy(c->host, hp, hl);
                    c->host[hl] = 0;
                    strncpy(c->path, slash, sizeof(c->path) - 1);
                    c->path[sizeof(c->path) - 1] = 0;
                }
            }
            free(u);
        }
    }
    if (api_model && api_model[0]) {
        char* m = w2u(api_model);
        if (m) { strncpy(c->model, m, sizeof(c->model) - 1); c->model[sizeof(c->model) - 1] = 0; free(m); }
    }
    if (api_effort && api_effort[0]) {
        char* e = w2u(api_effort);
        if (e) { strncpy(c->effort, e, sizeof(c->effort) - 1); c->effort[sizeof(c->effort) - 1] = 0; free(e); }
    }
    if (api_key_w && api_key_w[0]) {
        char* k = w2u(api_key_w);
        if (k) { strncpy(c->key, k, sizeof(c->key) - 1); c->key[sizeof(c->key) - 1] = 0; free(k); }
    } else {
        const char* env = getenv("CHAT_API_KEY");
        if (env) strncpy(c->key, env, sizeof(c->key) - 1), c->key[sizeof(c->key) - 1] = 0;
    }
}

static void chat_push(Chat* c, const char* role, const char* content)
{
    if (c->n < CHAT_HIST_MAX) {
        c->hist[c->n].role = _strdup(role);
        c->hist[c->n].content = _strdup(content);
        c->n++;
    } else {
        free(c->hist[c->drop].role);
        free(c->hist[c->drop].content);
        c->hist[c->drop].role = _strdup(role);
        c->hist[c->drop].content = _strdup(content);
        c->drop = (c->drop + 1) % CHAT_HIST_MAX;
    }
}

static void chat_reset(Chat* c)
{
    for (int i = 0; i < c->n; i++) {
        free(c->hist[i].role);
        free(c->hist[i].content);
    }
    c->n = 0;
    c->drop = 0;
    printf("chat     : 对话已清空\n");
}

/* POST JSON 并读回整个响应体 (NUL 结尾); 失败返回 NULL */
static char* chat_post(Chat* c, const char* body)
{
    DWORD tmo = 30000;
    HINTERNET net = InternetOpenA("moss_tts", INTERNET_OPEN_TYPE_PRECONFIG,
                                  NULL, NULL, 0);
    if (!net) { fprintf(stderr, "  [chat] InternetOpen 失败\n"); return NULL; }
    InternetSetOptionA(net, INTERNET_OPTION_CONNECT_TIMEOUT, &tmo, sizeof(tmo));
    InternetSetOptionA(net, INTERNET_OPTION_RECEIVE_TIMEOUT, &tmo, sizeof(tmo));
    InternetSetOptionA(net, INTERNET_OPTION_SEND_TIMEOUT, &tmo, sizeof(tmo));

    HINTERNET conn = InternetConnectA(net, c->host, INTERNET_DEFAULT_HTTPS_PORT,
                                      NULL, NULL, INTERNET_SERVICE_HTTP, 0, 0);
    if (!conn) {
        fprintf(stderr, "  [chat] 连不上 %s\n", c->host);
        InternetCloseHandle(net);
        return NULL;
    }
    HINTERNET req = HttpOpenRequestA(conn, "POST", c->path, NULL, NULL, NULL,
                                     INTERNET_FLAG_SECURE |
                                     INTERNET_FLAG_RELOAD, 0);
    if (!req) { InternetCloseHandle(conn); InternetCloseHandle(net); return NULL; }

    char hdr[512];
    snprintf(hdr, sizeof(hdr),
             "Content-Type: application/json\r\nAuthorization: Bearer %s\r\n",
             c->key);
    HttpAddRequestHeadersA(req, hdr, (DWORD)-1L, HTTP_ADDREQ_FLAG_ADD);

    char* resp = NULL;
    if (!HttpSendRequestA(req, NULL, 0, (void*)body, (DWORD)strlen(body))) {
        fprintf(stderr, "  [chat] 请求发送失败 (%lu)\n", GetLastError());
        goto done;
    }
    {
        DWORD st = 0, sz = sizeof(st);
        HttpQueryInfoA(req, HTTP_QUERY_STATUS_CODE | HTTP_QUERY_FLAG_NUMBER,
                       &st, &sz, 0);
        if (st != 200) {
            fprintf(stderr, "  [chat] HTTP %lu\n", st);
            goto done;
        }
    }
    {
        size_t cap = 1 << 16, len = 0;
        resp = (char*)xmalloc(cap);
        for (;;) {
            if (len + 4096 + 1 > cap) {
                cap *= 2;
                char* nb = (char*)realloc(resp, cap);
                if (!nb) die("out of memory");
                resp = nb;
            }
            DWORD got = 0;
            if (!InternetReadFile(req, resp + len, (DWORD)(cap - len - 1), &got))
                break;
            if (got == 0) break;
            len += got;
        }
        resp[len] = 0;
        if (len == 0) { free(resp); resp = NULL; }
    }

done:
    InternetCloseHandle(req);
    InternetCloseHandle(conn);
    InternetCloseHandle(net);
    return resp;
}

/* 去掉增补平面字符 (emoji 等, 4 字节 UTF-8): 进 TTS 无意义还可能出怪音 */
static void strip_emoji(char* s)
{
    char* w = s;
    for (const unsigned char* r = (const unsigned char*)s; *r; ) {
        if (r[0] >= 0xF0) {              /* 4 字节序列整体跳过 */
            int k = 1;
            while (r[k] && (r[k] & 0xC0) == 0x80 && k < 4) k++;
            r += k;
            continue;
        }
        *w++ = (char)*r++;
    }
    *w = 0;
}

/* 提取最终答案: 顶层 output_text; 退化时取第一个 text (summary 之外也可能有) */
static char* chat_reply_text(const char* resp)
{
    const char* end = resp + strlen(resp);
    char out[32768];
    if (js_find_str(resp, end, "output_text", out, sizeof(out)) && out[0])
        return _strdup(out);
    if (js_find_str(resp, end, "text", out, sizeof(out)) && out[0])
        return _strdup(out);
    return NULL;
}

/* 组请求体: {"model":"...","input":[{"role":..,"content":..}, ...],"stream":false} */
static char* chat_build_body(Chat* c, int stream)
{
    size_t cap = 1 << 16, len = 0;
    char* b = (char*)xmalloc(cap);
    b[0] = 0;
    char esc[16384];

#define BAPPEND(s)                                                          \
    do {                                                                     \
        size_t _n = strlen(s);                                               \
        while (len + _n + 1 > cap) {                                         \
            cap *= 2;                                                        \
            char* nb = (char*)realloc(b, cap);                               \
            if (!nb) die("out of memory");                                   \
            b = nb;                                                          \
        }                                                                    \
        memcpy(b + len, s, _n);                                              \
        len += _n;                                                           \
        b[len] = 0;                                                          \
    } while (0)

    BAPPEND("{\"model\":\"");
    json_escape_into(c->model, esc, (int)sizeof(esc));
    BAPPEND(esc);
    BAPPEND("\",\"input\":[");
    for (int i = 0; i < c->n; i++) {
        ChatMsg* m = &c->hist[(c->drop + i) % CHAT_HIST_MAX];
        if (c->n < CHAT_HIST_MAX) m = &c->hist[i];
        if (i) BAPPEND(",");
        BAPPEND("{\"role\":\"");
        BAPPEND(m->role);
        BAPPEND("\",\"content\":\"");
        json_escape_into(m->content, esc, (int)sizeof(esc));
        BAPPEND(esc);
        BAPPEND("\"}");
    }
    BAPPEND("]");
    if (strcmp(c->effort, "none")) {
        BAPPEND(",\"reasoning\":{\"effort\":\"");
        BAPPEND(c->effort);
        BAPPEND("\"}");
    }
    BAPPEND(stream ? ",\"stream\":true}" : ",\"stream\":false}");
#undef BAPPEND
    return b;
}

/* ---- SSE 流式对话: 模型边吐答案增量, 按句切开即播 ---- */

/* 一轮对话的累计统计 */
typedef struct {
    double   audio_s;      /* 累计音频时长 */
    uint64_t first_ms;     /* 从提问到首个出声 (0=未出声) */
    int      n;            /* 已合成句数 */
} TurnStats;

/* 统一的"朗读一段文本": 归一化 -> (可选)流式播放 -> 编号写盘 -> 记账 */
static void speak_text(Models* m, SP* sp, const char* text,
                       const int16_t* ref, size_t ref_frames, uint64_t seed,
                       int max_frames, int verbose, int use_eos,
                       Player* pl, int out_given, const wchar_t* out_path,
                       int* req, TurnStats* ts, uint64_t t0, int quiet)
{
    char* norm = normalize_for_tts(text);
    if (!norm || !norm[0]) { free(norm); return; }
    uint64_t rq = now_ms();
    if (pl) player_request(pl);
    size_t tot = 0;
    float* audio = synthesize(m, sp, norm, ref, ref_frames, seed, max_frames,
                              verbose, use_eos, L"", pl, &tot);
    (*req)++;
    if (audio) {
        if (out_given) {
            wchar_t stem[MAX_PATH * 2], ext[32], cur[MAX_PATH * 2 + 32];
            wcscpy(stem, out_path);
            wchar_t* dot = wcsrchr(stem, L'.');
            if (dot) { wcsncpy(ext, dot, 31); ext[31] = 0; *dot = 0; }
            else wcscpy(ext, L".wav");
            _snwprintf(cur, MAX_PATH * 2 + 32, L"%s_%03d%s", stem, *req, ext);
            wav_write_stereo(cur, audio, tot, CHANNELS, SAMPLE_RATE);
            char* cu = w2u(cur);
            printf("wrote     : %s\n", cu ? cu : "?");
            free(cu);
        }
        double as = (double)tot / SAMPLE_RATE;
        double gs = (double)(now_ms() - rq) / 1000.0;
        if (ts) {
            ts->audio_s += as;
            ts->n++;
            if (!ts->first_ms && pl && pl->t_first && pl->t_first > t0)
                ts->first_ms = pl->t_first - t0;
        }
        if (!quiet) {
            if (pl && pl->live)
                printf("[stats]   : 首字 %llu ms | 音频 %.2f s | 生成 %.2f s | RTF %.2f | 播放 %.2f s\n",
                       (unsigned long long)(pl->t_first ? pl->t_first - rq : 0),
                       as, gs, as > 0.0 ? gs / as : 0.0,
                       (double)pl->emitted / SAMPLE_RATE);
            else
                printf("[stats]   : 首字 %llu ms | 音频 %.2f s | 生成 %.2f s | RTF %.2f\n",
                       (unsigned long long)(pl->t_first ? pl->t_first - rq : 0),
                       as, gs, as > 0.0 ? gs / as : 0.0);
        }
        free(audio);
    } else {
        fprintf(stderr, "  [warn] 没有生成音频\n");
    }
    free(norm);
}

/* 等声卡队列播完 (回提示符前用, 免得"打完字听到的还是上一句的尾巴") */
static void player_drain(Player* p)
{
    if (!p->live) return;
    for (int t = 0; t < 3000; t++) {
        int busy = 0;
        for (int k = 0; k < PL_NBUF; k++)
            if (p->hdr[k].dwFlags & WHDR_INQUEUE) { busy = 1; break; }
        if (!busy) break;
        Sleep(20);
    }
}

/* 句子队列: SSE 线程 push, 主线程 pop 即播 */
typedef struct {
    CRITICAL_SECTION cs;
    HANDLE    ev;          /* auto-reset: 有新句或流结束 */
    char**    q;
    int       n, cap;
    int       done;
} SentQ;

static void sq_init(SentQ* s)
{
    memset(s, 0, sizeof(*s));
    InitializeCriticalSection(&s->cs);
    s->ev = CreateEvent(NULL, FALSE, FALSE, NULL);
}

static void sq_free(SentQ* s)
{
    DeleteCriticalSection(&s->cs);
    CloseHandle(s->ev);
    free(s->q);
}

static void sq_push(SentQ* s, char* owned)
{
    EnterCriticalSection(&s->cs);
    if (s->n == s->cap) {
        s->cap = s->cap ? s->cap * 2 : 16;
        char** nq = (char**)realloc(s->q, sizeof(char*) * (size_t)s->cap);
        if (!nq) die("out of memory");
        s->q = nq;
    }
    s->q[s->n++] = owned;
    LeaveCriticalSection(&s->cs);
    SetEvent(s->ev);
}

/* 取一句; 队列空且流已结束返回 NULL */
static char* sq_pop(SentQ* s)
{
    for (;;) {
        EnterCriticalSection(&s->cs);
        if (s->n > 0) {
            char* r = s->q[0];
            memmove(s->q, s->q + 1, sizeof(char*) * (size_t)(s->n - 1));
            s->n--;
            LeaveCriticalSection(&s->cs);
            return r;
        }
        int done = s->done;
        LeaveCriticalSection(&s->cs);
        if (done) return NULL;
        WaitForSingleObject(s->ev, 500);
    }
}

static void sq_finish(SentQ* s)
{
    EnterCriticalSection(&s->cs);
    s->done = 1;
    LeaveCriticalSection(&s->cs);
    SetEvent(s->ev);
}

/* SSE 流式 POST: 逐行解析事件, 答案增量 (response.output_text.delta 的
 * "delta" 字段) 通过回调吐出; 思维链增量 (reasoning_summary_text.delta)
 * 跳过。读到流结束返回 1 (HTTP 层成功与否)。 */
typedef void (*delta_fn)(void* ctx, const char* delta);

static int chat_post_stream(Chat* c, const char* body, delta_fn cb, void* ctx)
{
    DWORD tmo = 120000;
    HINTERNET net = InternetOpenA("moss_tts", INTERNET_OPEN_TYPE_PRECONFIG,
                                  NULL, NULL, 0);
    if (!net) { fprintf(stderr, "  [chat] InternetOpen 失败\n"); return 0; }
    InternetSetOptionA(net, INTERNET_OPTION_CONNECT_TIMEOUT, &tmo, sizeof(tmo));
    InternetSetOptionA(net, INTERNET_OPTION_RECEIVE_TIMEOUT, &tmo, sizeof(tmo));
    InternetSetOptionA(net, INTERNET_OPTION_SEND_TIMEOUT, &tmo, sizeof(tmo));

    HINTERNET conn = InternetConnectA(net, c->host, INTERNET_DEFAULT_HTTPS_PORT,
                                      NULL, NULL, INTERNET_SERVICE_HTTP, 0, 0);
    HINTERNET req = conn ? HttpOpenRequestA(conn, "POST", c->path, NULL, NULL,
                                            NULL, INTERNET_FLAG_SECURE |
                                            INTERNET_FLAG_RELOAD, 0) : NULL;
    if (!req) {
        fprintf(stderr, "  [chat] 连不上 %s\n", c->host);
        if (conn) InternetCloseHandle(conn);
        InternetCloseHandle(net);
        return 0;
    }
    char hdr[512];
    snprintf(hdr, sizeof(hdr),
             "Content-Type: application/json\r\nAuthorization: Bearer %s\r\n",
             c->key);
    HttpAddRequestHeadersA(req, hdr, (DWORD)-1L, HTTP_ADDREQ_FLAG_ADD);

    int ok = 0;
    if (HttpSendRequestA(req, NULL, 0, (void*)body, (DWORD)strlen(body))) {
        DWORD st = 0, sz = sizeof(st);
        HttpQueryInfoA(req, HTTP_QUERY_STATUS_CODE | HTTP_QUERY_FLAG_NUMBER,
                       &st, &sz, 0);
        if (st != 200) {
            fprintf(stderr, "  [chat] HTTP %lu\n", st);
        } else {
            ok = 1;
            /* 行缓冲: SSE 事件按行分隔, data: 行内是 JSON */
            char* line = (char*)xmalloc(65536);
            size_t ln = 0;
            char chunk[16384];
            for (;;) {
                DWORD got = 0;
                if (!InternetReadFile(req, chunk, sizeof(chunk) - 1, &got))
                    break;
                if (!got) break;
                chunk[got] = 0;
                for (const char* p = chunk; *p; p++) {
                    if (*p == '\n' || ln + 2 >= 65536) {
                        line[ln] = 0;
                        if (!strncmp(line, "data:", 5)) {
                            const char* j = line + 5;
                            while (*j == ' ') j++;
                            const char* jend = j + strlen(j);
                            char d[16384];
                            if (strstr(j, "\"type\":\"response.output_text.delta\"") &&
                                js_find_str(j, jend, "delta", d, sizeof(d)) && d[0]) {
                                cb(ctx, d);
                            }
                        }
                        ln = 0;
                        if (*p != '\n') line[ln++] = *p;
                    } else {
                        line[ln++] = *p;
                    }
                }
            }
            free(line);
        }
    } else {
        fprintf(stderr, "  [chat] 请求发送失败 (%lu)\n", GetLastError());
    }
    InternetCloseHandle(req);
    InternetCloseHandle(conn);
    InternetCloseHandle(net);
    return ok;
}

/* SSE 工作线程上下文 */
typedef struct {
    Chat*    c;
    char*    body;
    SentQ*   q;
    CRITICAL_SECTION cs;     /* 保护 full/pend */
    char*    full;           /* 完整答案 (入对话历史) */
    size_t   full_n, full_cap;
    char     pend[16384];    /* 未成句的增量 */
    size_t   pend_n;
    int      saw_delta;
    int      ok;
} SseJob;

static void sse_append(SseJob* j, const char* s, size_t n)
{
    EnterCriticalSection(&j->cs);
    if (j->full_n + n + 1 > j->full_cap) {
        size_t nc = j->full_cap ? j->full_cap : 65536;
        while (nc < j->full_n + n + 1) nc *= 2;
        char* nb = (char*)realloc(j->full, nc);
        if (!nb) die("out of memory");
        j->full = nb;
        j->full_cap = nc;
    }
    memcpy(j->full + j->full_n, s, n);
    j->full_n += n;
    j->full[j->full_n] = 0;
    LeaveCriticalSection(&j->cs);
}

/* 句末标点: 增量攒够一句就 push 给主线程 */
static const char* k_sent_enders[] = {
    "\xe3\x80\x82",          /* 。 */
    "!", "?", ";",
    "\xef\xbc\x81",          /* ! */
    "\xef\xbc\x9f",          /* ? */
    "\xef\xbc\x9b",          /* ; */
    "\xe2\x80\xa6",          /* … */
    "\n",
};

static void sse_cut_sentences(SseJob* j)
{
    for (;;) {
        const char* best = NULL;
        size_t best_len = 0;
        for (size_t e = 0; e < sizeof(k_sent_enders) / sizeof(k_sent_enders[0]); e++) {
            size_t el = strlen(k_sent_enders[e]);
            const char* hit = NULL;
            for (size_t i = 0; i + el <= j->pend_n; i++) {
                if (!memcmp(j->pend + i, k_sent_enders[e], el)) { hit = j->pend + i; break; }
            }
            if (hit && (!best || hit < best)) { best = hit; best_len = el; }
        }
        /* 无句末标点但积压太长 (约 300 字节, CJK ~100 字): 强制断句 */
        if (!best && j->pend_n >= 300) {
            best = j->pend + 300;
            best_len = 0;
        }
        if (!best) return;
        size_t cut = (size_t)(best - j->pend) + best_len;
        if (cut < 2) {                       /* 只剩标点本身, 不成句 */
            memmove(j->pend, j->pend + cut, j->pend_n - cut);
            j->pend_n -= cut;
            continue;
        }
        char* sent = (char*)xmalloc(cut + 1);
        memcpy(sent, j->pend, cut);
        sent[cut] = 0;
        memmove(j->pend, j->pend + cut, j->pend_n - cut);
        j->pend_n -= cut;
        sq_push(j->q, sent);
    }
}

static void on_delta(void* ctx, const char* delta)
{
    SseJob* j = (SseJob*)ctx;
    char d[16384];
    strncpy(d, delta, sizeof(d) - 1);
    d[sizeof(d) - 1] = 0;
    strip_emoji(d);
    if (!d[0]) return;
    j->saw_delta = 1;
    sse_append(j, d, strlen(d));
    EnterCriticalSection(&j->cs);
    size_t dl = strlen(d);
    if (j->pend_n + dl < sizeof(j->pend)) {
        memcpy(j->pend + j->pend_n, d, dl);
        j->pend_n += dl;
        j->pend[j->pend_n] = 0;
    }
    LeaveCriticalSection(&j->cs);
    sse_cut_sentences(j);
}

static DWORD WINAPI sse_worker(LPVOID p)
{
    SseJob* j = (SseJob*)p;
    j->ok = chat_post_stream(j->c, j->body, on_delta, j);
    EnterCriticalSection(&j->cs);
    if (j->pend_n) {
        j->pend[j->pend_n] = 0;
        char* tail = _strdup(j->pend);
        LeaveCriticalSection(&j->cs);
        strip_emoji(tail);
        if (tail[strspn(tail, " \t\r")]) sq_push(j->q, tail);
        else free(tail);
    } else {
        LeaveCriticalSection(&j->cs);
    }
    sq_finish(j->q);
    return 0;
}

/* ---------------------------------------------------------------------------
 * 8.6 P 核自动钉核
 *
 * Intel 12 代起的混合架构: P 核带超线程 (1 物理 2 逻辑), E 核没有 (1:1)。
 * 把进程钉在所有"带 SMT 的核"上, ORT 工作线程就不会漂到能效核 ---- 实测
 * 每帧 30+30ms -> 26+26ms (~15%)。非混合 CPU 检测结果是全量掩码, 等于不钉。
 * -------------------------------------------------------------------------*/
static DWORD_PTR auto_p_core_mask(void)
{
    DWORD len = 0;
    /* 第一次调用预期失败 (ERROR_INSUFFICIENT_BUFFER), 只要 len 填上了就继续 */
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    if (!len) return 0;
    BYTE* buf = (BYTE*)xmalloc(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
                                          (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) {
        free(buf);
        return 0;
    }
    DWORD_PTR smt = 0, all = 0;
    BYTE* p = buf;
    while (p < buf + len) {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* e = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)p;
        if (e->Relationship == RelationProcessorCore) {
            DWORD_PTR m = e->Processor.GroupMask[0].Mask;   /* 单处理器组假设 (<64 LP) */
            all |= m;
            int bits = 0;
            for (DWORD_PTR t = m; t; t >>= 1) bits += (int)(t & 1);
            if (bits >= 2) smt |= m;                        /* 带超线程的核 = P 核 */
        }
        if (e->Size == 0) break;
        p += e->Size;
    }
    free(buf);
    return smt ? smt : all;
}

/* ---------------------------------------------------------------------------
 * 9. main
 * -------------------------------------------------------------------------*/
/* 崩溃时打印出错模块, 方便定位 (ONNX Runtime 内部崩溃时很有用) */
static LONG WINAPI crash_filter(EXCEPTION_POINTERS* ep)
{
    void* addr = (void*)ep->ExceptionRecord->ExceptionAddress;
    HMODULE mod = NULL;
    char name[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCSTR)addr, &mod) && mod)
        GetModuleFileNameA(mod, name, MAX_PATH);
    fprintf(stderr, "[crash] code=0x%08lX addr=%p module=%s\n",
            (unsigned long)ep->ExceptionRecord->ExceptionCode, addr, name);
    return EXCEPTION_EXECUTE_HANDLER;
}
static void usage(void)
{
    printf(
        "MOSS-TTS-Nano 单文件 C 推理\n"
        "\n"
        "用法: moss_tts [文本] [选项]\n"
        "\n"
        "  --out <file>        输出 wav (默认 out.wav)\n"
        "  --model-dir <dir>   模型目录 (默认 models)\n"
        "  --voice <name>      内置音色 (默认 神里绫华, 见 --list-voices)\n"
        "  --ref-wav <file>    用自己的参考音频克隆音色 (优先于 --voice)\n"
        "  --save-voice <name> 把 --ref-wav 编出的参考码存进 manifest, 变成内置音色\n"
        "  --voice-desc <txt>  新音色的描述 (配合 --save-voice, 默认用名字)\n"
        "  --list-voices       列出 manifest 里的全部内置音色\n"
        "  --play              边生成边播放 (RTF<1, 流式不卡顿)\n"
        "  --server            常驻模式: 每行一段文本, 模型只加载一次 (含 --play)\n"
        "  --chat              对话模式 (配合 --server): 输入发给大模型, 回复实时语音播报\n"
        "  --api-key <k>       对话模式密钥 (或环境变量 CHAT_API_KEY)\n"
        "  --api-effort <s>    推理强度 low/medium/high/none (默认 low, 实测快 ~30%%)\n"
        "  --api-url <u>       对话 API 地址 (默认 xfyun /v1/responses)\n"
        "  --api-model <m>     对话模型 (默认 spark-x2.6)\n"
        "  --no-play           配合 --server: 只合成不播放\n"
        "  --fp32              强制原始精度 (models/ 有 .int8.onnx 变体时默认用 int8)\n"
        "  --seed <n>          随机种子 (默认 1234)\n"
        "  --max-frames <n>    单块最大帧数 (默认 375, 12.5 帧/秒)\n"
        "  --threads <n>       ONNX 线程数 (默认 4, 实测再多反而更慢)\n"
        "  --affinity <hex>    进程钉核掩码 (默认自动: 混合架构只钉 P 核, 如 ff)\n"
        "  --tokens            只打印分词结果\n"
        "  --verbose           打印生成过程\n"
        "  --no-eos            关闭 token 重复检测 (只依赖能量判据)\n"
        "  --dump-frames <f>   把生成的帧 token 导出成文本 (调试用)\n"
        "  -h, --help          帮助\n");
}

int wmain(int argc, wchar_t** argv)
{
    SetConsoleOutputCP(CP_UTF8);
    timeBeginPeriod(1);   /* 默认 15.6ms; 背压/等头的 Sleep(1) 要真睡 1ms */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    SetUnhandledExceptionFilter(crash_filter);

    char* text = NULL;
    wchar_t out_path[MAX_PATH] = L"out.wav";
    wchar_t model_dir[MAX_PATH] = L"models";
    wchar_t ref_wav[MAX_PATH] = L"";
    wchar_t save_voice[MAX_PATH] = L"";
    wchar_t voice_desc[MAX_PATH] = L"";
    uint64_t seed = 1234;
    int max_frames = MAX_FRAMES_DEF;
    int tokens_only = 0, verbose = 0, no_eos = 0, list_voices = 0;
    int play = 0, server = 0, no_play = 0, out_given = 0, fp32 = 0, chat_on = 0;
    wchar_t api_key[160] = L"", api_url[200] = L"", api_model[64] = L"", api_effort[16] = L"";
    const wchar_t* voice_w = L"神里绫华";   /* 默认音色: 用户克隆的音色 */
    wchar_t dump_frames[MAX_PATH] = L"";
    int threads = 0;
    unsigned long affinity = 0;

    /* 文本: 拼接所有非选项参数 */
    {
        size_t cap = 4096, used = 0;
        text = (char*)xcalloc(cap, 1);
        for (int i = 1; i < argc; i++) {
            const wchar_t* a = argv[i];
            if (a[0] == L'-') {
                if (!wcscmp(a, L"--out") && i + 1 < argc) { wcscpy(out_path, argv[++i]); out_given = 1; }
                else if (!wcscmp(a, L"--model-dir") && i + 1 < argc) { wcscpy(model_dir, argv[++i]); }
                else if (!wcscmp(a, L"--ref-wav") && i + 1 < argc) { wcscpy(ref_wav, argv[++i]); }
                else if (!wcscmp(a, L"--save-voice") && i + 1 < argc) { wcscpy(save_voice, argv[++i]); }
                else if (!wcscmp(a, L"--voice-desc") && i + 1 < argc) { wcscpy(voice_desc, argv[++i]); }
                else if (!wcscmp(a, L"--voice") && i + 1 < argc) { voice_w = argv[++i]; }
                else if (!wcscmp(a, L"--list-voices")) { list_voices = 1; }
                else if (!wcscmp(a, L"--play")) { play = 1; }
                else if (!wcscmp(a, L"--server")) { server = 1; }
                else if (!wcscmp(a, L"--no-play")) { no_play = 1; }
                else if (!wcscmp(a, L"--fp32")) { fp32 = 1; }
                else if (!wcscmp(a, L"--chat")) { chat_on = 1; }
                else if (!wcscmp(a, L"--api-key") && i + 1 < argc) { wcscpy(api_key, argv[++i]); }
                else if (!wcscmp(a, L"--api-effort") && i + 1 < argc) { wcscpy(api_effort, argv[++i]); }
                else if (!wcscmp(a, L"--api-url") && i + 1 < argc) { wcscpy(api_url, argv[++i]); }
                else if (!wcscmp(a, L"--api-model") && i + 1 < argc) { wcscpy(api_model, argv[++i]); }
                else if (!wcscmp(a, L"--seed") && i + 1 < argc) { seed = _wcstoui64(argv[++i], NULL, 10); }
                else if (!wcscmp(a, L"--max-frames") && i + 1 < argc) { max_frames = _wtoi(argv[++i]); }
                else if (!wcscmp(a, L"--threads") && i + 1 < argc) { threads = _wtoi(argv[++i]); }
                else if (!wcscmp(a, L"--affinity") && i + 1 < argc) { affinity = wcstoul(argv[++i], NULL, 16); }
                else if (!wcscmp(a, L"--tokens")) { tokens_only = 1; }
                else if (!wcscmp(a, L"--verbose")) { verbose = 1; }
                else if (!wcscmp(a, L"--no-eos")) { no_eos = 1; }
                else if (!wcscmp(a, L"--dump-frames") && i + 1 < argc) { wcscpy(dump_frames, argv[++i]); }
                else if (!wcscmp(a, L"-h") || !wcscmp(a, L"--help")) { usage(); return 0; }
                else {
                    char* u = w2u(a);
                    fprintf(stderr, "unknown option: %s\n", u ? u : "?");
                    free(u);
                    return 2;
                }
                continue;
            }
            char* u = w2u(a);
            size_t ul = strlen(u);
            if (used + ul + 2 > cap) {
                cap = (used + ul + 2) * 2;
                text = (char*)realloc(text, cap);
                if (!text) die("out of memory");
            }
            if (used) text[used++] = ' ';
            memcpy(text + used, u, ul);
            used += ul;
            text[used] = 0;
            free(u);
        }
    }
    if (text[0] == 0 && !server) {
        free(text);
        text = _strdup("\xE6\xAC\xA2\xE8\xBF\x8E\xE5\x85\xB3\xE6\xB3\xA8\xE6\xA8\xA1\xE6\x80\x9D"
                       "\xE6\x99\xBA\xE8\x83\xBD\xEF\xBC\x8C\xE8\xBF\x99\xE6\x98\xAF MOSS-TTS-Nano "
                       "\xE7\x9A\x84\xE7\xBA\xAF C \xE6\x8E\xA8\xE7\x90\x86\xE5\xAE\x9E\xE7\x8E\xB0\xE3\x80\x82");
    }

    if (threads <= 0) {
        /* 实测: 这个 100M 模型在 4 线程附近最快; 线程开太多 (8/16) 会因为
         * ORT 自旋等待 + 线程争抢而慢 3~5 倍。 */
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        threads = (int)si.dwNumberOfProcessors;
        if (threads <= 0) threads = 4;
        if (threads > 4) threads = 4;
    }

    /* 钉核: 显式 --affinity 优先; 否则混合架构自动钉 P 核 (带 SMT 的核) */
    {
        DWORD_PTR m = affinity ? (DWORD_PTR)affinity : auto_p_core_mask();
        if (m && SetProcessAffinityMask(GetCurrentProcess(), m))
            printf("affinity  : 0x%llX%s\n", (unsigned long long)m,
                   affinity ? "" : " (自动, 带 SMT 的核)");
        else if (affinity)
            fprintf(stderr, "[warn] SetProcessAffinityMask(0x%lX) 失败 (%lu)\n", affinity, GetLastError());
    }

    /* ---- 分词器 ---- */
    wchar_t tok_path[MAX_PATH * 2];
    _snwprintf(tok_path, MAX_PATH * 2, L"%s\\tokenizer.model", model_dir);

    uint64_t t0 = now_ms();
    SP sp;
    sp_load(&sp, tok_path);

    char* norm = normalize_for_tts(text);
    if (!server) printf("text      : %s\n", norm);

    if (tokens_only) {
        int32_t* ids = (int32_t*)xmalloc(sizeof(int32_t) * (strlen(norm) * 2 + 16));
        int n = sp_encode(&sp, norm, ids, (int)strlen(norm) * 2 + 16);
        printf("tokens(%d): [", n);
        for (int k = 0; k < n; k++) printf("%s%d", k ? ", " : "", ids[k]);
        printf("]\n");
        free(ids); free(norm); free(text);
        return 0;
    }

    /* ---- 音色 / 参考音频 ---- */
    wchar_t manifest_path[MAX_PATH * 2];
    _snwprintf(manifest_path, MAX_PATH * 2, L"%s\\browser_poc_manifest.json", model_dir);
    if (save_voice[0] && !ref_wav[0])
        die("--save-voice 需要配合 --ref-wav: 先用参考音频编出参考码才能保存");

    if (list_voices) {
        printf("内置音色 (来自 %ls):\n", manifest_path);
        if (!manifest_list_voices(manifest_path))
            fprintf(stderr, "  (读不到 manifest, 只有内嵌的 神里绫华 可用)\n");
        free(norm); free(text);
        return 0;
    }

    const int16_t* ref = k_ref_codes_ayaka;
    size_t ref_frames = REF_FRAMES_AYAKA;

    Models models;
    models_open(&models, model_dir, threads, fp32);

    int16_t* ref_owned = NULL;
    if (ref_wav[0]) {
        size_t nfr = 0;
        float* wav = wav_read_48k_stereo(ref_wav, &nfr);
        if (!wav) die("cannot read --ref-wav (需要 WAV: 16/24/32-bit PCM 或 float32)");
        nfr = prep_reference(wav, nfr);
        if (nfr < (size_t)(SAMPLE_RATE / 2)) die("--ref-wav 有效音频太短 (至少 0.5 秒)");
        /* 参考音频越长, prompt 越长, 每帧注意力越慢。15 秒足够克隆音色了。 */
        if (nfr > (size_t)SAMPLE_RATE * 15) {
            nfr = (size_t)SAMPLE_RATE * 15;
            fprintf(stderr, "  [note] 参考音频超过 15 秒, 只取前 15 秒\n");
        }
        ref_owned = codec_encode(&models, wav, nfr, &ref_frames);
        free(wav);
        ref = ref_owned;
        printf("ref-wav   : %.2f s -> %zu 帧 (声音克隆)\n", (double)nfr / SAMPLE_RATE, ref_frames);
        if (save_voice[0]) {
            char* sv = w2u(save_voice);
            char* sd = w2u(voice_desc[0] ? voice_desc : save_voice);
            if (!manifest_save_voice(manifest_path, sv, sd, ref_wav, ref_owned, ref_frames))
                printf("saved     : \"%s\" (%zu 帧) -> manifest, 之后可直接 --voice %s\n",
                       sv, ref_frames, sv);
            free(sv); free(sd);
        }
    } else {
        if (resolve_voice(manifest_path, voice_w, &ref, &ref_frames, &ref_owned) != 0)
            exit(2);
    }
    printf("threads   : %d\n", threads);
    if (verbose) {
        size_t priv = 0, ws = cur_working_set(&priv);
        fprintf(stderr, "  [mem] 模型加载后 working set %.0f MB, private %.0f MB\n",
                (double)ws / 1048576.0, (double)priv / 1048576.0);
    }

    /* ---- 流式播放器 + 合成 ---- */
    Player player;
    memset(&player, 0, sizeof(player));
    int use_player = (play || server);
    if (use_player) player_open(&player, !no_play);
    Player* pl = use_player ? &player : NULL;

    if (server) {
        /* 常驻模式: 模型只加载一次, 每行一段文本; 热启动后免掉 3s+ 的加载 */
        SetConsoleCP(CP_UTF8);        /* 交互终端下让 stdin 按 UTF-8 收 */
        SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
        printf("server    : 每行一段文本回车合成; /voice <名> 换音色, /seed <n> 换种子, /quit 退出\n");
        /* 预热: ORT 首次 Run 要建执行计划, 不预热的话首请求首字 ~2.6s */
        {
            uint64_t tw = now_ms();
            size_t wu_tot = 0;
            float* wu = synthesize(&models, &sp, "预热。", ref, ref_frames, seed,
                                  8, 0, !no_eos, L"", NULL, &wu_tot);
            free(wu);
            printf("warmup    : %llu ms (首请求不再慢)\n", (unsigned long long)(now_ms() - tw));
        }
        Chat chat;
        if (chat_on) {
            chat_init(&chat, api_url, api_model, api_key, api_effort);
            if (!chat.key[0]) {
                fprintf(stderr, "[chat] 缺少 API key (--api-key 或环境变量 CHAT_API_KEY), 对话模式未开启\n");
                chat_on = 0;
            } else {
                printf("chat      : %s%s (模型 %s); 直接输入对话, /reset 清空, /say <文本> 跳过大模型直接朗读\n",
                       chat.host, chat.path, chat.model);
            }
        }
        char line[16384];
        int req = 0;
        int tty = _isatty(_fileno(stdin));
        for (;;) {
            if (tty) printf("\n> ");
            if (!fgets(line, sizeof(line), stdin)) break;
            size_t ll = strlen(line);
            while (ll && (line[ll - 1] == '\n' || line[ll - 1] == '\r')) line[--ll] = 0;
            if (!ll) continue;

            int direct_say = 0;
            if (line[0] == '/') {
                if (!strncmp(line, "/quit", 5)) break;
                if (!strncmp(line, "/say", 4) && (line[4] == ' ' || line[4] == '\t')) {
                    memmove(line, line + 5, strlen(line + 5) + 1);
                    ll = strlen(line);
                    if (!ll) continue;
                    direct_say = 1;      /* 跳过大模型直接朗读 */
                } else if (!strncmp(line, "/reset", 6) && chat_on) {
                    chat_reset(&chat);
                    continue;
                } else if (!strncmp(line, "/voice", 6) && (line[6] == ' ' || line[6] == '\t')) {
                    const char* vn = line + 6;
                    while (*vn == ' ' || *vn == '\t') vn++;
                    wchar_t* vw = u2w(vn);
                    const int16_t* nref = NULL;
                    size_t nfr = 0;
                    int16_t* nown = NULL;
                    if (vw && resolve_voice(manifest_path, vw, &nref, &nfr, &nown) == 0) {
                        free(ref_owned);
                        ref_owned = nown;
                        ref = nref;
                        ref_frames = nfr;
                    }
                    free(vw);
                    continue;
                } else if (!strncmp(line, "/seed", 5) && (line[5] == ' ' || line[5] == '\t')) {
                    seed = _strtoui64(line + 5, NULL, 10);
                    printf("seed      : %llu\n", (unsigned long long)seed);
                    continue;
                } else {
                    fprintf(stderr, "未知命令: %s (可用: /voice <名> /seed <n> /reset /say <文本> /quit)\n", line);
                    continue;
                }
            }

            /* 对话模式: SSE 流式取回回复, 按句切开边收边说 (思维链增量跳过) */
            if (!direct_say && chat_on) {
                uint64_t tq = now_ms();
                chat_push(&chat, "user", line);
                char* body = chat_build_body(&chat, 1);
                SentQ sq;
                sq_init(&sq);
                SseJob job;
                memset(&job, 0, sizeof(job));
                job.c = &chat;
                job.body = body;
                job.q = &sq;
                InitializeCriticalSection(&job.cs);
                HANDLE th = CreateThread(NULL, 0, sse_worker, &job, 0, NULL);

                TurnStats ts;
                memset(&ts, 0, sizeof(ts));
                for (;;) {
                    char* s = sq_pop(&sq);
                    if (!s) break;
                    if (ts.n == 0)
                        printf("AI (首句): %s\n", s);
                    speak_text(&models, &sp, s, ref, ref_frames, seed, max_frames,
                               verbose, !no_eos, pl, out_given, out_path,
                               &req, &ts, tq, 1);
                    free(s);
                }
                WaitForSingleObject(th, INFINITE);
                CloseHandle(th);
                char* reply = job.full;      /* 完整答案, 入对话历史 */
                int ok = job.ok;
                DeleteCriticalSection(&job.cs);
                sq_free(&sq);
                free(body);

                /* 一个答案增量都没有 (老网关/被拦): 回退非流式整段 */
                if (!reply || !reply[0]) {
                    fprintf(stderr, "  [chat] 流里没有答案增量, 回退非流式\n");
                    body = chat_build_body(&chat, 0);
                    char* resp = chat_post(&chat, body);
                    free(body);
                    reply = resp ? chat_reply_text(resp) : NULL;
                    free(resp);
                    if (reply && reply[0]) {
                        strip_emoji(reply);
                        printf("AI       : %s  (%llu ms)\n", reply,
                               (unsigned long long)(now_ms() - tq));
                        speak_text(&models, &sp, reply, ref, ref_frames, seed,
                                   max_frames, verbose, !no_eos, pl, out_given,
                                   out_path, &req, &ts, tq, 0);
                    }
                } else {
                    printf("AI       : %s  (%llu ms)\n", reply,
                           (unsigned long long)(now_ms() - tq));
                }

                if (reply && reply[0])
                    chat_push(&chat, "assistant", reply);
                free(reply);

                if (ts.n > 0)
                    printf("[stats]   : 首句出声 %llu ms | %d 句 | 音频 %.2f s | 本轮 %llu ms\n",
                           (unsigned long long)ts.first_ms, ts.n, ts.audio_s,
                           (unsigned long long)(now_ms() - tq));
                player_drain(&player);
                continue;
            }

            /* /say 与非对话模式: 直接朗读原文 */
            speak_text(&models, &sp, line, ref, ref_frames, seed, max_frames,
                       verbose, !no_eos, pl, out_given, out_path, &req, NULL, 0, 0);
            player_drain(&player);
        }
        printf("\nserver    : 共 %d 次请求, 退出\n", req);
    } else {
        size_t total_samples = 0;
        float* all = synthesize(&models, &sp, norm, ref, ref_frames, seed,
                                max_frames, verbose, !no_eos, dump_frames, pl, &total_samples);
        if (!all) die("nothing generated");

        wav_write_stereo(out_path, all, total_samples, CHANNELS, SAMPLE_RATE);

        uint64_t t4 = now_ms();
        char* op = w2u(out_path);
        printf("\nwrote %s : %.2f s @ %d Hz stereo\n", op ? op : "?",
               (double)total_samples / SAMPLE_RATE, SAMPLE_RATE);
        printf("total     : %llu ms\n", (unsigned long long)(t4 - t0));
        free(op);
        free(all);
    }

    if (use_player) player_close(&player);

    g_stats.peak_rss = peak_working_set();
    if (verbose) {
        size_t load_sum = g_stats.load_prefill + g_stats.load_decode + g_stats.load_frame +
                            g_stats.load_dec + g_stats.load_enc;
        fprintf(stderr, "\n--- 耗时明细 ---\n");
        fprintf(stderr, "  模型加载 : %5llu ms  (prefill %llu / decode %llu / frame %llu / codec-dec %llu / codec-enc %llu)\n",
                (unsigned long long)load_sum,
                (unsigned long long)g_stats.load_prefill, (unsigned long long)g_stats.load_decode,
                (unsigned long long)g_stats.load_frame, (unsigned long long)g_stats.load_dec,
                (unsigned long long)g_stats.load_enc);
        fprintf(stderr, "  prefill  : %5llu ms\n", (unsigned long long)g_stats.prefill);
        fprintf(stderr, "  采样帧   : %5llu ms  (%d 次, 均 %.1f ms)\n",
                (unsigned long long)g_stats.frame, g_stats.steps,
                g_stats.steps ? (double)g_stats.frame / g_stats.steps : 0.0);
        fprintf(stderr, "  decode步 : %5llu ms  (均 %.1f ms)\n",
                (unsigned long long)g_stats.step,
                g_stats.steps ? (double)g_stats.step / g_stats.steps : 0.0);
        fprintf(stderr, "  流式解码 : %5llu ms  (含能量判据)\n", (unsigned long long)g_stats.probe);
        size_t priv2 = 0, ws2 = cur_working_set(&priv2);
        fprintf(stderr, "  峰值内存 : %.1f MB (结束时 %.1f MB / private %.1f MB)\n",
                (double)g_stats.peak_rss / 1048576.0, (double)ws2 / 1048576.0, (double)priv2 / 1048576.0);
    }
    free(ref_owned);
    free(norm);
    free(text);

    models_close(&models);
    if (g_meminfo) ort->ReleaseMemoryInfo(g_meminfo);
    if (g_env) ort->ReleaseEnv(g_env);
    return 0;
}
