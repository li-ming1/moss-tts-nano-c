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
 * ==========================================================================*/

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
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
    OK(ort->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL));
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

/* ---- built-in voice "Junhao" (CN 欢迎关注模思智能), 98 frames x 16 codebooks ---- */
static const int16_t k_ref_codes_junhao[1568] = {
    922, 507, 253, 140, 700, 228, 523, 612, 20, 194, 877, 682, 415, 244, 181, 195,
    503, 720, 733, 615, 153, 759, 1023, 541, 744, 1015, 245, 68, 323, 183, 307, 854,
    829, 958, 621, 750, 839, 865, 683, 955, 102, 557, 317, 968, 391, 539, 671, 425,
    295, 776, 279, 19, 61, 173, 65, 189, 923, 586, 42, 125, 442, 341, 462, 90,
    556, 787, 992, 553, 436, 483, 618, 160, 504, 550, 402, 169, 607, 576, 72, 507,
    537, 11, 699, 752, 135, 176, 122, 587, 722, 1016, 732, 103, 818, 680, 794, 605,
    314, 426, 584, 58, 853, 859, 349, 113, 295, 128, 939, 692, 225, 203, 107, 881,
    568, 471, 263, 796, 587, 485, 874, 63, 918, 571, 191, 528, 308, 807, 979, 412,
    363, 1018, 425, 985, 140, 7, 264, 833, 943, 111, 568, 241, 630, 759, 444, 164,
    101, 956, 551, 11, 250, 581, 458, 487, 105, 74, 2, 382, 141, 818, 414, 652,
    650, 814, 47, 264, 24, 509, 691, 425, 518, 997, 802, 711, 960, 154, 400, 93,
    64, 296, 383, 755, 333, 962, 864, 917, 561, 1022, 403, 329, 5, 461, 527, 18,
    522, 631, 510, 341, 247, 290, 782, 670, 758, 395, 102, 571, 394, 356, 746, 857,
    522, 304, 672, 148, 205, 180, 900, 807, 665, 302, 47, 480, 638, 258, 518, 304,
    522, 631, 200, 526, 535, 378, 231, 868, 807, 798, 634, 784, 969, 993, 96, 15,
    522, 631, 366, 561, 420, 58, 557, 372, 961, 190, 812, 571, 800, 757, 672, 767,
    522, 258, 944, 526, 334, 378, 724, 1014, 759, 582, 812, 451, 547, 130, 116, 114,
    681, 489, 801, 220, 779, 901, 1007, 1000, 786, 439, 214, 508, 972, 678, 53, 951,
    129, 487, 7, 540, 588, 487, 627, 760, 717, 20, 174, 309, 567, 76, 1006, 510,
    762, 580, 568, 606, 634, 603, 541, 33, 796, 43, 239, 708, 168, 974, 135, 422,
    843, 472, 764, 106, 669, 796, 760, 587, 158, 82, 170, 13, 169, 374, 474, 975,
    643, 186, 140, 62, 748, 359, 433, 56, 102, 646, 470, 3, 887, 185, 782, 421,
    303, 237, 900, 853, 806, 193, 994, 876, 150, 202, 897, 551, 227, 286, 167, 662,
    511, 662, 411, 60, 970, 669, 153, 316, 1004, 380, 753, 260, 1006, 730, 309, 98,
    699, 511, 808, 376, 266, 113, 694, 474, 412, 436, 6, 906, 155, 120, 435, 371,
    100, 19, 600, 915, 143, 807, 851, 899, 209, 174, 594, 929, 567, 104, 503, 267,
    931, 549, 132, 1008, 530, 577, 729, 382, 983, 114, 807, 8, 213, 369, 41, 160,
    358, 787, 131, 179, 83, 173, 205, 338, 427, 12, 681, 69, 806, 115, 925, 948,
    116, 445, 200, 935, 584, 320, 558, 720, 501, 187, 67, 428, 676, 682, 758, 799,
    522, 502, 349, 1006, 791, 619, 449, 579, 879, 1017, 936, 703, 487, 175, 339, 41,
    387, 860, 824, 652, 389, 624, 494, 37, 202, 758, 306, 805, 107, 794, 771, 227,
    593, 846, 555, 723, 420, 360, 16, 211, 926, 592, 978, 35, 439, 1013, 282, 554,
    563, 864, 853, 711, 329, 664, 330, 722, 64, 767, 303, 961, 547, 282, 1013, 180,
    939, 7, 224, 860, 633, 328, 231, 69, 782, 513, 609, 528, 373, 41, 663, 356,
    559, 1007, 147, 797, 253, 828, 655, 777, 205, 732, 989, 394, 194, 25, 414, 108,
    453, 558, 815, 434, 234, 626, 387, 631, 338, 907, 78, 999, 696, 705, 477, 978,
    884, 955, 482, 679, 625, 809, 561, 513, 735, 973, 940, 39, 745, 313, 293, 698,
    627, 79, 920, 153, 672, 555, 569, 808, 257, 172, 9, 755, 59, 770, 167, 899,
    231, 746, 386, 19, 898, 406, 452, 804, 770, 437, 131, 759, 277, 490, 821, 806,
    12, 347, 140, 818, 815, 1014, 857, 827, 464, 964, 208, 772, 71, 65, 835, 798,
    774, 888, 332, 1013, 751, 167, 218, 896, 863, 743, 683, 365, 193, 502, 569, 373,
    624, 397, 383, 716, 407, 701, 795, 23, 389, 737, 1006, 217, 660, 1005, 477, 753,
    550, 338, 24, 563, 279, 896, 626, 478, 835, 829, 307, 231, 426, 928, 339, 625,
    902, 198, 166, 453, 977, 490, 655, 169, 885, 41, 370, 814, 650, 959, 827, 337,
    112, 938, 282, 551, 610, 202, 973, 932, 629, 1003, 443, 338, 329, 734, 626, 251,
    244, 593, 407, 154, 105, 789, 349, 740, 332, 499, 605, 905, 443, 83, 693, 531,
    393, 433, 112, 157, 138, 614, 188, 52, 903, 477, 65, 433, 938, 219, 747, 69,
    636, 959, 383, 117, 229, 88, 0, 427, 753, 912, 141, 737, 664, 538, 238, 205,
    78, 82, 377, 364, 41, 453, 648, 129, 589, 139, 588, 429, 358, 998, 624, 118,
    673, 971, 838, 161, 186, 996, 976, 116, 715, 12, 81, 3, 578, 971, 867, 390,
    745, 777, 935, 956, 996, 712, 58, 248, 589, 952, 3, 706, 794, 10, 307, 422,
    584, 445, 138, 198, 281, 56, 960, 429, 255, 418, 133, 749, 897, 814, 424, 231,
    358, 687, 1021, 664, 551, 46, 846, 448, 361, 36, 966, 519, 121, 674, 243, 5,
    116, 687, 897, 952, 266, 107, 23, 214, 601, 558, 51, 484, 610, 554, 623, 233,
    940, 304, 21, 373, 225, 187, 911, 502, 986, 725, 552, 721, 420, 662, 165, 726,
    522, 258, 569, 535, 250, 276, 852, 909, 334, 151, 446, 573, 873, 999, 157, 366,
    522, 631, 670, 815, 547, 865, 969, 290, 397, 953, 68, 293, 665, 681, 998, 943,
    522, 258, 904, 341, 726, 573, 793, 431, 363, 279, 957, 896, 800, 6, 443, 549,
    522, 355, 474, 148, 909, 367, 303, 233, 397, 573, 622, 978, 112, 330, 272, 20,
    830, 344, 609, 85, 742, 18, 871, 632, 465, 651, 269, 809, 807, 343, 153, 470,
    760, 73, 241, 326, 415, 44, 343, 806, 421, 733, 485, 989, 635, 786, 959, 26,
    653, 632, 860, 748, 455, 143, 49, 233, 244, 388, 30, 318, 362, 584, 850, 533,
    605, 460, 153, 544, 484, 615, 669, 203, 450, 638, 969, 363, 262, 889, 507, 1000,
    82, 292, 704, 788, 453, 144, 390, 425, 204, 196, 1014, 538, 865, 910, 178, 264,
    503, 985, 68, 969, 295, 821, 850, 874, 66, 644, 165, 751, 652, 132, 318, 486,
    92, 162, 180, 582, 603, 465, 959, 453, 202, 919, 316, 691, 506, 393, 792, 254,
    225, 1004, 501, 957, 454, 159, 801, 90, 200, 847, 855, 967, 277, 76, 526, 255,
    457, 974, 97, 674, 565, 803, 268, 469, 385, 895, 969, 1007, 862, 327, 122, 733,
    183, 144, 492, 12, 570, 660, 207, 996, 998, 462, 485, 958, 558, 838, 272, 14,
    733, 633, 557, 117, 334, 41, 1012, 694, 12, 286, 88, 44, 1007, 665, 185, 837,
    796, 7, 7, 46, 548, 846, 88, 1008, 272, 541, 21, 179, 727, 693, 967, 950,
    158, 743, 122, 41, 7, 280, 191, 839, 825, 395, 408, 758, 537, 903, 612, 312,
    64, 355, 107, 91, 836, 845, 900, 659, 1009, 201, 727, 1019, 742, 90, 157, 424,
    1002, 304, 21, 759, 803, 482, 422, 575, 397, 786, 466, 614, 611, 838, 1017, 98,
    522, 631, 474, 179, 301, 272, 825, 208, 759, 649, 500, 933, 723, 480, 89, 622,
    522, 666, 371, 179, 909, 180, 740, 2, 759, 396, 634, 854, 225, 689, 32, 281,
    522, 258, 576, 341, 686, 302, 1019, 131, 799, 953, 757, 144, 592, 999, 664, 40,
    825, 933, 493, 628, 898, 210, 384, 364, 391, 890, 831, 212, 488, 776, 177, 774,
    64, 35, 672, 178, 270, 721, 934, 876, 961, 736, 778, 53, 925, 987, 212, 819,
    64, 841, 107, 565, 448, 99, 1004, 557, 596, 37, 403, 168, 128, 102, 565, 705,
    64, 841, 107, 220, 686, 251, 588, 733, 1016, 669, 330, 967, 110, 821, 600, 153,
    522, 666, 144, 449, 243, 58, 384, 515, 438, 994, 400, 166, 838, 648, 226, 165,
    522, 494, 485, 297, 460, 180, 467, 346, 116, 786, 838, 923, 194, 113, 520, 548,
    11, 188, 931, 627, 505, 410, 46, 368, 484, 1008, 545, 361, 736, 381, 763, 641,
    129, 484, 881, 641, 227, 587, 19, 977, 825, 759, 996, 13, 531, 690, 944, 551,
    882, 431, 301, 491, 493, 461, 135, 153, 325, 144, 221, 937, 143, 57, 371, 661,
    392, 974, 859, 978, 58, 500, 153, 492, 133, 368, 990, 333, 270, 662, 709, 641,
    474, 904, 1012, 549, 320, 216, 101, 117, 199, 740, 685, 75, 296, 996, 874, 48,
    1020, 210, 660, 614, 621, 267, 305, 261, 283, 748, 24, 884, 172, 32, 348, 675,
    107, 828, 120, 479, 1021, 587, 244, 654, 462, 821, 1023, 166, 44, 604, 373, 899,
    656, 779, 299, 809, 373, 916, 465, 582, 1020, 115, 1009, 3, 788, 44, 840, 879,
    652, 960, 562, 55, 627, 764, 660, 97, 240, 958, 526, 956, 889, 151, 137, 789,
    337, 919, 190, 813, 186, 474, 441, 906, 408, 741, 143, 564, 362, 366, 332, 989,
    118, 107, 185, 67, 554, 146, 31, 349, 548, 237, 503, 429, 820, 103, 167, 454,
    67, 251, 845, 321, 184, 752, 540, 678, 768, 468, 646, 81, 256, 817, 986, 342,
    494, 349, 524, 629, 31, 505, 222, 417, 300, 433, 827, 103, 802, 589, 731, 71,
    136, 756, 28, 584, 214, 388, 946, 245, 175, 120, 839, 689, 702, 313, 158, 771,
    160, 902, 270, 381, 978, 152, 465, 694, 257, 544, 649, 142, 1019, 442, 409, 211,
};

#define REF_FRAMES_JUNHAO 98

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

static void models_open(Models* m, const wchar_t* dir, int threads)
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
        LoadJob jobs[4] = {
            { &m->prefill, dir, L"moss_tts_prefill.onnx",                    threads, &g_stats.load_prefill, 0 },
            { &m->decode,  dir, L"moss_tts_decode_step.onnx",                threads, &g_stats.load_decode, 0 },
            { &m->frame,   dir, L"moss_tts_local_fixed_sampled_frame.onnx",  threads, &g_stats.load_frame, 0 },
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

/* 生成语音: 帧生成与 codec 解码交错进行, 音频直接累积到 out_pcm
 * 返回生成的帧数 */
static size_t generate_frames(Models* m, const int32_t* text_ids, size_t n_text,
                              const int16_t* ref, size_t ref_frames,
                              float** out_pcm, size_t* out_samples,
                              uint64_t seed, int max_frames,
                              int verbose, int use_eos, const wchar_t* dump_frames)
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
    float hidden[HIDDEN];
    memcpy(hidden, hidden_all + (S - 1) * HIDDEN, sizeof(float) * HIDDEN);

    ort->ReleaseValue(pout[0]);
    pout[0] = NULL;

    OrtValue* kv[GLOBAL_LAYERS * 2];
    for (int k = 0; k < GLOBAL_LAYERS * 2; k++) kv[k] = pout[1 + k];

    ort->ReleaseValue(pin[0]);
    ort->ReleaseValue(pin[1]);
    free(ids);
    free(amask);

    /* ---- 自回归帧循环 (边生成边解码) ---- */
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

    int32_t past_len = (int32_t)S;
    int silent_batches = 0;

    int16_t* out = (int16_t*)xmalloc(sizeof(int16_t) * (size_t)max_frames * N_VQ);
    size_t n_frames = 0;

    StreamDec sd;
    sd_init(&sd);

    /* 待解码的帧: 攒够 SD_BATCH 帧一起解码 (太小调度开销大, 太大峰值内存高) */
    enum { SD_BATCH = MOSS_SD_BATCH };
    int16_t pend[SD_BATCH * N_VQ];
    int n_pend = 0;

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

        /* 攒够一批就解码; 顺便用解码出来的能量判断是不是已经说完 */
        if (n_pend == SD_BATCH) {
            uint64_t _tp = now_ms();
            sd_run(m, &sd, pend, (size_t)n_pend);
            g_stats.probe += now_ms() - _tp;
            n_pend = 0;
            if (sd.last_energy < 1.0e-5) {
                if (++silent_batches >= 2) {
                    if (verbose) fprintf(stderr, "  [step %d] trailing silence, stop\n", step);
                    break;
                }
            } else {
                silent_batches = 0;
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

    /* 把最后不足一批的帧也解码掉 */
    if (n_pend > 0) {
        uint64_t _tp = now_ms();
        sd_run(m, &sd, pend, (size_t)n_pend);
        g_stats.probe += now_ms() - _tp;
        n_pend = 0;
    }

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

    *out_pcm = sd.pcm;
    *out_samples = sd.n_samples;
    sd.pcm = NULL;
    sd_free(&sd);
    return n_frames;
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
 * 8. 内置音色表 (从 browser_poc_manifest.json 读取)
 *
 * manifest 的 builtin_voices 里有 18 个预置音色, 每个带一段参考音频的 RVQ 码。
 * 为了不把两万多个数字塞进源码, 这里用一个极简的 JSON 扫描器在运行时读取;
 * 文件缺失时退回内嵌的 Junhao。
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
        if (*p == '\\' && p + 1 < end) p++;
        if (n + 1 < cap) out[n++] = *p;
        p++;
    }
    if (p >= end) return NULL;
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
        "  --voice <name>      内置音色 (默认 Yuewen, 见 --list-voices)\n"
        "  --ref-wav <file>    用自己的参考音频克隆音色 (优先于 --voice)\n"
        "  --list-voices       列出 manifest 里的全部内置音色\n"
        "  --seed <n>          随机种子 (默认 1234)\n"
        "  --max-frames <n>    单块最大帧数 (默认 375, 12.5 帧/秒)\n"
        "  --threads <n>       ONNX 线程数 (默认 4, 实测再多反而更慢)\n"
        "  --tokens            只打印分词结果\n"
        "  --verbose           打印生成过程\n"
        "  --no-eos            关闭 token 重复检测 (只依赖能量判据)\n"
        "  --dump-frames <f>   把生成的帧 token 导出成文本 (调试用)\n"
        "  -h, --help          帮助\n");
}

int wmain(int argc, wchar_t** argv)
{
    SetConsoleOutputCP(CP_UTF8);
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    SetUnhandledExceptionFilter(crash_filter);

    char* text = NULL;
    wchar_t out_path[MAX_PATH] = L"out.wav";
    wchar_t model_dir[MAX_PATH] = L"models";
    wchar_t ref_wav[MAX_PATH] = L"";
    uint64_t seed = 1234;
    int max_frames = MAX_FRAMES_DEF;
    int tokens_only = 0, verbose = 0, no_eos = 0, list_voices = 0;
    const wchar_t* voice_w = L"Yuewen";   /* 默认音色: 台湾腔女声 */
    wchar_t dump_frames[MAX_PATH] = L"";
    int threads = 0;

    /* 文本: 拼接所有非选项参数 */
    {
        size_t cap = 4096, used = 0;
        text = (char*)xcalloc(cap, 1);
        for (int i = 1; i < argc; i++) {
            const wchar_t* a = argv[i];
            if (a[0] == L'-') {
                if (!wcscmp(a, L"--out") && i + 1 < argc) { wcscpy(out_path, argv[++i]); }
                else if (!wcscmp(a, L"--model-dir") && i + 1 < argc) { wcscpy(model_dir, argv[++i]); }
                else if (!wcscmp(a, L"--ref-wav") && i + 1 < argc) { wcscpy(ref_wav, argv[++i]); }
                else if (!wcscmp(a, L"--voice") && i + 1 < argc) { voice_w = argv[++i]; }
                else if (!wcscmp(a, L"--list-voices")) { list_voices = 1; }
                else if (!wcscmp(a, L"--seed") && i + 1 < argc) { seed = _wcstoui64(argv[++i], NULL, 10); }
                else if (!wcscmp(a, L"--max-frames") && i + 1 < argc) { max_frames = _wtoi(argv[++i]); }
                else if (!wcscmp(a, L"--threads") && i + 1 < argc) { threads = _wtoi(argv[++i]); }
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
    if (text[0] == 0) {
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

    /* ---- 分词器 ---- */
    wchar_t tok_path[MAX_PATH * 2];
    _snwprintf(tok_path, MAX_PATH * 2, L"%s\\tokenizer.model", model_dir);

    uint64_t t0 = now_ms();
    SP sp;
    sp_load(&sp, tok_path);

    char* norm = normalize_for_tts(text);
    printf("text      : %s\n", norm);

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

    if (list_voices) {
        printf("内置音色 (来自 %ls):\n", manifest_path);
        if (!manifest_list_voices(manifest_path))
            fprintf(stderr, "  (读不到 manifest, 只有内嵌的 Junhao 可用)\n");
        free(norm); free(text);
        return 0;
    }

    const int16_t* ref = k_ref_codes_junhao;
    size_t ref_frames = REF_FRAMES_JUNHAO;
    char* voice_u8 = w2u(voice_w);

    Models models;
    models_open(&models, model_dir, threads);

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
    } else if (voice_u8 && strcmp(voice_u8, "Junhao") != 0) {
        ref_owned = manifest_voice(manifest_path, voice_u8, &ref_frames);
        if (!ref_owned) {
            fprintf(stderr, "找不到音色 \"%s\", 用 --list-voices 看看有哪些\n", voice_u8);
            exit(2);
        }
        ref = ref_owned;
        printf("voice     : %s (%zu 帧)\n", voice_u8, ref_frames);
    } else {
        printf("voice     : Yuewen (%zu 帧)\n", ref_frames);
    }
    free(voice_u8);
    printf("threads   : %d\n", threads);
    if (verbose) {
        size_t priv = 0, ws = cur_working_set(&priv);
        fprintf(stderr, "  [mem] 模型加载后 working set %.0f MB, private %.0f MB\n",
                (double)ws / 1048576.0, (double)priv / 1048576.0);
    }

    /* ---- 分块合成 ---- */
    char** chunks = NULL;
    int nch = split_chunks(&sp, norm, &chunks, MAX_TEXT_TOKENS);

    size_t total_samples = 0;
    float* all = NULL;

    for (int ci = 0; ci < nch; ci++) {
        int32_t* tids = (int32_t*)xmalloc(sizeof(int32_t) * (strlen(chunks[ci]) * 2 + 16));
        int nt = sp_encode(&sp, chunks[ci], tids, (int)strlen(chunks[ci]) * 2 + 16);
        if (nt <= 0) { free(tids); continue; }
        if (verbose || nch > 1) printf("chunk %d/%d: %d tokens\n", ci + 1, nch, nt);

        uint64_t t1 = now_ms();
        float* pcm = NULL;
        size_t nsamp = 0;
        size_t frames = generate_frames(&models, tids, (size_t)nt, ref, ref_frames,
                                        &pcm, &nsamp, seed, max_frames, verbose, !no_eos, dump_frames);
        uint64_t t2 = now_ms();
        if (frames == 0 || !pcm || nsamp == 0) {
            fprintf(stderr, "  [warn] no frames generated for this chunk\n");
            free(pcm); free(tids);
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

        free(pcm); free(tids);
    }

    for (int k = 0; k < nch; k++) free(chunks[k]);
    free(chunks);

    if (!all || total_samples == 0) die("nothing generated");

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

    wav_write_stereo(out_path, all, total_samples, CHANNELS, SAMPLE_RATE);

    uint64_t t4 = now_ms();
    char* op = w2u(out_path);
    printf("\nwrote %s : %.2f s @ %d Hz stereo\n", op ? op : "?", (double)total_samples / SAMPLE_RATE, SAMPLE_RATE);
    printf("total     : %llu ms\n", (unsigned long long)(t4 - t0));

    g_stats.peak_rss = peak_working_set();
    if (verbose) {
        uint64_t load_sum = g_stats.load_prefill + g_stats.load_decode + g_stats.load_frame +
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
    free(op);

    free(all);
    free(ref_owned);
    free(norm);
    free(text);

    models_close(&models);
    if (g_meminfo) ort->ReleaseMemoryInfo(g_meminfo);
    if (g_env) ort->ReleaseEnv(g_env);
    return 0;
}
