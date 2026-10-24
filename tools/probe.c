/* Minimal ORT probe: load a model, dump input/output types+shapes, optionally run
 * with zero inputs.  Build with the same flags as moss_tts.c.
 *   probe.exe <model.onnx> [run]
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnxruntime_c_api.h"

static LONG WINAPI cf(EXCEPTION_POINTERS* ep)
{
    HMODULE hm = GetModuleHandleA("onnxruntime.dll");
    void* a = ep->ExceptionRecord->ExceptionAddress;
    unsigned long long rva = (unsigned long long)((char*)a - (char*)hm);
    printf("[crash] code=%lx rip=%p ort_rva=%llx info0=%llx info1=%llx\n", ep->ExceptionRecord->ExceptionCode, a, rva, (unsigned long long)ep->ExceptionRecord->ExceptionInformation[0], (unsigned long long)ep->ExceptionRecord->ExceptionInformation[1]);
    unsigned long long* sp = (unsigned long long*)ep->ContextRecord->Rsp;
    printf("stack scan (values inside onnxruntime):\n");
    for (int i = 0; i < 200; i++) {
        unsigned long long v = sp[i];
        if (v > (unsigned long long)hm && v < (unsigned long long)hm + 0x800000)
            printf("  [sp+%3d] %llx\n", i * 8, v - (unsigned long long)hm);
    }
    return EXCEPTION_EXECUTE_HANDLER;
}

static const OrtApi* ort;
static OrtEnv* env;
static OrtMemoryInfo* mi;

#define OK(e) do { OrtStatus* s=(e); if(s){ printf("ERR: %s\n", ort->GetErrorMessage(s)); ort->ReleaseStatus(s); exit(1);} } while(0)

int wmain(int argc, wchar_t** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetUnhandledExceptionFilter(cf);
    int SEQ = argc > 3 ? _wtoi(argv[3]) : 179;
    HMODULE h = LoadLibraryA("onnxruntime.dll");
    typedef const OrtApiBase* (ORT_API_CALL* F)(void);
    const OrtApiBase* b = ((F)(void*)GetProcAddress(h, "OrtGetApiBase"))();
    printf("runtime: %s\n", b->GetVersionString());
    ort = b->GetApi(ORT_API_VERSION);
    { char buf[8]; DWORD n = GetEnvironmentVariableA("PROBE_LOG", buf, 8); int lv = n ? atoi(buf) : 3; OK(ort->CreateEnv((OrtLoggingLevel)lv, "probe", &env)); }
    OK(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));

    {
        HMODULE hm = GetModuleHandleA("onnxruntime.dll");
        printf("dll base=%p size-of-OrtApi-members probe://n", (void*)hm);
        printf("  CreateEnv      =%p (rva %llx)\n", (void*)ort->CreateEnv, (unsigned long long)((char*)(void*)ort->CreateEnv - (char*)hm));
        printf("  CreateSession  =%p (rva %llx)\n", (void*)ort->CreateSession, (unsigned long long)((char*)(void*)ort->CreateSession - (char*)hm));
        printf("  Run            =%p (rva %llx)\n", (void*)ort->Run, (unsigned long long)((char*)(void*)ort->Run - (char*)hm));
        printf("  RunAsync       =%p (rva %llx)\n", (void*)ort->RunAsync, (unsigned long long)((char*)(void*)ort->RunAsync - (char*)hm));
        printf("  SessionGetInputCount=%p (rva %llx)\n", (void*)ort->SessionGetInputCount, (unsigned long long)((char*)(void*)ort->SessionGetInputCount - (char*)hm));
        printf("  ReleaseEnv     =%p (rva %llx)\n", (void*)ort->ReleaseEnv, (unsigned long long)((char*)(void*)ort->ReleaseEnv - (char*)hm));
    }
    OrtSessionOptions* so;
    OK(ort->CreateSessionOptions(&so));
    { char buf[16]; DWORD n = GetEnvironmentVariableA("PROBE_THREADS", buf, 16); int th = n ? atoi(buf) : 4; OK(ort->SetIntraOpNumThreads(so, th)); printf("intra threads=%d\n", th); }
    { char buf[8]; DWORD n = GetEnvironmentVariableA("PROBE_OPT", buf, 8); int lv = n ? atoi(buf) : 99;
      if (lv != 99) { OK(ort->SetSessionGraphOptimizationLevel(so, (GraphOptimizationLevel)lv)); printf("opt level=%d\n", lv); } }
    OrtSession* sess;
    OK(ort->CreateSession(env, argv[1], so, &sess));

    OrtAllocator* alloc;
    OK(ort->GetAllocatorWithDefaultOptions(&alloc));

    size_t nin, nout;
    OK(ort->SessionGetInputCount(sess, &nin));
    OK(ort->SessionGetOutputCount(sess, &nout));
    printf("inputs=%zu outputs=%zu\n", nin, nout);

    char* names[64];
    ONNXTensorElementDataType types[64];
    int64_t* dims[64];
    size_t ndim[64];
    size_t total[64];

    for (size_t i = 0; i < nin; i++) {
        char* nm;
        OK(ort->SessionGetInputName(sess, i, alloc, &nm));
        OrtTypeInfo* ti;
        OK(ort->SessionGetInputTypeInfo(sess, i, &ti));
        const OrtTensorTypeAndShapeInfo* tsi;
        OK(ort->CastTypeInfoToTensorInfo(ti, &tsi));
        ONNXTensorElementDataType t;
        OK(ort->GetTensorElementType(tsi, &t));
        size_t nd;
        OK(ort->GetDimensionsCount(tsi, &nd));
        int64_t d[8] = {0};
        OK(ort->GetDimensions(tsi, d, nd));
        names[i] = nm; types[i] = t; ndim[i] = nd;
        dims[i] = (int64_t*)malloc(sizeof(int64_t) * (nd ? nd : 1));
        memcpy(dims[i], d, sizeof(int64_t) * nd);
        for (size_t k = 0; k < nd; k++) if (d[k] < 0) d[k] = (k == 0) ? 1 : SEQ;
        memcpy(dims[i], d, sizeof(int64_t) * nd);
        total[i] = 1;
        printf("  IN  %-24s type=%d dims=[", nm, (int)t);
        for (size_t k = 0; k < nd; k++) { printf("%lld%s", (long long)d[k], k + 1 < nd ? "," : ""); total[i] *= (size_t)d[k]; }
        printf("]\n");
        ort->ReleaseTypeInfo(ti);
    }
    for (size_t i = 0; i < nout && i < 4; i++) {
        char* nm;
        OK(ort->SessionGetOutputName(sess, i, alloc, &nm));
        printf("  OUT %s\n", nm);
    }

    if (argc > 2) {
        int mode = 0;
        { char buf[16]; DWORD n = GetEnvironmentVariableA("PROBE_MODE", buf, 16); if (n) mode = atoi(buf); }
        printf("running with zero inputs... mode=%d\n", mode);
        OrtRunOptions* ro = NULL;
        if (mode & 1) { OK(ort->CreateRunOptions(&ro)); printf("  runoptions ok\n"); }
        OrtValue* in[64] = {0};
        for (size_t i = 0; i < nin; i++) {
            size_t n = total[i] ? total[i] : 1;
            if (mode & 4) {
                OK(ort->CreateTensorAsOrtValue(alloc, dims[i], ndim[i], types[i], &in[i]));
                void* q = NULL; OK(ort->GetTensorMutableData(in[i], &q)); memset(q, 0, n * 4);
            } else {
                void* p = calloc(n, 4);
                OK(ort->CreateTensorWithDataAsOrtValue(mi, p, n * 4, dims[i], ndim[i], types[i], &in[i]));
            }
        }
        for (size_t i = 0; i < nin; i++) {
            OrtTensorTypeAndShapeInfo* tsi2 = NULL;
            OrtStatus* st2 = ort->GetTensorTypeAndShape(in[i], &tsi2);
            if (st2) { printf("  input %zu: BAD %s\n", i, ort->GetErrorMessage(st2)); ort->ReleaseStatus(st2); continue; }
            ONNXTensorElementDataType t2; size_t nd2; int64_t d2[8];
            ort->GetTensorElementType(tsi2, &t2);
            ort->GetDimensionsCount(tsi2, &nd2);
            ort->GetDimensions(tsi2, d2, nd2);
            printf("  input %zu ok: type=%d dims=[", i, (int)t2);
            for (size_t k = 0; k < nd2; k++) printf("%lld%s", (long long)d2[k], k + 1 < nd2 ? "," : "");
            printf("]\n");
            ort->ReleaseTensorTypeAndShapeInfo(tsi2);
        }
        char* on[64];
        for (size_t i = 0; i < nout; i++) OK(ort->SessionGetOutputName(sess, i, alloc, &on[i]));
        OrtValue* out[64] = {0};
        if (mode & 128) { printf("  run nin=0 nout=%zu\n", nout); OrtStatus* st = ort->Run(sess, ro, NULL, NULL, 0, NULL, nout, out); printf("  status=%p msg=%s\n", (void*)st, st ? ort->GetErrorMessage(st) : "(null)"); }
        else if (mode & 256) { printf("  run nin=1 nout=0\n"); OrtStatus* st = ort->Run(sess, ro, NULL, (const OrtValue* const*)in, 1, NULL, 0, NULL); printf("  status=%p msg=%s\n", (void*)st, st ? ort->GetErrorMessage(st) : "(null)"); }
        else if (mode & 512) { printf("  run nin=1 nout=1\n"); OrtStatus* st = ort->Run(sess, ro, NULL, (const OrtValue* const*)in, 1, NULL, 1, out); printf("  status=%p msg=%s\n", (void*)st, st ? ort->GetErrorMessage(st) : "(null)"); }
        else if (mode & 64) { printf("  run with nin=0 nout=0\n"); OrtStatus* st = ort->Run(sess, ro, NULL, NULL, 0, NULL, 0, NULL); printf("  status=%p msg=%s\n", (void*)st, st ? ort->GetErrorMessage(st) : "(null)"); }
        else if (mode & 16) { printf("  run with nout=0\n"); OK(ort->Run(sess, ro, NULL, (const OrtValue* const*)in, nin, NULL, 0, out)); }
        else if (mode & 32) { printf("  run with nout=1\n"); OK(ort->Run(sess, ro, NULL, (const OrtValue* const*)in, nin, NULL, 1, out)); }
        else if (mode & 8) {
            printf("  run with NULL names\n");
            OK(ort->Run(sess, ro, NULL, (const OrtValue* const*)in, nin, NULL, nout, out));
        } else {
            OK(ort->Run(sess, ro, (const char* const*)names, (const OrtValue* const*)in, nin,
                        (const char* const*)on, nout, out));
        }
        printf("run ok\n");
    }
    return 0;
}
