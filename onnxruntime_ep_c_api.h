// Placeholder for onnxruntime_ep_c_api.h.
//
// onnxruntime_c_api.h (ORT_API_VERSION 31) #includes this header at its very end
// for the newer execution-provider API. moss_tts.c only uses the classic OrtApi
// struct, so an empty stub is enough -- it keeps the build to a single .c file
// without vendoring the whole ONNX Runtime include/ tree.
#pragma once
