// Minimal stand-in for the official onnxruntime_error_code.h (shipped with the
// ONNX Runtime release). onnxruntime_c_api.h includes it by name.
#pragma once

typedef enum OrtErrorCode {
  ORT_OK = 0,
  ORT_FAIL = 1,
  ORT_INVALID_ARGUMENT = 2,
  ORT_NO_SUCHFILE = 3,
  ORT_NO_MODEL = 4,
  ORT_ENGINE_ERROR = 5,
  ORT_RUNTIME_EXCEPTION = 6,
  ORT_INVALID_PROTOBUF = 7,
  ORT_MODEL_LOADED = 8,
  ORT_NOT_IMPLEMENTED = 9,
  ORT_INVALID_GRAPH = 10,
  ORT_EP_FAIL = 11,
  ORT_MODEL_LOAD_CANCELED = 12,
  ORT_MODEL_REQUIRES_COMPILATION = 13,
  ORT_NOT_FOUND = 14
} OrtErrorCode;
