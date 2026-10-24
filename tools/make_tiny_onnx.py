"""Hand-build a minimal ONNX model (z = x + y) with raw protobuf, no deps.
Used to check whether the local onnxruntime.dll can execute *anything*.
"""
import struct

def varint(v):
    out = b""
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out += bytes([b | 0x80])
        else:
            out += bytes([b])
            return out

def tag(field, wt):
    return varint((field << 3) | wt)

def ld(field, payload):
    return tag(field, 2) + varint(len(payload)) + payload

def s(field, text):
    return ld(field, text.encode())

def i(field, v):
    return tag(field, 0) + varint(v)

def dim(value):
    return ld(1, i(1, value))

def value_info(name, shape):
    shape_pb = b"".join(dim(d) for d in shape)
    tensor = i(1, 1) + ld(2, shape_pb)          # elem_type=float, shape
    type_pb = ld(1, tensor)
    return s(1, name) + ld(2, type_pb)

node = s(1, "x") + s(1, "y") + s(2, "z") + s(4, "Add")
graph = ld(1, node) + s(2, "g") + ld(11, value_info("x", [2])) \
        + ld(11, value_info("y", [2])) + ld(12, value_info("z", [2]))
opset = i(2, 17)
model = i(1, 8) + ld(8, opset) + ld(7, graph)
open("tools/tiny_add.onnx", "wb").write(model)
print("wrote tools/tiny_add.onnx", len(model), "bytes")
