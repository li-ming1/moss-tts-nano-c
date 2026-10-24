"""Minimal ONNX protobuf reader: dump graph inputs/outputs with dtype+shape.
No external deps -- pure-python protobuf wire-format walk.
"""
import sys, struct

def read_varint(b, i):
    r = 0; s = 0
    while True:
        x = b[i]; i += 1
        r |= (x & 0x7F) << s
        if not (x & 0x80):
            return r, i
        s += 7

def fields(b):
    i = 0; n = len(b)
    while i < n:
        key, i = read_varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = read_varint(b, i)
            yield fn, wt, v
        elif wt == 1:
            yield fn, wt, b[i:i+8]; i += 8
        elif wt == 2:
            ln, i = read_varint(b, i)
            yield fn, wt, b[i:i+ln]; i += ln
        elif wt == 5:
            yield fn, wt, b[i:i+4]; i += 4
        else:
            raise ValueError("wiretype %d" % wt)

DT = {1:"float32",2:"uint8",3:"int8",4:"uint16",5:"int16",6:"int32",7:"int64",
      8:"string",9:"bool",10:"float16",11:"float64",12:"uint32",13:"uint64",14:"complex64",15:"complex128"}

def parse_valueinfo(b):
    name = ""
    dtype = None; dims = []
    for fn, wt, v in fields(b):
        if fn == 1 and wt == 2:
            name = v.decode("utf-8", "replace")
        elif fn == 2 and wt == 2:  # TypeProto
            for f2, w2, v2 in fields(v):
                if f2 == 1 and w2 == 2:  # tensor_type
                    for f3, w3, v3 in fields(v2):
                        if f3 == 1 and w3 == 0:  # elem_type
                            dtype = v3
                        elif f3 == 2 and w3 == 2:  # shape
                            for f4, w4, v4 in fields(v3):
                                if f4 == 1 and w4 == 2:  # dim
                                    d = "?"
                                    for f5, w5, v5 in fields(v4):
                                        if f5 == 1 and w5 == 0: d = v5
                                        elif f5 == 2 and w5 == 2: d = v5.decode()
                                    dims.append(d)
    return name, DT.get(dtype, str(dtype)), dims

def parse_model(path):
    with open(path, "rb") as f:
        b = f.read()
    graph = None
    for fn, wt, v in fields(b):
        if fn == 7 and wt == 2:
            graph = v; break
    if graph is None:
        print("no graph"); return
    ins, outs, ops = [], [], []
    for fn, wt, v in fields(graph):
        if fn == 11 and wt == 2: ins.append(parse_valueinfo(v))
        elif fn == 12 and wt == 2: outs.append(parse_valueinfo(v))
        elif fn == 1 and wt == 2:
            for f2, w2, v2 in fields(v):
                if f2 == 4 and w2 == 2: ops.append(v2.decode("utf-8", "replace"))
    print("=" * 70)
    print(path)
    print("  ops used:", ",".join(sorted(set(ops))))
    print("  INPUTS (%d):" % len(ins))
    for n, d, s in ins: print("    %-34s %-8s %s" % (n, d, s))
    print("  OUTPUTS (%d):" % len(outs))
    for n, d, s in outs[:6]: print("    %-34s %-8s %s" % (n, d, s))
    if len(outs) > 6: print("    ... %d more, last: %s %s %s" % (len(outs)-6, outs[-1][0], outs[-1][1], outs[-1][2]))

for p in sys.argv[1:]:
    parse_model(p)
