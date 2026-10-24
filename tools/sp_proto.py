"""Prototype of the SentencePiece unigram tokenizer (charsmap trie + Viterbi)
so we can validate the algorithm before porting it to C.
"""
import struct, sys, json

def read_varint(b, i):
    r = 0; s = 0
    while True:
        x = b[i]; i += 1
        r |= (x & 0x7F) << s
        if not (x & 0x80): return r, i
        s += 7

def fields(b):
    i = 0; n = len(b)
    while i < n:
        key, i = read_varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = read_varint(b, i); yield fn, wt, v
        elif wt == 1:
            yield fn, wt, b[i:i+8]; i += 8
        elif wt == 2:
            ln, i = read_varint(b, i); yield fn, wt, b[i:i+ln]; i += ln
        elif wt == 5:
            yield fn, wt, b[i:i+4]; i += 4

path = sys.argv[1] if len(sys.argv) > 1 else "models/tokenizer.model"
data = open(path, "rb").read()

pieces = []       # (str, score, type)
charsmap = b""
add_dummy_prefix = True
remove_extra_ws = True
escape_ws = True
unk_id = 0
model_type = 1
for fn, wt, v in fields(data):
    if fn == 1 and wt == 2:
        nm = ""; sc = 0.0; ty = 1
        for f2, w2, v2 in fields(v):
            if f2 == 1 and w2 == 2: nm = v2.decode("utf-8")
            elif f2 == 2 and w2 == 5: sc = struct.unpack("<f", v2)[0]
            elif f2 == 3 and w2 == 0: ty = v2
        pieces.append((nm, sc, ty))
    elif fn == 2 and wt == 2:
        for f2, w2, v2 in fields(v):
            if f2 == 3: model_type = v2
            elif f2 == 40: unk_id = v2
    elif fn == 3 and wt == 2:
        for f2, w2, v2 in fields(v):
            if f2 == 2 and w2 == 2: charsmap = v2
            elif f2 == 3: add_dummy_prefix = bool(v2)
            elif f2 == 4: remove_extra_ws = bool(v2)
            elif f2 == 5: escape_ws = bool(v2)

print("pieces=%d model_type=%d unk=%d charsmap=%d dummy=%s rmws=%s esc=%s"
      % (len(pieces), model_type, unk_id, len(charsmap), add_dummy_prefix, remove_extra_ws, escape_ws))

# ---- darts-clone double array ----
trie_size = struct.unpack("<I", charsmap[:4])[0]
trie = memoryview(charsmap)[4:4 + trie_size]
norm_blob = charsmap[4 + trie_size:]

def unit(i):
    return struct.unpack_from("<I", trie, i * 4)[0]

def label(u):   return u & 0xFF
def has_leaf(u): return (u >> 8) & 1
def value(u):   return u & 0x7FFFFFFF
def offset(u):  return (u >> 10) << ((u & (1 << 9)) >> 6)

def common_prefix_search(s):
    """s: bytes -> list of (length, value)"""
    res = []
    node_pos = 0
    u = unit(node_pos)
    node_pos ^= offset(u)
    for i, ch in enumerate(s):
        node_pos ^= ch
        u = unit(node_pos)
        if label(u) != ch:
            return res
        node_pos ^= offset(u)
        if has_leaf(u):
            res.append((i + 1, value(u)))
    return res

def norm_prefix(s):
    """Longest match replacement. Returns (replacement_bytes, consumed)."""
    best = None
    for ln, val in common_prefix_search(s):
        if best is None or ln > best[0]:
            best = (ln, val)
    if best is None:
        # copy one UTF-8 char
        b0 = s[0]
        n = 1 if b0 < 0x80 else (2 if b0 < 0xE0 else (3 if b0 < 0xF0 else 4))
        return s[:n], n
    ln, val = best
    end = norm_blob.find(b"\0", val)
    return norm_blob[val:end], ln

# ---- normalize ----
def normalize(text):
    src = text.encode("utf-8")
    out = bytearray()
    if add_dummy_prefix:
        out += "\u2581".encode()
    prev_space = add_dummy_prefix
    i = 0
    # sentencepiece: normalize prefix by prefix
    while i < len(src):
        rep, used = norm_prefix(src[i:])
        i += used
        # feed replacement through whitespace logic
        j = 0
        while j < len(rep):
            b = rep[j]
            if b == 0x20:
                if escape_ws:
                    if not prev_space:
                        out += "\u2581".encode()
                        prev_space = True
                    if not remove_extra_ws:
                        out += "\u2581".encode()
                else:
                    if (not remove_extra_ws) or (not prev_space):
                        out.append(b)
                        prev_space = True
                j += 1
            else:
                # copy one utf-8 char
                n = 1 if b < 0x80 else (2 if b < 0xE0 else (3 if b < 0xF0 else 4))
                out += rep[j:j + n]
                prev_space = False
                j += n
    return bytes(out)

# ---- viterbi ----
piece_to_id = {}
for i, (p, s, t) in enumerate(pieces):
    piece_to_id[p.encode()] = i
min_score = min(s for _, s, _ in pieces)
maxlen = max(len(p.encode()) for p, _, _ in pieces)
print("min_score=%.4f maxpiece_bytes=%d" % (min_score, maxlen))

def encode(text, verbose=False):
    nb = normalize(text)
    if verbose:
        print("normalized:", nb.decode("utf-8", "replace"))
    n = len(nb)
    best = [(-1e30, 0, -1)] * (n + 1)
    best[0] = (0.0, 0, -1)
    for pos in range(n):
        if best[pos][0] <= -1e29:
            continue
        found = False
        for end in range(pos + 1, min(n, pos + maxlen) + 1):
            pid = piece_to_id.get(nb[pos:end])
            if pid is None:
                continue
            p, sc, ty = pieces[pid]
            if ty == 5:
                continue
            cand = best[pos][0] + sc
            if cand > best[end][0]:
                best[end] = (cand, pos, pid)
            found = True
        if not found and best[pos + 1][2] == -1:
            best[pos + 1] = (best[pos][0] + min_score - 10.0, pos, unk_id)
    ids = []
    pos = n
    while pos > 0:
        sc, st, pid = best[pos]
        ids.append(pid)
        pos = st
    ids.reverse()
    return ids

manifest = json.load(open(path.replace("tokenizer.model", "browser_poc_manifest.json"), encoding="utf-8"))
for s in manifest["text_samples"]:
    got = encode(s["text"], verbose=True)
    exp = s["text_token_ids"]
    print(s["id"])
    print("  text:", s["text"])
    print("  got :", got)
    print("  exp :", exp)
    print("  MATCH" if got == exp else "  DIFF len %d vs %d" % (len(got), len(exp)))
