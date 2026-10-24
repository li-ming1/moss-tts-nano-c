"""Generate the C source snippet that embeds the prompt templates and the
built-in Junhao reference audio codes (extracted from browser_poc_manifest.json).
"""
import json, textwrap

man = json.load(open("models/browser_poc_manifest.json", encoding="utf-8"))
pt = man["prompt_templates"]
voice = man["builtin_voices"][0]
codes = voice["prompt_audio_codes"]

def arr(name, vals, per=16, ctype="int32_t"):
    body = []
    for i in range(0, len(vals), per):
        body.append("    " + ", ".join(str(v) for v in vals[i:i + per]) + ",")
    return "static const %s %s[%d] = {\n%s\n};\n" % (ctype, name, len(vals), "\n".join(body))

out = []
out.append("/* ---- prompt templates (from browser_poc_manifest.json) ---- */")
out.append(arr("k_user_prefix", pt["user_prompt_prefix_token_ids"]))
out.append(arr("k_after_ref", pt["user_prompt_after_reference_token_ids"]))
out.append(arr("k_asst_prefix", pt["assistant_prompt_prefix_token_ids"]))
out.append("/* ---- built-in voice \"%s\" (%s), %d frames x 16 codebooks ---- */"
           % (voice["voice"], voice["display_name"], len(codes)))
flat = [c for f in codes for c in f]
out.append(arr("k_ref_codes_junhao", flat, per=16, ctype="int16_t"))
out.append("#define REF_FRAMES_JUNHAO %d" % len(codes))

snippet = "\n".join(out)
src = open("moss_tts.c", encoding="utf-8").read()
marker = "/*__PROMPT_DATA__*/"
assert marker in src, "placeholder missing"
src = src.replace(marker, snippet)
open("moss_tts.c", "w", encoding="utf-8", newline="\n").write(src)
print("injected: %d prefix, %d after_ref, %d asst, %d ref codes"
      % (len(pt["user_prompt_prefix_token_ids"]),
         len(pt["user_prompt_after_reference_token_ids"]),
         len(pt["assistant_prompt_prefix_token_ids"]), len(flat)))
