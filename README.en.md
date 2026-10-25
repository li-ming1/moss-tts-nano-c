# MOSS-TTS-Nano in Pure C

English | [中文](README.md)

A single-file C11 inference implementation of
[OpenMOSS MOSS-TTS-Nano (0.1B)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX):
one translation unit, depending only on `onnxruntime.dll` (loaded at runtime via
LoadLibrary, no import lib) and the static CRT. The goal is near-realtime
streaming speech synthesis on Windows CPUs.

## Build

```
build.bat          (requires MSVC 2022, produces moss_tts.exe)
```

Running requires `onnxruntime.dll` and a `models\` directory (the ONNX exports
from the two HF repos — see the note in `.gitignore`).
Invocation: **cmd runs `moss_tts.exe ...` directly; PowerShell requires the
path prefix `./moss_tts.exe ...`** (a bare name is not on PATH); git-bash
accepts both.

## Quick Start

```
moss_tts.exe "Hello, world"                 # synthesize to out.wav
moss_tts.exe "Hello, world" --play          # play while generating
moss_tts.exe --server                       # resident mode, one text per line
moss_tts.exe --list-voices                  # list all built-in voices
```

`--server` mode: type a line, press Enter, and it is synthesized and played
streaming. Supports `/voice <name>`, `/seed <n>`, `/quit`; with `--out x.wav`
each request is written as `x_001.wav…`. The `[stats]` line reports
time-to-first-audio, generation/playback durations and RTF.

**Chat mode** (`--chat`, with `--server`): your input goes to an LLM and the
reply is spoken aloud in your current voice — **SSE streaming**: as the model
emits answer deltas, the TTS splits them into sentences (。!?;…), starts
speaking the first sentence as soon as it arrives and keeps going while the
rest streams in; reasoning-chain deltas are skipped automatically. Multi-turn
context is kept (`/reset` clears it), `/say <text>` bypasses the LLM, and
`/retry` re-speaks the last sentence with a new seed (for the occasional
timbre wobble). Leading filler words (嗯/呃/啊/哦) are stripped automatically:

```
moss_tts.exe --server --chat --api-key ak-xxxx
> Introduce the Great Wall in three sentences
AI (first): In the Spring and Autumn and Warring States periods, ...
[stats]   : first sentence 15000 ms | 3 sentences | 24.16 s audio | 38047 ms turn
```

Time to first sentence = LLM thinking time + ~1s (TTS first sound); later
sentences are mostly spoken before the turn ends. Requests default to
`reasoning: {"effort": "low"}` (~30% less thinking time; adjustable via
`--api-effort none/medium/high`); the server warms up ORT execution plans at
startup so the first request is not slow. The API key comes from `--api-key`
or the `CHAT_API_KEY` environment variable (do not commit it); endpoint and
model can be overridden with `--api-url` / `--api-model` — the default is the
xfyun MaaS `/v1/responses` endpoint (`spark-x2.6`, consuming
`response.output_text.delta` SSE events and skipping
`reasoning_summary_text.delta`, with a non-streaming fallback).

## Voices

- 18 official preset voices (6 Chinese / 5 English / 7 Japanese) live in
  `models/browser_poc_manifest.json`, used directly via `--voice <name>`;
  the default voice is embedded in the exe as a fallback so it works even
  without the manifest.
- `--ref-wav file.wav` clones from any reference audio in real time
  (silence trimming / normalization / 15-second cap, all automatic).
- `--save-voice <name> --voice-desc <text>` writes the cloned reference codes
  back into the manifest in the official format; afterwards it behaves exactly
  like a built-in voice. Reference audio tips: target language, continuous
  natural speech, about 10 seconds, no gaps, no background music.

```
moss_tts.exe --ref-wav voice.wav --save-voice MyVoice "A test sentence"
```

## Performance (i5-1240P, 4 threads, measured)

| Metric | Value |
|---|---|
| Sustained RTF | ~0.48–0.66 (decode int8 + frame int4) |
| Warm first sound in server | ~0.45s ready, ~1.1s audible (incl. prebuffer) |
| Model load | ~5s (only noticeable in one-shot mode) |
| Memory | working set ~900MB |

**Quantized acceleration** (both enabled by default after A/B listening
confirmed no perceptible difference; `--fp32` reverts everything, deleting
the file reverts one graph):

| Graph | Variant | Measured |
|---|---|---|
| decode_step | `.int8.onnx` (dynamic) | 21.7→13.9ms per step (−36%, AVX-VNNI) |
| sampling frame | `.int4.onnx` (MatMulNBits, block32) | 33.5→29.2ms per frame (−13%) |

Quantized output is a different rendition of the same text (autoregressive
sampling amplifies perturbation chaotically), so judge quality by ear. Note:
benchmark only back-to-back within the same load window — on this machine the
fp32 sampling time drifts 25–35ms with background load, and cross-window
comparisons produce wrong conclusions (learned the hard way).

Key design (details in the source header comment): four sessions loaded in
parallel, per-frame input tensor reuse, streaming codec decode with a fixed
13MB KV cache, codec decoding pipelined on an async thread, automatic P-core
pinning on Intel hybrid CPUs, cross-chunk prefill pipelining, playback
prebuffering + silence holding, and guarded generation for short inputs
(catches model rambling).

## Known Limits

- The 0.1B model occasionally rambles (normal sentence followed by stray
  syllables). Short single-chunk inputs are detected and retried with a new
  seed; long texts may still show it occasionally — a model capability limit.
- Cross-language cloning carries an accent (a Japanese reference speaking
  Chinese, etc.); the reference audio determines tone and pace.
- Publishing cloned voices of real people has consent implications — use
  your own judgement.

## Reference Implementation & Credits

The pipeline structure and key algorithms follow the Rust reference
implementation [mzdk100/moss-tts-nano](https://github.com/mzdk100/moss-tts-nano)
(also Apache-2.0): prompt layout, end-of-speech detection via frame repetition
(their 8-frame window is widened to 48 frames here), and the interleaved
frame-generation/codec-decode streaming design all originate there; this
project adds async pipelining, quantization and playback engineering on top.

## License

Consistent with the models, this project uses the **Apache License 2.0**
(see `LICENSE`; both model cards — OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX and
MOSS-Audio-Tokenizer-Nano-ONNX — are apache-2.0). The onnxruntime dependency
is MIT (not redistributed in this repo). Note that the license covers code
and models; **a cloned person's voice remains subject to their own rights**.

## Layout

```
moss_tts.c            the whole implementation (single file)
build.bat             MSVC build script
tools/                helper scripts (embedded-data generation, ONNX checks)
models/               models and manifest (download separately, gitignored)
```
