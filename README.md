# MOSS-TTS-Nano 纯 C 推理

[OpenMOSS MOSS-TTS-Nano (0.1B)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX) 的单文件 C11 推理实现:
一个翻译单元,只依赖 `onnxruntime.dll`(运行时 LoadLibrary,免导入库)与静态 CRT。
目标是 Windows CPU 上的准实时流式语音合成。

## 构建

```
build.bat          (需要 MSVC 2022, 产物 moss_tts.exe)
```

运行需要 `onnxruntime.dll` 与 `models\`(两个 HF 仓库的 ONNX 导出,见 `.gitignore` 注释)。

## 快速开始

```
moss_tts.exe "你好,世界"                     # 合成到 out.wav
moss_tts.exe "你好,世界" --play              # 边生成边播放
moss_tts.exe --server                        # 常驻模式, 每行一段文本 (流式播放)
moss_tts.exe --list-voices                   # 列出全部内置音色
```

`--server` 模式: 每行一段文本回车即合成播放,支持 `/voice <名>` 换音色、
`/seed <n>` 换种子、`/quit` 退出;带 `--out x.wav` 时每条请求写 `x_001.wav…`。
`[stats]` 行报告首字延迟、生成/播放时长与 RTF。

**对话模式**(`--chat`,配合 `--server`):输入发给大模型,回复用当前音色实时
语音播报,支持多轮上下文(`/reset` 清空),`/say <文本>` 跳过大模型直接朗读:

```
moss_tts.exe --server --chat --api-key ak-xxxx
> 你好,介绍一下你自己          <- 打字,大模型回复被语音播出
```

密钥来自 `--api-key` 或环境变量 `CHAT_API_KEY`(不要写进仓库);API 地址与
模型可用 `--api-url` / `--api-model` 覆盖,默认 xfyun MaaS `/v1/responses`
(`spark-x2.6`,取回复的 `output_text` 字段,跳过推理思维链)。

## 音色

- 18 个官方预置音色(6 中/5 英/7 日)存于 `models/browser_poc_manifest.json`,
  `--voice <名>` 直接使用;默认音色为神里绫华(参考码同时内嵌 exe 兜底,
  manifest 缺失也能启动)。
- `--ref-wav file.wav` 用任意参考音频实时克隆(自动裁静音/归一化/限 15 秒)。
- `--save-voice 名字 --voice-desc 描述` 把克隆结果按官方格式写回 manifest,
  之后与内置音色同等对待(`--voice 名字` 直接调)。参考音频建议:目标语言、
  连续自然说话、约 10 秒、无空隙无背景音乐。

```
moss_tts.exe --ref-wav 声音.wav --save-voice 新音色 "试听一句话"   # 入库+试听
```

## 性能 (i5-1240P, 4 线程, 参考实现数据)

| 指标 | 数值 |
|---|---|
| 持续合成 RTF | ~0.51–0.66 (decode_step 已 int8 量化) |
| server 热启动首字 | ~0.45s 就绪, ~1.1s 出声 (含预缓冲) |
| 模型加载 | ~5s (仅一次性模式可感) |
| 内存 | working set ~900MB |

**int8 量化**:`models/` 里放 `moss_tts_decode_step.int8.onnx` 即自动启用
(decode步 -36%);`--fp32` 强制原始精度,删文件即完全回退。量化输出与
fp32 是同文本的不同演绎(自回归采样对扰动混沌放大),音质需自行试听取舍。

关键设计(详见源码头注释): 四 session 并行加载、逐帧复用输入张量、
codec 流式解码(固定 13MB KV)、codec 解码异步线程与主循环流水并行、
Intel 混合架构自动钉 P 核、chunk 间 prefill 流水线、播放预缓冲 +
静音扣留、短句守护生成(拦模型跑偏)。

## 已知边界

- 0.1B 模型偶发跑偏(一句正常话后零碎冒音节)。单块短文本自动检测并换
  种子重试;长文本仍可能偶发,属模型能力边界。
- 跨语言克隆带口音(日语参考说中文等),参考音频决定语气与语速。
- 克隆真人声音对外发布有授权问题,自行把握。

## 目录

```
moss_tts.c            全部实现 (单文件)
build.bat             MSVC 构建脚本
tools/                辅助脚本 (内嵌数据生成 / ONNX 检查等)
models/               模型与 manifest (自行下载, gitignore)
```
