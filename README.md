# MOSS-TTS-Nano 纯 C 推理

[English](README.en.md) | 中文

[OpenMOSS MOSS-TTS-Nano (0.1B)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX) 的单文件 C11 推理实现:
一个翻译单元,只依赖 `onnxruntime.dll`(运行时 LoadLibrary,免导入库)与静态 CRT。
目标是 Windows CPU 上的准实时流式语音合成。

## 构建

```
build.bat          (需要 MSVC 2022, 产物 moss_tts.exe)
```

运行需要 `onnxruntime.dll` 与 `models\`(两个 HF 仓库的 ONNX 导出,见 `.gitignore` 注释)。
调用方式:**cmd 直接 `moss_tts.exe ...`;PowerShell 必须带路径前缀 `./moss_tts.exe ...`**(裸文件名不在 PATH 里会报"无法识别");git-bash 两者皆可。

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
语音播报 —— **SSE 流式**:模型边吐答案增量,TTS 按句(。!?;…)切开,第一句
一到就开嗓,后句边收边说;推理思维链增量自动跳过。支持多轮上下文
(`/reset` 清空),`/say <文本>` 跳过大模型直接朗读,`/retry` 换种子重说上一句
(偶发音色漂移时手动重掷);回复句首的 嗯/呃/啊/哦 填充词自动剥离:

```
moss_tts.exe --server --chat --api-key ak-xxxx
> 用三句话介绍长城的历史
AI (首句): 春秋战国时期,各诸侯国为了防御敌国的进攻,纷纷修筑了早期的城墙。
[stats]   : 首句出声 15000 ms | 3 句 | 音频 24.16 s | 本轮 38047 ms
```

首句出声 = 大模型思考时间 + ~1s(TTS 首字);后句在整轮结束前就已播完大半。
默认请求带 `reasoning: {"effort": "low"}`(实测思考时间约省 30%,`--api-effort
none/medium/high` 可调);server 启动时自动预热 ORT 执行计划,首请求不再慢。
密钥来自 `--api-key` 或环境变量 `CHAT_API_KEY`(不要写进仓库);API 地址与
模型可用 `--api-url` / `--api-model` 覆盖,默认 xfyun MaaS `/v1/responses`
(`spark-x2.6`,SSE 事件取 `response.output_text.delta`,跳过
`reasoning_summary_text.delta`;无增量时自动回退非流式)。

终端交互(interactive 终端生效,管道/重定向自动退回纯文本):回复正文
随模型增量**流式上屏**(打字机效果),底部一行**青色高亮**显示 TTS 正在
播报的句子(`>> 当前句`),逐句推进、结束后擦除接统计行。

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

## 试听样例 (`examples/`)

`examples/` 里是仓库自带的实测音频,**全部由本项目的模型克隆合成,不是任何原声录音**:

| 文件 | 音色来源 | 内容 | 时长 |
|---|---|---|---|
| `examples/神里绫华_demo.wav` | 内置预置音色 | 短句 demo | 9.8 s |
| `examples/神里绫华_大模型漫谈.wav` | 内置预置音色 | 长文连续朗读 | 88.5 s |
| `examples/哥伦比娅_demo.wav` | 参考音频克隆 (`--ref-wav`) | 短句 demo | 8.7 s |
| `examples/哥伦比娅_大模型发展史.wav` | 参考音频克隆 (`--ref-wav`) | 长文连续朗读 | 81.9 s |

复现方式:

```
moss_tts.exe --voice 神里绫华 "要合成的话" --out demo.wav
moss_tts.exe --ref-wav 参考音频.wav "要合成的话" --out demo.wav
```

**关于"像不像原声"**: 这些音色是模型从参考音频里学出来的,**并非原声**,和真人/原角色
必然存在差异。0.1B 是很小的模型,它抓得住音色的主干特征(音高区间、音质倾向、说话
节奏),但细节会走样: 轻微口音与咬字偏差、语气起伏不如真人自然、偶发音色漂移(长文
更明显),以及模型自身的跑偏(见"已知边界")。另外量化推理(decode int8 + frame int4)
与 fp32 也不是逐帧一致的演绎 —— 自回归采样下同一句话每次生成都不同,音质取舍请以自己
的耳朵为准。所以这些样例适合回答"这个小模型能做到什么程度",不适合用来判断"能不能
以假乱真"。

样例中的音色名仅用于标识参考音频的来源,音频本身是模型生成的合成语音,仅作技术演示,
不作商业用途;对外使用克隆音色请先确认授权(见"许可证")。

## 性能 (i5-1240P, 4 线程, 参考实现数据)

| 指标 | 数值 |
|---|---|
| 持续合成 RTF | ~0.48–0.66 (decode int8 + frame int4 双量化) |
| server 热启动首字 | ~0.45s 就绪, ~1.1s 出声 (含预缓冲) |
| 模型加载 | ~5s (仅一次性模式可感) |
| 内存 | working set ~900MB |

**量化加速**(均经人工试听确认无感差异后默认启用;`--fp32` 一键全回退,
删对应文件单点回退):

| 图 | 变体 | 实测 |
|---|---|---|
| decode_step | `.int8.onnx`(动态量化) | decode步 21.7→13.9ms(-36%,AVX-VNNI) |
| 采样 frame | `.int4.onnx`(MatMulNBits, **block128**, accuracy_level=4) | 采样帧 33.5→~23ms(-30%, 分档实测) |

量化输出与 fp32 是同文本的不同演绎(自回归采样对扰动混沌放大),音质
取舍需自行试听。调参结论(同窗口交替实测): accuracy_level 4(浮点反量化
计算)比 auto/int8 档快 ~14%;block 128 比 32 快 ~12%(更粗,听感需自行
确认,回退件 `moss_tts_local_fixed_sampled_frame.int4.b32.bak`);decode
配 int4 与 int8 打平(int8 已吃到 AVX-VNNI),保持 int8。注意:性能对比
必须同负载窗口背靠背做,本机 fp32 采样
耗时随负载在 25~35ms 漂移,跨窗口对比会得出错误结论(实测踩过)。

关键设计(详见源码头注释): 四 session 并行加载、逐帧复用输入张量、
codec 流式解码(固定 13MB KV)、codec 解码异步线程与主循环流水并行、
Intel 混合架构自动钉 P 核、chunk 间 prefill 流水线、播放预缓冲 +
静音扣留、短句守护生成(拦模型跑偏)。

## 已知边界

- 0.1B 模型偶发跑偏(一句正常话后零碎冒音节)。单块短文本自动检测并换
  种子重试;长文本仍可能偶发,属模型能力边界。
- 跨语言克隆带口音(日语参考说中文等),参考音频决定语气与语速。
- 克隆只还原音色主干,与真实原声必然有差异(咬字、口音、语气起伏),0.1B
  的模型能力就到这儿;长文比短句更容易听出"合成感",试听见 `examples/`。
- 克隆真人声音对外发布有授权问题,自行把握。

## 参考实现与致谢

本 C 实现的流水线结构与关键算法参照了 Rust 参考实现
[mzdk100/moss-tts-nano](https://github.com/mzdk100/moss-tts-nano)(同为
Apache-2.0):prompt 组织、帧重复分析判语音结束(其原版 8 帧窗口在本项目
放宽到 48 帧)、帧生成与 codec 解码交错的流式设计均源于此,并在此基础上
做了异步流水线、量化与播放工程化。

## 许可证

本项目与所用模型一致,采用 **Apache License 2.0**(见 `LICENSE`;
模型卡:OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX 与 MOSS-Audio-Tokenizer-Nano-ONNX
均为 apache-2.0)。依赖的 onnxruntime 为 MIT 许可(不随本仓库分发)。
注意:许可证覆盖代码与模型,**克隆出的具体人声音色仍受其本人权利约束**,
对外发布自行把握。

## 目录

```
moss_tts.c            全部实现 (单文件)
build.bat             MSVC 构建脚本
tools/                辅助脚本 (内嵌数据生成 / ONNX 检查等)
examples/             试听样例 (模型克隆音色的实测 wav)
models/               模型与 manifest (自行下载, gitignore)
```
