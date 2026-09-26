# vedioAI · 课程视频 AI 理解

播放本地课程视频，用 AI 理解**全部内容**：语音转写、课件 OCR、带时间戳问答与学习文档。

核心思路只有一句：**一次入库、多次查询**。

```
本地课程视频
   ├─ FFmpeg remux ──────────► H.264/AAC 播放代理（浏览器能播）
   ├─ FFmpeg 抽音频 ─────────► 16kHz 单声道 mp3
   │      └─ 火山录音文件识别 ─► 字级时间戳转写
   └─ 课件抽帧 + RapidOCR ───► 幻灯片文字
                    ↓
              课程中间表示（IR）
                    ↓
        分层摘要树 + SQLite FTS5/向量索引
                    ↓
   DeepSeek V4 Flash 问答与写文档 ──► 带可点击时间戳的回答 / Markdown 笔记
                    ↓
        豆包 Seed 2.1 Turbo 看图答题（视觉型问题）
```

## 为什么是这套架构

旧项目（`D:\videoplayer\MediaPlayer`）已证明「播放器 + ASR + LLM」能跑通，
但它效果差是**架构问题**，不是缺几个功能。三处刻意换掉的假设：

| 会被期待落空的假设 | 换成什么 | 为什么 |
|---|---|---|
| 分块 RAG 是主架构 | **分层摘要树 + 长上下文直答** | 2 小时中文课全稿约 3 万字 ≈ 2–4 万 token，塞得进 1M 窗口。分块检索结构上做不到「整门课总结」和跨段聚合问答 |
| Whisper 做中文主力 | **火山录音文件识别**（字级时间戳） | Whisper-large-v3 中文 CER 20%，约每 5 字错 1 个；错字会毒化召回、文档与时间戳定位 |
| OCR 推到第二期 | **OCR 进第一期** | 中文网课大量信息在 PPT/板书。只有转写的检索在 PPT 课上必然低于预期 |

另外修掉三处会白吃工期的坑：

- **HTML5 `<video>` 打不开很多课程文件**：Chromium 不支持 AC3/EAC3，MKV/FLV 不可靠
  → 入库时一次性 remux 成 H.264/AAC MP4 播放代理。
- **一上来就上 Electron + sidecar + monorepo 是负债** → MVP 用 FastAPI + 单页 HTML，
  `<video>` 加 `currentTime = t` 就能实现点击时间戳跳转。
- **PaddleOCR 是打包地狱** → RapidOCR + PP-OCRv5 ONNX（模型 22MB），全链坚持 ONNX。

## 模型选型

| 角色 | 模型 | 价格 | 关键点 |
|---|---|---|---|
| 转写（字级时间戳） | 火山**录音文件识别 2.0** | 0.8 元/小时，免费 20 小时 | 字/词时间戳、热词、智能分句；旧项目火山账号可复用 |
| 文本问答 + 写文档 | **DeepSeek Flash**（模型 ID：`deepseek-flash`） | 峰 ¥3/¥9，闲 ¥1.5/¥4.5；**缓存 ¥0.10/¥0.05** | 1M 上下文、输出 384K、缓存最便宜 |
| 视觉（课件 / 板书） | 豆包 **Seed 2.1 Turbo** | ¥3/¥15，缓存 ¥0.6 | 256K；与 ASR 同属火山方舟，一个账号 |
| 本地兜底（可选） | bge-m3 int8 嵌入 / bge-reranker-v2-m3 fp32 / RapidOCR / FunASR | 免费 | 不装也能跑，检索退化为关键词 |

一门 2 小时课全链路成本约 **2 元**。

> **模型 ID 的坑**：DeepSeek 的可调用 ID 是 `deepseek-flash` 和 `deepseek-v4-pro`。
> 「DeepSeek V4 Flash」是宣传名，写成 `deepseek-v4-flash` 会直接 404。
> 用 `GET https://api.deepseek.com/models` 可以列出实际可用值。

**两条必须遵守的纪律：**

1. **转写走「录音文件识别」，不要走「音频理解」。** 后者是让模型听懂音频并推理，
   单次时长上限小、贵、而且时间戳是生成的而非对齐出来的。前者为长音频设计，
   支持数小时文件，返回精确对齐的时间戳。
2. **整稿必须作为稳定前缀。** 同一份 3 万字转写要重发数十次，命中上下文缓存后
   输入成本降到 ¥0.05–0.10/M，不命中要 ¥1.5–3/M，**差 6–30 倍**。
   所以绝不把「当前播放进度」塞进前缀 —— 它只放在问题那一侧。

## 快速开始

```powershell
# 1. 环境
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .

# 2. 课件 OCR（强烈建议：中文网课大量信息在 PPT/板书/代码录屏里，不在语音里）
.\.venv\Scripts\python.exe -m pip install rapidocr-onnxruntime
#    注意：一帧 OCR 约 3–22 秒（随文字量增长），一门课上百张就是几十分钟。
#    这是纯本地 CPU 开销，一次性。之后重跑入库会复用已有课件文字，不会重做。

# 3. 可选：本地检索模型（不装也能跑，检索退化为关键词；以后再装可 `vedioai reindex`）
.\.venv\Scripts\python.exe -m pip install tokenizers onnxruntime
#    放到 models/bge-m3-onnx/ 与 models/bge-reranker-v2-m3-onnx/
#    每个目录需要 model.onnx 和 tokenizer.json
#    `tokenizers` 是最容易漏的一个依赖：缺了它模型加载会静默失败，
#    表现就是「模型文件明明放对了，但检索还是纯关键词」。用 `vedioai check` 可确认。

# 4. 密钥
copy .env.example .env
#    编辑 .env，填入 VOLC_SPEECH_API_KEY（或 APP_ID + ACCESS_TOKEN）、
#    DEEPSEEK_API_KEY、ARK_API_KEY

# 5. 自检：会告诉你缺什么
.\.venv\Scripts\vedioai.exe check

# 6. 配置好密钥后，发真实请求验证「密钥能不能用、服务有没有开通」
.\.venv\Scripts\vedioai.exe check --live
```

第 6 步别跳过。最常见的失败不是「忘了填密钥」，而是**密钥填了但服务/资源没开通**，
或者资源 ID 填错 —— 这两种情况不测的话，会等到真正入库到一半时才暴露，白等十几分钟。
`--live` 会提交 2 秒静音给 ASR 验证鉴权与资源开通（服务回「无语音」即代表链路全通），
并对两个 LLM 各发一次 `max_tokens=1` 的对话验证密钥与模型名。

### 关于火山 ASR 的两个坑

**坑一：标准版收不了本地文件。** 标准版 `volc.seedasr.auc`（0.8 元/小时）**只接受公网
URL**，不能直接提交本地文件 —— 想用它得先把音频传到对象存储。

本项目默认走**极速版 `volc.bigasr.auc_turbo`**，它支持 `audio.data`（base64），
**本地文件开箱即用、无需对象存储**。等你有 TOS 了，把 `.env` 里的
`VOLC_ASR_RESOURCE_ID` 换成标准版即可。

**坑二：`20000003` 不是错误。** 它的含义是「请求合法，但音频里没有语音」。静音段、
纯音乐、无人说话都会返回它。把它当错误处理会连锁出三个问题：自检永远失败、
课程含静音段时整次入库被误判失败、以及不可重试的参数错误被白重试。

代码里已按「传输层异常 + 服务端 5xx 才重试，4xxxxxxx 立刻失败并给出指引」处理，
并有回归测试钉住（`tests/test_asr_client.py`）。

### 关于 DeepSeek 推理模型的坑（实测踩过）

**`deepseek-flash` 是推理模型，默认开启思考。它的思维链按输出 token 计费，
而且计入 `max_tokens` 预算。**

实测数据（片段摘要任务，同一段输入重复调用）：

| `max_tokens` | 失败率 | 失败样本的 `finish_reason` |
|---|---|---|
| 1024 | **4/8 = 50%** | 全部是 `length`（被截断） |
| 4096 | 0/4 | — |

4 个失败样本里有 3 个 `content` 长度正好是 0——**思考过程把 1024 个 token 全吃光了，
答案一个字都没输出**。另一个输出了 925 字符后被截断，JSON 不完整。

对这个现象，代码里有三层防护：

1. **结构化抽取一律关闭思考**（`chat_json` 默认 `thinking=False`）。摘要、术语抽取
   这类任务是「抄写」而不是「推理」，开思考只会更慢更贵。
2. **被截断时自动放大预算重试**，并把截断报成「输出被 max_tokens 截断」——
   而不是含糊的「无法解析 JSON」。
3. **摘要失败不再作废整次入库**。这是当初最严重的后果：视频摘要那一步返回被截断，
   异常一路抛出，前面**已经付过费**的转写成果全部被丢弃，课程变成 `failed`。

**另一个隐藏副作用**：思考模式下 DeepSeek 会**忽略 `temperature`**。而评估集靠
`temperature=0` 保证可复现——也就是说，开着思考时同一套题的分数会飘，无法用来做
前后对比。评估链路已强制关闭思考。

问答默认也关思考（更快、更便宜、且答案的证据已经在上下文里）。觉得某些难题答得不够
好时可以打开对比：`vedioai.config.yaml` 里设 `llm.thinking: true`。

### 重跑入库不会重复付转写费

转写是整条链路里**唯一按小时计费、也唯一不可复现**的环节。所以入库带复用逻辑：
只要库里已有该课程的转写结果，重跑就会跳过 ASR。

```
复用已有播放代理 data\library\<id>\proxy.mp4
复用已有转写：228 句（跳过 ASR，不产生费用）
```

课程 ID 由「路径 + 大小 + 修改时间」生成，所以换了文件就是另一门课，不存在用到
过期转写的风险。

### 第一步：先建评估集（不要跳过）

旧项目就是死在「效果差但不知道差在哪」：改了提示词、换了模型，没有任何办法判断
是变好还是变差。所以先做这件事。

```powershell
# 入库一门真实课程
.\.venv\Scripts\vedioai.exe ingest "D:\courses\lesson1.mp4"

# 按课程真实章节生成 40 题骨架（事实/跨段/视觉/全局各 10）
.\.venv\Scripts\vedioai.exe eval --scaffold <video_id>

# 一边看视频一边把题目填满，然后跑基线
.\.venv\Scripts\vedioai.exe eval evals\questions.yaml
```

评分标准写在 [`evals/SCORING.md`](evals/SCORING.md)，包含各类型的**门槛线** ——
没到门槛时该动哪一层，那里也写了。

### 日常使用

```powershell
.\.venv\Scripts\vedioai.exe serve            # 打开 http://127.0.0.1:17831
```

界面里：左边课程库与入库，中间播放器与章节，右边问答与学习文档。
回答里的 `[12:30]` 可以直接点击跳转。

命令行也可以：

```powershell
vedioai list                                  # 课程库
vedioai info <video_id>                       # 章节与摘要
vedioai ask <video_id> "这门课一共讲了几种排序？"
vedioai ask <video_id> "这里最坏复杂度是多少" --at 12:30
vedioai notes <video_id>                      # 导出学习文档
vedioai reindex [video_id]                    # 后装嵌入模型后重建向量索引
```

## RAG：已经做了，但**不是主力**，而且你不需要「安装」它

先澄清一个常见误解：**RAG 是架构模式，不是软件包。** 没有什么「本地没有 RAG」这回事，
它已经写在 `retrieve.py` 里了。

本项目的检索是**混合检索**：FTS5 关键词（jieba 预分词）+ 向量召回，用 RRF 融合，
可选重排。它的作用是：

- 给出回答的**可点击引用位置**（`[12:30]` 能跳转）；
- 为视觉型问题**挑出证据课件图**送给多模态模型；
- 多集课程库降本（不必每次都塞全稿）。

**但它不是主路径。** 这是刻意的设计：2 小时中文课的全稿约 3 万字，塞得进 1M 上下文，
所以**局部事实型问题直接长上下文直答**——证据完整、无需召回、不会因为 top-k 召回不中
而答错。只有全局聚合型问题（「一共讲了几种 X」）和视觉型问题才依赖检索的产物。

**所以：单课问答不需要 RAG 也能很好用。** 检索在这里是加分项，不是前提。

### 你缺的只是两个可选模型文件（不装也能跑）

| 模型 | 作用 | 不装的后果 | 放置位置 |
|---|---|---|---|
| bge-m3 ONNX **int8** | 向量召回 | 检索退化为纯关键词 | `models/bge-m3-onnx/{model.onnx,tokenizer.json}` |
| bge-reranker-v2-m3 ONNX **fp32** | 精排前 5 | 不做重排 | `models/bge-reranker-v2-m3-onnx/{model.onnx,tokenizer.json}` |

重排**必须用 fp32**：int8 量化会改变 top-1 排序——而排序器量化的恰好就是这个「第一名」。

### 已经有本地 Ollama 的话，不用再下一份 ONNX

本机 Ollama 已拉过 `bge-m3` 时，直接把嵌入指过去，省掉一份 ONNX 权重：

```yaml
retrieve:
  embed_backend: ollama        # auto | onnx | ollama | none
  ollama_url: http://127.0.0.1:11434
  ollama_embed_model: bge-m3
```

`auto`（默认）先找 ONNX，找不到再退回 Ollama，都没有才退化为纯关键词。

**已知坑：新版 Ollama 在旧显卡驱动上会连 GPU 后端一起崩。** 现象是
`/api/embed` 与 `/api/embeddings` 都返回 500，报
`CUDA error: device kernel image is invalid`，日志里另有
`llama-server GPU discovery watchdog timed out`。根因是 Ollama 自带的 CUDA
运行时比驱动新：例如驱动 546.30 只到 CUDA 12.3，而新版 llama.cpp 的 PDL
内核探测（`ggml_cuda_kernel_can_use_pdl`）在它上面过不去——**注意这不是
「错选了 cuda_v13」，Ollama 用 cuda_v12 同样失败**，改库版本解决不了。

**强制 CPU 即可**（bge-m3 走 CPU 完全够用，53 个文本块约 3 秒、维度 1024 正确）：

```powershell
[Environment]::SetEnvironmentVariable('OLLAMA_LLM_LIBRARY','cpu','User')
# 再重启 Ollama（托盘图标退出后重新打开）
```

两个容易踩的点：
- **必须重启 Ollama** 让新进程继承环境变量；在已有 shell 里
  `Start-Process` 起的进程拿到的是旧环境块，不生效。
- 若想保留 GPU，可用 Vulkan（`OLLAMA_LLM_LIBRARY=vulkan`，实测可用但更慢），
  或把驱动升到支持 CUDA 13 的版本后再撤掉这个变量。

### 装了模型之后，别重新入库

```powershell
vedioai reindex            # 全部课程
vedioai reindex <video_id> # 指定课程
```

`reindex` 只读库里已有的文本块做本地编码，**零 API 成本**，而且比 `ingest` 快得多。

现在重跑 `ingest` 也不会重复付 ASR 的钱了（转写与播放代理、课件都会复用），但
`reindex` 更直接：它只做向量这一件事，不会顺带重跑摘要与 OCR。

### 装好 OCR 之后，重跑入库即可补齐课件文字

课件抽帧与 OCR 同样带复用逻辑，规则是：

- 已有课件**且已有文字** → 复用，不重做（省下几十分钟本地计算）；
- 已有课件**但一张都没有文字**（典型场景：先入库、后装 OCR）→ **自动重做**，
  避免装好之后重跑却静默复用了空文字；
- 改了抽帧参数（如 `sample_interval_ms`）→ 需要显式强制：

```powershell
vedioai ingest "D:\courses\a.mp4" --refresh-slides
```

## 目录结构

```
src/vedioai/
  config.py        配置（环境变量 > .env > yaml > 默认）
  schema.py        课程中间表示：Video / Segment / Chunk / Chapter / Slide
  store.py         SQLite + FTS5（jieba 预分词）+ 向量（numpy 暴力检索）
  pipeline.py      入库编排：探测→代理→抽音频→ASR→课件→分段→摘要→索引
  ingest/
    media.py       FFmpeg：探测、播放代理、抽音频、静音检测与切段
    asr_volc.py    火山录音文件识别（base64 直传 / 公网 URL 双模式）
    slides.py      课件抽帧（帧差 + 感知哈希）与 RapidOCR
    segment.py     句级转写 → 语义块 → 章节
  summarize.py     分层摘要树：块要点 → 章节摘要 → 全课摘要
  embedding.py     本地 bge-m3 嵌入 / bge-reranker 重排（可选，缺失时自动退化）
  retrieve.py      混合检索（FTS5 + 向量 + RRF 融合 + 可选重排）
  context.py       稳定前缀构建（缓存命中的关键）
  ask.py           问答：长上下文直答 / 全局聚合 / 视觉
  notes.py         学习文档：分层 map-reduce，不一次写完
  jobs.py          后台任务与进度
  selfcheck.py     凭证实时自检（check --live）
  eval_runner.py   评估集运行器与打分
  server.py        FastAPI + Range 视频流
  cli.py           命令行
web/index.html     单页界面
evals/             评估集、评分标准、运行器
tests/             单元测试 + 端到端集成测试（无需网络）
```

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

- `tests/test_local.py`：存储、分段、检索（含向量召回接线）、上下文、评分、自检逻辑
- `tests/test_asr_client.py`：ASR 错误码分类与重试策略（MockTransport，不发真实请求）
- `tests/test_llm_client.py`：截断处理、思考开关、JSON 容错解析（MockTransport）
- `tests/test_integration_pipeline.py`：合成视频跑完整入库 + 问答 + HTTP 服务，
  用桩替换 ASR/LLM，**不需要网络与密钥**；含「摘要失败不作废入库」
  与「重跑不重复付转写费」两项回归

当前状态：**141 项全部通过**（含会话、用量账本、评估拓展剥离、字幕/搜索/系列、集成管线）。

> 测试会生成视频、音频与 SQLite 文件。`pyproject.toml` 已把 pytest 临时目录
> 指向项目内的 `.pytest-tmp/`，避免写入系统盘 —— 系统盘只剩几百 MB 时会直接报
> `database or disk is full`。

## 已落地的学习体验（单课为主）

| 能力 | 说明 |
|---|---|
| 会话 | 一门课多个命名会话；多轮问答落库；侧栏可新建 / 重命名 / 归档 |
| 拓展标注 | 问答允许 `> **拓展**` 课外补充；摘要/笔记仍禁止；评估评分会忽略拓展段 |
| 用量账本 | 界面「用量」页；可答「这门课花了多少钱」；含 ASR / LLM / 缓存命中 |
| 字幕与转写 | `GET /api/subtitles/{id}.vtt`；播放器字幕轨 + 同步高亮转写面板 |
| 进度续播 | 按 `video_id` 记在浏览器 localStorage，下次打开续播 |
| 课内搜索 | `GET /api/search?video_id=&q=`，点结果跳到对应时间 |
| 流式回答 | `POST /api/ask/stream`（SSE）；前缀结构不变，保住 DeepSeek 缓存 |
| 入库 / 删除 | 拖拽或选文件填路径；课程库可删除（走 purge） |
| 错词纠正 | 转写里选中提交「错→对」，写入术语表并 `repair`（不重跑 ASR） |
| 课程系列 | 按源文件父目录自动分组；系列内可跨课问答（各课摘要 + 跨课检索） |

**评估集**：`evals/questions.yaml`（Android 加密课 40 题）+
`evals/questions.0665b84d44c3131e.yaml`（Root 课 20 题）。两门课在拓展提示词下
问答门禁均可通过。

## 各阶段目标

| 阶段 | 内容 | 出口条件 |
|---|---|---|
| 第一期 | 评估集、入库管线、FastAPI + 单页、时间戳跳转、摘要树、笔记导出 | 评估集达标（见 SCORING.md 门槛） |
| 第一期+ | 会话、账本、字幕/进度/搜索/流式、界面纠错、目录级课程系列 | 本地单人学习闭环顺手 |
| 第二期 | Electron 壳、媒体库、入库断点续跑、本地 FunASR 离线模式、密钥进凭据库、PyInstaller onedir 打包与模型首启下载 | 能装给不写代码的人用 |
| 第三期 | 测验/闪卡、更细的跨系列管理 | 一个学期多门课复习闭环 |

**刻意不做**：社交、直播、AI 生视频、知识图谱。

## 已知风险

- **音频上云**：需要显式开关、费用提示、按文件 hash 缓存避免重复提交。
- **ASR 错字会进文档**：需要置信度可视化 + 术语热词表 + 局部纠错后重嵌入。
  不要指望用户手改 3 万字。
- **缓存被破坏会让成本失控**：任何把变化量塞进前缀的改动，都要在 PR 里被拦下。
- **向量召回静默失效**：嵌入模型是可选的，接线错了（例如构造检索器时漏传
  `embedder`）不会报错，只会悄悄退化成关键词检索，看起来「能用」。已有回归测试
  直接断言 `AskService.retriever.embedder`，不要删。
- **转写质量调优会反复迭代**：没有评估集时无法收敛。
- **打包与首启体验**（第二期）：PyInstaller 的问题总量通常远超预期。
- **不要自研播放内核**：旧项目最大的时间坑就在这里。

## 环境备注

当前机器上 `PATH` 里的 FFmpeg 是旧项目的 4.2.2 版本。已验证它能完成 remux、
抽音频、静音检测与抽帧，够用。但若要支持较新的编码（如 AV1 软解）或遇到
解码异常，建议换成 6.x/7.x 并把 `vedioai.config.yaml` 里的 `media.ffmpeg`
指向新路径。
