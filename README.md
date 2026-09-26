# vedioAI · 课程视频 AI 理解

播放本地课程视频，用 AI 理解**全部内容**：语音转写、课件 OCR、带时间戳问答与学习文档。

核心思路：**一次入库、多次查询**。

```
本地课程视频
   ├─ FFmpeg remux ──────────► H.264/AAC 播放代理
   ├─ FFmpeg 抽音频 ─────────► 16kHz 单声道 mp3
   │      └─ 火山录音文件识别 ─► 字级时间戳转写
   └─ 课件抽帧 + RapidOCR ───► 幻灯片文字
                    ↓
              课程中间表示（IR）
                    ↓
        分层摘要树 + SQLite FTS5 / 向量索引
                    ↓
   DeepSeek 问答与写文档 ──► 带可点击时间戳的回答 / Markdown 笔记
                    ↓
        豆包 Seed 看图答题（视觉型问题）
```

## 能力

| 能力 | 说明 |
|---|---|
| 入库 | 本地视频 → 转写 + 课件 OCR + 摘要 + 索引 |
| 问答 | 带时间戳引用，可点击跳转；支持多轮会话与流式输出 |
| 学习文档 | 分层生成 Markdown 笔记 |
| 字幕 / 转写 | WebVTT 字幕轨 + 同步高亮转写面板 |
| 课内搜索 | 转写、课件 OCR、章节标题一起搜，点结果跳转 |
| 进度续播 | 浏览器按课程记住播放位置 |
| 错词纠正 | 界面标记「错→对」，写入术语表并局部修复（不重跑 ASR） |
| 用量账本 | 查看本课 / 全库花了多少钱 |
| 课程系列 | 按源文件目录分组，支持系列内跨课问答 |

单课事实型问题以**整稿长上下文直答**为主；检索用于引用定位、视觉取证和跨课场景。

## 模型

| 角色 | 模型 | 说明 |
|---|---|---|
| 转写 | 火山录音文件识别（默认 `volc.bigasr.auc_turbo`） | 字级时间戳，支持本地文件 base64 |
| 文本 | DeepSeek Flash（`deepseek-flash`） | 长上下文 + 前缀缓存 |
| 视觉 | 豆包 Seed 2.1 Turbo | 课件 / 板书相关问题 |
| 本地可选 | bge-m3 / reranker / RapidOCR | 不装也能跑，检索退化为关键词 |

一门约 2 小时课，全链路费用大约 **2 元**量级（视缓存命中与课时而定）。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install rapidocr-onnxruntime   # 课件 OCR，建议安装

copy .env.example .env
# 编辑 .env：VOLC_SPEECH_API_KEY、DEEPSEEK_API_KEY、ARK_API_KEY

.\.venv\Scripts\vedioai.exe check
.\.venv\Scripts\vedioai.exe check --live
```

可选本地检索模型（不装则关键词检索）：

```powershell
.\.venv\Scripts\python.exe -m pip install tokenizers onnxruntime
# 放入 models/bge-m3-onnx/ 与 models/bge-reranker-v2-m3-onnx/
# 各需 model.onnx、tokenizer.json
```

若本机已有 Ollama 的 `bge-m3`，可在 `vedioai.config.yaml` 中设置：

```yaml
retrieve:
  embed_backend: ollama
  ollama_url: http://127.0.0.1:11434
  ollama_embed_model: bge-m3
```

### 入库与使用

```powershell
.\.venv\Scripts\vedioai.exe ingest "D:\courses\lesson1.mp4"
.\.venv\Scripts\vedioai.exe serve --host 0.0.0.0 --port 17831
# 打开 http://127.0.0.1:17831
```

命令行：

```powershell
vedioai list
vedioai info <video_id>
vedioai ask <video_id> "这门课一共讲了几种排序？"
vedioai ask <video_id> "这里最坏复杂度是多少" --at 12:30
vedioai notes <video_id>
vedioai reindex [video_id]          # 补装嵌入模型后重建向量
vedioai usage                       # 用量账本
```

重跑入库会复用已有转写、播放代理与课件文字，**不会重复付 ASR 费用**。只重建向量用 `reindex`。先入库后装 OCR 时，再跑一次 `ingest` 会自动补齐课件文字；改了抽帧参数可用 `--refresh-slides`。

### 评估（可选）

```powershell
vedioai eval --scaffold <video_id>
vedioai eval evals/questions.yaml
```

评分标准见 [`evals/SCORING.md`](evals/SCORING.md)。仓库内附两门课评估集（Android 加密 40 题、Root 原理 20 题）。

## 目录结构

```
src/vedioai/     入库、检索、问答、笔记、Web 服务
web/index.html   单页界面
evals/           评估集与评分标准
tests/           单元测试与集成测试
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

当前约 **141** 项测试。临时文件写在项目内 `.pytest-tmp/`。

## 配置要点

- 密钥只放在 `.env`（已 gitignore），参考 `.env.example`
- 默认 ASR 资源 ID：`volc.bigasr.auc_turbo`（本地文件可直接传）
- 问答默认关闭模型「思考」，摘要 / 结构化抽取同样关闭，以控制费用与稳定性
- 整课转写作为稳定前缀发送，以命中 DeepSeek 上下文缓存；播放进度只放在问题侧
