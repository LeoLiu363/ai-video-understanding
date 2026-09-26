"""本机 FastAPI 服务 + 单页前端。

MVP 刻意不做 Electron：`<video src="/media/<id>#t=1234">` 加
`video.currentTime = t` 就能实现「点击引用跳转」，零 UI 工程。
桌面壳等管线被验证有效之后再补。

服务只监听 127.0.0.1，视频文件只在本机读取。
"""

from __future__ import annotations

import logging
import mimetypes
import re
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel
import shutil

from .ask import AskService
from .config import Config
from .context import summary_without_outline
from .embedding import Embedder, Reranker, build_embedder
from . import ledger
from .glossary import append_course_correction
from .ingest.asr_volc import VolcASRClient
from .jobs import JobRegistry
from .llm.client import LLMClient, from_llm_config
from .notes import NotesService
from .pipeline import IngestPipeline, repair_videos, video_id_for
from .schema import ms_to_hms
from .series import (
    annotate_library_items,
    build_series_prefix,
    list_series,
    search_across,
)
from .store import Store
from .subtitles import segments_to_webvtt

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class IngestRequest(BaseModel):
    path: str
    skip_summary: bool = False
    no_slides: bool = False


class AskRequest(BaseModel):
    video_id: str
    question: str
    current_ms: int | None = None
    top_k: int | None = None
    # 可选：带上会话 ID 就进多轮 + 落库；不带则仍是单轮（CLI / 评估兼容）
    session_id: str | None = None


class SeriesAskRequest(BaseModel):
    series_id: str
    question: str
    video_ids: list[str] | None = None


class FixRequest(BaseModel):
    video_id: str
    wrong: str
    right: str
    reason: str = "界面标记"


class SessionCreateRequest(BaseModel):
    video_id: str
    title: str = ""


class SessionUpdateRequest(BaseModel):
    title: str | None = None
    archived: bool | None = None


class NotesRequest(BaseModel):
    video_id: str


def create_app(cfg: Config | None = None) -> FastAPI:
    """创建应用。cfg 省略时自行加载配置（便于 `uvicorn --factory` 直接启动）。"""
    if cfg is None:
        from .config import load_config

        cfg = load_config()

    app = FastAPI(title="vedioAI", version="0.1.0")
    store = Store(cfg.db_path)
    jobs = JobRegistry()

    embedder = build_embedder(cfg)
    reranker = Reranker(cfg.rerank_model_dir)

    def make_llm() -> LLMClient:
        # 挂上记账：挂在客户端出口上，所以问答/摘要/文档全都自动被记，
        # 新增调用点也不会漏（见 ledger.recorder）
        return ledger.attach(from_llm_config(cfg.llm), store)

    def make_vision() -> LLMClient | None:
        if not cfg.vision.api_key:
            return None
        return ledger.attach(
            LLMClient(cfg.vision.api_key, cfg.vision.base_url, cfg.vision.model), store
        )

    app.state.store = store
    app.state.jobs = jobs

    # ------------------------------------------------------------------ 页面

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        page = WEB_DIR / "index.html"
        if not page.exists():
            return HTMLResponse("<h1>web/index.html 缺失</h1>", status_code=500)
        return HTMLResponse(page.read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- 课程库

    @app.get("/api/library")
    def library() -> dict:
        items = []
        for v in store.list_videos():
            # 列表接口不带 video_summary：那是一份 4500 字左右的摘要，
            # 列表 UI 一个字都用不到（只用标题/时长/章数/状态）。以前每门课都
            # 白传一遍，课程一多就是纯浪费。
            v.pop("video_summary", None)
            items.append(
                {
                    **v,
                    "duration_label": ms_to_hms(v["duration_ms"]),
                }
            )
        items = annotate_library_items(items)
        return {"items": items}

    @app.get("/api/library/{video_id}")
    def library_item(video_id: str, with_ocr: bool = False) -> dict:
        video = store.get_video(video_id)
        if video is None:
            raise HTTPException(404, "课程不存在")
        chapters = store.get_chapters(video_id)
        return {
            "video": {**video.to_row(), "duration_label": ms_to_hms(video.duration_ms)},
            # 剥掉摘要自带的「大纲：」列表：下面 chapters 字段已经列了同一批章节
            "summary": summary_without_outline(store.get_video_summary(video_id) or ""),

            "chapters": [
                {
                    "chapter_id": ch.chapter_id,
                    "idx": ch.idx,
                    "title": ch.title or f"第 {ch.idx + 1} 章",
                    "summary": ch.summary,
                    "start_ms": ch.start_ms,
                    "end_ms": ch.end_ms,
                    "label": ms_to_hms(ch.start_ms),
                }
                for ch in chapters
            ],
            "slides": [
                {
                    "idx": s.idx,
                    "start_ms": s.start_ms,
                    "label": ms_to_hms(s.start_ms),
                    # 课件 OCR 占了整个响应的 90%（实测 222KB 里的 162KB），
                    # 而界面点开课程时并不用它。默认不发，需要时显式加 ?with_ocr=1。
                    "ocr_text": (s.ocr_text if with_ocr else ""),
                }
                for s in store.get_slides(video_id)
            ],
            "stats": {
                "chunks": len(store.get_chunks(video_id)),
                "segments": len(store.get_segments(video_id)),
            },
        }

    @app.delete("/api/library/{video_id}")
    def library_delete(video_id: str) -> dict:
        IngestPipeline(cfg, store).purge(video_id)
        return {"ok": True}

    # ------------------------------------------------------------------ 入库

    def _start_ingest_job(path: Path, *, skip_summary: bool = False, no_slides: bool = False) -> dict:
        if not path.exists():
            raise HTTPException(400, f"文件不存在：{path}")
        video_id = video_id_for(path)
        running = jobs.find_running("ingest", video_id)
        if running:
            return {"job_id": running.job_id, "video_id": video_id, "reused": True}

        if no_slides:
            cfg.slides.enabled = False

        def work(on_progress) -> dict:
            asr = VolcASRClient(cfg.asr)
            llm = None if skip_summary else make_llm()
            pipeline = IngestPipeline(cfg, store, llm=llm, embedder=embedder, asr=asr)
            try:
                video = pipeline.run(
                    path, skip_summary=skip_summary, progress=on_progress
                )
                if llm is not None:
                    AskService(
                        cfg, store, llm, make_vision(),
                        embedder=embedder, reranker=reranker,
                    ).warm(video.video_id)
                return {"video_id": video.video_id, "title": video.title}
            finally:
                asr.close()

        job = jobs.submit("ingest", video_id, work)
        return {"job_id": job.job_id, "video_id": video_id}

    @app.post("/api/ingest")
    def ingest(req: IngestRequest) -> dict:
        return _start_ingest_job(
            Path(req.path), skip_summary=req.skip_summary, no_slides=req.no_slides
        )

    @app.post("/api/ingest/upload")
    async def ingest_upload(file: UploadFile = File(...)) -> dict:
        """浏览器选文件/拖拽：先落到 data/inbox/，再走同一条入库任务。"""
        name = Path(file.filename or "upload.bin").name
        if not name or name in (".", ".."):
            raise HTTPException(400, "文件名无效")
        inbox = cfg.data_dir / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        dest = inbox / name
        # 同名文件加短后缀，避免覆盖
        if dest.exists():
            dest = inbox / f"{dest.stem}-{datetime.now().strftime('%H%M%S')}{dest.suffix}"
        try:
            with dest.open("wb") as out:
                shutil.copyfileobj(file.file, out)
        finally:
            await file.close()
        return _start_ingest_job(dest)

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "任务不存在")
        return job.to_dict()

    @app.get("/api/jobs")
    def job_list() -> dict:
        return {"items": [j.to_dict() for j in jobs.list()]}

    # ------------------------------------------------------------------ 问答

    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict:
        if not cfg.llm.api_key:
            raise HTTPException(400, "未配置 DEEPSEEK_API_KEY")
        llm = make_llm()
        try:
            service = AskService(
                cfg, store, llm, make_vision(),
                embedder=embedder, reranker=reranker,
            )
            answer = service.ask(
                req.video_id,
                req.question,
                current_ms=req.current_ms,
                top_k=req.top_k,
                session_id=req.session_id,
            )
            return answer.to_dict()
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            llm.close()

    @app.post("/api/ask/stream")
    def ask_stream(req: AskRequest):
        """SSE 流式问答。事件：meta / delta / done / error。"""
        if not cfg.llm.api_key:
            raise HTTPException(400, "未配置 DEEPSEEK_API_KEY")

        import json as _json

        def event_stream():
            llm = make_llm()
            try:
                service = AskService(
                    cfg, store, llm, make_vision(),
                    embedder=embedder, reranker=reranker,
                )
                for kind, payload in service.ask_stream(
                    req.video_id,
                    req.question,
                    current_ms=req.current_ms,
                    top_k=req.top_k,
                    session_id=req.session_id,
                ):
                    if kind == "meta":
                        data = payload
                    elif kind == "delta":
                        data = {"text": payload}
                    elif kind == "done":
                        data = payload.to_dict()
                    else:
                        data = {"raw": str(payload)}
                    yield f"event: {kind}\ndata: {_json.dumps(data, ensure_ascii=False)}\n\n"
            except Exception as exc:  # noqa: BLE001
                yield (
                    "event: error\ndata: "
                    + _json.dumps({"message": str(exc)}, ensure_ascii=False)
                    + "\n\n"
                )
            finally:
                llm.close()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/subtitles/{video_id}.vtt")
    def subtitles_vtt(video_id: str) -> Response:
        if store.get_video(video_id) is None:
            raise HTTPException(404, "课程不存在")
        vtt = segments_to_webvtt(store.get_segments(video_id))
        return Response(
            content=vtt,
            media_type="text/vtt; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/segments/{video_id}")
    def list_segments(video_id: str) -> dict:
        if store.get_video(video_id) is None:
            raise HTTPException(404, "课程不存在")
        items = [
            {
                "idx": s.idx,
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "text": s.text,
                "label": ms_to_hms(s.start_ms),
            }
            for s in store.get_segments(video_id)
        ]
        return {"items": items}

    @app.get("/api/search")
    def search(video_id: str, q: str, top_k: int = 20) -> dict:
        if store.get_video(video_id) is None:
            raise HTTPException(404, "课程不存在")
        q = (q or "").strip()
        if not q:
            return {"items": []}
        items = store.search_segments(video_id, q, top_k=top_k)
        for it in items:
            it["label"] = ms_to_hms(it["start_ms"])
        return {"items": items, "q": q}

    @app.post("/api/fix")
    def fix_term(req: FixRequest) -> dict:
        """界面标记错词：写入自动术语表并对本课执行 repair（含重嵌入）。"""
        try:
            auto_path = append_course_correction(
                cfg.glossary_path,
                req.video_id,
                req.wrong,
                req.right,
                reason=req.reason or "界面标记",
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        report = repair_videos(
            cfg, store, [req.video_id], embedder=embedder, dry_run=False
        )
        return {
            "ok": True,
            "glossary": str(auto_path),
            "report": report.get(req.video_id) or {},
        }

    @app.get("/api/series")
    def series_list() -> dict:
        return {"items": list_series(store)}

    @app.post("/api/ask-series")
    def ask_series(req: SeriesAskRequest) -> dict:
        """系列内跨课问答：各课摘要前缀 + 跨课检索证据。"""
        if not cfg.llm.api_key:
            raise HTTPException(400, "未配置 DEEPSEEK_API_KEY")
        groups = {g["series_id"]: g for g in list_series(store)}
        group = groups.get(req.series_id)
        if group is None:
            raise HTTPException(404, "系列不存在")
        video_ids = req.video_ids or [v["video_id"] for v in group["videos"]]
        video_ids = [vid for vid in video_ids if store.get_video(vid) is not None]
        if not video_ids:
            raise HTTPException(400, "系列内没有可用课程")

        prefix = build_series_prefix(store, video_ids)
        hits = search_across(store, video_ids, req.question)
        evidence_lines = []
        for h in hits:
            evidence_lines.append(
                f"- 《{h['title']}》[{h['label']}] {h['text']}"
            )
        evidence = "\n".join(evidence_lines) if evidence_lines else "（本系列关键词检索暂无命中）"
        question_text = (
            f"问题：{req.question}\n\n"
            f"【跨课检索证据】\n{evidence}\n\n"
            "请综合各课摘要与证据回答；引用时写清课名与时间，格式如《课名》[MM:SS]。"
        )

        from .llm import prompts as _prompts

        llm = make_llm()
        try:
            with ledger.usage_scope("ask", video_ids[0]):
                full_prompt = f"{_prompts.QA_GLOBAL}\n\n---\n{prefix}"
                reply = llm.ask(full_prompt, question_text)
            citations = [
                {
                    "start_ms": h["start_ms"],
                    "end_ms": h["end_ms"],
                    "label": h["label"],
                    "text": h["text"],
                    "video_id": h["video_id"],
                    "title": h["title"],
                }
                for h in hits[:8]
            ]
            return {
                "text": reply.text,
                "intent": "series",
                "citations": citations,
                "usage": {
                    "prompt_tokens": reply.usage.prompt_tokens,
                    "completion_tokens": reply.usage.completion_tokens,
                    "cached_tokens": reply.usage.cached_tokens,
                    "cache_hit_rate": round(reply.usage.cache_hit_rate, 3),
                },
                "series_id": req.series_id,
                "video_ids": video_ids,
            }
        finally:
            llm.close()

    # ------------------------------------------------------------------ 会话

    @app.get("/api/sessions")
    def list_sessions(video_id: str, include_archived: bool = False) -> dict:
        if store.get_video(video_id) is None:
            raise HTTPException(404, "课程不存在")
        return {
            "items": store.list_sessions(video_id, include_archived=include_archived)
        }

    @app.post("/api/sessions")
    def create_session(req: SessionCreateRequest) -> dict:
        if store.get_video(req.video_id) is None:
            raise HTTPException(404, "课程不存在")
        return store.create_session(req.video_id, title=req.title)

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        session = store.get_session(session_id)
        if session is None:
            raise HTTPException(404, "会话不存在")
        messages = store.get_messages(session_id)
        first = next((m["content"] for m in messages if m["role"] == "user"), "")
        from .store import _short_title

        display = session.get("title") or _short_title(first)
        return {
            **session,
            "display_title": display,
            "messages": messages,
            "message_count": len(messages),
        }

    @app.patch("/api/sessions/{session_id}")
    def update_session(session_id: str, req: SessionUpdateRequest) -> dict:
        if req.title is None and req.archived is None:
            raise HTTPException(400, "没有可更新的字段")
        updated = store.update_session(
            session_id, title=req.title, archived=req.archived
        )
        if updated is None:
            raise HTTPException(404, "会话不存在")
        return updated

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str) -> dict:
        if not store.delete_session(session_id):
            raise HTTPException(404, "会话不存在")
        return {"ok": True}

    @app.post("/api/notes")
    def notes(req: NotesRequest) -> dict:
        if not cfg.llm.api_key:
            raise HTTPException(400, "未配置 DEEPSEEK_API_KEY")
        video_id = req.video_id

        running = jobs.find_running("notes", video_id)
        if running:
            return {"job_id": running.job_id, "reused": True}

        def work(on_progress) -> dict:
            llm = make_llm()
            try:
                service = NotesService(cfg, store, llm)
                result = service.generate(
                    video_id,
                    progress=lambda d, t, m: on_progress(_NotesProgress(d, t, m)),
                )
                return result.to_dict()
            finally:
                llm.close()

        job = jobs.submit("notes", video_id, work)
        return {"job_id": job.job_id}

    @app.get("/api/notes/{video_id}")
    def notes_document(video_id: str) -> dict:
        """读取磁盘上已生成的学习文档。

        以前只有 POST（生成），没有 GET（读取），于是界面只能显示「本次会话刚
        生成、还在内存里」的那一份——服务一重启就再也看不到已有的 notes.md。
        这个接口让界面直接读文件，生成与阅读解耦。
        """
        video = store.get_video(video_id)
        if video is None:
            raise HTTPException(404, "课程不存在")

        out_dir = Path(cfg.library_dir) / video_id
        notes_path = out_dir / "notes.md"
        if not notes_path.exists():
            return {"exists": False, "markdown": "", "concepts": "", "meta": {}}

        markdown = notes_path.read_text(encoding="utf-8")
        st = notes_path.stat()
        concepts_path = out_dir / "concepts.md"
        return {
            "exists": True,
            "markdown": markdown,
            "concepts": (
                concepts_path.read_text(encoding="utf-8") if concepts_path.exists() else ""
            ),
            "meta": {
                "chars": len(markdown),
                "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
                # 生成失败时 NotesService 不会覆盖良品，会留一份 .bak；
                # 一并告知界面，用户才知道手上这份是不是被保留的旧版。
                "has_backup": (out_dir / "notes.md.bak").exists(),
            },
        }

    # ------------------------------------------------------------------ 媒体

    @app.get("/media/{video_id}")
    def media(video_id: str, request: Request) -> Response:
        video = store.get_video(video_id)
        if video is None:
            raise HTTPException(404, "课程不存在")
        target = Path(video.proxy_path or video.path)
        if not target.exists():
            # 代理被删时回退到原片
            target = Path(video.path)
        if not target.exists():
            raise HTTPException(404, "媒体文件不存在")
        return _range_response(target, request, "video/mp4")

    @app.get("/api/slides/{video_id}/{idx}")
    def slide_image(video_id: str, idx: int) -> Response:
        slides = {s.idx: s for s in store.get_slides(video_id)}
        slide = slides.get(idx)
        if slide is None:
            raise HTTPException(404, "幻灯片不存在")
        path = Path(slide.image_path)
        if not path.exists():
            raise HTTPException(404, "幻灯片图片已清理")
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return FileResponse(path, media_type=mime)

    @app.get("/api/usage")
    def usage(video_id: str = "") -> dict:
        """用量账本。video_id 省略 = 全库汇总。

        带上单价表与「高峰/闲时」判定依据，因为估价随计费时段翻倍——
        界面要能说清这个数是怎么来的。
        """
        data = store.usage_summary(video_id or None)
        data["peak_now"] = ledger.is_peak()
        data["prices"] = {
            model: {
                tier: {"hit": p.hit, "miss": p.miss, "out": p.out}
                for tier, p in tiers.items()
            }
            for model, tiers in ledger.PRICES.items()
        }
        data["asr_yuan_per_hour"] = ledger.ASR_YUAN_PER_HOUR
        return data

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": True,
            "asr_ready": cfg.asr.ready,
            "llm_ready": bool(cfg.llm.api_key),
            "vision_ready": bool(cfg.vision.api_key),
            "embedder": embedder.available,
            "reranker": reranker.available,
        }

    return app


class _NotesProgress:
    """把 notes 的回调适配成统一的 Progress 形状。"""

    __slots__ = ("done", "total", "message")

    def __init__(self, done: int, total: int, message: str):
        self.done = done
        self.total = total
        self.message = message

    @property
    def percent(self) -> float:
        return (self.done / self.total * 100.0) if self.total else 0.0

    @property
    def stage(self) -> str:
        return "notes"


def _range_response(path: Path, request: Request, media_type: str) -> Response:
    """带 Range 支持的静态文件响应。

    自己实现而不是依赖框架：`<video>` 拖动进度条依赖 206 与
    Content-Range，处理不当会导致无法 seek。
    """
    size = path.stat().st_size
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(path, media_type=media_type)

    match = _RANGE_RE.match(range_header)
    if not match:
        return FileResponse(path, media_type=media_type)

    start_raw, end_raw = match.groups()
    start = int(start_raw) if start_raw else 0
    end = int(end_raw) if end_raw else size - 1
    end = min(end, size - 1)
    if start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    length = end - start + 1

    def stream():
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                data = fh.read(min(1024 * 512, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    }
    return StreamingResponse(stream(), status_code=206, headers=headers, media_type=media_type)
