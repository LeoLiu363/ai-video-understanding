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
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from .ask import AskService
from .config import Config
from .embedding import Embedder, Reranker, build_embedder
from .ingest.asr_volc import VolcASRClient
from .jobs import JobRegistry
from .llm.client import LLMClient, from_llm_config
from .notes import NotesService
from .pipeline import IngestPipeline, video_id_for
from .schema import ms_to_hms
from .store import Store

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
        return from_llm_config(cfg.llm)

    def make_vision() -> LLMClient | None:
        if not cfg.vision.api_key:
            return None
        return LLMClient(cfg.vision.api_key, cfg.vision.base_url, cfg.vision.model)

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
            items.append(
                {
                    **v,
                    "duration_label": ms_to_hms(v["duration_ms"]),
                }
            )
        return {"items": items}

    @app.get("/api/library/{video_id}")
    def library_item(video_id: str) -> dict:
        video = store.get_video(video_id)
        if video is None:
            raise HTTPException(404, "课程不存在")
        chapters = store.get_chapters(video_id)
        return {
            "video": {**video.to_row(), "duration_label": ms_to_hms(video.duration_ms)},
            "summary": store.get_video_summary(video_id),
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
                    "ocr_text": s.ocr_text,
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

    @app.post("/api/ingest")
    def ingest(req: IngestRequest) -> dict:
        path = Path(req.path)
        if not path.exists():
            raise HTTPException(400, f"文件不存在：{path}")

        video_id = video_id_for(path)
        running = jobs.find_running("ingest", video_id)
        if running:
            return {"job_id": running.job_id, "video_id": video_id, "reused": True}

        if req.no_slides:
            cfg.slides.enabled = False

        def work(on_progress) -> dict:
            asr = VolcASRClient(cfg.asr)
            llm = None if req.skip_summary else make_llm()
            pipeline = IngestPipeline(cfg, store, llm=llm, embedder=embedder, asr=asr)
            try:
                video = pipeline.run(
                    path, skip_summary=req.skip_summary, progress=on_progress
                )
                if llm is not None:
                    # 入完立刻预热前缀，让第一次提问就命中上下文缓存
                    AskService(
                        cfg, store, llm, make_vision(),
                        embedder=embedder, reranker=reranker,
                    ).warm(video.video_id)
                return {"video_id": video.video_id, "title": video.title}
            finally:
                asr.close()

        job = jobs.submit("ingest", video_id, work)
        return {"job_id": job.job_id, "video_id": video_id}

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
                req.video_id, req.question, current_ms=req.current_ms, top_k=req.top_k
            )
            return answer.to_dict()
        finally:
            llm.close()

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
