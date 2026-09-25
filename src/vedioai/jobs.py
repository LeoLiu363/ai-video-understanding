"""后台任务注册表。

入库、生成文档都是长任务（转写可能几分钟），必须异步执行 + 可查询进度，
否则前端只能干等，用户会以为卡死。
"""

from __future__ import annotations

import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from .pipeline import Progress


@dataclass
class Job:
    job_id: str
    kind: str
    video_id: str
    status: str = "running"
    stage: str = ""
    percent: float = 0.0
    message: str = ""
    error: str = ""
    result: dict = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished_at: str = ""

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "video_id": self.video_id,
            "status": self.status,
            "stage": self.stage,
            "percent": round(self.percent, 1),
            "message": self.message,
            "error": self.error,
            "result": self.result,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 20) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)
        return jobs[:limit]

    def find_running(self, kind: str, video_id: str) -> Job | None:
        with self._lock:
            for job in self._jobs.values():
                if job.kind == kind and job.video_id == video_id and job.status == "running":
                    return job
        return None

    def submit(
        self,
        kind: str,
        video_id: str,
        fn: Callable[[Callable[[Progress], None]], dict],
    ) -> Job:
        """在后台线程跑 fn，fn 通过回调上报进度。"""
        job = Job(job_id=uuid.uuid4().hex[:12], kind=kind, video_id=video_id)
        with self._lock:
            self._jobs[job.job_id] = job

        def on_progress(progress: Progress) -> None:
            job.stage = progress.stage
            job.message = progress.message or job.stage
            job.percent = progress.percent

        def runner() -> None:
            try:
                job.result = fn(on_progress) or {}
                job.status = "done"
                job.percent = 100.0
                job.message = "完成"
            except Exception as exc:  # noqa: BLE001
                job.status = "failed"
                job.error = str(exc)
                job.message = f"失败：{exc}"
            finally:
                job.finished_at = datetime.now().isoformat(timespec="seconds")

        threading.Thread(target=runner, name=f"{kind}-{job.job_id}", daemon=True).start()
        return job


def format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()
