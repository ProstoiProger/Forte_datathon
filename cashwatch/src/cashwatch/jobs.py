import json
import logging
import os
import queue
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cashwatch.analyzers import AnalyzerError, threshold_for
from cashwatch.analyzers.base import Analyzer
from cashwatch.config import Settings
from cashwatch.results import public_violation, render_violations
from cashwatch.rules import Registry
from cashwatch.video import VideoError, probe

log = logging.getLogger("cashwatch.jobs")

JOB_ID_RE = re.compile(r"^[0-9a-f]{16}$")
PENDING = ("queued", "processing")


class QueueFullError(Exception):
    pass


def job_budget(remaining: float, waiting: int, fair_share: bool, floor: float, cap: float | None = None) -> float:
    """Seconds this job may use. With fair sharing the time left is split with the jobs waiting behind it, so
    five uploads at once each get about a fifth instead of the first taking everything and starving the rest.
    `cap` bounds a job that starts alone, so uploads arriving a few seconds later still find time left."""
    share = remaining / (1 + max(waiting, 0)) if fair_share else remaining
    budget = max(min(remaining, share), floor)
    return min(budget, cap) if cap else budget


@dataclass
class JobRecord:
    job_id: str
    filename: str
    upload: str
    status: str = "queued"
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    duration_sec: float | None = None
    analyzer: str | None = None
    verdict: str | None = None
    violations: list[dict[str, Any]] | None = None
    error: str | None = None


def job_view(record: JobRecord, detail: bool) -> dict[str, Any]:
    if record.status in PENDING:
        return {"job_id": record.job_id, "status": "processing", "progress": round(record.progress, 3)}
    if record.status == "failed":
        return {"job_id": record.job_id, "status": "failed", "error": record.error or "processing failed"}
    view: dict[str, Any] = {
        "job_id": record.job_id,
        "status": "done",
        "verdict": record.verdict,
        "violations": [public_violation(v, detail) for v in record.violations or []],
    }
    if detail:
        view["duration_sec"] = record.duration_sec
        view["analyzer"] = record.analyzer
        if record.started_at and record.finished_at:
            view["processing_sec"] = round(record.finished_at - record.started_at, 2)
    return view


class JobManager:
    def __init__(self, settings: Settings, analyzer: Analyzer, registry: Registry) -> None:
        self.settings = settings
        self.analyzer = analyzer
        self.registry = registry
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue(maxsize=settings.queue_max)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._recover()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="cashwatch-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def queue_size(self) -> int:
        return self._queue.qsize()

    def new_job_id(self) -> str:
        return secrets.token_hex(8)

    def upload_path(self, job_id: str, suffix: str) -> Path:
        return self.settings.uploads_dir / f"{job_id}{suffix}"

    def submit(self, job_id: str, filename: str, upload: Path) -> JobRecord:
        record = JobRecord(job_id=job_id, filename=filename, upload=upload.name)
        with self._lock:
            self._jobs[job_id] = record
        self._save(record)
        try:
            self._queue.put_nowait(job_id)
        except queue.Full:
            with self._lock:
                self._jobs.pop(job_id, None)
            self._job_file(job_id).unlink(missing_ok=True)
            raise QueueFullError from None
        return record

    def get(self, job_id: str) -> JobRecord | None:
        if not JOB_ID_RE.match(job_id):
            return None
        with self._lock:
            record = self._jobs.get(job_id)
        if record is not None:
            return record
        path = self._job_file(job_id)
        if path.exists():
            record = self._load(path)
            with self._lock:
                self._jobs.setdefault(job_id, record)
            return record
        return None

    def _job_file(self, job_id: str) -> Path:
        return self.settings.jobs_dir / f"{job_id}.json"

    def _save(self, record: JobRecord) -> None:
        with self._lock:
            payload = json.dumps(asdict(record), ensure_ascii=False)
        path = self._job_file(record.job_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)

    def _load(self, path: Path) -> JobRecord:
        return JobRecord(**json.loads(path.read_text(encoding="utf-8")))

    def _recover(self) -> None:
        for path in sorted(self.settings.jobs_dir.glob("*.json")):
            try:
                record = self._load(path)
            except (json.JSONDecodeError, TypeError, OSError):
                log.warning("skipping unreadable job file %s", path.name)
                continue
            if record.status in PENDING:
                if (self.settings.uploads_dir / record.upload).exists() and not self._queue.full():
                    record.status, record.progress = "queued", 0.0
                    self._queue.put_nowait(record.job_id)
                else:
                    record.status, record.error = "failed", "interrupted before completion"
                self._save(record)
            with self._lock:
                self._jobs[record.job_id] = record

    def _update(self, record: JobRecord, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(record, key, value)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            record = self.get(job_id)
            if record is not None:
                self._process(record)
            self._queue.task_done()

    def _log_trace(self, record: JobRecord) -> None:
        """One log line per job with what the analyzer saw (windows, candidates, verifications), for diagnosis."""
        stats = getattr(getattr(self.analyzer, "primary", self.analyzer), "last_stats", None)
        if not stats:
            return
        verifications = [
            {k: v.get(k) for k in ("window", "type_id", "rule", "confirmed", "frame_t", "description")}
            for v in stats.get("verifications", [])
        ]
        trace = {k: stats.get(k) for k in ("windows", "perceived", "facts_failed", "facts_skipped", "candidates", "seconds")}
        log.info("job %s (%s) trace %s verifications %s", record.job_id, record.filename,
                 json.dumps(trace, ensure_ascii=False), json.dumps(verifications, ensure_ascii=False))

    def _process(self, record: JobRecord) -> None:
        upload = self.settings.uploads_dir / record.upload
        remaining = self.settings.time_budget_sec - (time.time() - record.created_at)
        budget = job_budget(remaining, self._queue.qsize(), self.settings.fair_share_budget, self.settings.min_job_sec,
                            self.settings.max_job_sec)
        deadline = time.monotonic() + budget
        self._update(record, status="processing", started_at=time.time(), analyzer=self.analyzer.name)
        self._save(record)

        def progress(value: float) -> None:
            self._update(record, progress=0.9 * max(0.0, min(1.0, value)))

        try:
            info = probe(upload)
            threshold = threshold_for(self.settings, self.analyzer)
            events = [e for e in self.analyzer.analyze(upload, progress, deadline) if e.confidence >= threshold]
            self._log_trace(record)
            violations = render_violations(events, upload, info, self.registry)
            self._update(
                record,
                status="done",
                progress=1.0,
                duration_sec=round(info.duration_sec, 2),
                verdict="violation" if violations else "no_violation",
                violations=violations,
                finished_at=time.time(),
            )
        except AnalyzerError as exc:
            # nothing could be analyzed in time: an explicit failure, never a guessed "no violation"
            log.warning("job %s not analyzed: %s", record.job_id, exc)
            self._update(record, status="failed", error=f"analysis incomplete: {exc}", finished_at=time.time())
        except VideoError as exc:
            self._update(record, status="failed", error=str(exc), finished_at=time.time())
        except Exception:
            log.exception("job %s failed", record.job_id)
            self._update(record, status="failed", error="internal processing error", finished_at=time.time())
        finally:
            if not self.settings.keep_uploads:
                upload.unlink(missing_ok=True)
        self._save(record)
        log.info("job %s %s in %.1fs", record.job_id, record.status, (record.finished_at or time.time()) - (record.started_at or time.time()))
