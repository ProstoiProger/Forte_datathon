import hmac
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, Security, UploadFile, status
from fastapi.security import APIKeyHeader

from cashwatch.jobs import JobManager, QueueFullError, job_view

CHUNK = 1024 * 1024
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(request: Request, key: str | None = Security(api_key_header)) -> None:
    keys = request.app.state.settings.keys
    if not key or not any(hmac.compare_digest(key.encode(), k.encode()) for k in keys):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")


def _manager(request: Request) -> JobManager:
    return request.app.state.manager


def accept_upload(request: Request, upload: UploadFile, allowed: tuple[str, ...]) -> dict[str, Any]:
    settings = request.app.state.settings
    manager = _manager(request)
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > settings.max_upload_bytes + CHUNK:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "file is too large")
    filename = upload.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"expected {', '.join(allowed)} file")
    job_id = manager.new_job_id()
    dest = manager.upload_path(job_id, suffix)
    size = 0
    try:
        with dest.open("wb") as out:
            while chunk := upload.file.read(CHUNK):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "file is too large")
                out.write(chunk)
        if size == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "file is empty")
        manager.submit(job_id, filename, dest)
    except QueueFullError:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "queue is full, retry later") from None
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return {"job_id": job_id, "status": "processing"}


def read_job(request: Request, job_id: str, detail: bool) -> dict[str, Any]:
    record = _manager(request).get(job_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return job_view(record, detail)


api_router = APIRouter(prefix="/api/v1")


@api_router.get("/health")
def health(request: Request) -> dict[str, Any]:
    manager = _manager(request)
    ready, detail = getattr(request.app.state, "analyzer_status", (True, manager.analyzer.name))
    return {"status": "ok", "analyzer": manager.analyzer.name, "ready": ready, "detail": detail, "queue": manager.queue_size()}


@api_router.post("/analyze", dependencies=[Depends(require_api_key)])
def analyze(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    return accept_upload(request, file, request.app.state.settings.allowed_extensions)


@api_router.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(request: Request, job_id: str) -> dict[str, Any]:
    return read_job(request, job_id, request.app.state.settings.api_extra_fields)


ui_router = APIRouter(prefix="/ui/api")


@ui_router.post("/analyze")
def ui_analyze(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    return accept_upload(request, file, request.app.state.settings.ui_allowed_extensions)


@ui_router.get("/jobs/{job_id}")
def ui_get_job(request: Request, job_id: str) -> dict[str, Any]:
    return read_job(request, job_id, True)


@ui_router.get("/types")
def ui_types(request: Request) -> list[dict[str, Any]]:
    registry = request.app.state.registry
    return [
        {
            "type_id": t.type_id,
            "title": t.title,
            "rationale": t.rationale,
            "citations": [{"clause": c.clause, "document": c.act.short, "verified": c.verified} for c in t.citations],
        }
        for t in registry.all()
        if t.type_id != "stub_motion"
    ]
