"""FastAPI application: submit batches of TTS / voice-training jobs and poll them."""

from __future__ import annotations

import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from . import __version__
from .config import Settings, settings as default_settings
from .engine import Engine, ModelNotReady, get_engine, gpu_info, torch_device
from .paths import NotFound, list_styles, list_voices, probe_audio, resolve_style, unique_path
from .schemas import (
    AVAILABLE_LANGS,
    BatchAccepted,
    BatchRequest,
    BatchView,
    GenerateRequest,
    HealthView,
    JobListView,
    JobStatus,
    JobType,
    JobView,
    QueueStats,
    StyleView,
    SubmitAccepted,
    SystemView,
    TrainRequest,
    VoiceView,
)
from .store import JobStore
from .workers import WorkerPool, read_log_tail

log = logging.getLogger("supertonic.api")

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
MEDIA_TYPES = {".wav": "audio/wav", ".flac": "audio/flac", ".ogg": "audio/ogg"}
_STARTED_AT = time.time()


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or default_settings
    settings.ensure_dirs()

    store = JobStore(settings.db_path)
    engine = get_engine(settings)
    pool = WorkerPool(settings, store, engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.start_workers:
            pool.restore()
            pool.start()
            if settings.preload_engine and engine.model_ready:
                # Warm the ONNX sessions now so the first request isn't the one
                # that pays for loading four graphs.
                try:
                    engine.load()
                    log.info("engine warm on providers %s", engine.providers)
                except Exception as exc:  # pragma: no cover - depends on host
                    log.warning("engine preload failed, will retry lazily: %s", exc)
        try:
            yield
        finally:
            pool.shutdown()
            store.close()

    app = FastAPI(
        title="Supertonic Voice Clone Batch API",
        description=(
            "Queue voice-style training and speech synthesis jobs, run many of them in "
            "parallel on a single GPU box, and poll their status."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.store = store
    app.state.engine = engine
    app.state.pool = pool

    # ------------------------------------------------------------------ auth

    def require_api_key(request: Request) -> None:
        if not settings.api_key:
            return
        supplied = request.headers.get("x-api-key") or ""
        if not supplied:
            authorization = request.headers.get("authorization", "")
            if authorization.lower().startswith("bearer "):
                supplied = authorization[7:]
        if not secrets.compare_digest(supplied, settings.api_key):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key")

    guard = [Depends(require_api_key)]

    # --------------------------------------------------------------- helpers

    def to_view(job: dict[str, Any], include_log: bool = False) -> JobView:
        payload = dict(job)
        if include_log:
            payload["log_tail"] = read_log_tail(
                settings.logs_dir / f"{job['id']}.log", settings.log_tail_lines
            )
        return JobView(**payload)

    def fetch_job(job_id: str) -> dict[str, Any]:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"job '{job_id}' not found")
        return job

    def validate_generate(params: dict[str, Any]) -> None:
        if len(params.get("text", "")) > settings.max_text_chars:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"text exceeds MAX_TEXT_CHARS ({settings.max_text_chars})",
            )
        try:
            resolve_style(params["style"], settings)
        except NotFound as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    def enqueue(
        job_type: JobType,
        params: dict[str, Any],
        *,
        batch_id: Optional[str] = None,
        priority: int = 0,
        metadata: Optional[dict[str, Any]] = None,
        webhook_url: Optional[str] = None,
    ) -> dict[str, Any]:
        job = store.create_job(
            job_type,
            params,
            batch_id=batch_id,
            priority=priority,
            metadata=metadata,
            webhook_url=webhook_url,
        )
        pool.submit(job)
        return job

    # ---------------------------------------------------------------- health

    @app.get("/health", response_model=HealthView, tags=["system"])
    def health() -> HealthView:
        return HealthView(
            status="ok",
            version=__version__,
            model_ready=engine.model_ready,
            uptime_seconds=round(time.time() - _STARTED_AT, 1),
        )

    @app.get("/v1/system", response_model=SystemView, tags=["system"], dependencies=guard)
    def system() -> SystemView:
        counts = store.counts()
        return SystemView(
            version=__version__,
            model_ready=engine.model_ready,
            engine_loaded=engine.loaded,
            providers=engine.providers,
            device=torch_device(),
            gpu=gpu_info(),
            workers={
                "generate": settings.generate_concurrency,
                "train": settings.train_concurrency,
                "busy": len(pool.active_jobs()),
            },
            queue_depth=pool.queue_depth(),
            jobs=QueueStats(**{k: counts.get(k, 0) for k in QueueStats.model_fields}),
            data_dir=str(settings.data_dir.resolve()),
        )

    @app.get("/v1/languages", tags=["system"], dependencies=guard)
    def languages() -> dict[str, Any]:
        return {"languages": AVAILABLE_LANGS}

    # ------------------------------------------------------------------ jobs

    @app.post(
        "/v1/jobs/generate",
        response_model=SubmitAccepted,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["jobs"],
        dependencies=guard,
    )
    def submit_generate(payload: GenerateRequest) -> SubmitAccepted:
        params = payload.model_dump(exclude={"priority", "webhook_url", "metadata"})
        validate_generate(params)
        job = enqueue(
            JobType.generate,
            params,
            priority=payload.priority,
            metadata=payload.metadata,
            webhook_url=payload.webhook_url,
        )
        return SubmitAccepted(job=to_view(job))

    @app.post(
        "/v1/jobs/train",
        response_model=SubmitAccepted,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["jobs"],
        dependencies=guard,
    )
    def submit_train(payload: TrainRequest) -> SubmitAccepted:
        params = payload.model_dump(exclude={"priority", "webhook_url", "metadata"})
        job = enqueue(
            JobType.train,
            params,
            priority=payload.priority,
            metadata=payload.metadata,
            webhook_url=payload.webhook_url,
        )
        return SubmitAccepted(job=to_view(job))

    @app.post(
        "/v1/batches",
        response_model=BatchAccepted,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["batches"],
        dependencies=guard,
    )
    def submit_batch(payload: BatchRequest) -> BatchAccepted:
        if len(payload.items) > settings.max_batch_items:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"batch has {len(payload.items)} items, limit is {settings.max_batch_items}",
            )
        # Validate every item before creating any job, so a bad item can't leave
        # half a batch queued.
        prepared: list[tuple[JobType, dict[str, Any], int, dict[str, Any]]] = []
        for index, item in enumerate(payload.items):
            if item.type == JobType.generate:
                params = item.generate.model_dump()  # type: ignore[union-attr]
                try:
                    validate_generate(params)
                except HTTPException as exc:
                    raise HTTPException(exc.status_code, f"items[{index}]: {exc.detail}") from exc
            else:
                params = item.train.model_dump()  # type: ignore[union-attr]
            prepared.append((item.type, params, item.priority, item.metadata))

        batch_id = store.create_batch(payload.metadata, payload.webhook_url)
        jobs = [
            enqueue(
                job_type,
                params,
                batch_id=batch_id,
                priority=priority,
                metadata=metadata,
                webhook_url=payload.webhook_url,
            )
            for job_type, params, priority, metadata in prepared
        ]
        return BatchAccepted(
            batch_id=batch_id, accepted=len(jobs), jobs=[to_view(job) for job in jobs]
        )

    @app.get("/v1/jobs", response_model=JobListView, tags=["jobs"], dependencies=guard)
    def list_jobs(
        job_status: Optional[list[JobStatus]] = Query(None, alias="status"),
        job_type: Optional[JobType] = Query(None, alias="type"),
        batch_id: Optional[str] = None,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> JobListView:
        jobs, total = store.list_jobs(
            status=[s.value for s in job_status] if job_status else None,
            job_type=job_type.value if job_type else None,
            batch_id=batch_id,
            limit=limit,
            offset=offset,
        )
        return JobListView(
            total=total, limit=limit, offset=offset, jobs=[to_view(job) for job in jobs]
        )

    @app.get("/v1/jobs/{job_id}", response_model=JobView, tags=["jobs"], dependencies=guard)
    def get_job(job_id: str, include_log: bool = True) -> JobView:
        return to_view(fetch_job(job_id), include_log=include_log)

    @app.get("/v1/jobs/{job_id}/logs", tags=["jobs"], dependencies=guard)
    def get_job_logs(job_id: str, lines: int = Query(200, ge=1, le=10000)) -> PlainTextResponse:
        fetch_job(job_id)
        tail = read_log_tail(settings.logs_dir / f"{job_id}.log", lines)
        return PlainTextResponse("\n".join(tail) or "(no output yet)")

    @app.get("/v1/jobs/{job_id}/result", tags=["jobs"], dependencies=guard)
    def get_job_result(job_id: str) -> FileResponse:
        job = fetch_job(job_id)
        if job["status"] != JobStatus.succeeded.value:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"job is '{job['status']}', results are only available once it succeeds",
            )
        result = job.get("result") or {}
        raw = result.get("audio_path") or result.get("style_path")
        if not raw or not Path(raw).is_file():
            raise HTTPException(status.HTTP_410_GONE, "result file is no longer on disk")
        path = Path(raw)
        return FileResponse(
            path,
            media_type=MEDIA_TYPES.get(path.suffix, "application/octet-stream"),
            filename=path.name,
        )

    @app.post("/v1/jobs/{job_id}/cancel", response_model=JobView, tags=["jobs"], dependencies=guard)
    def cancel_job(job_id: str) -> JobView:
        job, was_running = store.cancel_job(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"job '{job_id}' not found")
        if job["status"] in (
            JobStatus.succeeded.value,
            JobStatus.failed.value,
        ):
            raise HTTPException(
                status.HTTP_409_CONFLICT, f"job already finished with status '{job['status']}'"
            )
        pool.cancel(job_id)
        if was_running:
            # The worker notices the flag and records the terminal state itself.
            return to_view(fetch_job(job_id))
        return to_view(job)

    @app.delete("/v1/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["jobs"], dependencies=guard)
    def delete_job(job_id: str) -> Response:
        job = fetch_job(job_id)
        if job["status"] in (JobStatus.queued.value, JobStatus.running.value):
            raise HTTPException(
                status.HTTP_409_CONFLICT, "cancel the job before deleting it"
            )
        _remove_tree(settings.outputs_dir / job_id)
        (settings.logs_dir / f"{job_id}.log").unlink(missing_ok=True)
        store.delete_job(job_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # --------------------------------------------------------------- batches

    @app.get("/v1/batches/{batch_id}", response_model=BatchView, tags=["batches"], dependencies=guard)
    def get_batch(batch_id: str) -> BatchView:
        batch = store.get_batch(batch_id)
        if batch is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"batch '{batch_id}' not found")
        jobs, total = store.list_jobs(batch_id=batch_id, limit=settings.max_batch_items)
        counts: dict[str, int] = {}
        for job in jobs:
            counts[job["status"]] = counts.get(job["status"], 0) + 1
        done = sum(
            counts.get(s, 0)
            for s in (JobStatus.succeeded.value, JobStatus.failed.value, JobStatus.canceled.value)
        )
        if total == 0:
            overall = "empty"
        elif done < total:
            overall = "running" if counts.get(JobStatus.running.value) else "queued"
        elif counts.get(JobStatus.failed.value):
            overall = "completed_with_failures"
        elif counts.get(JobStatus.canceled.value) == total:
            overall = "canceled"
        else:
            overall = "succeeded"
        progress = (
            sum(float(job.get("progress") or 0.0) for job in jobs) / total if total else 0.0
        )
        return BatchView(
            id=batch_id,
            created_at=batch["created_at"],
            total=total,
            counts=counts,
            status=overall,
            progress=round(progress, 4),
            jobs=[to_view(job) for job in jobs],
        )

    @app.post("/v1/batches/{batch_id}/cancel", response_model=BatchView, tags=["batches"], dependencies=guard)
    def cancel_batch(batch_id: str) -> BatchView:
        if store.get_batch(batch_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"batch '{batch_id}' not found")
        jobs, _ = store.list_jobs(
            batch_id=batch_id,
            status=[JobStatus.queued.value, JobStatus.running.value],
            limit=settings.max_batch_items,
        )
        for job in jobs:
            store.cancel_job(job["id"])
            pool.cancel(job["id"])
        return get_batch(batch_id)

    # ---------------------------------------------------------------- voices

    @app.post(
        "/v1/voices",
        response_model=VoiceView,
        status_code=status.HTTP_201_CREATED,
        tags=["voices"],
        dependencies=guard,
    )
    async def upload_voice(
        file: UploadFile = File(..., description="Target speaker audio (WAV recommended)."),
        name: Optional[str] = Form(None, description="Voice id; defaults to the filename stem."),
    ) -> VoiceView:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in AUDIO_SUFFIXES:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f"unsupported audio type '{suffix or file.filename}'; "
                f"accepted: {', '.join(sorted(AUDIO_SUFFIXES))}",
            )
        stem = _safe_id(name or Path(file.filename or "voice").stem)
        target = unique_path(settings.voices_dir, stem, suffix)

        written = 0
        try:
            with target.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    written += len(chunk)
                    if written > settings.max_upload_bytes:
                        raise HTTPException(
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"upload exceeds MAX_UPLOAD_BYTES ({settings.max_upload_bytes})",
                        )
                    handle.write(chunk)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

        stat = target.stat()
        return VoiceView(
            id=target.stem,
            filename=target.name,
            bytes=stat.st_size,
            created_at=stat.st_mtime,
            **probe_audio(target),
        )

    @app.get("/v1/voices", response_model=list[VoiceView], tags=["voices"], dependencies=guard)
    def get_voices() -> list[VoiceView]:
        return [VoiceView(**entry) for entry in list_voices(settings)]

    @app.delete(
        "/v1/voices/{voice_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["voices"],
        dependencies=guard,
    )
    def delete_voice(voice_id: str) -> Response:
        matches = [
            path
            for path in settings.voices_dir.glob(f"{_safe_id(voice_id)}.*")
            if path.suffix.lower() in AUDIO_SUFFIXES
        ]
        if not matches:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"voice '{voice_id}' not found")
        for path in matches:
            path.unlink(missing_ok=True)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # ---------------------------------------------------------------- styles

    @app.get("/v1/styles", response_model=list[StyleView], tags=["styles"], dependencies=guard)
    def get_styles() -> list[StyleView]:
        return [StyleView(**entry) for entry in list_styles(settings)]

    @app.get("/v1/styles/{style_id}", tags=["styles"], dependencies=guard)
    def download_style(style_id: str) -> FileResponse:
        try:
            path = resolve_style(style_id, settings)
        except NotFound as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        return FileResponse(path, media_type="application/json", filename=path.name)

    @app.delete(
        "/v1/styles/{style_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["styles"],
        dependencies=guard,
    )
    def delete_style(style_id: str) -> Response:
        path = settings.styles_dir / f"{_safe_id(style_id)}.json"
        if not path.is_file():
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"trained style '{style_id}' not found (built-in styles cannot be deleted)",
            )
        path.unlink()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # ------------------------------------------------------------ exceptions

    @app.exception_handler(ModelNotReady)
    async def _model_not_ready(_: Request, exc: ModelNotReady) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(NotFound)
    async def _not_found(_: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    return app


def _safe_id(value: str) -> str:
    import re

    cleaned = re.sub(r"[^\w.-]+", "_", (value or "").strip(), flags=re.UNICODE).strip("._")
    return cleaned[:80] or "item"


def _remove_tree(path: Path) -> None:
    import shutil

    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


app = create_app()
