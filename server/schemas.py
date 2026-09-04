"""Request/response models for the batch API."""

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

AVAILABLE_LANGS = [
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr",
    "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk",
    "sl", "sv", "tr", "uk", "vi",
]


class JobType(str, Enum):
    generate = "generate"
    train = "train"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    canceled = "canceled"


TERMINAL_STATUSES = {JobStatus.succeeded, JobStatus.failed, JobStatus.canceled}


class GenerateParams(BaseModel):
    """One synthesis request."""

    text: str = Field(..., min_length=1, description="Text to synthesize.")
    style: str = Field(
        ...,
        description=(
            "Voice style to use: a style id returned by /v1/styles, a built-in name "
            "such as 'F4', or a path to a style JSON file."
        ),
    )
    lang: str = Field("en", description="Language code.")
    total_step: Optional[int] = Field(
        None, ge=1, le=64, description="Diffusion steps. Higher is slower but cleaner."
    )
    speed: float = Field(1.05, gt=0.1, le=3.0, description="Speech speed multiplier.")
    silence_duration: float = Field(
        0.3, ge=0.0, le=5.0, description="Silence inserted between text chunks, in seconds."
    )
    output_format: Literal["wav", "flac", "ogg"] = "wav"
    name: Optional[str] = Field(None, description="Optional label used in the output filename.")

    @field_validator("lang")
    @classmethod
    def _known_lang(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in AVAILABLE_LANGS:
            raise ValueError(f"unsupported lang '{v}'; supported: {', '.join(AVAILABLE_LANGS)}")
        return v

    @field_validator("style", "name")
    @classmethod
    def _strip(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if isinstance(v, str) else v


class TrainParams(BaseModel):
    """One voice-style training request."""

    name: str = Field(..., min_length=1, max_length=64, description="Name of the resulting style.")
    voice: str = Field(
        ...,
        description="Voice id returned by POST /v1/voices, or a path to a target WAV file.",
    )
    gender: Literal["F", "M"] = "F"
    reference_style: str = Field(
        "auto", description="'auto', 'none', or a style id / path to seed the optimization."
    )
    seed: int = 49
    speed: float = Field(1.05, gt=0.1, le=3.0)
    num_steps: int = Field(3000, ge=1, le=100_000)
    learning_rate: float = Field(2e-4, gt=0.0, le=1.0)
    vocoder_steps: int = Field(6, ge=1, le=64)
    save_steps: int = Field(500, ge=1)
    early_stop_loss_threshold: float = Field(0.015, ge=0.0, le=2.0)

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        v = v.strip()
        if not v or any(c in v for c in "/\\\0") or v in (".", ".."):
            raise ValueError("name must not be empty or contain path separators")
        return v


def _validate_http_url(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    if not v.startswith(("http://", "https://")):
        raise ValueError("webhook_url must be an http(s) URL")
    return v


class JobSubmit(BaseModel):
    """Common submission envelope."""

    priority: int = Field(0, ge=-100, le=100, description="Higher runs first.")
    webhook_url: Optional[str] = Field(
        None, description="POSTed a job summary when the job reaches a terminal state."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("webhook_url")
    @classmethod
    def _http_url(cls, v: Optional[str]) -> Optional[str]:
        return _validate_http_url(v)


class GenerateRequest(JobSubmit, GenerateParams):
    pass


class TrainRequest(JobSubmit, TrainParams):
    pass


class BatchItem(BaseModel):
    type: JobType
    priority: int = Field(0, ge=-100, le=100)
    metadata: dict[str, Any] = Field(default_factory=dict)
    generate: Optional[GenerateParams] = None
    train: Optional[TrainParams] = None

    @model_validator(mode="after")
    def _one_payload(self) -> "BatchItem":
        if self.type == JobType.generate and self.generate is None:
            raise ValueError("items with type='generate' need a 'generate' payload")
        if self.type == JobType.train and self.train is None:
            raise ValueError("items with type='train' need a 'train' payload")
        return self


class BatchRequest(BaseModel):
    """Submit many jobs at once; they run in parallel across the worker pools."""

    items: list[BatchItem] = Field(..., min_length=1)
    webhook_url: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("webhook_url")
    @classmethod
    def _http_url(cls, v: Optional[str]) -> Optional[str]:
        return _validate_http_url(v)


class JobView(BaseModel):
    id: str
    type: JobType
    status: JobStatus
    batch_id: Optional[str] = None
    priority: int = 0
    progress: float = 0.0
    progress_message: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    attempts: int = 0
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    queue_seconds: Optional[float] = None
    run_seconds: Optional[float] = None
    log_tail: Optional[list[str]] = None


class JobListView(BaseModel):
    total: int
    limit: int
    offset: int
    jobs: list[JobView]


class BatchView(BaseModel):
    id: str
    created_at: float
    total: int
    counts: dict[str, int]
    status: str
    progress: float
    jobs: list[JobView]


class SubmitAccepted(BaseModel):
    job: JobView


class BatchAccepted(BaseModel):
    batch_id: str
    accepted: int
    jobs: list[JobView]


class VoiceView(BaseModel):
    id: str
    filename: str
    bytes: int
    created_at: float
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None


class StyleView(BaseModel):
    id: str
    source: Literal["builtin", "trained"]
    bytes: int
    created_at: float
    job_id: Optional[str] = None


class QueueStats(BaseModel):
    queued: int = 0
    running: int = 0
    succeeded: int = 0
    failed: int = 0
    canceled: int = 0


class SystemView(BaseModel):
    version: str
    model_ready: bool
    engine_loaded: bool
    providers: list[str]
    device: str
    gpu: Optional[dict[str, Any]] = None
    workers: dict[str, int]
    queue_depth: dict[str, int]
    jobs: QueueStats
    data_dir: str


class HealthView(BaseModel):
    status: str
    version: str
    model_ready: bool
    uptime_seconds: float
