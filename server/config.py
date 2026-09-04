"""Runtime configuration, read once from the environment."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _default_generate_concurrency() -> int:
    cpu = os.cpu_count() or 2
    return max(1, min(4, cpu // 2))


@dataclass
class Settings:
    # --- storage ---
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DATA_DIR", "data")))
    model_dir: Path = field(default_factory=lambda: Path(os.environ.get("MODEL_DIR", "supertonic3")))
    train_script: Path = field(
        default_factory=lambda: Path(
            os.environ.get("TRAIN_SCRIPT", str(Path(__file__).resolve().parent.parent / "train_style.py"))
        )
    )

    # --- auth ---
    api_key: str = os.environ.get("API_KEY", "")

    # --- workers ---
    generate_concurrency: int = field(
        default_factory=lambda: _env_int("GENERATE_CONCURRENCY", _default_generate_concurrency())
    )
    train_concurrency: int = field(default_factory=lambda: _env_int("TRAIN_CONCURRENCY", 1))
    start_workers: bool = field(default_factory=lambda: _env_bool("START_WORKERS", True))
    preload_engine: bool = field(default_factory=lambda: _env_bool("PRELOAD_ENGINE", True))

    # --- inference ---
    # auto | cpu | cuda | tensorrt
    ort_provider: str = os.environ.get("ORT_PROVIDER", "auto").strip().lower()
    ort_intra_threads: int = field(default_factory=lambda: _env_int("ORT_INTRA_THREADS", 0))
    default_total_step: int = field(default_factory=lambda: _env_int("DEFAULT_TOTAL_STEP", 6))

    # --- limits ---
    max_text_chars: int = field(default_factory=lambda: _env_int("MAX_TEXT_CHARS", 20000))
    max_batch_items: int = field(default_factory=lambda: _env_int("MAX_BATCH_ITEMS", 512))
    max_upload_bytes: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_BYTES", 200 * 1024 * 1024))
    max_attempts: int = field(default_factory=lambda: _env_int("MAX_ATTEMPTS", 2))
    train_timeout_sec: int = field(default_factory=lambda: _env_int("TRAIN_TIMEOUT_SEC", 6 * 3600))
    log_tail_lines: int = field(default_factory=lambda: _env_int("LOG_TAIL_LINES", 40))

    # --- webhooks ---
    webhook_timeout_sec: int = field(default_factory=lambda: _env_int("WEBHOOK_TIMEOUT_SEC", 15))

    def __post_init__(self) -> None:
        # Accept plain strings for the path fields; callers construct Settings by
        # hand in tests and scripts, not only from the environment.
        for name in ("data_dir", "model_dir", "train_script"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                setattr(self, name, Path(value))

    @property
    def onnx_dir(self) -> Path:
        return self.model_dir / "onnx"

    @property
    def builtin_styles_dir(self) -> Path:
        return self.model_dir / "voice_styles"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jobs.db"

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    @property
    def styles_dir(self) -> Path:
        return self.data_dir / "styles"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.voices_dir, self.styles_dir, self.outputs_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
