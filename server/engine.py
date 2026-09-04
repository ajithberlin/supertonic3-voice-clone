"""ONNX Runtime engine: one warm TextToSpeech shared by every generate worker.

``InferenceSession.run`` is thread-safe, so a single set of sessions serves all
generate workers concurrently — which is what makes many parallel jobs cheap.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .config import Settings

_ENGINE_LOCK = threading.Lock()
_ENGINE: Optional["Engine"] = None


def available_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    return list(ort.get_available_providers())


def resolve_providers(preference: str) -> list[str]:
    """Map the ORT_PROVIDER setting onto providers this build actually has."""
    have = available_providers()
    cpu = ["CPUExecutionProvider"]
    if not have:
        return cpu
    pref = (preference or "auto").lower()
    if pref == "cpu":
        return cpu
    if pref == "tensorrt" and "TensorrtExecutionProvider" in have:
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", *cpu]
    if pref in ("cuda", "gpu", "auto") and "CUDAExecutionProvider" in have:
        return ["CUDAExecutionProvider", *cpu]
    if pref in ("cuda", "gpu", "tensorrt"):
        # Explicitly requested but unavailable: fall back rather than refuse to boot.
        return cpu
    return cpu


def gpu_info() -> Optional[dict[str, Any]]:
    """Best-effort GPU description; never raises."""
    try:
        import torch  # noqa: PLC0415  (optional dependency in inference-only images)

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "backend": "torch",
                "count": torch.cuda.device_count(),
                "name": props.name,
                "total_memory_mb": round(props.total_memory / 1024 / 1024),
                "capability": f"{props.major}.{props.minor}",
                "cuda": torch.version.cuda,
            }
    except Exception:  # pragma: no cover - depends on the host
        pass
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            lines = [ln.strip() for ln in out.stdout.strip().splitlines() if ln.strip()]
            name, _, mem = lines[0].partition(",")
            return {"backend": "nvidia-smi", "count": len(lines), "name": name.strip(),
                    "total_memory": mem.strip()}
    except Exception:
        pass
    return None


class ModelNotReady(RuntimeError):
    pass


class Engine:
    """Warm ONNX sessions plus the generation loop."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.providers = resolve_providers(settings.ort_provider)
        self.loaded_at: Optional[float] = None
        self._tts = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ load

    @property
    def model_ready(self) -> bool:
        onnx_dir = self.settings.onnx_dir
        required = (
            "duration_predictor.onnx",
            "text_encoder.onnx",
            "vector_estimator.onnx",
            "vocoder.onnx",
            "tts.json",
            "unicode_indexer.json",
        )
        return all((onnx_dir / f).exists() for f in required)

    @property
    def loaded(self) -> bool:
        return self._tts is not None

    def load(self):
        if self._tts is not None:
            return self._tts
        with self._load_lock:
            if self._tts is not None:
                return self._tts
            if not self.model_ready:
                raise ModelNotReady(
                    f"Supertonic model files not found under {self.settings.onnx_dir}. "
                    "Run ./setup_infer.sh (or set MODEL_DIR) before serving requests."
                )
            import onnxruntime as ort

            from helper import (
                TextToSpeech,
                load_cfgs,
                load_onnx_all,
                load_text_processor,
            )

            opts = ort.SessionOptions()
            if self.settings.ort_intra_threads > 0:
                opts.intra_op_num_threads = self.settings.ort_intra_threads
            onnx_dir = str(self.settings.onnx_dir)
            dp, text_enc, vector_est, vocoder = load_onnx_all(onnx_dir, opts, self.providers)
            self._tts = TextToSpeech(
                load_cfgs(onnx_dir),
                load_text_processor(onnx_dir),
                dp,
                text_enc,
                vector_est,
                vocoder,
            )
            self.loaded_at = time.time()
            return self._tts

    @property
    def sample_rate(self) -> int:
        return int(self.load().sample_rate)

    # -------------------------------------------------------------- generate

    def generate(
        self,
        *,
        text: str,
        lang: str,
        style_path: Path,
        out_path: Path,
        total_step: int,
        speed: float,
        silence_duration: float = 0.3,
        progress: Optional[Callable[[float, str], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> dict[str, Any]:
        """Synthesize ``text`` chunk by chunk so progress and cancellation are live."""
        import numpy as np
        import soundfile as sf

        from helper import chunk_text, load_voice_style

        tts = self.load()
        style = load_voice_style([str(style_path)])
        max_len = 120 if lang in ("ko", "ja") else 300
        chunks = chunk_text(text, max_len=max_len) or [text.strip()]

        pieces: list[Any] = []
        total_duration = 0.0
        silence = np.zeros((1, int(silence_duration * tts.sample_rate)), dtype=np.float32)
        started = time.time()

        for index, chunk in enumerate(chunks):
            if should_cancel and should_cancel():
                raise JobCanceled("canceled during synthesis")
            wav, duration = tts.batch([chunk], [lang], style, total_step, speed)
            trimmed = wav[0, : int(tts.sample_rate * float(duration[0]))]
            if pieces:
                pieces.append(silence[0])
                total_duration += silence_duration
            pieces.append(trimmed)
            total_duration += float(duration[0])
            if progress:
                progress(
                    (index + 1) / len(chunks),
                    f"synthesized chunk {index + 1}/{len(chunks)}",
                )

        audio = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, tts.sample_rate)
        return {
            "audio_path": str(out_path),
            "filename": out_path.name,
            "sample_rate": tts.sample_rate,
            "duration_seconds": round(float(len(audio)) / tts.sample_rate, 3),
            "reported_duration_seconds": round(total_duration, 3),
            "chunks": len(chunks),
            "bytes": out_path.stat().st_size,
            "synthesis_seconds": round(time.time() - started, 3),
            "real_time_factor": round(
                (time.time() - started) / max(len(audio) / tts.sample_rate, 1e-6), 3
            ),
            "providers": self.providers,
        }


class JobCanceled(RuntimeError):
    """Raised inside a worker when the job was canceled mid-flight."""


def get_engine(settings: Optional[Settings] = None) -> Engine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            from .config import settings as default_settings

            _ENGINE = Engine(settings or default_settings)
        return _ENGINE


def reset_engine() -> None:
    """Test hook: drop the process-wide engine."""
    global _ENGINE
    with _ENGINE_LOCK:
        _ENGINE = None


def torch_device() -> str:
    try:
        from utils.device import get_device

        return get_device()
    except Exception:
        return "cpu" if not os.environ.get("CUDA_VISIBLE_DEVICES") else "cuda:0"
