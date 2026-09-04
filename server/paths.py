"""Resolution of voice/style references to files, with the sandbox enforced.

Clients pass short ids ("F4", "vox_1a2b", "my-voice"). Callers may also pass a
path, but only one that lands inside a directory the server owns -- otherwise a
crafted request could read arbitrary files off a shared GPU host.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from .config import Settings


class NotFound(LookupError):
    """A referenced voice or style does not exist."""


def _roots(settings: Settings) -> list[Path]:
    roots = [settings.data_dir, settings.model_dir, Path.cwd()]
    resolved: list[Path] = []
    for root in roots:
        try:
            resolved.append(root.resolve())
        except OSError:  # pragma: no cover - unreadable mount
            continue
    return resolved


def within_sandbox(path: Path, settings: Settings) -> bool:
    try:
        target = path.resolve()
    except OSError:
        return False
    for root in _roots(settings):
        if target == root or root in target.parents:
            return True
    return False


def resolve_style(ref: str, settings: Settings) -> Path:
    """Map a style reference onto a JSON file on disk."""
    ref = (ref or "").strip()
    if not ref:
        raise NotFound("style reference is empty")

    stem = ref[:-5] if ref.endswith(".json") else ref
    candidates: list[Path] = []
    if "/" not in ref and "\\" not in ref:
        candidates.append(settings.styles_dir / f"{stem}.json")
        candidates.append(settings.builtin_styles_dir / f"{stem}.json")
    else:
        candidate = Path(ref if ref.endswith(".json") else f"{ref}.json")
        if within_sandbox(candidate, settings):
            candidates.append(candidate)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise NotFound(
        f"style '{ref}' not found. List available styles with GET /v1/styles."
    )


def resolve_voice(ref: str, settings: Settings) -> Path:
    """Map a voice reference onto a WAV file on disk."""
    ref = (ref or "").strip()
    if not ref:
        raise NotFound("voice reference is empty")

    candidates: list[Path] = []
    if "/" not in ref and "\\" not in ref:
        stem = ref
        for suffix in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        for suffix in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
            candidates.append(settings.voices_dir / f"{stem}{suffix}")
    else:
        candidate = Path(ref)
        if within_sandbox(candidate, settings):
            candidates.append(candidate)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise NotFound(
        f"voice '{ref}' not found. Upload one with POST /v1/voices or pass a path "
        "inside the server's data directory."
    )


def list_styles(settings: Settings) -> list[dict[str, Any]]:
    """Trained styles first, then the built-ins shipped with the model."""
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source, directory in (("trained", settings.styles_dir), ("builtin", settings.builtin_styles_dir)):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.stem in seen:
                continue
            seen.add(path.stem)
            stat = path.stat()
            entries.append(
                {
                    "id": path.stem,
                    "source": source,
                    "bytes": stat.st_size,
                    "created_at": stat.st_mtime,
                    "job_id": _style_job_id(path),
                }
            )
    return entries


def _style_job_id(path: Path) -> Optional[str]:
    try:
        with path.open("r") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if isinstance(metadata, dict):
        job_id = metadata.get("job_id")
        return job_id if isinstance(job_id, str) else None
    return None


def list_voices(settings: Settings) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not settings.voices_dir.is_dir():
        return entries
    for path in sorted(settings.voices_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        stat = path.stat()
        entry: dict[str, Any] = {
            "id": path.stem,
            "filename": path.name,
            "bytes": stat.st_size,
            "created_at": stat.st_mtime,
        }
        entry.update(probe_audio(path))
        entries.append(entry)
    return entries


def probe_audio(path: Path) -> dict[str, Any]:
    """Duration/sample rate when soundfile can read the file; empty otherwise."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return {
            "duration_seconds": round(float(info.frames) / info.samplerate, 3),
            "sample_rate": int(info.samplerate),
        }
    except Exception:
        return {}


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """A non-clobbering path: ``stem.wav``, then ``stem-1.wav``, and so on."""
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate


def timestamped_name(stem: str, suffix: str) -> str:
    return f"{stem}_{int(time.time())}{suffix}"
