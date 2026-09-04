"""Worker pools that drain the job queues.

Two independent pools so the two workloads never starve each other:

* ``generate`` workers call the shared warm ONNX engine in-process. Sessions are
  thread-safe, so N workers really do run N syntheses at once.
* ``train`` workers shell out to ``train_style.py`` in a subprocess. Training
  holds a lot of GPU memory and pins a CUDA context, so isolating it means a
  crashed or canceled run frees everything, and cancel is a real SIGTERM.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .config import Settings
from .engine import Engine, JobCanceled
from .paths import NotFound, resolve_style, resolve_voice, unique_path
from .schemas import JobStatus, JobType
from .store import JobStore

log = logging.getLogger("supertonic.workers")

_STEP_RE = re.compile(r"Step\s+(\d+)\s*/\s*(\d+)")
_LOSS_RE = re.compile(r"Loss:\s*([0-9.]+)")
_BEST_RE = re.compile(r"Best:\s*([0-9.]+)")
_PROGRESS_EVERY_SEC = 2.0


class JobLog:
    """Append-only per-job log file with an in-memory tail for the API."""

    def __init__(self, path: Path, tail_lines: int):
        self.path = path
        self.tail_lines = tail_lines
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", errors="replace")

    def write(self, line: str) -> None:
        stamped = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {line.rstrip()}"
        self._handle.write(stamped + "\n")
        self._handle.flush()

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:  # pragma: no cover
            pass


def read_log_tail(path: Path, lines: int) -> list[str]:
    if lines <= 0 or not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return [ln.rstrip("\n") for ln in handle.readlines()[-lines:]]
    except OSError:
        return []


class WorkerPool:
    def __init__(self, settings: Settings, store: JobStore, engine: Engine):
        self.settings = settings
        self.store = store
        self.engine = engine
        self._queues: dict[str, queue.PriorityQueue] = {
            JobType.generate.value: queue.PriorityQueue(),
            JobType.train.value: queue.PriorityQueue(),
        }
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._canceled: set[str] = set()
        self._processes: dict[str, subprocess.Popen] = {}
        self._active: dict[str, str] = {}  # job_id -> worker name
        self._seq = 0

    # ----------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        pools = (
            (JobType.generate.value, max(1, self.settings.generate_concurrency)),
            (JobType.train.value, max(1, self.settings.train_concurrency)),
        )
        for kind, count in pools:
            for index in range(count):
                name = f"{kind}-{index}"
                thread = threading.Thread(target=self._run, args=(kind, name), name=name, daemon=True)
                thread.start()
                self._threads.append(thread)
        log.info(
            "worker pools started: generate=%d train=%d",
            self.settings.generate_concurrency,
            self.settings.train_concurrency,
        )

    def shutdown(self, timeout: float = 10.0) -> None:
        self._stop.set()
        with self._lock:
            processes = list(self._processes.values())
        for process in processes:
            _terminate(process)
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads.clear()

    def restore(self) -> int:
        """Re-enqueue everything the database still considers queued."""
        requeued = self.store.requeue_interrupted(self.settings.max_attempts)
        pending = self.store.pending_jobs()
        for job in pending:
            self.submit(job)
        if pending:
            log.info("restored %d queued job(s) (%d interrupted)", len(pending), requeued)
        return len(pending)

    # ------------------------------------------------------------- queueing

    def submit(self, job: dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            order = self._seq
        # PriorityQueue pops the smallest tuple: negate priority so high runs first.
        self._queues[job["type"]].put((-int(job.get("priority", 0)), order, job["id"]))

    def queue_depth(self) -> dict[str, int]:
        return {kind: q.qsize() for kind, q in self._queues.items()}

    def active_jobs(self) -> dict[str, str]:
        with self._lock:
            return dict(self._active)

    # --------------------------------------------------------- cancellation

    def cancel(self, job_id: str) -> None:
        with self._lock:
            self._canceled.add(job_id)
            process = self._processes.get(job_id)
        if process is not None:
            _terminate(process)

    def is_canceled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._canceled

    def _clear_cancel(self, job_id: str) -> None:
        with self._lock:
            self._canceled.discard(job_id)

    # ------------------------------------------------------------ main loop

    def _run(self, kind: str, worker_name: str) -> None:
        work_queue = self._queues[kind]
        while not self._stop.is_set():
            try:
                _, _, job_id = work_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._execute(job_id, worker_name)
            except Exception:  # pragma: no cover - defensive
                log.exception("worker %s crashed handling %s", worker_name, job_id)
            finally:
                work_queue.task_done()

    def _execute(self, job_id: str, worker_name: str) -> None:
        if self.is_canceled(job_id):
            self.store.finish_job(job_id, JobStatus.canceled, error="canceled before start")
            self._clear_cancel(job_id)
            self._notify(job_id)
            return

        job = self.store.mark_running(job_id)
        if job is None:
            return  # canceled or deleted between enqueue and pickup

        with self._lock:
            self._active[job_id] = worker_name

        job_log = JobLog(self.settings.logs_dir / f"{job_id}.log", self.settings.log_tail_lines)
        job_log.write(f"job {job_id} ({job['type']}) started on {worker_name}")
        try:
            if job["type"] == JobType.generate.value:
                result = self._run_generate(job, job_log)
            else:
                result = self._run_train(job, job_log)
            job_log.write("job succeeded")
            self.store.finish_job(job_id, JobStatus.succeeded, result=result)
        except JobCanceled as exc:
            job_log.write(f"job canceled: {exc}")
            self.store.finish_job(job_id, JobStatus.canceled, error=str(exc))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            job_log.write(f"job failed: {message}")
            log.warning("job %s failed: %s", job_id, message)
            attempts = int(job.get("attempts", 1))
            if attempts < self.settings.max_attempts and not self.is_canceled(job_id):
                job_log.write(f"retrying (attempt {attempts + 1}/{self.settings.max_attempts})")
                retried = self.store.update_job(
                    job_id,
                    status=JobStatus.queued,
                    error=message,
                    progress=0.0,
                    progress_message="queued for retry",
                    started_at=None,
                )
                job_log.close()
                with self._lock:
                    self._active.pop(job_id, None)
                if retried:
                    self.submit(retried)
                return
            self.store.finish_job(job_id, JobStatus.failed, error=message)
        finally:
            job_log.close()
            self._clear_cancel(job_id)
            with self._lock:
                self._active.pop(job_id, None)
                self._processes.pop(job_id, None)
        self._notify(job_id)

    # ------------------------------------------------------------- generate

    def _run_generate(self, job: dict[str, Any], job_log: JobLog) -> dict[str, Any]:
        params = job["params"]
        job_id = job["id"]
        style_path = resolve_style(params["style"], self.settings)
        job_log.write(f"style resolved to {style_path}")

        suffix = "." + params.get("output_format", "wav")
        stem = params.get("name") or style_path.stem
        out_path = unique_path(self.settings.outputs_dir / job_id, _safe_stem(stem), suffix)

        last_update = [0.0]

        def report(fraction: float, message: str) -> None:
            now = time.time()
            if now - last_update[0] < _PROGRESS_EVERY_SEC and fraction < 1.0:
                return
            last_update[0] = now
            job_log.write(message)
            self.store.update_job(job_id, progress=round(fraction, 4), progress_message=message)

        result = self.engine.generate(
            text=params["text"],
            lang=params.get("lang", "en"),
            style_path=style_path,
            out_path=out_path,
            total_step=int(params.get("total_step") or self.settings.default_total_step),
            speed=float(params.get("speed", 1.05)),
            silence_duration=float(params.get("silence_duration", 0.3)),
            progress=report,
            should_cancel=lambda: self.is_canceled(job_id),
        )
        result["style"] = style_path.stem
        result["download_url"] = f"/v1/jobs/{job_id}/result"
        job_log.write(
            f"wrote {result['filename']} "
            f"({result['duration_seconds']}s audio, RTF {result['real_time_factor']})"
        )
        return result

    # ---------------------------------------------------------------- train

    def _run_train(self, job: dict[str, Any], job_log: JobLog) -> dict[str, Any]:
        params = job["params"]
        job_id = job["id"]
        voice_path = resolve_voice(params["voice"], self.settings)
        job_log.write(f"target wav resolved to {voice_path}")

        reference = params.get("reference_style", "auto")
        if reference not in ("auto", "none"):
            reference = str(resolve_style(reference, self.settings))
            job_log.write(f"reference style resolved to {reference}")

        work_dir = self.settings.outputs_dir / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        num_steps = int(params.get("num_steps", 3000))

        argv = [
            sys.executable,
            "-u",
            str(self.settings.train_script),
            "--name", params["name"],
            "--gender", params.get("gender", "F"),
            "--target-wav-path", str(voice_path),
            "--reference-style", reference,
            "--seed", str(params.get("seed", 49)),
            "--speed", str(params.get("speed", 1.05)),
            "--num-steps", str(num_steps),
            "--learning-rate", str(params.get("learning_rate", 2e-4)),
            "--vocoder-steps", str(params.get("vocoder_steps", 6)),
            "--save-steps", str(params.get("save_steps", 500)),
            "--early-stop-loss-threshold", str(params.get("early_stop_loss_threshold", 0.015)),
            "--model-dir", str(self.settings.model_dir),
            "--output-dir", str(work_dir),
        ]
        job_log.write("running: " + " ".join(argv))

        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        process = subprocess.Popen(
            argv,
            cwd=str(Path(__file__).resolve().parent.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        with self._lock:
            self._processes[job_id] = process

        deadline = time.time() + self.settings.train_timeout_sec
        last_update = 0.0
        best_loss: Optional[float] = None
        timed_out = False

        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line:
                job_log.write(line)
            step_match = _STEP_RE.search(line)
            best_match = _BEST_RE.search(line) or _LOSS_RE.search(line)
            if best_match:
                try:
                    best_loss = float(best_match.group(1))
                except ValueError:
                    pass
            if step_match and time.time() - last_update >= _PROGRESS_EVERY_SEC:
                last_update = time.time()
                done, total = int(step_match.group(1)), max(int(step_match.group(2)), 1)
                message = f"step {done}/{total}"
                if best_loss is not None:
                    message += f" | best loss {best_loss:.4f}"
                self.store.update_job(
                    job_id,
                    progress=round(min(done / total, 0.99), 4),
                    progress_message=message,
                )
            if time.time() > deadline:
                timed_out = True
                job_log.write(f"exceeded TRAIN_TIMEOUT_SEC ({self.settings.train_timeout_sec}s)")
                _terminate(process)
                break

        returncode = process.wait()
        with self._lock:
            self._processes.pop(job_id, None)

        if self.is_canceled(job_id):
            raise JobCanceled("canceled during training")
        if timed_out:
            raise RuntimeError(f"training timed out after {self.settings.train_timeout_sec}s")
        if returncode != 0:
            tail = read_log_tail(job_log.path, 15)
            raise RuntimeError(
                f"train_style.py exited with code {returncode}. Last output:\n"
                + "\n".join(tail)
            )

        produced = work_dir / params["name"] / f"{params['name']}.json"
        if not produced.is_file():
            raise RuntimeError(f"training finished but {produced} was not written")

        style_path = self.settings.styles_dir / f"{_safe_stem(params['name'])}.json"
        self.settings.styles_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, style_path)
        _stamp_style_metadata(style_path, job_id)
        job_log.write(f"published style {style_path}")

        checkpoints = sorted(p.name for p in (work_dir / params["name"]).glob("*.json"))
        return {
            "style_id": style_path.stem,
            "style_path": str(style_path),
            "best_loss": best_loss,
            "steps_requested": num_steps,
            "checkpoints": checkpoints,
            "download_url": f"/v1/jobs/{job_id}/result",
            "generate_with": {"style": style_path.stem},
        }

    # -------------------------------------------------------------- webhook

    def _notify(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if not job or not job.get("webhook_url"):
            return
        threading.Thread(
            target=self._post_webhook, args=(job["webhook_url"], job), daemon=True
        ).start()

    def _post_webhook(self, url: str, job: dict[str, Any]) -> None:
        payload = {
            "event": f"job.{job['status']}",
            "job": {
                key: job.get(key)
                for key in (
                    "id", "type", "status", "batch_id", "progress", "result",
                    "error", "metadata", "created_at", "started_at", "finished_at",
                )
            },
        }
        body = json.dumps(payload).encode("utf-8")
        try:
            import urllib.request

            request = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(request, timeout=self.settings.webhook_timeout_sec):
                pass
        except Exception as exc:
            log.warning("webhook POST to %s failed: %s", url, exc)


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", value.strip(), flags=re.UNICODE).strip("._")
    return cleaned[:80] or "output"


def _stamp_style_metadata(path: Path, job_id: str) -> None:
    """Record which job produced a style so /v1/styles can link them."""
    try:
        with path.open("r") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload.setdefault("metadata", {})["job_id"] = job_id
            with path.open("w") as handle:
                json.dump(payload, handle)
    except (OSError, ValueError) as exc:  # pragma: no cover
        log.warning("could not stamp job id into %s: %s", path, exc)


def _terminate(process: subprocess.Popen) -> None:
    """SIGTERM the whole process group, then SIGKILL if it clings on."""
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.terminate()
        except Exception:
            return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
