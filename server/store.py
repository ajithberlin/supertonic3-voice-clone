"""SQLite-backed job store.

The database is the durable record of every job; the in-memory dispatch queues in
``server.workers`` are rebuilt from it on startup so a container restart (common on
RunPod/Vast spot instances) resumes rather than loses queued work.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

from .schemas import JobStatus, JobType

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                TEXT PRIMARY KEY,
    type              TEXT NOT NULL,
    status            TEXT NOT NULL,
    batch_id          TEXT,
    priority          INTEGER NOT NULL DEFAULT 0,
    progress          REAL NOT NULL DEFAULT 0.0,
    progress_message  TEXT,
    params            TEXT NOT NULL DEFAULT '{}',
    result            TEXT,
    error             TEXT,
    metadata          TEXT NOT NULL DEFAULT '{}',
    webhook_url       TEXT,
    attempts          INTEGER NOT NULL DEFAULT 0,
    created_at        REAL NOT NULL,
    started_at        REAL,
    finished_at       REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_type    ON jobs(type);
CREATE INDEX IF NOT EXISTS idx_jobs_batch   ON jobs(batch_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);

CREATE TABLE IF NOT EXISTS batches (
    id          TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    webhook_url TEXT
);
"""

_JOB_COLUMNS = (
    "id", "type", "status", "batch_id", "priority", "progress", "progress_message",
    "params", "result", "error", "metadata", "webhook_url", "attempts",
    "created_at", "started_at", "finished_at",
)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def _loads(raw: Optional[str], default: Any) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    job = {k: row[k] for k in _JOB_COLUMNS}
    job["params"] = _loads(job["params"], {})
    job["metadata"] = _loads(job["metadata"], {})
    job["result"] = _loads(job["result"], None)
    started, finished, created = job["started_at"], job["finished_at"], job["created_at"]
    job["queue_seconds"] = round(started - created, 3) if started else None
    job["run_seconds"] = round(finished - started, 3) if (started and finished) else None
    return job


class JobStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ jobs

    def create_job(
        self,
        job_type: JobType,
        params: dict[str, Any],
        *,
        batch_id: Optional[str] = None,
        priority: int = 0,
        metadata: Optional[dict[str, Any]] = None,
        webhook_url: Optional[str] = None,
    ) -> dict[str, Any]:
        job_id = new_id("job")
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, type, status, batch_id, priority, params, metadata,"
                " webhook_url, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    JobType(job_type).value,
                    JobStatus.queued.value,
                    batch_id,
                    priority,
                    json.dumps(params),
                    json.dumps(metadata or {}),
                    webhook_url,
                    now,
                ),
            )
            self._conn.commit()
        return self.get_job(job_id)  # type: ignore[return-value]

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return row_to_dict(row) if row else None

    def list_jobs(
        self,
        *,
        status: Optional[Iterable[str]] = None,
        job_type: Optional[str] = None,
        batch_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        where, args = [], []
        statuses = [str(s) for s in (status or [])]
        if statuses:
            where.append(f"status IN ({','.join('?' * len(statuses))})")
            args.extend(statuses)
        if job_type:
            where.append("type = ?")
            args.append(job_type)
        if batch_id:
            where.append("batch_id = ?")
            args.append(batch_id)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._lock:
            total = self._conn.execute(f"SELECT COUNT(*) AS c FROM jobs {clause}", args).fetchone()["c"]
            rows = self._conn.execute(
                f"SELECT * FROM jobs {clause} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (*args, limit, offset),
            ).fetchall()
        return [row_to_dict(r) for r in rows], total

    def update_job(self, job_id: str, **fields: Any) -> Optional[dict[str, Any]]:
        if not fields:
            return self.get_job(job_id)
        for key in ("params", "result", "metadata"):
            if key in fields and not isinstance(fields[key], (str, type(None))):
                fields[key] = json.dumps(fields[key])
        if "status" in fields:
            fields["status"] = JobStatus(fields["status"]).value
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._conn.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?", (*fields.values(), job_id)
            )
            self._conn.commit()
        return self.get_job(job_id)

    def mark_running(self, job_id: str) -> Optional[dict[str, Any]]:
        """Move queued -> running. Returns None if the job was canceled or is gone."""
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status = ?, started_at = ?, finished_at = NULL, error = NULL,"
                " progress = 0.0, attempts = attempts + 1 WHERE id = ? AND status = ?",
                (JobStatus.running.value, now, job_id, JobStatus.queued.value),
            )
            self._conn.commit()
            changed = cur.rowcount
        return self.get_job(job_id) if changed else None

    def finish_job(
        self,
        job_id: str,
        status: JobStatus,
        *,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        progress: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        fields: dict[str, Any] = {
            "status": status,
            "finished_at": time.time(),
            "error": error,
        }
        if result is not None:
            fields["result"] = result
        if progress is not None:
            fields["progress"] = progress
        elif status == JobStatus.succeeded:
            fields["progress"] = 1.0
        return self.update_job(job_id, **fields)

    def cancel_job(self, job_id: str) -> tuple[Optional[dict[str, Any]], bool]:
        """Cancel a queued job outright. Returns (job, was_running)."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None, False
            status = row["status"]
            if status == JobStatus.queued.value:
                self._conn.execute(
                    "UPDATE jobs SET status = ?, finished_at = ?, error = ? WHERE id = ?",
                    (JobStatus.canceled.value, time.time(), "canceled before start", job_id),
                )
                self._conn.commit()
                return self.get_job(job_id), False
            return row_to_dict(row), status == JobStatus.running.value

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def pending_jobs(self) -> list[dict[str, Any]]:
        """Queued jobs in dispatch order, used to rebuild the in-memory queues."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY priority DESC, created_at ASC",
                (JobStatus.queued.value,),
            ).fetchall()
        return [row_to_dict(r) for r in rows]

    def requeue_interrupted(self, max_attempts: int) -> int:
        """Recover jobs left ``running`` by a crashed or restarted process."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, attempts FROM jobs WHERE status = ?", (JobStatus.running.value,)
            ).fetchall()
            requeued = 0
            for row in rows:
                if row["attempts"] < max_attempts:
                    self._conn.execute(
                        "UPDATE jobs SET status = ?, started_at = NULL, progress = 0.0,"
                        " progress_message = ? WHERE id = ?",
                        (JobStatus.queued.value, "requeued after server restart", row["id"]),
                    )
                    requeued += 1
                else:
                    self._conn.execute(
                        "UPDATE jobs SET status = ?, finished_at = ?, error = ? WHERE id = ?",
                        (
                            JobStatus.failed.value,
                            time.time(),
                            "interrupted by server restart and out of retries",
                            row["id"],
                        ),
                    )
            self._conn.commit()
        return requeued

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS c FROM jobs GROUP BY status"
            ).fetchall()
        return {r["status"]: r["c"] for r in rows}

    def queue_depth(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT type, COUNT(*) AS c FROM jobs WHERE status = ? GROUP BY type",
                (JobStatus.queued.value,),
            ).fetchall()
        depth = {t.value: 0 for t in JobType}
        depth.update({r["type"]: r["c"] for r in rows})
        return depth

    def purge_before(self, cutoff: float) -> list[dict[str, Any]]:
        """Delete terminal jobs finished before ``cutoff``; returns what was removed."""
        terminal = (JobStatus.succeeded.value, JobStatus.failed.value, JobStatus.canceled.value)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM jobs WHERE status IN ({','.join('?' * len(terminal))})"
                " AND finished_at IS NOT NULL AND finished_at < ?",
                (*terminal, cutoff),
            ).fetchall()
            jobs = [row_to_dict(r) for r in rows]
            for job in jobs:
                self._conn.execute("DELETE FROM jobs WHERE id = ?", (job["id"],))
            self._conn.commit()
        return jobs

    # --------------------------------------------------------------- batches

    def create_batch(
        self, metadata: Optional[dict[str, Any]] = None, webhook_url: Optional[str] = None
    ) -> str:
        batch_id = new_id("batch")
        with self._lock:
            self._conn.execute(
                "INSERT INTO batches (id, created_at, metadata, webhook_url) VALUES (?,?,?,?)",
                (batch_id, time.time(), json.dumps(metadata or {}), webhook_url),
            )
            self._conn.commit()
        return batch_id

    def get_batch(self, batch_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "metadata": _loads(row["metadata"], {}),
            "webhook_url": row["webhook_url"],
        }
