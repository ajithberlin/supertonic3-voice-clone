#!/usr/bin/env python3
"""Submit a batch of lines to the Supertonic batch server and download the audio.

    # plain text: one line per row, blank lines and '#' comments ignored
    python scripts/batch_client.py --base-url http://localhost:8000 \
        --style F4 --input lines.txt --out ./audio

    # or JSONL, where each row is a full generate payload
    echo '{"text": "Hello.", "style": "F4", "name": "hello"}' > lines.jsonl
    python scripts/batch_client.py --input lines.jsonl --out ./audio

Set API_KEY in the environment (or pass --api-key) if the server requires one.
Only the standard library is used, so this runs anywhere Python 3.9+ does.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


def call(
    base_url: str,
    method: str,
    path: str,
    *,
    api_key: str = "",
    payload: Optional[dict] = None,
    raw: bool = False,
) -> Any:
    request = urllib.request.Request(base_url.rstrip("/") + path, method=method)
    if api_key:
        request.add_header("X-API-Key", api_key)
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, data=data, timeout=300) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"{method} {path} failed with HTTP {exc.code}: {exc.read().decode(errors='replace')}"
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach {base_url}: {exc.reason}") from exc
    return body if raw else (json.loads(body) if body else None)


def read_items(path: Path, style: str, lang: str, total_step: Optional[int]) -> list[dict]:
    items: list[dict] = []
    is_jsonl = path.suffix.lower() in (".jsonl", ".ndjson")
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if is_jsonl:
            try:
                payload = json.loads(line)
            except ValueError as exc:
                raise SystemExit(f"{path}:{index}: invalid JSON - {exc}") from exc
            payload.setdefault("style", style)
            payload.setdefault("lang", lang)
        else:
            payload = {"text": line, "style": style, "lang": lang, "name": f"line-{index:04d}"}
        if total_step:
            payload.setdefault("total_step", total_step)
        items.append(payload)
    if not items:
        raise SystemExit(f"{path} contained no usable lines")
    return items


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    parser.add_argument("--input", required=True, type=Path,
                        help="Text file (one line per utterance) or .jsonl of generate payloads")
    parser.add_argument("--style", default="F4", help="Style for rows that do not name one")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--total-step", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("batch_output"))
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--no-download", action="store_true",
                        help="Submit and report status, but do not fetch the audio")
    args = parser.parse_args()

    payloads = read_items(args.input, args.style, args.lang, args.total_step)
    print(f">> submitting {len(payloads)} generate job(s) to {args.base_url}")

    batch = call(
        args.base_url, "POST", "/v1/batches", api_key=args.api_key,
        payload={
            "items": [{"type": "generate", "generate": item} for item in payloads],
            "metadata": {"client": "batch_client.py", "source": str(args.input)},
        },
    )
    batch_id = batch["batch_id"]
    print(f">> batch {batch_id}: {batch['accepted']} job(s) queued")

    deadline = time.time() + args.timeout
    status: dict = {}
    while time.time() < deadline:
        status = call(args.base_url, "GET", f"/v1/batches/{batch_id}", api_key=args.api_key)
        print(
            f"   {status['status']:<24} {int(status['progress'] * 100):3d}%  "
            + " ".join(f"{k}={v}" for k, v in sorted(status["counts"].items())),
            end="\r", flush=True,
        )
        if status["status"] in ("succeeded", "completed_with_failures", "canceled", "empty"):
            break
        time.sleep(args.poll_interval)
    else:
        raise SystemExit(f"\nbatch {batch_id} did not finish within {args.timeout}s")

    print()
    succeeded = [job for job in status["jobs"] if job["status"] == "succeeded"]
    failed = [job for job in status["jobs"] if job["status"] == "failed"]

    if succeeded and not args.no_download:
        args.out.mkdir(parents=True, exist_ok=True)
        for job in succeeded:
            filename = (job.get("result") or {}).get("filename") or f"{job['id']}.wav"
            audio = call(args.base_url, "GET", f"/v1/jobs/{job['id']}/result",
                         api_key=args.api_key, raw=True)
            (args.out / filename).write_bytes(audio)
        print(f">> downloaded {len(succeeded)} file(s) to {args.out}")

    for job in failed:
        print(f"   FAILED {job['id']}: {job['error']}", file=sys.stderr)

    print(f">> done: {len(succeeded)} succeeded, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
