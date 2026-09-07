#!/usr/bin/env python3
"""Batch voice cloner for Supertonic: upload audio, queue style training, and download style JSONs locally.

Usage:
    # Clone voices starting from index 12 to 78 (or specify --start and --end)
    python scripts/clone_voices_batch.py \
        --base-url https://9rnrmlgzfwi13a-8000.proxy.runpod.net \
        --start 12 \
        --end 78 \
        --out ./voices/cloned_styles
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


def http_call(
    base_url: str,
    method: str,
    path: str,
    *,
    api_key: str = "",
    payload: Optional[dict] = None,
    raw: bool = False,
    timeout: float = 120.0,
) -> Any:
    url = base_url.rstrip("/") + path
    headers = {"User-Agent": "curl/8.7.1"}
    if api_key:
        headers["X-API-Key"] = api_key
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        err_msg = exc.read().decode(errors="replace")
        raise RuntimeError(f"{method} {path} HTTP {exc.code}: {err_msg}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach {base_url}: {exc.reason}") from exc
    return body if raw else (json.loads(body) if body else None)


def upload_voice(base_url: str, file_path: Path, voice_name: str, api_key: str = "") -> str:
    """Upload reference voice file to /v1/voices using multipart/form-data."""
    boundary = "----SupertonicCloner" + os.urandom(12).hex()
    body = bytearray()

    # Form field: name
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="name"\r\n\r\n'.encode("utf-8"))
    body.extend(f"{voice_name}\r\n".encode("utf-8"))

    # Form field: file
    mime_type = "audio/mpeg" if file_path.suffix.lower() == ".mp3" else "audio/wav"
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'.encode(
            "utf-8"
        )
    )
    body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    headers = {
        "User-Agent": "curl/8.7.1",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    if api_key:
        headers["X-API-Key"] = api_key

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/voices",
        data=body,
        headers=headers,
        method="POST",
    )
    if api_key:
        req.add_header("X-API-Key", api_key)
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        return res["id"]


def sanitize_style_name(index: int, raw_name: str) -> str:
    """Generate a clean, safe, max-60 char style name."""
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_name).strip("_")
    name = f"{index:03d}_{clean}"
    if len(name) > 60:
        name = name[:60].rstrip("_")
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BASE_URL", "https://9rnrmlgzfwi13a-8000.proxy.runpod.net"),
        help="Supertonic batch server URL",
    )
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("voices/voice/voice_samples.json"),
        help="Path to voice_samples.json",
    )
    parser.add_argument(
        "--voices-dir",
        type=Path,
        default=Path("voices/voice"),
        help="Directory containing the voice audio files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("voices/cloned_styles"),
        help="Local directory to store downloaded style JSONs",
    )
    parser.add_argument("--start", type=int, default=12, help="1-based start voice index (default: 12)")
    parser.add_argument("--end", type=int, default=78, help="1-based end voice index (default: 78)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Number of training jobs to keep in-flight concurrently (default: 2)",
    )
    parser.add_argument("--num-steps", type=int, default=3000, help="Training optimization steps")
    parser.add_argument(
        "--poll-interval", type=float, default=10.0, help="Seconds between status checks"
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Verify connection
    print(f">> Connecting to server: {args.base_url}", flush=True)
    try:
        health = http_call(args.base_url, "GET", "/health", api_key=args.api_key)
        print(f"   Server status: {health.get('status')} | Version: {health.get('version')}", flush=True)
        system = http_call(args.base_url, "GET", "/v1/system", api_key=args.api_key)
        train_workers = system.get("workers", {}).get("train", 1)
        print(f"   Server worker pool: train={train_workers} | Client concurrency target={args.concurrency}", flush=True)
    except Exception as exc:
        print(f"!! Failed to reach server at {args.base_url}: {exc}", file=sys.stderr, flush=True)
        return 1

    # 2. Read manifest
    if not args.manifest.is_file():
        print(f"!! Manifest file not found: {args.manifest}", file=sys.stderr, flush=True)
        return 1

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    entries = manifest.get("entries", [])
    print(f">> Loaded {len(entries)} total entries from {args.manifest.name}", flush=True)

    # 3. Filter entries by [start, end]
    start_idx = max(1, args.start)
    end_idx = min(len(entries), args.end)
    selected = [
        (idx, entries[idx - 1]) for idx in range(start_idx, end_idx + 1)
    ]
    print(f">> Selected {len(selected)} voices (from #{start_idx} to #{end_idx}) for cloning", flush=True)

    # 4. Check existing local styles
    to_clone = []
    for idx, entry in selected:
        style_name = sanitize_style_name(idx, entry.get("name") or f"voice_{idx}")
        local_style_file = args.out / f"{style_name}.json"
        if local_style_file.is_file() and local_style_file.stat().st_size > 100:
            print(f"   [SKIP] Voice #{idx} already cloned locally: {local_style_file.name}", flush=True)
        else:
            to_clone.append((idx, entry, style_name, local_style_file))

    if not to_clone:
        print(">> All requested voices are already cloned and saved locally!", flush=True)
        return 0

    print(f">> {len(to_clone)} voices remaining to clone (concurrency = {args.concurrency}).\n", flush=True)

    # 5. Process voices with concurrent in-flight queue
    total_cloned = 0
    total_failed = 0
    pending = list(to_clone)
    # job_id -> metadata dict
    in_flight: dict[str, dict[str, Any]] = {}

    while pending or in_flight:
        # Fill in-flight jobs up to concurrency limit
        while pending and len(in_flight) < max(1, args.concurrency):
            idx, entry, style_name, local_file = pending.pop(0)
            rel_file = entry["file"]
            audio_path = args.voices_dir / rel_file
            if not audio_path.is_file():
                print(f"!! [ERROR] Audio file not found: {audio_path}, skipping #{idx}", flush=True)
                total_failed += 1
                continue

            gender_str = (entry.get("gender") or "female").lower()
            gender = "M" if gender_str in ("male", "m") else "F"

            print(f"\n>> Queuing Voice #{idx}: {style_name} (Gender: {gender})", flush=True)
            try:
                voice_id = upload_voice(args.base_url, audio_path, style_name, api_key=args.api_key)
                train_payload = {
                    "name": style_name,
                    "voice": voice_id,
                    "gender": gender,
                    "num_steps": args.num_steps,
                    "learning_rate": 2e-4,
                    "vocoder_steps": 6,
                    "save_steps": 500,
                    "early_stop_loss_threshold": 0.015,
                }
                sub_res = http_call(
                    args.base_url, "POST", "/v1/jobs/train", api_key=args.api_key, payload=train_payload
                )
                job_id = sub_res["job"]["id"]
                in_flight[job_id] = {
                    "idx": idx,
                    "style_name": style_name,
                    "local_file": local_file,
                    "last_msg": "",
                    "started": time.time(),
                }
                print(f"   [DISPATCHED] Job {job_id} | In-flight active: {len(in_flight)}/{args.concurrency}", flush=True)
            except Exception as exc:
                print(f"!! Failed to upload/dispatch voice #{idx}: {exc}", file=sys.stderr, flush=True)
                total_failed += 1

        if not in_flight:
            break

        time.sleep(args.poll_interval)

        for job_id in list(in_flight.keys()):
            info = in_flight[job_id]
            try:
                job_info = http_call(
                    args.base_url, "GET", f"/v1/jobs/{job_id}", api_key=args.api_key
                )
            except Exception as exc:
                print(f"   [Warning] Polling error for job {job_id}: {exc}", flush=True)
                continue

            status = job_info.get("status")
            progress = job_info.get("progress", 0.0)
            msg = job_info.get("progress_message") or ""
            elapsed = time.time() - info["started"]

            if msg != info["last_msg"]:
                print(f"   [{status.upper()}] Voice #{info['idx']} ({info['style_name']}): {progress*100:4.1f}% | {msg} ({elapsed:.0f}s elapsed)", flush=True)
                info["last_msg"] = msg

            if status == "succeeded":
                result = job_info.get("result", {})
                best_loss = result.get("best_loss")
                run_sec = job_info.get("run_seconds", elapsed)
                print(f"\n🎉 Voice #{info['idx']} ({info['style_name']}) SUCCEEDED in {run_sec:.1f}s! Best loss: {best_loss}", flush=True)
                try:
                    style_bytes = http_call(
                        args.base_url, "GET", f"/v1/jobs/{job_id}/result", api_key=args.api_key, raw=True
                    )
                    info["local_file"].write_bytes(style_bytes)
                    print(f"💾 Saved locally: {info['local_file'].name} ({len(style_bytes)} bytes)\n", flush=True)
                    total_cloned += 1
                except Exception as exc:
                    print(f"!! Failed to download style JSON for #{info['idx']}: {exc}", file=sys.stderr, flush=True)
                    total_failed += 1
                del in_flight[job_id]

            elif status in ("failed", "canceled"):
                print(f"\n❌ Voice #{info['idx']} ({info['style_name']}) {status.upper()}: {job_info.get('error')}\n", file=sys.stderr, flush=True)
                del in_flight[job_id]
                total_failed += 1

    print("\n=======================================================")
    print(f"Batch cloning summary:")
    print(f"  Successfully cloned & saved locally: {total_cloned}")
    print(f"  Failed: {total_failed}")
    print(f"  Destination directory: {args.out.resolve()}")
    print("=======================================================")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
