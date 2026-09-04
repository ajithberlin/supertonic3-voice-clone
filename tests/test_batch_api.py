"""End-to-end tests for the batch API.

The ONNX engine is stubbed out so these run without the Supertonic weights: what
is under test is the queue, the parallelism, the status transitions and the
artifact plumbing -- not the model.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import engine as engine_module  # noqa: E402
from server.app import create_app  # noqa: E402
from server.config import Settings  # noqa: E402

STYLE_FIXTURE = {
    "style_ttl": {"data": [[[0.0]]], "dims": [1, 50, 256], "type": "float32"},
    "style_dp": {"data": [[[0.0]]], "dims": [1, 8, 16], "type": "float32"},
    "metadata": {"source_file": "fixture.wav"},
}


class StubEngine:
    """Stand-in for the ONNX engine: writes a marker file and records overlap."""

    def __init__(self, settings, delay=0.25):
        self.settings = settings
        self.providers = ["StubExecutionProvider"]
        self.delay = delay
        self.model_ready = True
        self.loaded = True
        self.loaded_at = time.time()
        self.calls: list[str] = []
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def load(self):
        return self

    def generate(self, *, text, lang, style_path, out_path, total_step, speed,
                 silence_duration=0.3, progress=None, should_cancel=None):
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.calls.append(text)
        try:
            deadline = time.time() + self.delay
            while time.time() < deadline:
                if should_cancel and should_cancel():
                    raise engine_module.JobCanceled("canceled during synthesis")
                time.sleep(0.01)
            if progress:
                progress(1.0, "synthesized chunk 1/1")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"RIFF----WAVEfmt ")
            return {
                "audio_path": str(out_path),
                "filename": out_path.name,
                "sample_rate": 44100,
                "duration_seconds": 1.0,
                "reported_duration_seconds": 1.0,
                "chunks": 1,
                "bytes": out_path.stat().st_size,
                "synthesis_seconds": self.delay,
                "real_time_factor": self.delay,
                "providers": self.providers,
            }
        finally:
            with self._lock:
                self.in_flight -= 1


@pytest.fixture()
def env(tmp_path, monkeypatch):
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "model",
        generate_concurrency=4,
        train_concurrency=1,
        preload_engine=False,
        api_key="",
        max_attempts=1,
    )
    settings.ensure_dirs()
    settings.builtin_styles_dir.mkdir(parents=True, exist_ok=True)
    (settings.builtin_styles_dir / "F4.json").write_text(json.dumps(STYLE_FIXTURE))

    stub = StubEngine(settings)
    monkeypatch.setattr(engine_module, "get_engine", lambda s=None: stub)
    monkeypatch.setattr("server.app.get_engine", lambda s=None: stub)

    app = create_app(settings)
    with TestClient(app) as client:
        yield client, settings, stub


def wait_for(client, job_id, statuses=("succeeded", "failed", "canceled"), timeout=20.0):
    deadline = time.time() + timeout
    body = {}
    while time.time() < deadline:
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["status"] in statuses:
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} stuck in {body.get('status')!r}")


def test_health_and_system(env):
    client, _, _ = env
    health = client.get("/health").json()
    assert health["status"] == "ok"

    system = client.get("/v1/system").json()
    assert system["workers"]["generate"] == 4
    assert system["providers"] == ["StubExecutionProvider"]
    assert set(system["jobs"]) == {"queued", "running", "succeeded", "failed", "canceled"}


def test_generate_job_lifecycle(env):
    client, settings, stub = env
    response = client.post(
        "/v1/jobs/generate",
        json={"text": "Hello from the batch server.", "style": "F4", "name": "greeting"},
    )
    assert response.status_code == 202
    job_id = response.json()["job"]["id"]

    done = wait_for(client, job_id)
    assert done["status"] == "succeeded"
    assert done["progress"] == 1.0
    assert done["result"]["style"] == "F4"
    assert done["run_seconds"] is not None
    assert any("started" in line for line in done["log_tail"])
    assert stub.calls == ["Hello from the batch server."]

    audio = client.get(f"/v1/jobs/{job_id}/result")
    assert audio.status_code == 200
    assert audio.content.startswith(b"RIFF")

    logs = client.get(f"/v1/jobs/{job_id}/logs")
    assert "job succeeded" in logs.text


def test_batch_runs_jobs_in_parallel(env):
    client, _, stub = env
    items = [
        {"type": "generate", "generate": {"text": f"line {i}", "style": "F4"}}
        for i in range(8)
    ]
    response = client.post("/v1/batches", json={"items": items, "metadata": {"run": "test"}})
    assert response.status_code == 202
    batch_id = response.json()["batch_id"]
    assert response.json()["accepted"] == 8

    deadline = time.time() + 30
    while time.time() < deadline:
        batch = client.get(f"/v1/batches/{batch_id}").json()
        if batch["status"] in ("succeeded", "completed_with_failures"):
            break
        time.sleep(0.05)

    assert batch["status"] == "succeeded"
    assert batch["counts"]["succeeded"] == 8
    assert batch["progress"] == 1.0
    # Four generate workers were configured; the stub must have seen overlap.
    assert stub.max_in_flight > 1, f"expected parallel execution, saw {stub.max_in_flight}"


def test_batch_rejects_unknown_style_without_queueing_anything(env):
    client, _, _ = env
    response = client.post(
        "/v1/batches",
        json={
            "items": [
                {"type": "generate", "generate": {"text": "ok", "style": "F4"}},
                {"type": "generate", "generate": {"text": "bad", "style": "does-not-exist"}},
            ]
        },
    )
    assert response.status_code == 404
    assert "items[1]" in response.json()["detail"]
    assert client.get("/v1/jobs").json()["total"] == 0


def test_cancel_running_job(env):
    client, _, stub = env
    stub.delay = 5.0
    job_id = client.post(
        "/v1/jobs/generate", json={"text": "long one", "style": "F4"}
    ).json()["job"]["id"]

    deadline = time.time() + 10
    while time.time() < deadline:
        if client.get(f"/v1/jobs/{job_id}").json()["status"] == "running":
            break
        time.sleep(0.02)

    assert client.post(f"/v1/jobs/{job_id}/cancel").status_code == 200
    assert wait_for(client, job_id)["status"] == "canceled"


def test_job_filters_and_deletion(env):
    client, settings, _ = env
    job_id = client.post(
        "/v1/jobs/generate", json={"text": "filter me", "style": "F4"}
    ).json()["job"]["id"]
    wait_for(client, job_id)

    listing = client.get("/v1/jobs", params={"status": "succeeded", "type": "generate"}).json()
    assert listing["total"] == 1
    assert listing["jobs"][0]["id"] == job_id

    assert client.get("/v1/jobs", params={"status": "failed"}).json()["total"] == 0

    assert client.delete(f"/v1/jobs/{job_id}").status_code == 204
    assert client.get(f"/v1/jobs/{job_id}").status_code == 404
    assert not (settings.outputs_dir / job_id).exists()


def test_voice_upload_and_style_listing(env):
    client, settings, _ = env
    wav = b"RIFF" + b"\x00" * 40
    response = client.post(
        "/v1/voices",
        files={"file": ("My Voice.wav", wav, "audio/wav")},
        data={"name": "target one"},
    )
    assert response.status_code == 201
    voice_id = response.json()["id"]
    assert voice_id == "target_one"
    assert (settings.voices_dir / "target_one.wav").is_file()

    assert [v["id"] for v in client.get("/v1/voices").json()] == ["target_one"]

    styles = client.get("/v1/styles").json()
    assert [(s["id"], s["source"]) for s in styles] == [("F4", "builtin")]
    assert client.get("/v1/styles/F4").status_code == 200
    assert client.get("/v1/styles/nope").status_code == 404

    assert client.post(
        "/v1/voices", files={"file": ("x.txt", b"nope", "text/plain")}
    ).status_code == 415

    assert client.delete(f"/v1/voices/{voice_id}").status_code == 204
    assert client.get("/v1/voices").json() == []


def test_path_traversal_is_rejected(env):
    client, _, _ = env
    response = client.post(
        "/v1/jobs/generate", json={"text": "hi", "style": "../../../etc/passwd"}
    )
    assert response.status_code == 404


def test_api_key_enforced(tmp_path, monkeypatch):
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "model",
        api_key="s3cret",
        start_workers=False,
        preload_engine=False,
    )
    settings.ensure_dirs()
    stub = StubEngine(settings)
    monkeypatch.setattr("server.app.get_engine", lambda s=None: stub)
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200  # health stays open for probes
        assert client.get("/v1/system").status_code == 401
        assert client.get("/v1/system", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/v1/system", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert client.get(
            "/v1/system", headers={"Authorization": "Bearer s3cret"}
        ).status_code == 200


def test_queued_jobs_survive_a_restart(tmp_path, monkeypatch):
    """A restarted container must resume queued work rather than drop it."""
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "model",
        start_workers=False,
        preload_engine=False,
        api_key="",
    )
    settings.ensure_dirs()
    settings.builtin_styles_dir.mkdir(parents=True, exist_ok=True)
    (settings.builtin_styles_dir / "F4.json").write_text(json.dumps(STYLE_FIXTURE))

    stub = StubEngine(settings, delay=0.05)
    monkeypatch.setattr("server.app.get_engine", lambda s=None: stub)

    with TestClient(create_app(settings)) as client:  # workers off: nothing drains
        job_id = client.post(
            "/v1/jobs/generate", json={"text": "survive", "style": "F4"}
        ).json()["job"]["id"]
        assert client.get(f"/v1/jobs/{job_id}").json()["status"] == "queued"

    restarted = Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "model",
        generate_concurrency=2,
        preload_engine=False,
        api_key="",
    )
    with TestClient(create_app(restarted)) as client:
        assert wait_for(client, job_id)["status"] == "succeeded"


FAKE_TRAINER = '''
import argparse, json, os, sys, time

parser = argparse.ArgumentParser()
for flag in ("--name", "--gender", "--target-wav-path", "--reference-style", "--seed",
             "--speed", "--num-steps", "--learning-rate", "--vocoder-steps",
             "--save-steps", "--early-stop-loss-threshold", "--model-dir", "--output-dir"):
    parser.add_argument(flag)
args = parser.parse_args()

assert os.path.isfile(args.target_wav_path), "target wav was not passed through"
total = int(args.num_steps)
for step in range(1, total + 1):
    print(f"  Step {step}/{total} | Loss: 0.2000 | LR: 0.0002 | Best: 0.1234")
    sys.stdout.flush()
    time.sleep(float(os.environ.get("FAKE_STEP_SLEEP", "0")))

out_dir = os.path.join(args.output_dir, args.name)
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, args.name + ".json"), "w") as handle:
    json.dump({"style_ttl": {"dims": [1, 50, 256]}, "metadata": {"source_file": args.target_wav_path}}, handle)
print("Done!")
'''

FAILING_TRAINER = 'import sys; print("boom: cuda out of memory"); sys.exit(3)'


@pytest.fixture()
def train_env(tmp_path, monkeypatch):
    script = tmp_path / "fake_train.py"
    script.write_text(FAKE_TRAINER)
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "model",
        train_script=script,
        generate_concurrency=1,
        train_concurrency=1,
        preload_engine=False,
        api_key="",
        max_attempts=1,
    )
    settings.ensure_dirs()
    (settings.voices_dir / "speaker.wav").write_bytes(b"RIFF" + b"\x00" * 40)
    stub = StubEngine(settings)
    monkeypatch.setattr("server.app.get_engine", lambda s=None: stub)
    with TestClient(create_app(settings)) as client:
        yield client, settings, script


def test_train_job_publishes_a_usable_style(train_env):
    client, settings, _ = train_env
    response = client.post(
        "/v1/jobs/train",
        json={"name": "my-voice", "voice": "speaker", "num_steps": 5, "save_steps": 5},
    )
    assert response.status_code == 202
    job_id = response.json()["job"]["id"]

    done = wait_for(client, job_id, timeout=60)
    assert done["status"] == "succeeded", done.get("error")
    assert done["result"]["style_id"] == "my-voice"
    assert done["result"]["best_loss"] == pytest.approx(0.1234)
    assert done["result"]["steps_requested"] == 5

    published = settings.styles_dir / "my-voice.json"
    assert published.is_file()
    assert json.loads(published.read_text())["metadata"]["job_id"] == job_id

    # The freshly trained style is immediately usable by name.
    style_ids = [s["id"] for s in client.get("/v1/styles").json()]
    assert "my-voice" in style_ids
    follow_up = client.post(
        "/v1/jobs/generate", json={"text": "now with my voice", "style": "my-voice"}
    )
    assert follow_up.status_code == 202
    assert wait_for(client, follow_up.json()["job"]["id"])["status"] == "succeeded"


def test_train_job_reports_subprocess_failure(train_env, tmp_path):
    client, settings, script = train_env
    script.write_text(FAILING_TRAINER)
    job_id = client.post(
        "/v1/jobs/train", json={"name": "doomed", "voice": "speaker", "num_steps": 1}
    ).json()["job"]["id"]

    done = wait_for(client, job_id, timeout=60)
    assert done["status"] == "failed"
    assert "exited with code 3" in done["error"]
    assert "cuda out of memory" in done["error"]


def test_train_job_rejects_unknown_voice(train_env):
    client, _, _ = train_env
    job_id = client.post(
        "/v1/jobs/train", json={"name": "nope", "voice": "missing-speaker", "num_steps": 1}
    ).json()["job"]["id"]
    done = wait_for(client, job_id, timeout=60)
    assert done["status"] == "failed"
    assert "not found" in done["error"]


def test_cancel_running_training_kills_the_subprocess(train_env, monkeypatch):
    client, _, _ = train_env
    monkeypatch.setenv("FAKE_STEP_SLEEP", "0.2")
    job_id = client.post(
        "/v1/jobs/train", json={"name": "long-run", "voice": "speaker", "num_steps": 200}
    ).json()["job"]["id"]

    deadline = time.time() + 20
    while time.time() < deadline:
        if client.get(f"/v1/jobs/{job_id}").json()["status"] == "running":
            break
        time.sleep(0.05)

    client.post(f"/v1/jobs/{job_id}/cancel")
    assert wait_for(client, job_id, timeout=60)["status"] == "canceled"
