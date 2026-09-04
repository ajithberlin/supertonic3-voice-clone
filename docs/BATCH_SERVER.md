# Batch server

An HTTP service around this repo's two workloads — **synthesis** (`generate.py`) and
**voice-style training** (`train_style.py`) — so both can be driven as queued jobs,
many at a time, and polled for status.

- **Two worker pools.** Synthesis workers share one warm set of ONNX sessions and run
  concurrently. Training runs in its own subprocess pool. A 25-minute training job
  never blocks a two-second synthesis.
- **Durable queue.** Jobs live in SQLite under `DATA_DIR`. Restart the container and
  queued work resumes; anything interrupted mid-flight is re-queued.
- **Real status.** Per-job progress (synthesis chunks, training steps and loss),
  timings, log tails, artifact downloads, cancellation, and optional webhooks.

Interactive API docs are served at `/docs` once the server is up.

## Run it

```bash
./setup_infer.sh                                # weights + ONNX Runtime (synthesis only)
./setup.sh                                      # weights + torch stack (adds training)
pip install -r requirements_server.txt

python -m server                                # http://localhost:8000
```

Or in Docker, which is what you want on a rented GPU — see [deploy/README.md](../deploy/README.md):

```bash
docker compose up --build
```

## Quick tour

```bash
BASE=http://localhost:8000

# What am I running on?
curl -s $BASE/v1/system | python3 -m json.tool

# Synthesize one line with a built-in style
JOB=$(curl -s -X POST $BASE/v1/jobs/generate -H 'Content-Type: application/json' \
  -d '{"text":"Hello from the batch server.","style":"F4"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["job"]["id"])')

curl -s $BASE/v1/jobs/$JOB | python3 -m json.tool     # poll until succeeded
curl -s $BASE/v1/jobs/$JOB/result -o hello.wav
```

Many lines at once, with the bundled client:

```bash
python scripts/batch_client.py --base-url $BASE --input lines.txt --style F4 --out ./audio
```

Clone a voice, then use it:

```bash
curl -s -X POST $BASE/v1/voices -F file=@voices/F6.wav -F name=my-voice
curl -s -X POST $BASE/v1/jobs/train -H 'Content-Type: application/json' \
  -d '{"name":"my-voice","voice":"my-voice","num_steps":3000}'
# when it succeeds, result.style_id is usable as the "style" of a generate job
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness. Never requires the API key. |
| `GET` | `/v1/system` | Providers, GPU, worker counts, queue depth, job totals. |
| `GET` | `/v1/languages` | Accepted `lang` codes. |
| `POST` | `/v1/jobs/generate` | Queue one synthesis. `202` + job id. |
| `POST` | `/v1/jobs/train` | Queue one style training run. `202` + job id. |
| `POST` | `/v1/batches` | Queue many jobs of either type at once. |
| `GET` | `/v1/jobs` | List jobs; filter by `status` (repeatable), `type`, `batch_id`. |
| `GET` | `/v1/jobs/{id}` | Status, progress, result, error, log tail. |
| `GET` | `/v1/jobs/{id}/logs` | Full worker log as plain text. |
| `GET` | `/v1/jobs/{id}/result` | The WAV (generate) or style JSON (train). |
| `POST` | `/v1/jobs/{id}/cancel` | Cancel a queued or running job. |
| `DELETE` | `/v1/jobs/{id}` | Delete the record, its outputs and its log. |
| `GET` | `/v1/batches/{id}` | Aggregate status, counts and mean progress. |
| `POST` | `/v1/batches/{id}/cancel` | Cancel everything still pending in the batch. |
| `POST` | `/v1/voices` | Upload reference audio (multipart). |
| `GET` `DELETE` | `/v1/voices[/{id}]` | List / delete uploaded audio. |
| `GET` | `/v1/styles` | Built-in and trained styles. |
| `GET` `DELETE` | `/v1/styles/{id}` | Download / delete a style JSON. |

### Job lifecycle

```
queued ──> running ──> succeeded
   │           │
   │           ├─────> failed      (retried up to MAX_ATTEMPTS first)
   └───────────┴─────> canceled
```

`progress` runs 0.0 → 1.0. For synthesis it tracks text chunks; for training it tracks
optimization steps, with `progress_message` carrying the best loss so far.

### Generate parameters

| Field | Default | Notes |
| --- | --- | --- |
| `text` | *required* | Chunked automatically at sentence boundaries. |
| `style` | *required* | Built-in name, trained style id, or a path inside `DATA_DIR`. |
| `lang` | `en` | See `/v1/languages`. |
| `total_step` | `DEFAULT_TOTAL_STEP` (6) | Diffusion steps. Higher is slower and cleaner. |
| `speed` | `1.05` | Speech rate multiplier. |
| `silence_duration` | `0.3` | Seconds of silence between chunks. |
| `output_format` | `wav` | `wav`, `flac` or `ogg`. |
| `name` | style name | Used for the output filename. |
| `priority` | `0` | Higher runs first. |
| `webhook_url` | — | POSTed the job summary on completion. |

### Train parameters

Mirrors `train_style.py`: `name`, `voice`, `gender`, `reference_style`
(`auto` \| `none` \| a style id), `seed`, `speed`, `num_steps`, `learning_rate`,
`vocoder_steps`, `save_steps`, `early_stop_loss_threshold`.

On success the style is copied to `DATA_DIR/styles/<name>.json` and is immediately
usable as the `style` of a generate job. Intermediate checkpoints stay under
`DATA_DIR/outputs/<job_id>/`.

### Batch submission

```json
{
  "metadata": {"run": "chapter-1"},
  "items": [
    {"type": "generate", "priority": 10,
     "generate": {"text": "First line.", "style": "F4", "name": "line-01"}},
    {"type": "generate",
     "generate": {"text": "Second line.", "style": "my-voice"}},
    {"type": "train",
     "train": {"name": "another-voice", "voice": "vox-2", "num_steps": 2000}}
  ]
}
```

Validation is all-or-nothing: if any item names a style that does not exist, the whole
request is rejected with the offending index and nothing is queued.

## Configuration

Every knob is an environment variable; the full table is in
[deploy/README.md](../deploy/README.md#environment-variables). The ones that matter most:

- `API_KEY` — when set, `/v1/*` requires `X-API-Key` (or `Authorization: Bearer`).
  **Set this on any publicly reachable deployment.**
- `GENERATE_CONCURRENCY` / `TRAIN_CONCURRENCY` — the two pool sizes.
- `ORT_PROVIDER` — `auto` picks CUDA when onnxruntime-gpu can see it, else CPU.
- `DATA_DIR` — mount a volume here; it holds the job database and every artifact.

## Postman

`postman/supertonic-batch.postman_collection.json` covers every endpoint, and
`postman/supertonic-batch.postman_environment.json` holds the variables. Import both,
set `baseUrl` and `apiKey`, and the requests chain themselves: submitting a job saves
`jobId`, submitting a batch saves `batchId`, uploading a voice saves `voiceId`.

## Notes and limits

- The server runs as a **single process**. The queues and the warm ONNX sessions live
  in memory, so do not start it with multiple uvicorn workers — scale with
  `GENERATE_CONCURRENCY`, or run more containers behind a load balancer with separate
  `DATA_DIR`s.
- Style and voice references are resolved inside `DATA_DIR`, `MODEL_DIR` and the repo
  directory. Paths outside those return 404.
- Uploaded audio, trained styles and outputs are kept until deleted. On a spot instance,
  mount `DATA_DIR` on a persistent volume.
