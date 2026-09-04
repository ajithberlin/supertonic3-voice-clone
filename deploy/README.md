# Running the batch server on rented GPUs

The image is a single self-contained service: HTTP API, job queue, worker pools and
(optionally) the Supertone weights. It needs one NVIDIA GPU and one volume.

```
ghcr.io/<owner>/<repo>:latest             # full image: synthesis + style training
ghcr.io/<owner>/<repo>:latest-inference   # slim image: synthesis only, no torch
```

GHCR packages start out **private**. Either flip the package to public in its GitHub
package settings, or log in on the GPU host first:

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u <owner> --password-stdin
```

## Which GPU

| Card | Arch | VRAM | Notes |
| --- | --- | --- | --- |
| RTX 3090 | Ampere, sm_86 | 24 GB | Works well. Cheapest option on Vast. |
| RTX 4090 | Ada, sm_89 | 24 GB | ~2x the 3090 on training throughput. |
| RTX 5090 | Blackwell, sm_120 | 32 GB | **Needs the cu128 build** — the default image. |
| A100 / L40S | Ampere / Ada | 40-80 GB | Fine; raise `GENERATE_CONCURRENCY`. |

Training peaks around 2.6 GB of VRAM, so the card size mostly buys you parallelism,
not headroom. A 24 GB card comfortably runs `TRAIN_CONCURRENCY=2` alongside a handful
of generate workers.

## RunPod

**Pods → Deploy → Custom container.**

- Container image: `ghcr.io/<owner>/<repo>:latest`
- Container disk: 30 GB (the image with baked weights is ~12 GB)
- Volume: 20 GB mounted at `/data`
- Expose HTTP port: `8000`
- Environment variables: see the table below

`deploy/runpod-template.json` holds the same settings in RunPod's template format —
import it rather than filling the form by hand.

RunPod puts the pod behind `https://<POD_ID>-8000.proxy.runpod.net`, which is public.
**Always set `API_KEY`** on a RunPod deployment.

## Vast.ai

Search for an offer, then use the "Docker image" launch mode:

- Image: `ghcr.io/<owner>/<repo>:latest`
- Docker options: `-p 8000:8000 -e API_KEY=... -v /workspace/supertonic:/data`
- On-start script: leave empty; the entrypoint handles everything.

Or from a shell on any rented box:

```bash
docker run -d --name supertonic --gpus all \
  -p 8000:8000 \
  -e API_KEY="$(openssl rand -hex 24)" \
  -e GENERATE_CONCURRENCY=4 \
  -e TRAIN_CONCURRENCY=1 \
  -v /workspace/supertonic:/data \
  --restart unless-stopped \
  ghcr.io/<owner>/<repo>:latest
```

`deploy/vast-launch.sh` wraps that with sanity checks.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `API_KEY` | *(empty)* | When set, every `/v1/*` route requires `X-API-Key`. `/health` stays open. |
| `PORT` | `8000` | Listen port. |
| `DATA_DIR` | `/data` | Jobs database, uploads, styles, outputs, logs. Mount a volume here. |
| `MODEL_DIR` | `/opt/supertonic3` | Where the Supertone weights live. |
| `GENERATE_CONCURRENCY` | `4` | Parallel synthesis workers. |
| `TRAIN_CONCURRENCY` | `1` | Parallel training subprocesses. Raise only on a big card. |
| `ORT_PROVIDER` | `auto` | `auto`, `cuda`, `cpu`, or `tensorrt`. Falls back to CPU if unavailable. |
| `DEFAULT_TOTAL_STEP` | `6` | Default diffusion steps per request. |
| `MAX_ATTEMPTS` | `2` | Retries before a job is marked failed. |
| `TRAIN_TIMEOUT_SEC` | `21600` | Hard cap on one training run. |
| `MAX_UPLOAD_BYTES` | `209715200` | Voice upload size limit. |
| `PRELOAD_ENGINE` | `true` | Warm the ONNX sessions at boot instead of on first request. |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins. |

## First requests

```bash
BASE=https://<POD_ID>-8000.proxy.runpod.net
KEY=<your API_KEY>

curl -s $BASE/health
curl -s -H "X-API-Key: $KEY" $BASE/v1/system | python3 -m json.tool

# synthesize with a built-in style
curl -s -X POST $BASE/v1/jobs/generate \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"text":"Batch server is live.","style":"F4"}'
```

Then poll `GET /v1/jobs/{id}` and download `GET /v1/jobs/{id}/result`.

## Spot instances

Both providers can reclaim an interruptible instance mid-job. The job database lives
in `/data`, and on boot the server re-queues anything that was `queued` or `running`
when the process died (up to `MAX_ATTEMPTS`). Mount `/data` on a persistent volume and
interruptions cost you a restart, not the queue.
