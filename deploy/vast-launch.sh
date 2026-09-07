#!/usr/bin/env bash
#
# Launch the batch server on a freshly rented Vast.ai / RunPod box.
#
#   curl -fsSL <raw-url>/deploy/vast-launch.sh | bash -s -- \
#       ghcr.io/ajithberlin/supertonic3-voice-clone:latest
set -euo pipefail

IMAGE="${1:-${IMAGE:-}}"
if [[ -z "${IMAGE}" ]]; then
    echo "usage: $0 ghcr.io/ajithberlin/supertonic3-voice-clone:latest" >&2
    exit 2
fi

NAME="${CONTAINER_NAME:-supertonic-batch}"
PORT="${PORT:-8000}"
DATA_DIR="${HOST_DATA_DIR:-/workspace/supertonic}"
API_KEY="${API_KEY:-}"

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker is not installed on this host" >&2
    exit 1
fi

if ! docker info --format '{{.Runtimes}}' 2>/dev/null | grep -q nvidia; then
    echo "warning: the nvidia container runtime was not detected." >&2
    echo "         the server will start but run on CPU, which is very slow." >&2
fi

if [[ -z "${API_KEY}" ]]; then
    API_KEY="$(openssl rand -hex 24 2>/dev/null || head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    echo ">> generated API_KEY=${API_KEY}"
    echo "   (save it; every /v1 request needs 'X-API-Key: ${API_KEY}')"
fi

mkdir -p "${DATA_DIR}"

echo ">> pulling ${IMAGE}"
docker pull "${IMAGE}"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

echo ">> starting ${NAME}"
docker run -d --name "${NAME}" \
    --gpus all \
    --restart unless-stopped \
    --shm-size 2g \
    -p "${PORT}:8000" \
    -e "API_KEY=${API_KEY}" \
    -e "GENERATE_CONCURRENCY=${GENERATE_CONCURRENCY:-4}" \
    -e "TRAIN_CONCURRENCY=${TRAIN_CONCURRENCY:-1}" \
    -e "ORT_PROVIDER=${ORT_PROVIDER:-auto}" \
    -v "${DATA_DIR}:/data" \
    "${IMAGE}"

echo ">> waiting for the server to report healthy (weights may download on first boot)"
for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo ">> healthy"
        curl -fsS -H "X-API-Key: ${API_KEY}" "http://127.0.0.1:${PORT}/v1/system" || true
        echo
        exit 0
    fi
    sleep 5
done

echo "error: server did not become healthy in 10 minutes; check 'docker logs ${NAME}'" >&2
exit 1
