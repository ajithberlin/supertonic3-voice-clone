#!/usr/bin/env bash
# Container entrypoint: make sure the model and data directories exist, then serve.
#
#   serve            (default) run the batch API
#   train ...        run train_style.py directly
#   generate ...     run generate.py directly
#   <anything else>  executed verbatim
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/opt/supertonic3}"
DATA_DIR="${DATA_DIR:-/data}"
MODEL_REPO="${MODEL_REPO:-Supertone/supertonic-3}"

log() { printf '[entrypoint] %s\n' "$*" >&2; }

mkdir -p "${DATA_DIR}"/{voices,styles,outputs,logs}

# Torch ships its own CUDA/cuDNN libraries; putting them after the system ones
# lets onnxruntime-gpu find a cuDNN even on hosts with a leaner CUDA runtime.
if TORCH_LIB_ROOT=$(python -c 'import os,nvidia;print(os.path.dirname(nvidia.__file__))' 2>/dev/null); then
    EXTRA_LIBS=$(find "${TORCH_LIB_ROOT}" -maxdepth 2 -type d -name lib 2>/dev/null | tr '\n' ':')
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${EXTRA_LIBS}"
fi

if [ ! -f "${MODEL_DIR}/onnx/vocoder.onnx" ]; then
    log "model not found in ${MODEL_DIR}; downloading ${MODEL_REPO} (one-time, ~1 GB)"
    hf download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"
    log "model download complete"
else
    log "model present in ${MODEL_DIR}"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | while read -r line; do
        log "gpu: ${line}"
    done
else
    log "no nvidia-smi on PATH; the server will fall back to CPU execution"
fi

cd /app

case "${1:-serve}" in
    serve)
        shift || true
        log "starting batch API on ${HOST:-0.0.0.0}:${PORT:-8000}"
        exec python -m server "$@"
        ;;
    train)
        shift
        exec python train_style.py "$@"
        ;;
    generate)
        shift
        exec python generate.py "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
