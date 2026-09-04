#
# Supertonic voice-clone batch server.
#
# Built for consumer NVIDIA cards on RunPod / Vast.ai. The default CUDA 12.8 +
# torch cu128 combination covers Ampere through Blackwell -- RTX 3090 (sm_86),
# 4090 (sm_89) and 5090 (sm_120) all run on this one image. Older CUDA channels
# do NOT carry sm_120 kernels, so only drop TORCH_CHANNEL below cu128 if you
# know the target is pre-Blackwell.
#
#   docker build -f docker/Dockerfile -t supertonic-batch .
#   docker run --gpus all -p 8000:8000 -v supertonic-data:/data supertonic-batch

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# --------------------------------------------------------------------------- deps
FROM ${CUDA_IMAGE} AS deps

ARG TORCH_CHANNEL=cu128
ARG INSTALL_TRAINING=true
ARG PYTHON=python3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        ca-certificates curl git tini \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ${PYTHON} -m venv ${VIRTUAL_ENV} && pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt requirements_infer.txt requirements_server.txt ./

# Inference stack: ONNX Runtime with the CUDA provider, plus the API server.
RUN pip install -r requirements_infer.txt -r requirements_server.txt \
    && pip uninstall -y onnxruntime \
    && pip install "onnxruntime-gpu>=1.20,<2"

# Training stack (torch + speechbrain). Skip with --build-arg INSTALL_TRAINING=false
# for an inference-only image that is roughly 4 GB smaller.
RUN if [ "${INSTALL_TRAINING}" = "true" ]; then \
        pip install --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}" \
                    --extra-index-url https://pypi.org/simple \
                    -r requirements.txt \
        && pip uninstall -y onnxruntime \
        && pip install "onnxruntime-gpu>=1.20,<2" ; \
    else \
        echo "skipping training dependencies (INSTALL_TRAINING=${INSTALL_TRAINING})" ; \
    fi

RUN pip install "huggingface_hub[cli]>=0.25"

# -------------------------------------------------------------------------- model
# Split out so a code-only change does not re-download ~1 GB of weights.
FROM deps AS model

ARG BAKE_MODEL=true
ARG MODEL_REPO=Supertone/supertonic-3

RUN mkdir -p /opt/supertonic3 && \
    if [ "${BAKE_MODEL}" = "true" ]; then \
        hf download "${MODEL_REPO}" --local-dir /opt/supertonic3 --quiet && \
        echo "baked ${MODEL_REPO}" ; \
    else \
        echo "model not baked; the entrypoint will download it on first boot" ; \
    fi

# ------------------------------------------------------------------------- runtime
FROM deps AS runtime

LABEL org.opencontainers.image.title="supertonic3-voice-clone batch server" \
      org.opencontainers.image.description="Batch voice-style training and TTS over HTTP, CUDA-accelerated." \
      org.opencontainers.image.source="https://github.com/ajithberlin/supertonic3-voice-clone" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV MODEL_DIR=/opt/supertonic3 \
    DATA_DIR=/data \
    HOST=0.0.0.0 \
    PORT=8000 \
    ORT_PROVIDER=auto \
    GENERATE_CONCURRENCY=4 \
    TRAIN_CONCURRENCY=1 \
    HF_HOME=/data/hf \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=model /opt/supertonic3 /opt/supertonic3
COPY . /app
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && mkdir -p /data

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
