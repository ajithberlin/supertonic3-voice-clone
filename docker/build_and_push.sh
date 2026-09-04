#!/usr/bin/env bash
#
# Build the batch-server image and push it to GitHub Container Registry.
#
#   export GHCR_TOKEN=ghp_...                # PAT with write:packages
#   ./docker/build_and_push.sh --push
#   ./docker/build_and_push.sh --push --tag v1.0.0 --inference-only
#
# The image name defaults to ghcr.io/<owner>/<repo> derived from `git remote`.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE=""
TAGS=()
PUSH=false
PLATFORM="linux/amd64"
CUDA_IMAGE="nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04"
TORCH_CHANNEL="cu128"
INSTALL_TRAINING="true"
BAKE_MODEL="true"
NO_CACHE=""
# Single-platform builds land in the local daemon unless --push says otherwise.
LOAD="--load"

usage() {
    sed -n '3,9p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --image NAME          Full image name (default: ghcr.io/<owner>/<repo>)
  --tag TAG             Extra tag; repeatable. Always also tags the short git SHA.
  --push                Push to the registry (otherwise builds locally only)
  --load                Load the built image into the local docker daemon
  --platform PLATFORM   Build platform (default: linux/amd64)
  --cuda-image IMAGE    CUDA base image (default: nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04)
  --torch-channel CH    PyTorch wheel channel: cu128 (Blackwell/5090-safe), cu126, cu121
  --inference-only      Skip torch/speechbrain; TTS only, no style training
  --no-bake-model       Do not bake the Supertone weights into the image
  --no-cache            Build without the layer cache
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --tag) TAGS+=("$2"); shift 2 ;;
        --push) PUSH=true; shift ;;
        --load) LOAD="--load"; shift ;;
        --platform) PLATFORM="$2"; shift 2 ;;
        --cuda-image) CUDA_IMAGE="$2"; shift 2 ;;
        --torch-channel) TORCH_CHANNEL="$2"; shift 2 ;;
        --inference-only) INSTALL_TRAINING="false"; shift ;;
        --no-bake-model) BAKE_MODEL="false"; shift ;;
        --no-cache) NO_CACHE="--no-cache"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

# ---------------------------------------------------------------- image name
if [[ -z "${IMAGE}" ]]; then
    REMOTE="$(git config --get remote.origin.url || true)"
    if [[ -z "${REMOTE}" ]]; then
        echo "error: no git remote and no --image given" >&2
        exit 2
    fi
    SLUG="$(printf '%s' "${REMOTE}" \
        | sed -E 's#^git@[^:]+:##; s#^https?://[^/]+/##; s#\.git$##')"
    IMAGE="ghcr.io/$(printf '%s' "${SLUG}" | tr '[:upper:]' '[:lower:]')"
fi

SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
DIRTY=""
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    DIRTY="-dirty"
fi

TAGS+=("sha-${SHA}${DIRTY}")
if [[ "${INSTALL_TRAINING}" == "false" ]]; then
    TAGS+=("inference-latest")
elif [[ "${BRANCH}" == "main" || "${BRANCH}" == "master" ]]; then
    TAGS+=("latest")
else
    TAGS+=("$(printf '%s' "${BRANCH}" | tr '/' '-')")
fi

TAG_ARGS=()
for tag in "${TAGS[@]}"; do
    TAG_ARGS+=(--tag "${IMAGE}:${tag}")
done

# -------------------------------------------------------------------- login
if [[ "${PUSH}" == "true" ]]; then
    TOKEN="${GHCR_TOKEN:-${GITHUB_TOKEN:-${CR_PAT:-}}}"
    OWNER="${GHCR_USER:-$(printf '%s' "${IMAGE}" | cut -d/ -f2)}"
    if [[ -n "${TOKEN}" ]]; then
        echo ">> logging in to ghcr.io as ${OWNER}"
        printf '%s' "${TOKEN}" | docker login ghcr.io -u "${OWNER}" --password-stdin
    else
        echo ">> GHCR_TOKEN not set; assuming 'docker login ghcr.io' was already run" >&2
    fi
fi

# -------------------------------------------------------------------- build
if ! docker buildx inspect supertonic-builder >/dev/null 2>&1; then
    docker buildx create --name supertonic-builder --use >/dev/null
else
    docker buildx use supertonic-builder
fi

echo ">> building ${IMAGE}"
printf '   tags:       %s\n' "${TAGS[*]}"
printf '   platform:   %s\n' "${PLATFORM}"
printf '   cuda base:  %s\n' "${CUDA_IMAGE}"
printf '   torch:      %s (training=%s)\n' "${TORCH_CHANNEL}" "${INSTALL_TRAINING}"
printf '   bake model: %s\n' "${BAKE_MODEL}"

OUTPUT="${LOAD}"
if [[ "${PUSH}" == "true" ]]; then
    OUTPUT="--push"
fi

# shellcheck disable=SC2086
docker buildx build \
    --file docker/Dockerfile \
    --platform "${PLATFORM}" \
    --build-arg "CUDA_IMAGE=${CUDA_IMAGE}" \
    --build-arg "TORCH_CHANNEL=${TORCH_CHANNEL}" \
    --build-arg "INSTALL_TRAINING=${INSTALL_TRAINING}" \
    --build-arg "BAKE_MODEL=${BAKE_MODEL}" \
    --label "org.opencontainers.image.revision=${SHA}" \
    --label "org.opencontainers.image.version=${TAGS[0]}" \
    "${TAG_ARGS[@]}" \
    ${NO_CACHE} ${OUTPUT} \
    .

echo ">> done"
if [[ "${PUSH}" == "true" ]]; then
    echo "   pulled with: docker pull ${IMAGE}:${TAGS[-1]}"
    echo "   NOTE: GHCR packages are private by default. Make it public under"
    echo "         github.com/users/<owner>/packages -> package settings, or log in"
    echo "         on the GPU host with a read:packages token before pulling."
fi
