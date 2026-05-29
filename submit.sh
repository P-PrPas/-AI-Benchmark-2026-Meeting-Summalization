#!/bin/bash

set -e

REGISTRY="registry.ai.in.th"
IMAGE_PATH="2026-textsum/b35d39c9/peerapas.2eii"
USERNAME="peerapas.2eii"
FINAL_LOCAL_NAME="${FINAL_LOCAL_NAME:-camnet-p}"
BASE_LOCAL_IMAGE="${BASE_LOCAL_IMAGE:-camnet-deps-base:latest}"
EMBED_LOCAL_IMAGE="${EMBED_LOCAL_IMAGE:-camnet-weight-embed:latest}"
RERANK_LOCAL_IMAGE="${RERANK_LOCAL_IMAGE:-camnet-weight-rerank:latest}"
LLM_LOCAL_IMAGE="${LLM_LOCAL_IMAGE:-camnet-weight-llm:latest}"

EMBED_MODEL_NAME="${CAMNET_SUBMIT_EMBED_MODEL_NAME:-bge-m3}"
RERANK_MODEL_NAME="${CAMNET_SUBMIT_RERANK_MODEL_NAME:-reranker_phase_b_v1_final_model}"
LLM_MODEL_NAME="${CAMNET_SUBMIT_LLM_MODEL_NAME:-llm_best_run_c5_final_merged}"

if [ -z "$1" ]; then
  echo "Usage: ./submit.sh <USER_TAG>"
  exit 1
fi

USER_TAG="$1"
FULL_IMAGE="${REGISTRY}/${IMAGE_PATH}:${USER_TAG}"

require_dir() {
  if [ ! -d "$1" ]; then
    echo "Missing model directory: $1"
    exit 1
  fi
}

image_exists() {
  docker image inspect "$1" >/dev/null 2>&1
}

should_rebuild() {
  if [ "${CAMNET_REBUILD_ALL_MODELS:-0}" = "1" ]; then
    return 0
  fi
  if [ "${2:-0}" = "1" ]; then
    return 0
  fi
  if image_exists "$1"; then
    return 1
  fi
  return 0
}

echo "========================================"
echo "  CAMNET-P Submission Script"
echo "  Final image : ${FULL_IMAGE}"
echo "  Embed model : ${EMBED_MODEL_NAME}"
echo "  Rerank model: ${RERANK_MODEL_NAME}"
echo "  LLM model   : ${LLM_MODEL_NAME}"
echo "========================================"

require_dir "weight/${EMBED_MODEL_NAME}"
require_dir "weight/${RERANK_MODEL_NAME}"
require_dir "weight/${LLM_MODEL_NAME}"

echo ""
echo "Step 1/4 - Login to registry"
docker login ${REGISTRY} -u ${USERNAME}

echo ""
echo "Step 2/4 - Build reusable images"
if [ "${CAMNET_REBUILD_BASE:-0}" = "1" ] || ! image_exists "${BASE_LOCAL_IMAGE}"; then
  docker build -f Dockerfile.base -t ${BASE_LOCAL_IMAGE} .
else
  echo "Skipping base image build: ${BASE_LOCAL_IMAGE}"
fi

if should_rebuild "${EMBED_LOCAL_IMAGE}" "${CAMNET_REBUILD_EMBED:-0}"; then
  docker build -f Dockerfile.model \
    --build-arg MODEL_NAME=${EMBED_MODEL_NAME} \
    -t ${EMBED_LOCAL_IMAGE} \
    weight/${EMBED_MODEL_NAME}
else
  echo "Skipping embed model build: ${EMBED_LOCAL_IMAGE}"
fi

if should_rebuild "${RERANK_LOCAL_IMAGE}" "${CAMNET_REBUILD_RERANK:-0}"; then
  docker build -f Dockerfile.model \
    --build-arg MODEL_NAME=${RERANK_MODEL_NAME} \
    -t ${RERANK_LOCAL_IMAGE} \
    weight/${RERANK_MODEL_NAME}
else
  echo "Skipping rerank model build: ${RERANK_LOCAL_IMAGE}"
fi

if should_rebuild "${LLM_LOCAL_IMAGE}" "${CAMNET_REBUILD_LLM:-0}"; then
  docker build -f Dockerfile.model \
    --build-arg MODEL_NAME=${LLM_MODEL_NAME} \
    -t ${LLM_LOCAL_IMAGE} \
    weight/${LLM_MODEL_NAME}
else
  echo "Skipping LLM build: ${LLM_LOCAL_IMAGE}"
fi

docker build \
  --build-arg BASE_IMAGE=${BASE_LOCAL_IMAGE} \
  --build-arg EMBED_IMAGE=${EMBED_LOCAL_IMAGE} \
  --build-arg EMBED_MODEL_NAME=${EMBED_MODEL_NAME} \
  --build-arg RERANK_IMAGE=${RERANK_LOCAL_IMAGE} \
  --build-arg RERANK_MODEL_NAME=${RERANK_MODEL_NAME} \
  --build-arg LLM_IMAGE=${LLM_LOCAL_IMAGE} \
  --build-arg CAMNET_LLM_MODEL_NAME=${LLM_MODEL_NAME} \
  -t ${FINAL_LOCAL_NAME} .

echo ""
echo "Step 3/4 - Tag final image"
docker image tag ${FINAL_LOCAL_NAME} ${FULL_IMAGE}

echo ""
echo "Step 4/4 - Push final image"
docker image push ${FULL_IMAGE}

echo ""
echo "Done."
echo "Pushed image: ${FULL_IMAGE}"
