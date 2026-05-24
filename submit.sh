#!/bin/bash
# ─────────────────────────────────────────────────────────────
# submit.sh  — Build → Tag → Push ไปยัง registry จริง
#
# วิธีใช้:
#   chmod +x submit.sh
#   ./submit.sh <USER_TAG>
#
# ตัวอย่าง:
#   ./submit.sh v1
#   ./submit.sh best
# ─────────────────────────────────────────────────────────────

set -e   # หยุดทันทีถ้า error

# ── Config ──────────────────────────────────────────────────
REGISTRY="registry.ai.in.th"
IMAGE_PATH="2026-textsum/b35d39c9/peerapas.2eii"
LOCAL_NAME="camnet-p"
USERNAME="peerapas.2eii"
SUBMIT_MODEL_NAME="${CAMNET_SUBMIT_MODEL_NAME:-Qwen2.5-7B-Instruct}"

# ── รับ USER_TAG จาก argument ────────────────────────────────
if [ -z "$1" ]; then
  echo "❌  กรุณาระบุ USER_TAG เช่น: ./submit.sh v1"
  exit 1
fi
USER_TAG=$1
FULL_IMAGE="${REGISTRY}/${IMAGE_PATH}:${USER_TAG}"

echo "========================================"
echo "  CAMNET-P Submission Script"
echo "  Image : ${FULL_IMAGE}"
echo "  Model : ${SUBMIT_MODEL_NAME}"
echo "========================================"

# ── Step 1: Login ────────────────────────────────────────────
echo ""
echo "▶ Step 1/4 — Login to registry ..."
docker login ${REGISTRY} -u ${USERNAME}

# ── Step 2: Build ────────────────────────────────────────────
echo ""
echo "▶ Step 2/4 — Build Docker image ..."
if [ ! -d "weight/${SUBMIT_MODEL_NAME}" ]; then
  echo "❌  Missing model directory: weight/${SUBMIT_MODEL_NAME}"
  exit 1
fi
docker build \
  --build-arg CAMNET_LLM_MODEL_NAME=${SUBMIT_MODEL_NAME} \
  -t ${LOCAL_NAME} .

# ── Step 3: Tag ──────────────────────────────────────────────
echo ""
echo "▶ Step 3/4 — Tag image ..."
IMAGE_ID=$(docker images ${LOCAL_NAME} --format "{{.ID}}" | head -1)
echo "   Image ID: ${IMAGE_ID}"
docker image tag ${IMAGE_ID} ${FULL_IMAGE}

# ── Step 4: Push ─────────────────────────────────────────────
echo ""
echo "▶ Step 4/4 — Push to registry ..."
docker image push ${FULL_IMAGE}

echo ""
echo "========================================"
echo "✅  Done!  Image pushed:"
echo "   ${FULL_IMAGE}"
echo ""
echo "   ไปที่ benchmark.ai.in.th แล้วเลือก image:"
echo "   peerapas.2eii:${USER_TAG}"
echo "========================================"

# ── Optional: Logout ─────────────────────────────────────────
# docker logout ${REGISTRY}
