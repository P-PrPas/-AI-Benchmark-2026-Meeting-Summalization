#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gpus=1
#SBATCH -t 6:00:00
#SBATCH -A zz991011
#SBATCH -J evidence_set
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

PROJECT_ROOT="${CAMNET_PROJECT_ROOT:-/project/zz991000-zdeva/zz991011/CAMNET_P}"
MODEL_ROOT="${CAMNET_MODEL_DIR:-/project/zz991000-zdeva/zz991011/models}"
CACHE_DIR="${CAMNET_CACHE_DIR:-/project/zz991000-zdeva/zz991011/.cache}"
OUTPUT_PATH="${CAMNET_EVIDENCE_SET_OUTPUT_PATH:-${PROJECT_ROOT}/artifacts/evidence_set_v1/evidence_set.pkl}"

cd "${PROJECT_ROOT}"

export CAMNET_USE_RERANKER="${CAMNET_USE_RERANKER:-1}"
export CAMNET_RERANK_MODEL_PATH="${CAMNET_RERANK_MODEL_PATH:-${PROJECT_ROOT}/artifacts/reranker_phase_b_v1/final_model}"

conda run -n three_env python -u -m finetune.train_evidence_set_selector \
  --project-root "${PROJECT_ROOT}" \
  --train-json-path "${PROJECT_ROOT}/data/train/train_set.json" \
  --embed-model-name-or-path "${MODEL_ROOT}/bge-m3" \
  --rerank-model-name-or-path "${CAMNET_RERANK_MODEL_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --cache-dir "${CACHE_DIR}" \
  --retrieval-top-k "${CAMNET_RETRIEVAL_TOP_K:-20}" \
  "$@"
