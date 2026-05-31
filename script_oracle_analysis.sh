#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gpus=1
#SBATCH -t 4:00:00
#SBATCH -A zz991011
#SBATCH -J oracle_analysis
#SBATCH -o /project/zz991000-zdeva/zz991011/CAMNET_P/logs/slurm-%j.out
#SBATCH -e /project/zz991000-zdeva/zz991011/CAMNET_P/logs/slurm-%j.err

set -euo pipefail

PROJECT_ROOT="${CAMNET_PROJECT_ROOT:-/project/zz991000-zdeva/zz991011/CAMNET_P}"
MODEL_ROOT="${CAMNET_MODEL_DIR:-/project/zz991000-zdeva/zz991011/models}"
CACHE_DIR="${CAMNET_CACHE_DIR:-/project/zz991000-zdeva/zz991011/.cache}"
OUTPUT_DIR="${CAMNET_ORACLE_OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/oracle_analysis_v1}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

export CAMNET_USE_RERANKER="${CAMNET_USE_RERANKER:-1}"
export CAMNET_RERANK_MODEL_PATH="${CAMNET_RERANK_MODEL_PATH:-${PROJECT_ROOT}/artifacts/reranker_phase_b_v1/final_model}"
export PYTHONUNBUFFERED=1

echo "Job starts at: $(date)"
echo "Running on node: $(hostname)"
echo "Oracle output dir: ${OUTPUT_DIR}"
echo "Rerank model path: ${CAMNET_RERANK_MODEL_PATH}"
echo "Retrieval top-k: ${CAMNET_RETRIEVAL_TOP_K:-20}"

conda run -n three_env python -u -m finetune.oracle_analysis \
  --project-root "${PROJECT_ROOT}" \
  --train-json-path "${PROJECT_ROOT}/data/train/train_set.json" \
  --embed-model-name-or-path "${MODEL_ROOT}/bge-m3" \
  --rerank-model-name-or-path "${CAMNET_RERANK_MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --retrieval-top-k "${CAMNET_RETRIEVAL_TOP_K:-20}" \
  "$@"

echo "Job finished at: $(date)"
