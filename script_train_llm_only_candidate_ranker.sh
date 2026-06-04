#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -A zz991011
#SBATCH -J llm_candidate_ranker
#SBATCH -o /project/zz991000-zdeva/zz991011/CAMNET_P/logs/slurm-%j.out
#SBATCH -e /project/zz991000-zdeva/zz991011/CAMNET_P/logs/slurm-%j.err

set -euo pipefail

PROJECT_ROOT="${CAMNET_PROJECT_ROOT:-/project/zz991000-zdeva/zz991011/CAMNET_P}"
MODEL_ROOT="${CAMNET_MODEL_DIR:-/project/zz991000-zdeva/zz991011/models}"
CACHE_ROOT="/project/zz991000-zdeva/zz991011/.cache"
CANDIDATE_PATH="${CAMNET_LLM_ONLY_CANDIDATE_PATH:-${PROJECT_ROOT}/artifacts/llm_only_raw_bestofn_oracle/llm_only_candidates.json}"
OUTPUT_PATH="${CAMNET_LLM_ONLY_CANDIDATE_RANKER_OUTPUT_PATH:-${PROJECT_ROOT}/artifacts/llm_only_candidate_ranker_v1/ranker.pkl}"
SEMANTIC_MODEL_PATH="${CAMNET_SEMANTIC_MODEL_PATH:-${MODEL_ROOT}/bge-m3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-three_env}"

mkdir -p "${PROJECT_ROOT}/logs" "$(dirname "${OUTPUT_PATH}")"
cd "${PROJECT_ROOT}"

ml Mamba

export HF_HOME="$CACHE_ROOT"
export HF_HUB_CACHE="$CACHE_ROOT"
export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Job starts at: $(date)"
echo "Running on node: $(hostname)"
echo "Candidate path: ${CANDIDATE_PATH}"
echo "Output path: ${OUTPUT_PATH}"
echo "Semantic model path: ${SEMANTIC_MODEL_PATH}"

conda run -n "$CONDA_ENV_NAME" python -u -m finetune.train_llm_only_candidate_ranker \
  --project-root "${PROJECT_ROOT}" \
  --candidate-path "${CANDIDATE_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --semantic-model-name-or-path "${SEMANTIC_MODEL_PATH}" \
  --cache-dir "${CACHE_ROOT}" \
  "$@"

echo "Job finished at: $(date)"
