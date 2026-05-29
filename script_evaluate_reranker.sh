#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH -A zz991011
#SBATCH -J eval_reranker
#SBATCH -o /project/zz991000-zdeva/zz991011/CAMNET_P/logs/slurm-reranker-eval-%j.out
#SBATCH -e /project/zz991000-zdeva/zz991011/CAMNET_P/logs/slurm-reranker-eval-%j.err

set -euo pipefail

REPO_ROOT="/project/zz991000-zdeva/zz991011/CAMNET_P"
MODEL_ROOT="${CAMNET_MODEL_DIR:-/project/zz991000-zdeva/zz991011/models}"
CACHE_ROOT="/project/zz991000-zdeva/zz991011/.cache"
OUTPUT_DIR="${CAMNET_RERANK_OUTPUT_DIR:-$REPO_ROOT/artifacts/qwen3_reranker_4b_qlora}"
CONDA_ENV_NAME="three_env"
MODEL_PATH="${CAMNET_RERANK_EVAL_MODEL_PATH:-$OUTPUT_DIR/final_model}"

mkdir -p "$REPO_ROOT/logs" "$OUTPUT_DIR"
cd "$REPO_ROOT"

ml Mamba

export HF_HOME="$CACHE_ROOT"
export HF_HUB_CACHE="$CACHE_ROOT"
export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

conda run -n "$CONDA_ENV_NAME" python -u -m finetune.evaluate_reranker \
  "$@" \
  --project-root "$REPO_ROOT" \
  --train-json-path "$REPO_ROOT/data/train/train_set.json" \
  --model-name-or-path "$MODEL_PATH" \
  --embed-model-name-or-path "$MODEL_ROOT/bge-m3" \
  --output-dir "$OUTPUT_DIR" \
  --cache-dir "$CACHE_ROOT"
