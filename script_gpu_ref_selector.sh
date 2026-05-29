#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 6:00:00
#SBATCH -A zz991011
#SBATCH -J refsel_train
#SBATCH -o /project/zz991000-zdeva/zz991011/CAMNET_P/logs/slurm-%j.out
#SBATCH -e /project/zz991000-zdeva/zz991011/CAMNET_P/logs/slurm-%j.err

set -euo pipefail

REPO_ROOT="/project/zz991000-zdeva/zz991011/CAMNET_P"
MODEL_ROOT="${CAMNET_MODEL_DIR:-/project/zz991000-zdeva/zz991011/models}"
CACHE_ROOT="/project/zz991000-zdeva/zz991011/.cache"
OUTPUT_DIR="${CAMNET_REF_SELECTOR_OUTPUT_DIR:-$REPO_ROOT/artifacts/ref_selector_v1}"
CONDA_ENV_NAME="three_env"

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
export CAMNET_REF_SELECTOR_OUTPUT_DIR="$OUTPUT_DIR"

echo "Job starts at: $(date)"
echo "Running on node: $(hostname)"
echo "Repo root: $REPO_ROOT"
echo "Output dir: $OUTPUT_DIR"

conda run -n "$CONDA_ENV_NAME" python -u -m finetune.train_ref_selector \
  "$@" \
  --project-root "$REPO_ROOT" \
  --train-json-path "$REPO_ROOT/data/train/train_set.json" \
  --embed-model-name-or-path "$MODEL_ROOT/bge-m3" \
  --rerank-model-name-or-path "$REPO_ROOT/artifacts/reranker_phase_b_v1/final_model" \
  --output-path "$OUTPUT_DIR/ref_selector.pkl" \
  --cache-dir "$CACHE_ROOT" \
  --val-doc-ratio "${CAMNET_VAL_DOC_RATIO:-0.2}" \
  --seed "${CAMNET_SEED:-42}" \
  --retrieval-top-k "${CAMNET_RETRIEVAL_TOP_K:-20}"

echo "Job finished at: $(date)"
