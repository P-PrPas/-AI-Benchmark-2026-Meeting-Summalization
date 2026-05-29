#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -A zz991011
#SBATCH -J finetune_model
#SBATCH -o /project/zz991000-zdeva/zz991011/CAMNET_P/logs/slurm-%j.out
#SBATCH -e /project/zz991000-zdeva/zz991011/CAMNET_P/logs/slurm-%j.err

set -euo pipefail

REPO_ROOT="/project/zz991000-zdeva/zz991011/CAMNET_P"
MODEL_ROOT="${CAMNET_MODEL_DIR:-/project/zz991000-zdeva/zz991011/models}"
CACHE_ROOT="/project/zz991000-zdeva/zz991011/.cache"
OUTPUT_DIR="${CAMNET_FINETUNE_OUTPUT_DIR:-$REPO_ROOT/artifacts/typhoon25_qwen3_4b_rag_qa_qlora}"
CONDA_ENV_NAME="three_env"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
NUM_LOGGING_STEPS="${NUM_LOGGING_STEPS:-10}"

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
export CAMNET_FINETUNE_OUTPUT_DIR="$OUTPUT_DIR"
export CAMNET_ENABLE_DYNAMIC_REF_SELECTION="${CAMNET_ENABLE_DYNAMIC_REF_SELECTION:-1}"
export CAMNET_ENABLE_LEARNED_REF_SELECTOR="${CAMNET_ENABLE_LEARNED_REF_SELECTOR:-0}"
export CAMNET_ENABLE_LLM_REF_ARBITER="${CAMNET_ENABLE_LLM_REF_ARBITER:-0}"
export CAMNET_ENABLE_QUERY_REFINEMENT="${CAMNET_ENABLE_QUERY_REFINEMENT:-0}"
export CAMNET_ENABLE_EVIDENCE_COMPRESSION="${CAMNET_ENABLE_EVIDENCE_COMPRESSION:-0}"
export CAMNET_ENABLE_FACT_ANSWER_REWRITE="${CAMNET_ENABLE_FACT_ANSWER_REWRITE:-0}"
export CAMNET_REF_SELECTION_TOP2_MIN="${CAMNET_REF_SELECTION_TOP2_MIN:-0.35}"
export CAMNET_REF_SELECTION_TOP3_MIN="${CAMNET_REF_SELECTION_TOP3_MIN:-0.30}"
export CAMNET_REF_SELECTION_FACT_MAX_GAP="${CAMNET_REF_SELECTION_FACT_MAX_GAP:-0.08}"
export CAMNET_REF_SELECTION_AGG_MAX_GAP="${CAMNET_REF_SELECTION_AGG_MAX_GAP:-0.12}"

echo "Job starts at: $(date)"
echo "Running on node: $(hostname)"
echo "Repo root: $REPO_ROOT"

conda run -n "$CONDA_ENV_NAME" python -u -m finetune.train \
  "$@" \
  --project-root "$REPO_ROOT" \
  --train-json-path "$REPO_ROOT/data/train/train_set.json" \
  --model-name-or-path "$MODEL_ROOT/typhoon2.5-qwen3-4b" \
  --embed-model-name-or-path "$MODEL_ROOT/bge-m3" \
  --output-dir "$OUTPUT_DIR" \
  --cache-dir "$CACHE_ROOT" \
  --num-train-epochs "$NUM_TRAIN_EPOCHS" \
  --logging-steps "$NUM_LOGGING_STEPS"

echo "Job finished at: $(date)"
