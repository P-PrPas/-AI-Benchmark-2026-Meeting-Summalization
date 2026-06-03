#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 3:00:00
#SBATCH -A zz991011
#SBATCH -J eval_llm_only
#SBATCH -o /project/zz991000-zdeva/zz991011/CAMNET_P/logs/slurm-%j.out
#SBATCH -e /project/zz991000-zdeva/zz991011/CAMNET_P/logs/slurm-%j.err

set -euo pipefail

REPO_ROOT="/project/zz991000-zdeva/zz991011/CAMNET_P"
MODEL_ROOT="${CAMNET_MODEL_DIR:-/project/zz991000-zdeva/zz991011/models}"
CACHE_ROOT="/project/zz991000-zdeva/zz991011/.cache"
OUTPUT_DIR="${CAMNET_FINETUNE_OUTPUT_DIR:-$REPO_ROOT/artifacts/llm_only_${CAMNET_LLM_ONLY_PROMPT_MODE:-minimal}}"
LLM_ONLY_MODEL_PATH="${CAMNET_LLM_ONLY_MODEL_PATH:-$MODEL_ROOT/Qwen3.5-9B-finetuned-bf16}"
SEMANTIC_MODEL_PATH="${CAMNET_SEMANTIC_MODEL_PATH:-$MODEL_ROOT/bge-m3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-three_env}"

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
export CAMNET_LLM_ONLY_PROMPT_MODE="${CAMNET_LLM_ONLY_PROMPT_MODE:-minimal}"
export CAMNET_LLM_ONLY_MAX_SEQ_LEN="${CAMNET_LLM_ONLY_MAX_SEQ_LEN:-32768}"
export CAMNET_LLM_ONLY_MAX_NEW_TOKENS="${CAMNET_LLM_ONLY_MAX_NEW_TOKENS:-384}"
export CAMNET_LLM_ONLY_BATCH_SIZE="${CAMNET_LLM_ONLY_BATCH_SIZE:-1}"
export CAMNET_LLM_ONLY_MAX_DOC_CHARS="${CAMNET_LLM_ONLY_MAX_DOC_CHARS:-0}"

echo "Job starts at: $(date)"
echo "Running on node: $(hostname)"
echo "LLM-only evaluation"
echo "Model path: $LLM_ONLY_MODEL_PATH"
echo "Semantic model path: $SEMANTIC_MODEL_PATH"
echo "Prompt mode: $CAMNET_LLM_ONLY_PROMPT_MODE"
echo "Output dir: $OUTPUT_DIR"

conda run -n "$CONDA_ENV_NAME" python -u -m finetune.evaluate_llm_only \
  --project-root "$REPO_ROOT" \
  --train-json-path "$REPO_ROOT/data/train/train_set.json" \
  --model-name-or-path "$LLM_ONLY_MODEL_PATH" \
  --semantic-model-name-or-path "$SEMANTIC_MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --cache-dir "$CACHE_ROOT" \
  --prompt-mode "$CAMNET_LLM_ONLY_PROMPT_MODE" \
  --max-seq-len "$CAMNET_LLM_ONLY_MAX_SEQ_LEN" \
  --max-new-tokens "$CAMNET_LLM_ONLY_MAX_NEW_TOKENS" \
  --max-doc-chars "$CAMNET_LLM_ONLY_MAX_DOC_CHARS" \
  --batch-size "$CAMNET_LLM_ONLY_BATCH_SIZE" \
  "$@"

echo "Job finished at: $(date)"
