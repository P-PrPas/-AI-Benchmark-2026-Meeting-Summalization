#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -A zz991011
#SBATCH -J answer_ranker
#SBATCH -o /project/zz991000-zdeva/zz991011/CAMNET_P/logs/slurm-%j.out
#SBATCH -e /project/zz991000-zdeva/zz991011/CAMNET_P/logs/slurm-%j.err

set -euo pipefail

PROJECT_ROOT="${CAMNET_PROJECT_ROOT:-/project/zz991000-zdeva/zz991011/CAMNET_P}"
ANSWER_CANDIDATES_PATH="${CAMNET_ANSWER_CANDIDATES_PATH:-${PROJECT_ROOT}/artifacts/eval_answer_candidates_oracle_v2/answer_candidates.json}"
OUTPUT_PATH="${CAMNET_ANSWER_RANKER_OUTPUT_PATH:-${PROJECT_ROOT}/artifacts/answer_ranker_v1/answer_ranker.pkl}"

mkdir -p "${PROJECT_ROOT}/logs" "$(dirname "${OUTPUT_PATH}")"
cd "${PROJECT_ROOT}"

echo "Job starts at: $(date)"
echo "Running on node: $(hostname)"
echo "Answer candidates path: ${ANSWER_CANDIDATES_PATH}"
echo "Output path: ${OUTPUT_PATH}"

conda run -n three_env python -u -m finetune.train_answer_ranker \
  --project-root "${PROJECT_ROOT}" \
  --answer-candidates-path "${ANSWER_CANDIDATES_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  "$@"

echo "Job finished at: $(date)"
