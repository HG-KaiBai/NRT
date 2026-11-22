#!/usr/bin/env bash
#SBATCH --account=def-anthree
#SBATCH --job-name=tune_regularized_nll
#SBATCH --output=/home/y679zhao/projects/def-anthree/y679zhao/NRT/nrt-29cases-20261113_223129/source/regularization/tuning/log/%x-%A_%a.out
#SBATCH --error=/home/y679zhao/projects/def-anthree/y679zhao/NRT/nrt-29cases-20261113_223129/source/regularization/tuning/log/%x-%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-28%29
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
NRT_ROOT=/home/y679zhao/projects/def-anthree/y679zhao/NRT/nrt-29cases-20261113_223129/source
cd "$NRT_ROOT"
sha256sum --quiet -c "/home/y679zhao/projects/def-anthree/y679zhao/NRT/nrt-29cases-20261113_223129/SOURCE_SHA256SUMS"
source "$NRT_ROOT/runtime.sh"
TASK_ARGS=()
if [[ -n ${SLURM_ARRAY_TASK_ID:-${TUNING_TASK_ID:-}} ]]; then
  TASK_ARGS+=(--task-id "${SLURM_ARRAY_TASK_ID:-${TUNING_TASK_ID}}")
fi
nrt_python regularization.tuning.main \
  --methods "${METHODS:-all}" --cases "${CASES:-all}" \
  --penalty "${PENALTY:-partition_kl}" \
  --output-dir "${OUTDIR:-$NRT_ROOT/regularization/tuning/output}" \
  --n-samples "${N_SAMPLES:-5000}" --seed "${BASE_SEED:-123}" \
  --device "${DEVICE:-cuda:0}" --threads "${SLURM_CPUS_PER_TASK:-4}" \
  --epochs "${EPOCHS:-200}" --trials "${TRIALS:-200}" \
  --initial-trials "${INITIAL_TRIALS:-20}" "${TASK_ARGS[@]}" "$@"
