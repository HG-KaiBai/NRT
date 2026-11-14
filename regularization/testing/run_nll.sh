#!/usr/bin/env bash
#SBATCH --account=def-anthree
#SBATCH --job-name=simulate_regularized_nll
#SBATCH --output=/home/y679zhao/projects/def-anthree/y679zhao/NRT/regularization/testing/log/%x-%A_%a.out
#SBATCH --error=/home/y679zhao/projects/def-anthree/y679zhao/NRT/regularization/testing/log/%x-%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-2%3
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
if [[ -z ${NRT_ROOT:-} ]]; then
  if [[ -n ${SLURM_JOB_ID:-} ]]; then
    NRT_ROOT=/home/y679zhao/projects/def-anthree/y679zhao/NRT
  else
    NRT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
  fi
fi
source "$NRT_ROOT/runtime.sh"
METHOD_NAMES=(fixed_nll_cv adaptive_nll_cv fixed_at_adaptive_architecture)
SELECTED_METHODS=${METHODS:-all}
if [[ -n ${SLURM_ARRAY_TASK_ID:-${NLL_TASK_ID:-}} ]]; then
  TASK_ID=${SLURM_ARRAY_TASK_ID:-${NLL_TASK_ID}}
  if ! [[ $TASK_ID =~ ^[0-2]$ ]]; then
    echo "NLL task ID must be 0, 1 or 2" >&2
    exit 2
  fi
  SELECTED_METHODS=${METHOD_NAMES[$TASK_ID]}
fi
nrt_python regularization.testing.simulation "${COMMAND:-run}" \
  --methods "$SELECTED_METHODS" --summary-methods "${SUMMARY_METHODS:-all}" \
  --cases "${CASES:-all}" \
  --hyper-dir "${HYPER_DIR:-$NRT_ROOT/regularization/tuning/output/hyper_para}" \
  --output-dir "${OUTDIR:-$NRT_ROOT/regularization/testing/output}" \
  --n-samples "${N_SAMPLES:-5000}" --base-seed "${BASE_SEED:-123}" \
  --replications "${REPLICATIONS:-500}" --grid-size "${GRID_SIZE:-256}" \
  --device "${DEVICE:-cuda:0}" --threads "${SLURM_CPUS_PER_TASK:-4}" "$@"
