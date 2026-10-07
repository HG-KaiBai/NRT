#!/usr/bin/env bash
#SBATCH --account=def-anthree
#SBATCH --job-name=simulate_cpu
#SBATCH --output=/home/y679zhao/projects/def-anthree/y679zhao/NRT/testing/log/%x-%A_%a.out
#SBATCH --error=/home/y679zhao/projects/def-anthree/y679zhao/NRT/testing/log/%x-%A_%a.err
#SBATCH --time=4-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-500

#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
# Slurm executes a spool copy; source runtime.sh from the shared project.
if [[ -z ${NRT_ROOT:-} ]]; then
  if [[ -n ${SLURM_JOB_ID:-} || -n ${SLURM_SUBMIT_DIR:-} ]]; then
    NRT_ROOT=/home/y679zhao/projects/def-anthree/y679zhao/NRT
  else
    NRT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
  fi
fi
source "$NRT_ROOT/runtime.sh"
if type module >/dev/null 2>&1; then
  module load "${R_MODULE:-r/4.4.0}"
fi
EXTRA_ARGS=()
if [[ -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
  EXTRA_ARGS+=(--replication "$SLURM_ARRAY_TASK_ID")
fi
# One array task per replication: other CPU methods first, then ICcforest alone.
nrt_simulate deep_distribution_regression,ctree,iccforest,cox,aft cpu \
  --parallel-fits "${PARALLEL_FITS:-${SLURM_CPUS_PER_TASK:-8}}" \
  --iccforest-parallel-fits "${ICCFOREST_PARALLEL_FITS:-1}" \
  --threads-per-fit 1 "${EXTRA_ARGS[@]}" "$@"
