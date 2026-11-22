#!/usr/bin/env bash
#SBATCH --account=def-anthree
#SBATCH --job-name=simulate_cpu
#SBATCH --output=/home/y679zhao/projects/def-anthree/y679zhao/NRT/nrt-29cases-20261113_223129/source/testing/log/%x-%A_%a.out
#SBATCH --error=/home/y679zhao/projects/def-anthree/y679zhao/NRT/nrt-29cases-20261113_223129/source/testing/log/%x-%A_%a.err
#SBATCH --time=4-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-500

#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
# Slurm executes a spool copy; source runtime.sh from the shared project.
if [[ -z ${NRT_ROOT:-} ]]; then
  if [[ -n ${SLURM_JOB_ID:-} || -n ${SLURM_SUBMIT_DIR:-} ]]; then
    NRT_ROOT=/home/y679zhao/projects/def-anthree/y679zhao/NRT/nrt-29cases-20261113_223129/source
  else
    NRT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
  fi
fi
cd "$NRT_ROOT"
sha256sum --quiet -c "/home/y679zhao/projects/def-anthree/y679zhao/NRT/nrt-29cases-20261113_223129/SOURCE_SHA256SUMS"
source "$NRT_ROOT/runtime.sh"
EXTRA_ARGS=()
if [[ -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
  EXTRA_ARGS+=(--replication "$SLURM_ARRAY_TASK_ID")
fi
# One array task per replication: other CPU methods first, then ICcforest alone.
nrt_simulate deep_distribution_regression,ctree,iccforest,cox,aft cpu \
  --parallel-fits "${PARALLEL_FITS:-${SLURM_CPUS_PER_TASK:-8}}" \
  --iccforest-parallel-fits "${ICCFOREST_PARALLEL_FITS:-1}" \
  --threads-per-fit 1 "${EXTRA_ARGS[@]}" "$@"
