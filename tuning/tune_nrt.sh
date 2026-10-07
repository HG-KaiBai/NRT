#!/usr/bin/env bash
#SBATCH --account=def-anthree
#SBATCH --job-name=tune_nrt
#SBATCH --output=/home/y679zhao/projects/def-anthree/y679zhao/NRT/tuning/log/%x-%A_%a.out
#SBATCH --error=/home/y679zhao/projects/def-anthree/y679zhao/NRT/tuning/log/%x-%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-104%105

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
EXTRA_ARGS=()
if [[ -n ${NRT_TASK_ID:-} && -z ${SLURM_ARRAY_TASK_ID:-${TUNING_TASK_ID:-}} ]]; then
  # Preserve the local loss selector; TUNING_TASK_ID selects one loss/case.
  if ! [[ $NRT_TASK_ID =~ ^[0-2]$ ]]; then
    echo "NRT_TASK_ID must be 0, 1 or 2; use TUNING_TASK_ID for a loss/case task" >&2
    exit 2
  fi
  METHODS=(NRT_NLL NRT_Brier NRT_2Loss)
  EXTRA_ARGS+=(--nrt-methods "${METHODS[$NRT_TASK_ID]}")
fi
# Each task owns one loss/case combination and runs that case's full CV.
nrt_tune NRT "${EXTRA_ARGS[@]}" "$@"
