#!/usr/bin/env bash
#SBATCH --account=def-anthree
#SBATCH --job-name=simulate_nrt
#SBATCH --output=/home/y679zhao/projects/def-anthree/y679zhao/NRT/testing/log/%x-%A_%a.out
#SBATCH --error=/home/y679zhao/projects/def-anthree/y679zhao/NRT/testing/log/%x-%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --array=0-2%3

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
METHODS=(NRT_NLL NRT_Brier NRT_2Loss)
if [[ -n ${SLURM_ARRAY_TASK_ID:-${NRT_TASK_ID:-}} ]]; then
  TASK_ID=${SLURM_ARRAY_TASK_ID:-${NRT_TASK_ID}}
  if ! [[ $TASK_ID =~ ^[0-2]$ ]]; then
    echo "NRT task ID must be 0, 1 or 2" >&2
    exit 2
  fi
  # Each array task owns one GPU and fits its 500 replications sequentially.
  nrt_simulate "${METHODS[$TASK_ID]}" "${DEVICE:-cuda:0}"     --parallel-fits 1 --threads-per-fit "${SLURM_CPUS_PER_TASK:-4}" "$@"
else
  # Local execution needs three visible GPUs; one process per training loss.
  PIDS=()
  for TASK_ID in 0 1 2; do
    nrt_simulate "${METHODS[$TASK_ID]}" "cuda:$TASK_ID"       --parallel-fits 1 --threads-per-fit "${THREADS_PER_FIT:-4}" "$@" &
    PIDS+=("$!")
  done
  STATUS=0
  for PID in "${PIDS[@]}"; do
    wait "$PID" || STATUS=1
  done
  exit "$STATUS"
fi
