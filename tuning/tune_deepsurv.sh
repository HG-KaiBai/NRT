#!/usr/bin/env bash
#SBATCH --account=def-anthree
#SBATCH --job-name=tune_deepsurv
#SBATCH --output=/home/y679zhao/projects/def-anthree/y679zhao/NRT/tuning/log/%x-%A_%a.out
#SBATCH --error=/home/y679zhao/projects/def-anthree/y679zhao/NRT/tuning/log/%x-%A_%a.err
#SBATCH --time=4-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-14%15

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
# One supported exact/right-censored case per task.
nrt_tune DeepSurv "$@"
