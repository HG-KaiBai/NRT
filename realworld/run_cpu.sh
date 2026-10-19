#!/usr/bin/env bash
#SBATCH --account=def-anthree
#SBATCH --chdir=/home/y679zhao/links/projects/def-anthree/y679zhao/NRT
#SBATCH --job-name=support_cpu
#SBATCH --output=/home/y679zhao/links/projects/def-anthree/y679zhao/NRT/realworld/log/%x-%A_%a.out
#SBATCH --error=/home/y679zhao/links/projects/def-anthree/y679zhao/NRT/realworld/log/%x-%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
# Slurm copies batch scripts to its spool; neither BASH_SOURCE nor the
# submission directory identifies the project reliably inside a batch job.
if [[ -z ${NRT_ROOT:-} ]]; then
  if [[ -n ${SLURM_JOB_ID:-} ]]; then
    NRT_ROOT=/home/y679zhao/links/projects/def-anthree/y679zhao/NRT
  else
    NRT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
  fi
fi
if [[ ! -r "$NRT_ROOT/runtime.sh" ]]; then
  echo "NRT runtime not found: $NRT_ROOT/runtime.sh" >&2
  exit 2
fi
source "$NRT_ROOT/runtime.sh"
groups=(Cox Ctree ICcforest)
task=${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}
if ! [[ $task =~ ^[0-2]$ ]]; then echo 'TASK_ID must be 0..2' >&2; exit 2; fi
nrt_run "$PY" -u -m realworld.run run --methods "${groups[$task]}" \
  --data "${DATA_CSV:-$NRT_ROOT/realworld/data/support/support_deepsurv.csv}" \
  --output-dir "${OUTDIR:-$NRT_ROOT/realworld/output/support}" \
  --device cpu --threads "${SLURM_CPUS_PER_TASK:-4}" "$@"
