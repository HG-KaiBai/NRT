#!/usr/bin/env bash
#SBATCH --account=def-anthree
#SBATCH --chdir=/home/y679zhao/projects/def-anthree/y679zhao/NRT
#SBATCH --job-name=support_summary
#SBATCH --output=/home/y679zhao/projects/def-anthree/y679zhao/NRT/realworld/log/%x-%j.out
#SBATCH --error=/home/y679zhao/projects/def-anthree/y679zhao/NRT/realworld/log/%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
# Slurm copies batch scripts to its spool; neither BASH_SOURCE nor the
# submission directory identifies the project reliably inside a batch job.
if [[ -z ${NRT_ROOT:-} ]]; then
  if [[ -n ${SLURM_JOB_ID:-} ]]; then
    NRT_ROOT=/home/y679zhao/projects/def-anthree/y679zhao/NRT
  else
    NRT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
  fi
fi
if [[ ! -r "$NRT_ROOT/runtime.sh" ]]; then
  echo "NRT runtime not found: $NRT_ROOT/runtime.sh" >&2
  exit 2
fi
source "$NRT_ROOT/runtime.sh"
DATASET=${DATASET:-support}
nrt_python realworld.run summarize --dataset "$DATASET" \
  --output-dir "${OUTDIR:-$NRT_ROOT/realworld/output/$DATASET}" "$@"
