#!/usr/bin/env bash
#SBATCH --account=def-anthree
#SBATCH --job-name=prepare_cases
#SBATCH --output=/home/y679zhao/projects/def-anthree/y679zhao/NRT/tuning/log/%x-%j.out
#SBATCH --error=/home/y679zhao/projects/def-anthree/y679zhao/NRT/tuning/log/%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
if [[ -z ${NRT_ROOT:-} ]]; then
  if [[ -n ${SLURM_JOB_ID:-} || -n ${SLURM_SUBMIT_DIR:-} ]]; then
    NRT_ROOT=/home/y679zhao/projects/def-anthree/y679zhao/NRT
  else
    NRT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
  fi
fi
source "$NRT_ROOT/runtime.sh"
nrt_python tuning.prepare_cases \
  --output-dir "${OUTDIR:-$NRT_ROOT/tuning/output}" \
  --n-samples "${N_SAMPLES:-5000}" --seed "${BASE_SEED:-123}" \
  --cases "${CASES:-all}" --threads "${SLURM_CPUS_PER_TASK:-1}" "$@"
