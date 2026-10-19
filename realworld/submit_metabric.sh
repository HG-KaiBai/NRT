#!/usr/bin/env bash
#SBATCH --account=def-anthree
#SBATCH --job-name=metabric_submit
#SBATCH --chdir=/home/y679zhao/links/projects/def-anthree/y679zhao/NRT
#SBATCH --output=/home/y679zhao/links/projects/def-anthree/y679zhao/NRT/realworld/log/%x-%j.out
#SBATCH --error=/home/y679zhao/links/projects/def-anthree/y679zhao/NRT/realworld/log/%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
# Submit with sbatch from any directory, or run with bash on the login node.
set -euo pipefail
# Reuse the common submission pipeline, including Slurm-spool-safe root lookup.
export DATASET=metabric
if [[ -z ${NRT_ROOT:-} ]]; then
  if [[ -n ${SLURM_JOB_ID:-} ]]; then
    NRT_ROOT=/home/y679zhao/links/projects/def-anthree/y679zhao/NRT
  else
    NRT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
  fi
fi
exec bash "$NRT_ROOT/realworld/submit_support.sh" "$@"
