#!/usr/bin/env bash
#SBATCH --account=def-anthree
#SBATCH --job-name=simulate_node_mean_kl
#SBATCH --output=/home/y679zhao/links/projects/def-anthree/y679zhao/NRT/regularization/testing/log/%x-%A_%a.out
#SBATCH --error=/home/y679zhao/links/projects/def-anthree/y679zhao/NRT/regularization/testing/log/%x-%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-2%2
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
if [[ -z ${NRT_ROOT:-} ]]; then
  if [[ -n ${SLURM_JOB_ID:-} ]]; then
    NRT_ROOT=/home/y679zhao/links/projects/def-anthree/y679zhao/NRT
  else
    NRT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
  fi
fi
export NRT_ROOT
export METHODS=adaptive_nll_cv,fixed_at_adaptive_architecture
export SUMMARY_METHODS=$METHODS
export HYPER_DIR=${HYPER_DIR:-$NRT_ROOT/regularization/tuning/output_node_mean_kl/hyper_para}
export OUTDIR=${OUTDIR:-$NRT_ROOT/regularization/testing/output_node_mean_kl}
exec bash "$NRT_ROOT/regularization/testing/run_nll.sh" "$@"
