#!/usr/bin/env bash
#SBATCH --account=def-anthree
#SBATCH --job-name=compare_nrt_tau_abs_bias
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail

PROJECT_ROOT=/home/y679zhao/projects/def-anthree/y679zhao/NRT
NRT_ROOT="$PROJECT_ROOT/nrt-29cases-20261021_105101/tau_bias_source"
source "$NRT_ROOT/runtime.sh"

RUNS=(
  "$PROJECT_ROOT/nrt-29cases-20261021_105101/partition_kl/testing"
  "$PROJECT_ROOT/nrt-body-tail-20261110_140942/partition_kl/testing"
  "$PROJECT_ROOT/nrt-root-tail-20261111_010058/partition_kl/testing"
)
COMPARISON_ARGS=(
  --run "finite_1p1tau=${RUNS[0]}"
  --run "body_tail=${RUNS[1]}"
  --run "root_tail=${RUNS[2]}"
  --replications 500 --cases all --methods all
)

for output in "${RUNS[@]}"; do
  nrt_python regularization.testing.simulation summarize \
    --output-dir "$output" --methods all --cases all --replications 500 \
    --metric survival_tau_absolute_bias --device cpu --threads "${SLURM_CPUS_PER_TASK:-2}"
  nrt_run "$PY" -m regularization.testing.compare_structures "${COMPARISON_ARGS[@]}" \
    --output-dir "$output/structure_comparison"
done
