#!/usr/bin/env bash
#SBATCH --account=def-anthree
#SBATCH --job-name=realworld
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
NRT_ROOT=${NRT_ROOT:-/home/y679zhao/projects/def-anthree/y679zhao/NRT}
source "$NRT_ROOT/runtime.sh"
MODE=${MODE:-run}
EXTRA=()
if [[ "$MODE" == run && -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
    EXTRA+=(--task-id "$SLURM_ARRAY_TASK_ID")
fi
if [[ "$MODE" == summarize ]]; then
    EXTRA+=(--require-complete)
fi
nrt_python realworld.run "$MODE" \
    --output "${REALWORLD_OUTPUT:?Set REALWORLD_OUTPUT}" \
    --profile "${PROFILE:-full}" --device "${DEVICE:-cpu}" \
    --threads "${SLURM_CPUS_PER_TASK:-1}" "${EXTRA[@]}" "$@"
