#!/usr/bin/env bash
# Submit a fresh body/tail run from the original, fixed Fir project path.
set -euo pipefail
export NRT_ROOT
NRT_ROOT=/home/y679zhao/projects/def-anthree/y679zhao/NRT
cd "$NRT_ROOT"
RUN_DIR=${NRT_RUN_DIR:-$NRT_ROOT/reproduction/nrt-body-tail-$(date +%Y%m%d_%H%M%S)}
if [[ "$RUN_DIR" != /* ]]; then
  echo "NRT_RUN_DIR must be an absolute path" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/log"

TUNING_JOB=$(OUTDIR="$RUN_DIR/tuning" sbatch --parsable --export=ALL \
  --array="${TUNING_ARRAY:-0-28%29}" \
  --output="$RUN_DIR/log/tuning-%A_%a.out" --error="$RUN_DIR/log/tuning-%A_%a.err" \
  "$NRT_ROOT/regularization/tuning/tune_nll.sh" "$@")
TUNING_JOB=${TUNING_JOB%%;*}
printf 'Tuning job: %s\nRun directory: %s\n' "$TUNING_JOB" "$RUN_DIR"

TESTING_JOB=$(HYPER_DIR="$RUN_DIR/tuning/hyper_para" OUTDIR="$RUN_DIR/testing" \
  sbatch --parsable --export=ALL --dependency="afterok:$TUNING_JOB" \
  --array="${TESTING_ARRAY:-0-2%3}" \
  --output="$RUN_DIR/log/testing-%A_%a.out" --error="$RUN_DIR/log/testing-%A_%a.err" \
  "$NRT_ROOT/regularization/testing/run_nll.sh")
printf 'Testing job: %s (after tuning succeeds)\n' "${TESTING_JOB%%;*}"
