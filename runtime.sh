#!/usr/bin/env bash
# Common setup for launchers. PY and OUTDIR/HYPER_DIR/TUNING_DIR can be overridden.
NRT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# Load the rebuilt NRT Python/R environment for the default interpreter.
# An explicit PY override keeps control of its own module and library setup.
if [[ -z ${PY:-} || ${PY:-} == /home/y679zhao/envs/nrt-runtime/bin/python \
    || ${PY:-} == /home/y679zhao/envs/nrt-python311/bin/python ]]; then
  source /home/y679zhao/envs/nrt-runtime/modules.sh
  nrt_runtime_modules
  nrt_runtime_paths
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1
cd "$NRT_ROOT"

# Run probes and fits on the allocated node, using the job's visible devices.
nrt_step() {
  if [[ -n ${SLURM_JOB_ID:-} ]]; then
    srun --nodes=1 --ntasks=1 --cpu-bind=cores "$@"
  else
    "$@"
  fi
}

nrt_requeue_current_task() {
  local reason=$1
  local max_requeues=${NRT_MAX_PYTHON_REQUEUES:-4}
  local restart_count=${SLURM_RESTART_COUNT:-0}
  local target

  if [[ -z ${SLURM_JOB_ID:-} ]]; then
    return 1
  fi
  if ! [[ $max_requeues =~ ^[0-9]+$ && $restart_count =~ ^[0-9]+$ ]]; then
    echo "NRT_MAX_PYTHON_REQUEUES and SLURM_RESTART_COUNT must be nonnegative integers" >&2
    return 1
  fi
  if ((restart_count >= max_requeues)); then
    echo "Not requeueing: restart limit $max_requeues reached (reason: $reason)" >&2
    return 1
  fi
  if ! command -v scontrol >/dev/null 2>&1; then
    echo "Cannot requeue the failed task because scontrol is unavailable" >&2
    return 1
  fi
  if [[ -n ${SLURM_ARRAY_JOB_ID:-} && -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
    target="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  else
    target=$SLURM_JOB_ID
  fi
  echo "Requeueing Slurm task $target after startup failure: $reason" >&2
  if scontrol requeue "$target"; then
    echo "Requeue request accepted for $target; stopping this attempt" >&2
    return 0
  fi
  echo "Slurm rejected the requeue request for $target" >&2
  return 1
}

nrt_resolve_python() {
  local interpreter=${PY:-/home/y679zhao/envs/nrt-runtime/bin/python}
  local attempts=${NRT_PYTHON_CHECK_ATTEMPTS:-4}
  local retry_delay=${NRT_PYTHON_CHECK_DELAY_SECONDS:-15}
  local attempt

  if ! [[ $attempts =~ ^[1-9][0-9]*$ && $retry_delay =~ ^[0-9]+$ ]]; then
    echo "NRT_PYTHON_CHECK_ATTEMPTS must be positive and NRT_PYTHON_CHECK_DELAY_SECONDS must be nonnegative" >&2
    return 1
  fi
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if [[ -x "$interpreter" ]] \
        && nrt_step "$interpreter" -c 'import torch, numpy, pandas' >/dev/null 2>&1; then
      printf '%s\n' "$interpreter"
      return 0
    fi
    if ((attempt < attempts)); then
      echo "Python environment check failed (attempt $attempt/$attempts); retrying in ${retry_delay}s: $interpreter" >&2
      sleep "$retry_delay"
    fi
  done
  echo "Python is not executable or lacks torch, numpy, or pandas after $attempts attempts: $interpreter" >&2
  nrt_requeue_current_task "Python environment check failed" || true
  return 1
}

nrt_run() {
  local status
  if nrt_step "$@"; then
    return 0
  else
    status=$?
  fi
  # The Python drivers reserve 75 for node/dependency preflight failures.
  if ((status == 75)); then
    nrt_requeue_current_task "GPU/CPU node preflight failed" || true
  fi
  return "$status"
}

if ! PY=$(nrt_resolve_python); then
  exit 2
fi

nrt_tune() {
  local method=$1
  local default_epochs=200
  local task_id=${SLURM_ARRAY_TASK_ID:-${TUNING_TASK_ID:-}}
  local task_args=()
  if [[ -n "$task_id" ]]; then
    task_args+=(--task-id "$task_id")
  fi
  if [[ "$method" == DeepSurv ]]; then
    default_epochs=1000
    task_args+=(--tuning-evals "${TUNING_EVALS:-50}")
  fi
  shift
  nrt_run "$PY" -u -m tuning.main --method "$method" \
    --output-dir "${OUTDIR:-$NRT_ROOT/tuning/output}" \
    --n-samples "${N_SAMPLES:-5000}" --cases "${CASES:-all}" \
    --device "${DEVICE:-cuda:0}" --threads "${SLURM_CPUS_PER_TASK:-4}" \
    --epochs "${EPOCHS:-$default_epochs}" "${task_args[@]}" "$@"
}

nrt_simulate() {
  local methods=$1
  local device=$2
  shift 2
  nrt_run "$PY" -u -m testing.simulation run --methods "$methods" --device "$device" \
    --hyper-dir "${HYPER_DIR:-$NRT_ROOT/testing/hyper_para}" \
    --tuning-dir "${TUNING_DIR:-$NRT_ROOT/tuning/output}" \
    --output-dir "${OUTDIR:-$NRT_ROOT/testing/output}" \
    --n-samples "${N_SAMPLES:-5000}" --cases "${CASES:-all}" \
    --summary-methods "${SUMMARY_METHODS:-all}" \
    --replications "${REPLICATIONS:-500}" --base-seed "${BASE_SEED:-123}" \
    --grid-size "${GRID_SIZE:-256}" "$@"
}
