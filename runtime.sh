#!/usr/bin/env bash
# Common setup for launchers. PY and OUTDIR/HYPER_DIR/TUNING_DIR can be overridden.
NRT_ROOT=${NRT_ROOT:-/home/y679zhao/projects/def-anthree/y679zhao/NRT}
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
  echo "Requeueing Slurm task $target: $reason" >&2
  if scontrol requeue "$target"; then
    echo "Requeue request accepted for $target; stopping this attempt" >&2
    return 0
  fi
  echo "Slurm rejected the requeue request for $target" >&2
  return 1
}

nrt_resolve_python() {
  local interpreter=$PY
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
  return 75
}

nrt_run() {
  local status
  if nrt_step "$@"; then
    return 0
  else
    status=$?
  fi
  # Drivers and cuda_retry reserve 75 for node/dependency/availability failures.
  if ((status == 75)); then
    nrt_requeue_current_task "GPU/CPU node or CUDA availability check failed" || true
  fi
  return "$status"
}

# Shared CPU/GPU precheck and CUDA availability handling for every launcher.
nrt_python() {
  nrt_run "$PY" -u -m node_runner "$@"
}

# Define requeue handling before loading modules, so a broken node's module
# environment can fail with the same retry status as the execution probes.
if [[ -z ${PY:-} ]]; then
  if type module >/dev/null 2>&1; then
    if ! module load "${NRT_PYTHON_MODULE:-python/3.14.2}"; then
      nrt_requeue_current_task "Python module load failed" || true
      exit 75
    fi
  fi
  PY="${NRT_VENV:-$HOME/envs/pt}/bin/python3.14"
fi
if type module >/dev/null 2>&1; then
  if ! module load "${R_MODULE:-r/4.4.0}"; then
    nrt_requeue_current_task "R module load failed" || true
    exit 75
  fi
fi
export R_LIBS_USER=${R_LIBS_USER:-$HOME/R/x86_64-pc-linux-gnu-library/4.4}

if ! PY=$(nrt_resolve_python); then
  exit 75
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
  nrt_python tuning.main --method "$method" \
    --output-dir "${OUTDIR:-$NRT_ROOT/tuning/output}" \
    --n-samples "${N_SAMPLES:-5000}" --seed "${BASE_SEED:-123}" --cases "${CASES:-all}" \
    --device "${DEVICE:-cuda:0}" --threads "${SLURM_CPUS_PER_TASK:-4}" \
    --epochs "${EPOCHS:-$default_epochs}" "${task_args[@]}" "$@"
}

nrt_simulate() {
  local methods=$1
  local device=$2
  shift 2
  nrt_python testing.simulation run --methods "$methods" --device "$device" \
    --hyper-dir "${HYPER_DIR:-${TUNING_DIR:-$NRT_ROOT/tuning/output}/hyper_para}" \
    --tuning-dir "${TUNING_DIR:-$NRT_ROOT/tuning/output}" \
    --output-dir "${OUTDIR:-$NRT_ROOT/testing/output}" \
    --n-samples "${N_SAMPLES:-5000}" --cases "${CASES:-all}" \
    --summary-methods "${SUMMARY_METHODS:-all}" \
    --replications "${REPLICATIONS:-500}" --base-seed "${BASE_SEED:-123}" \
    --grid-size "${GRID_SIZE:-256}" "$@"
}
