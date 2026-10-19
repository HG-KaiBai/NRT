#!/usr/bin/env bash
#SBATCH --account=def-anthree
#SBATCH --job-name=support_submit
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
export NRT_ROOT
cd "$NRT_ROOT"
mkdir -p realworld/log
source "$NRT_ROOT/runtime.sh"
export PY
export DATASET=${DATASET:-support}
if [[ "$DATASET" != support && "$DATASET" != metabric ]]; then
  echo 'DATASET must be support or metabric' >&2; exit 2
fi
export OUTDIR=${OUTDIR:-$NRT_ROOT/realworld/output/$DATASET}
export DATA_CSV=${DATA_CSV:-$NRT_ROOT/realworld/data/$DATASET/${DATASET}_deepsurv.csv}
search_args=()
if [[ -n ${NRT_SEARCH_SPACE:-} ]]; then
  export NRT_SEARCH_SPACE
  search_args+=(--nrt-search-space "$NRT_SEARCH_SPACE")
fi
nrt_run "$PY" -m realworld.run preflight --device cpu
nrt_run "$PY" -m realworld.run prepare --dataset "$DATASET" --data "$DATA_CSV" --output-dir "$OUTDIR" "${search_args[@]}"
# Explicit paths also support NRT_ROOT overrides; the log directory exists now.
submit_args=(--parsable --export=ALL --chdir="$NRT_ROOT"
  --output="$NRT_ROOT/realworld/log/%x-%A_%a.out"
  --error="$NRT_ROOT/realworld/log/%x-%A_%a.err")
gpu=$(sbatch "${submit_args[@]}" --job-name="${DATASET}_gpu" "$NRT_ROOT/realworld/run_gpu.sh")
cpu=$(sbatch "${submit_args[@]}" --job-name="${DATASET}_cpu" "$NRT_ROOT/realworld/run_cpu.sh")
gpu=${gpu%%;*}
cpu=${cpu%%;*}
summary=$(sbatch "${submit_args[@]}" --job-name="${DATASET}_summary" --dependency="afterok:$gpu:$cpu" "$NRT_ROOT/realworld/summarize.sh")
summary=${summary%%;*}
printf 'GPU array: %s\nCPU array: %s\nSummary: %s\nCSV: %s/comparison.csv\n' "$gpu" "$cpu" "$summary" "$OUTDIR"
