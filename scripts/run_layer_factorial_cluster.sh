#!/usr/bin/env bash

set -Eeuo pipefail

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: scripts/run_layer_factorial_cluster.sh MODEL [--dry-run]

Supported MODEL values:
  gpt2-small, gpt2-large, gpt2-xl,
  pythia-70m, pythia-160m, pythia-410m,
  pythia-14b, pythia-28b, pythia-69b.

Required environment:
  RUN_ROOT             Absolute scratch root for checkpoints/results/logs
  HF_HOME              Absolute Hugging Face cache populated by staging
  CUDA_VISIBLE_DEVICES Exactly one physical GPU identifier (real runs)

Optional environment:
  PYTHON_BIN, RSCRIPT_BIN, TUNED_LENS_ROOT, TUNED_LENS_PYTHONPATH
  LAYER_FACTORIAL_CONFIG, FACTORIAL_JOBS, THREADS_PER_JOB
  CHECKPOINT_ROOT, RESULTS_ROOT, LOG_ROOT
  EXPECTED_GIT_COMMIT, ALLOW_DIRTY=1
EOF
}

[[ $# -ge 1 && $# -le 2 ]] || {
  usage >&2
  exit 2
}
MODEL="$1"
DRY_RUN=0
if [[ $# -eq 2 ]]; then
  [[ "$2" == "--dry-run" ]] || {
    usage >&2
    exit 2
  }
  DRY_RUN=1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RSCRIPT_BIN="${RSCRIPT_BIN:-Rscript}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
REGISTRY="$REPOSITORY_ROOT/src/h01_data/layer_factorial_models.py"
CONFIG_TOOL="$REPOSITORY_ROOT/src/h01_data/layer_factorial_config.py"
LAYER_FACTORIAL_CONFIG="${LAYER_FACTORIAL_CONFIG:-$REPOSITORY_ROOT/configs/layer_factorial.json}"

command -v "$PYTHON_BIN" >/dev/null 2>&1 ||
  die "Python executable not found: $PYTHON_BIN"
command -v "$RSCRIPT_BIN" >/dev/null 2>&1 ||
  die "Rscript executable not found: $RSCRIPT_BIN"
command -v git >/dev/null 2>&1 || die "git is required"
command -v flock >/dev/null 2>&1 || die "flock is required"

mapfile -t CONFIG_MODELS < <(
  "$PYTHON_BIN" "$CONFIG_TOOL" \
    --config "$LAYER_FACTORIAL_CONFIG" --list-models
)
printf '%s\n' "${CONFIG_MODELS[@]}" | grep -Fxq "$MODEL" ||
  die "model is not enabled by $LAYER_FACTORIAL_CONFIG: $MODEL"
mapfile -t CONFIG_LENSES < <(
  "$PYTHON_BIN" "$CONFIG_TOOL" \
    --config "$LAYER_FACTORIAL_CONFIG" --get switches.lens_methods
)
USE_TUNED_LENS=0
for lens_method in "${CONFIG_LENSES[@]}"; do
  [[ "$lens_method" == "tuned-lens" ]] && USE_TUNED_LENS=1
done

FACTORIAL_JOBS="${FACTORIAL_JOBS:-$(
  "$PYTHON_BIN" "$CONFIG_TOOL" \
    --config "$LAYER_FACTORIAL_CONFIG" --get runtime.jobs
)}"
THREADS_PER_JOB="${THREADS_PER_JOB:-$(
  "$PYTHON_BIN" "$CONFIG_TOOL" \
    --config "$LAYER_FACTORIAL_CONFIG" --get runtime.threads_per_job
)}"
case "$THREADS_PER_JOB" in
  ''|*[!0-9]*) die "THREADS_PER_JOB must be a positive integer" ;;
esac
(( THREADS_PER_JOB >= 1 )) ||
  die "THREADS_PER_JOB must be a positive integer"
case "$FACTORIAL_JOBS" in
  ''|*[!0-9]*) die "FACTORIAL_JOBS must be an integer from 1 to 4" ;;
esac
(( FACTORIAL_JOBS >= 1 && FACTORIAL_JOBS <= 4 )) ||
  die "FACTORIAL_JOBS must be an integer from 1 to 4"

LENS_ARTIFACT="$("$PYTHON_BIN" "$REGISTRY" --model "$MODEL" --field lens_artifact)" ||
  die "unsupported model: $MODEL"

: "${RUN_ROOT:?RUN_ROOT must be set to an absolute scratch path}"
: "${HF_HOME:?HF_HOME must be set to an absolute cache path}"
[[ "$RUN_ROOT" == /* ]] || die "RUN_ROOT must be absolute"
[[ "$HF_HOME" == /* ]] || die "HF_HOME must be absolute"

TUNED_LENS_ROOT="${TUNED_LENS_ROOT:-$RUN_ROOT/resources/tuned-lens}"
TUNED_LENS_PYTHONPATH="${TUNED_LENS_PYTHONPATH:-$RUN_ROOT/python/tuned-lens-0.2.0}"
TUNED_LENS_PATH="$TUNED_LENS_ROOT/$LENS_ARTIFACT"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RUN_ROOT/checkpoints/layer-factorial/$MODEL}"
RESULTS_ROOT="${RESULTS_ROOT:-$RUN_ROOT/results/layer-factorial/$MODEL}"
LOG_ROOT="${LOG_ROOT:-$RUN_ROOT/logs/layer-factorial/$MODEL}"

TEXT_FILE="$(
  "$PYTHON_BIN" "$CONFIG_TOOL" --config "$LAYER_FACTORIAL_CONFIG" \
    --resolve-path text
)"
SENTENCE_MANIFEST="$(
  "$PYTHON_BIN" "$CONFIG_TOOL" --config "$LAYER_FACTORIAL_CONFIG" \
    --resolve-path sentence_manifest
)"
JOINT_FILE="$(
  "$PYTHON_BIN" "$CONFIG_TOOL" --config "$LAYER_FACTORIAL_CONFIG" \
    --resolve-path joint_template --model "$MODEL"
)"
PAPER_RT_FILE="$(
  "$PYTHON_BIN" "$CONFIG_TOOL" --config "$LAYER_FACTORIAL_CONFIG" \
    --resolve-path paper_rt
)"
FREQUENCY_FILE="$(
  "$PYTHON_BIN" "$CONFIG_TOOL" --config "$LAYER_FACTORIAL_CONFIG" \
    --resolve-path precomputed_frequency
)"

cd "$REPOSITORY_ROOT"
GIT_COMMIT="$(git rev-parse HEAD)"
if [[ -n "${EXPECTED_GIT_COMMIT:-}" &&
      "$GIT_COMMIT" != "$EXPECTED_GIT_COMMIT" ]]; then
  die "checkout is $GIT_COMMIT; expected $EXPECTED_GIT_COMMIT"
fi
if [[ "$ALLOW_DIRTY" != 1 ]] &&
   { ! git diff --quiet || ! git diff --cached --quiet; }; then
  die "tracked files are dirty; commit/stash them or set ALLOW_DIRTY=1"
fi

mkdir -p "$CHECKPOINT_ROOT" "$RESULTS_ROOT" "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/run-$(date -u +%Y%m%dT%H%M%SZ)-$GIT_COMMIT.log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'status=$?; trap - EXIT; printf "Finished UTC: %s (exit %s)\n" "$(date -u +%FT%TZ)" "$status"; exit "$status"' EXIT

printf 'Layer-factorial cluster run\n'
printf 'Started UTC: %s\n' "$(date -u +%FT%TZ)"
printf 'Repository: %s\nCommit: %s\nModel: %s\n' "$REPOSITORY_ROOT" "$GIT_COMMIT" "$MODEL"
printf 'Configuration: %s\n' "$LAYER_FACTORIAL_CONFIG"
printf 'Checkpoint root: %s\nResults root: %s\n' "$CHECKPOINT_ROOT" "$RESULTS_ROOT"

export HF_HOME
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$THREADS_PER_JOB"
export MKL_NUM_THREADS="$THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$THREADS_PER_JOB"
if [[ "$USE_TUNED_LENS" == 1 ]]; then
  export PYTHONPATH="$TUNED_LENS_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
fi

PREFLIGHT=(
  "$PYTHON_BIN" scripts/preflight_layer_factorial.py
  --model "$MODEL"
  --text-fname "$TEXT_FILE"
  --sentence-manifest-fname "$SENTENCE_MANIFEST"
  --joint-data-fname "$JOINT_FILE"
  --paper-rt-fname "$PAPER_RT_FILE"
  --precomputed-frequency-fname "$FREQUENCY_FILE"
  --lens-methods "${CONFIG_LENSES[@]}"
  --hf-home "$HF_HOME"
)
if [[ "$USE_TUNED_LENS" == 1 ]]; then
  PREFLIGHT+=(--tuned-lens-path "$TUNED_LENS_PATH")
fi

RUNNER=(
  "$PYTHON_BIN" scripts/run_layer_factorial.py
  --config "$LAYER_FACTORIAL_CONFIG"
  --model "$MODEL"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --results-root "$RESULTS_ROOT"
  --python "$PYTHON_BIN"
  --rscript "$RSCRIPT_BIN"
  --jobs "$FACTORIAL_JOBS"
  --threads-per-job "$THREADS_PER_JOB"
)
if [[ "$USE_TUNED_LENS" == 1 ]]; then
  RUNNER+=(
    --tuned-lens-path "$TUNED_LENS_PATH"
    --tuned-lens-pythonpath "$TUNED_LENS_PYTHONPATH"
  )
fi

"${PREFLIGHT[@]}"

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'Dry run; exact command:\n'
  printf '%q ' "${RUNNER[@]}"
  printf '\n'
  exit 0
fi

if [[ "$USE_TUNED_LENS" == 1 ]]; then
  [[ -d "$TUNED_LENS_PYTHONPATH" ]] ||
    die "tuned-lens package directory is missing: $TUNED_LENS_PYTHONPATH"
fi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" &&
   "$CUDA_VISIBLE_DEVICES" != *,* ]] ||
  die "CUDA_VISIBLE_DEVICES must name exactly one physical GPU"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
RUNTIME_PREFLIGHT=(
  "${PREFLIGHT[@]}" --check-runtime --require-cuda --check-model-cache
)
if [[ "$USE_TUNED_LENS" == 1 ]]; then
  RUNTIME_PREFLIGHT+=(--smoke-load)
fi
"${RUNTIME_PREFLIGHT[@]}"

exec 9>"$CHECKPOINT_ROOT/.run.lock"
flock -n 9 ||
  die "another layer-factorial process holds $CHECKPOINT_ROOT/.run.lock"

printf 'Python: %s\n' "$("$PYTHON_BIN" --version 2>&1)"
printf 'R: %s\n' "$("$RSCRIPT_BIN" --version 2>&1)"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "$CUDA_VISIBLE_DEVICES"
printf 'Offline flags: HF_HUB_OFFLINE=%s TRANSFORMERS_OFFLINE=%s\n' "$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE"

"${RUNNER[@]}"

[[ -s "$CHECKPOINT_ROOT/run-manifest.json" ]] ||
  die "runner finished without a run manifest"
[[ -s "$RESULTS_ROOT/combined/REPORT.md" ]] ||
  die "runner finished without a combined report"
printf 'Complete report: %s\n' "$RESULTS_ROOT/combined/REPORT.md"
