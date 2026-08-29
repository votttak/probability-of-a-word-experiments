#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python}"
REGISTRY="$REPOSITORY_ROOT/src/h01_data/layer_factorial_models.py"
DRY_RUN_ARGUMENT=()
if [[ $# -gt 1 ]]; then
  printf 'Usage: %s [--dry-run]\n' "$0" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  [[ "$1" == "--dry-run" ]] || {
    printf 'Usage: %s [--dry-run]\n' "$0" >&2
    exit 2
  }
  DRY_RUN_ARGUMENT=(--dry-run)
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
  exit 1
}
mapfile -t MODELS < <("$PYTHON_BIN" "$REGISTRY" --list)
(( ${#MODELS[@]} > 0 )) || {
  printf 'Layer-factorial model registry is empty\n' >&2
  exit 1
}

# Deliberately sequential: each model runs four extraction cells one at a
# time on the single GPU selected by CUDA_VISIBLE_DEVICES.
for model in "${MODELS[@]}"; do
  "$SCRIPT_DIR/run_layer_factorial_cluster.sh" "$model" "${DRY_RUN_ARGUMENT[@]}"
done
