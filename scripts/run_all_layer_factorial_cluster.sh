#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
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

# Deliberately sequential: each model runs four extraction cells one at a
# time on the single GPU selected by CUDA_VISIBLE_DEVICES.
for model in gpt2-small gpt2-large gpt2-xl; do
  "$SCRIPT_DIR/run_layer_factorial_cluster.sh" "$model" "${DRY_RUN_ARGUMENT[@]}"
done
