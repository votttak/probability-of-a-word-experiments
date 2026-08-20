#!/usr/bin/env bash

# Sequential fail-fast launcher for the ten models with completed C predictors.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

if [[ -n ${PYTHON:-} ]]; then
    PYTHON_BIN=${PYTHON}
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    echo "Python is required; activate the project environment." >&2
    exit 2
fi

MODEL_REGISTRY=src/h01_data/internal_layer_models.py
if (( $# > 0 )); then
    selected_models=("$@")
elif [[ -n ${MODELS:-} ]]; then
    read -r -a selected_models <<< "${MODELS}"
else
    mapfile -t selected_models < <(
        "${PYTHON_BIN}" "${MODEL_REGISTRY}" --list
    )
fi
(( ${#selected_models[@]} > 0 )) || {
    echo "No models selected." >&2
    exit 2
}

if [[ -n ${EXPECTED_MODEL_COMMIT:-} ]] &&
        (( ${#selected_models[@]} != 1 )); then
    echo "EXPECTED_MODEL_COMMIT may be used only with one selected model." >&2
    exit 2
fi

if [[ ${DRY_RUN:-0} != 1 ]]; then
    [[ -n ${CUDA_VISIBLE_DEVICES:-} ]] || {
        echo "Set CUDA_VISIBLE_DEVICES to exactly one free GPU." >&2
        exit 2
    }
    [[ ${CUDA_VISIBLE_DEVICES} != *,* ]] || {
        echo "CUDA_VISIBLE_DEVICES must expose exactly one GPU." >&2
        exit 2
    }
    echo "Sequential runner will use exposed GPU ${CUDA_VISIBLE_DEVICES}."
fi

declare -A seen=()
for model in "${selected_models[@]}"; do
    if ! canonical=$(
        "${PYTHON_BIN}" "${MODEL_REGISTRY}" \
            --model "${model}" --field alias 2>/dev/null
    ); then
        echo "Unsupported model ${model}; pythia-120b is intentionally excluded." >&2
        exit 2
    fi
    [[ -z ${seen[${canonical}]:-} ]] || {
        echo "Duplicate model selection: ${canonical}" >&2
        exit 2
    }
    seen[${canonical}]=1

    if [[ ${canonical} == pythia-69b ]]; then
        echo "WARNING: pythia-69b is the 6.9B single-GPU feasibility boundary."
        echo "Its mandatory offline load and final-layer anchor smoke are feasibility gates; failure stops the sequence."
    fi
    echo "===== Starting MODEL=${canonical} ====="
    MODEL=${canonical} "${SCRIPT_DIR}/run_layer_full.sh"
    echo "===== Completed MODEL=${canonical} ====="
done
