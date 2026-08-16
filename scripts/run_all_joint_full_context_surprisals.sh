#!/usr/bin/env bash

# Generate the Natural Stories context-limited predictor checkpoints used by
# MakefileJointFull, without running Infini-gram, the joint merge, or R models.
# Completed model TSVs are Make targets and are skipped on a resumed run.
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
    echo "Python is required." >&2
    exit 2
fi

CONTEXT_LENGTHS=${CONTEXT_LIMITED_CONTEXT_LENGTHS:-"1 2 3 4"}
DRY_RUN=${DRY_RUN:-0}
ALLOW_CPU=${ALLOW_CPU:-0}

if [[ -n ${RUN_ROOT:-} ]]; then
    CACHE_DIR=${CACHE_DIR:-"${RUN_ROOT}/cache"}
    CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${RUN_ROOT}/checkpoints/rt"}
    RESULTS_DIR=${RESULTS_DIR:-"${RUN_ROOT}/results/rt"}
    LOG_DIR=${LOG_DIR:-"${RUN_ROOT}/logs/context-limited-all-models"}
else
    CACHE_DIR=${CACHE_DIR:-.cache}
    CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/rt}
    RESULTS_DIR=${RESULTS_DIR:-results/rt}
    LOG_DIR=${LOG_DIR:-.cache/logs/context-limited-all-models}
fi

if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ ! -v CUDA_VISIBLE_DEVICES ]]; then
        echo "Set CUDA_VISIBLE_DEVICES to one free GPU before running." >&2
        exit 2
    fi
    if [[ -z "${CUDA_VISIBLE_DEVICES}" && "${ALLOW_CPU}" != "1" ]]; then
        echo "CUDA_VISIBLE_DEVICES is empty; set ALLOW_CPU=1 only for a deliberate CPU run." >&2
        exit 2
    fi
    if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
        echo "Warning: wordsprobability uses only the first visible GPU; it does not shard models." >&2
    fi
    if [[ "${ALLOW_CPU}" != "1" ]] &&
            ! "${PYTHON_BIN}" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
        echo "PyTorch cannot access CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; refusing an accidental CPU run." >&2
        exit 2
    fi
fi

if ! SUPPORTED_MODELS_OUTPUT=$(
    "${PYTHON_BIN}" -c 'from src.h01_data.get_context_limited_surprisals import SUPPORTED_MODELS; print("\n".join(SUPPORTED_MODELS))'
); then
    echo "Could not load the supported model list; activate the project environment." >&2
    exit 2
fi
if [[ -z "${SUPPORTED_MODELS_OUTPUT}" ]]; then
    echo "The supported model list is empty." >&2
    exit 2
fi
mapfile -t SUPPORTED_MODELS <<< "${SUPPORTED_MODELS_OUTPUT}"

if (( $# > 0 )); then
    SELECTED_MODELS=("$@")
elif [[ -n ${MODELS:-} ]]; then
    read -r -a SELECTED_MODELS <<< "${MODELS}"
else
    SELECTED_MODELS=("${SUPPORTED_MODELS[@]}")
fi

declare -A IS_SUPPORTED=()
for model in "${SUPPORTED_MODELS[@]}"; do
    IS_SUPPORTED["${model}"]=1
done
for model in "${SELECTED_MODELS[@]}"; do
    if [[ -z ${IS_SUPPORTED[$model]+x} ]]; then
        echo "Unsupported model: ${model}" >&2
        echo "Supported models: ${SUPPORTED_MODELS[*]}" >&2
        exit 2
    fi
done

default_batch_size() {
    case "$1" in
        gpt2-small|gpt2-medium|pythia-70m|pythia-160m|pythia-410m)
            echo 8
            ;;
        gpt2-large|pythia-14b)
            echo 4
            ;;
        gpt2-xl)
            echo 2
            ;;
        pythia-28b|pythia-69b|pythia-120b)
            echo 1
            ;;
        *)
            echo 1
            ;;
    esac
}

if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${LOG_DIR}"
    {
        echo "timestamp=$(date --iso-8601=seconds)"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        echo "HF_HOME=${HF_HOME:-<unset>}"
        "${PYTHON_BIN}" -c 'from importlib.metadata import version; import torch; print("torch={} transformers={} wordsprobability={}".format(version("torch"), version("transformers"), version("wordsprobability"))); print("cuda_available={} gpu={}".format(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))'
    } 2>&1 | tee -a "${LOG_DIR}/environment.log"
fi

CONTEXT_TAG=${CONTEXT_LENGTHS// /-}
for model in "${SELECTED_MODELS[@]}"; do
    if [[ -n ${CONTEXT_LIMITED_BATCH_SIZE:-} ]]; then
        batch_size=${CONTEXT_LIMITED_BATCH_SIZE}
    else
        batch_size=$(default_batch_size "${model}")
    fi

    make_args=(
        -f MakefileJointFull
        joint_full_context_surprisals
        DATASET=natural_stories
        "MODEL=${model}"
        "CONTEXT_LIMITED_CONTEXT_LENGTHS=${CONTEXT_LENGTHS}"
        "CONTEXT_LIMITED_BATCH_SIZE=${batch_size}"
        "CACHE_DIR=${CACHE_DIR}"
        "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
        "RESULTS_DIR=${RESULTS_DIR}"
    )

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "Dry run: model=${model} batch_size=${batch_size}"
        make -n "${make_args[@]}"
        continue
    fi

    log_file="${LOG_DIR}/${model}-contexts_${CONTEXT_TAG}.log"
    echo "Starting model=${model} batch_size=${batch_size} log=${log_file}" |
        tee -a "${log_file}"
    if ! make "${make_args[@]}" 2>&1 | tee -a "${log_file}"; then
        echo "Failed model=${model}; completed model checkpoints remain reusable." |
            tee -a "${log_file}" >&2
        exit 1
    fi
done

echo "Context-limited surprisal sweep complete for: ${SELECTED_MODELS[*]}"
