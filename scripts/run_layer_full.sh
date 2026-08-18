#!/usr/bin/env bash

# Run the canonical Natural Stories/GPT-2-small N-vs-L and C-vs-L analysis on
# one explicitly selected GPU.  Make and per-passage checkpoints provide safe
# restart behavior when this script is rerun with the same arguments.
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

if ! command -v Rscript >/dev/null 2>&1; then
    echo "Rscript is required; activate the project environment." >&2
    exit 2
fi
RSCRIPT_BIN=$(command -v Rscript)

if [[ -z ${RUN_ROOT:-} ]]; then
    echo "Set RUN_ROOT to the cluster scratch experiment directory." >&2
    exit 2
fi
if [[ ${RUN_ROOT} != /* ]]; then
    echo "RUN_ROOT must be an absolute path: ${RUN_ROOT}" >&2
    exit 2
fi
if [[ -z ${HF_HOME:-} ]]; then
    echo "Set HF_HOME to the populated cluster Hugging Face cache." >&2
    exit 2
fi

DRY_RUN=${DRY_RUN:-0}
if [[ ${DRY_RUN} != 0 && ${DRY_RUN} != 1 ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 2
fi

LAYER_FOLDS=${LAYER_FOLDS:-10}
LAYER_SEED=${LAYER_SEED:-42}
FINAL_LAYER_ANCHOR_TOLERANCE=${FINAL_LAYER_ANCHOR_TOLERANCE:-0.0005}
if [[ ${FINAL_LAYER_ANCHOR_TOLERANCE} != 0.0005 ]]; then
    echo "Canonical full run requires FINAL_LAYER_ANCHOR_TOLERANCE=0.0005." >&2
    exit 2
fi
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${RUN_ROOT}/checkpoints/rt"}
RESULTS_DIR=${RESULTS_DIR:-"${RUN_ROOT}/results/rt"}
LOG_DIR=${LOG_DIR:-"${RUN_ROOT}/logs/layer-full"}

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

CANONICAL_TAG=natural_stories-gpt2-small-ngram_v4_piletrain_llama-contexts_0-1-2-3-4-alpha_0_4-context_1-2-3-4
FULL_TEXT_FILE=${CHECKPOINT_DIR}/text_rt_data/natural_stories.txt
REFERENCE_FILE=${CHECKPOINT_DIR}/surprisals_rt_data/suprisal-natural_stories-gpt2-small.tsv
CANONICAL_JOINT_FILE=${CHECKPOINT_DIR}/joint_full/${CANONICAL_TAG}/joint-data.tsv
EXPECTED_TEXT_SHA256=04578a7187ec7edb779362f912df97befc74f7945c4d554902e2049041579da4
EXPECTED_REFERENCE_SHA256=cd86559a6ef0289ae006b04f365f623531ef8ebb9badb61e12538bd8ae886bc9
EXPECTED_JOINT_SHA256=d09c6b35e3a7ee8cb7dea83bc4ee5babc59f413ab0067a0d20ca5c4ab3450b1e
EXPECTED_MODEL_COMMIT=607a30d783dfa663caf39e06633721c8d4cfcd7e
INPUT_FILES=(
    "${FULL_TEXT_FILE}"
    "${REFERENCE_FILE}"
    "${CANONICAL_JOINT_FILE}"
)

for input_file in "${INPUT_FILES[@]}"; do
    if [[ ! -s ${input_file} ]]; then
        echo "Required canonical input is missing or empty: ${input_file}" >&2
        exit 2
    fi
done
if [[ ! -f scripts/preflight_layer_full.py ]]; then
    echo "Missing canonical input validator: scripts/preflight_layer_full.py" >&2
    exit 2
fi

make_common_args=(
    -f MakefileLayerFull
    "PYTHON=${PYTHON_BIN}"
    "RSCRIPT=${RSCRIPT_BIN}"
    "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
    "RESULTS_DIR=${RESULTS_DIR}"
    "LAYER_FOLDS=${LAYER_FOLDS}"
    "LAYER_SEED=${LAYER_SEED}"
    "FINAL_LAYER_ANCHOR_TOLERANCE=${FINAL_LAYER_ANCHOR_TOLERANCE}"
)

if [[ ${DRY_RUN} == 1 ]]; then
    echo "Dry run, in execution order: layer-12 anchor smoke, then full layers 1-12"
    make -n "${make_common_args[@]}" layer_full
    exit 0
fi

if [[ ! -v CUDA_VISIBLE_DEVICES || -z ${CUDA_VISIBLE_DEVICES} ]]; then
    echo "Set CUDA_VISIBLE_DEVICES to exactly one free GPU." >&2
    exit 2
fi
if [[ ${CUDA_VISIBLE_DEVICES} == *,* ]]; then
    echo "CUDA_VISIBLE_DEVICES must expose exactly one GPU, not: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi
if [[ ${HF_HUB_OFFLINE} != 1 || ${TRANSFORMERS_OFFLINE} != 1 ]]; then
    echo "This runner requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1." >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
environment_log=${LOG_DIR}/environment.log
run_log=${LOG_DIR}/gpt2-small-layers_1-12-folds_${LAYER_FOLDS}-seed_${LAYER_SEED}.log

{
    echo "run_start=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "repo_root=${REPO_ROOT}"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_status_begin"
    git status --short
    echo "git_status_end"
    echo "RUN_ROOT=${RUN_ROOT}"
    echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
    echo "RESULTS_DIR=${RESULTS_DIR}"
    echo "HF_HOME=${HF_HOME}"
    echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
    echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "EXPECTED_MODEL_COMMIT=${EXPECTED_MODEL_COMMIT}"
    echo "EXPECTED_TEXT_SHA256=${EXPECTED_TEXT_SHA256}"
    echo "EXPECTED_REFERENCE_SHA256=${EXPECTED_REFERENCE_SHA256}"
    echo "EXPECTED_JOINT_SHA256=${EXPECTED_JOINT_SHA256}"
    "${PYTHON_BIN}" --version
    "${RSCRIPT_BIN}" --version
    "${PYTHON_BIN}" -c 'from importlib.metadata import version; import torch; print("torch={} transformers={} wordsprobability={}".format(torch.__version__, version("transformers"), version("wordsprobability"))); print("cuda_available={} visible_devices={} gpu={}".format(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"))'
    sha256sum "${INPUT_FILES[@]}"
} 2>&1 | tee -a "${environment_log}"

if ! "${PYTHON_BIN}" scripts/preflight_layer_full.py \
        --text-fname "${FULL_TEXT_FILE}" \
        --joint-data-fname "${CANONICAL_JOINT_FILE}" \
        --reference-surprisal-fname "${REFERENCE_FILE}" \
        --model gpt2-small \
        --expected-final-layer 12 \
        --expected-rows 10256 \
        --expected-passages 10 \
        --expected-text-sha256 "${EXPECTED_TEXT_SHA256}" \
        --expected-reference-sha256 "${EXPECTED_REFERENCE_SHA256}" \
        --expected-joint-sha256 "${EXPECTED_JOINT_SHA256}" \
        2>&1 | tee -a "${environment_log}"; then
    echo "Canonical full-layer input preflight failed." >&2
    exit 2
fi

if ! "${PYTHON_BIN}" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() == 1 else 1)'; then
    echo "PyTorch must see exactly one CUDA GPU." >&2
    exit 2
fi

# This loads the actual wrapper in offline mode, verifying that model weights,
# tokenizer files, boundary masks, package integration, and pinned revision are
# cached before the long scored run starts.
if ! EXPECTED_MODEL_COMMIT="${EXPECTED_MODEL_COMMIT}" "${PYTHON_BIN}" -c 'import os; from src.h01_data.get_context_limited_surprisals import load_wordsprobability_model; wrapper=load_wordsprobability_model("gpt2-small"); model=wrapper.model; parameter=next(model.parameters()); commit=getattr(model.config, "_commit_hash", None); expected=os.environ["EXPECTED_MODEL_COMMIT"]; print("offline_cache_preflight=alias:{} name:{} commit:{} dtype:{} device:{}".format("gpt2-small", model.config._name_or_path, commit, parameter.dtype, parameter.device)); raise SystemExit(0 if commit == expected else "cached model revision mismatch: expected {} got {}".format(expected, commit))' 2>&1 | tee -a "${environment_log}"; then
    echo "Offline GPT-2-small cache/revision preflight failed under HF_HOME=${HF_HOME}." >&2
    exit 2
fi

echo "Starting full-corpus layer-12 anchor smoke; log=${run_log}" | tee -a "${run_log}"
if ! make "${make_common_args[@]}" layer_full_anchor_smoke 2>&1 | tee -a "${run_log}"; then
    echo "Anchor smoke failed; full 12-layer scoring was not started." | tee -a "${run_log}" >&2
    exit 1
fi

echo "Starting canonical full layer run; log=${run_log}" | tee -a "${run_log}"
if ! make "${make_common_args[@]}" layer_full 2>&1 | tee -a "${run_log}"; then
    echo "Layer run failed; rerun the identical command to reuse completed passage checkpoints." |
        tee -a "${run_log}" >&2
    exit 1
fi
echo "run_end=$(date --iso-8601=seconds)" | tee -a "${run_log}"
