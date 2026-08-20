#!/usr/bin/env bash

# Run one canonical Natural Stories N-vs-L/C-vs-L experiment on one explicitly
# exposed GPU. Model metadata comes exclusively from the ten-model registry.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

fail() {
    echo "$*" >&2
    exit 2
}

if [[ -n ${PYTHON:-} ]]; then
    PYTHON_BIN=${PYTHON}
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    fail "Python is required; activate the project environment."
fi

RSCRIPT_BIN=${RSCRIPT:-Rscript}
WORDSPROBABILITY_BIN=${WORDSPROBABILITY:-wordsprobability}
MODEL_REGISTRY=src/h01_data/internal_layer_models.py
MODEL=${MODEL:-gpt2-small}

if ! REGISTERED_MODEL=$(
    "${PYTHON_BIN}" "${MODEL_REGISTRY}" --model "${MODEL}" --field alias 2>/dev/null
); then
    echo "Unsupported MODEL=${MODEL}. Canonical models are:" >&2
    "${PYTHON_BIN}" "${MODEL_REGISTRY}" --list >&2
    exit 2
fi
[[ ${REGISTERED_MODEL} == "${MODEL}" ]] ||
    fail "Registry returned ${REGISTERED_MODEL} for MODEL=${MODEL}."

registry_field() {
    "${PYTHON_BIN}" "${MODEL_REGISTRY}" \
        --model "${MODEL}" --field "$1"
}

MODEL_HF_NAME=$(registry_field hf_name)
MODEL_FAMILY=$(registry_field family)
INTERNAL_FINAL_LAYER=$(registry_field final_layer)
INTERNAL_LAYER_IDS=$(registry_field layer_ids)
REFERENCE_POLICY=$(registry_field reference_policy)
REGISTRY_ANCHOR_TOLERANCE=$(registry_field default_anchor_tolerance)

if [[ ${MODEL} == pythia-69b ]]; then
    echo "WARNING: pythia-69b is the 6.9B single-GPU feasibility boundary."
    echo "The mandatory offline load and final-layer anchor smoke must both pass before all-layer scoring."
fi

if [[ -v FINAL_LAYER_ANCHOR_TOLERANCE ]] &&
        [[ ${FINAL_LAYER_ANCHOR_TOLERANCE} != "${REGISTRY_ANCHOR_TOLERANCE}" ]]; then
    fail "MODEL=${MODEL} requires FINAL_LAYER_ANCHOR_TOLERANCE=${REGISTRY_ANCHOR_TOLERANCE}; arbitrary overrides are forbidden."
fi
FINAL_LAYER_ANCHOR_TOLERANCE=${REGISTRY_ANCHOR_TOLERANCE}

DRY_RUN=${DRY_RUN:-0}
[[ ${DRY_RUN} == 0 || ${DRY_RUN} == 1 ]] ||
    fail "DRY_RUN must be 0 or 1."

[[ -n ${RUN_ROOT:-} ]] ||
    fail "Set RUN_ROOT to the cluster scratch experiment directory."
[[ ${RUN_ROOT} == /* ]] ||
    fail "RUN_ROOT must be an absolute path: ${RUN_ROOT}"
RUN_ROOT=${RUN_ROOT%/}

LAYER_FOLDS=${LAYER_FOLDS:-10}
LAYER_SEED=${LAYER_SEED:-42}
[[ ${LAYER_FOLDS} =~ ^[1-9][0-9]*$ ]] ||
    fail "LAYER_FOLDS must be a positive integer."
[[ ${LAYER_SEED} =~ ^[0-9]+$ ]] ||
    fail "LAYER_SEED must be a nonnegative integer."

CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${RUN_ROOT}/checkpoints/rt"}
INPUT_CHECKPOINT_DIR=${INPUT_CHECKPOINT_DIR:-"${CHECKPOINT_DIR}"}
RESULTS_DIR=${RESULTS_DIR:-"${RUN_ROOT}/results/rt"}
LOG_DIR=${LOG_DIR:-"${RUN_ROOT}/logs/layer-full"}
CHECKPOINT_DIR=${CHECKPOINT_DIR%/}
INPUT_CHECKPOINT_DIR=${INPUT_CHECKPOINT_DIR%/}
RESULTS_DIR=${RESULTS_DIR%/}
LOG_DIR=${LOG_DIR%/}
for directory in \
        "${CHECKPOINT_DIR}" "${INPUT_CHECKPOINT_DIR}" \
        "${RESULTS_DIR}" "${LOG_DIR}"; do
    [[ ${directory} == /* ]] ||
        fail "Experiment directories must be absolute: ${directory}"
done

DEFINITIVE_TAG=natural_stories-gpt2-small-ngram_v4_piletrain_llama-contexts_0-1-2-3-4-alpha_0_4-context_1-2-3-4
MODEL_TAG=natural_stories-${MODEL}-ngram_v4_piletrain_llama-contexts_0-1-2-3-4-alpha_0_4-context_1-2-3-4
EXPECTED_TEXT_SHA256=04578a7187ec7edb779362f912df97befc74f7945c4d554902e2049041579da4
EXPECTED_DEFINITIVE_JOINT_SHA256=d09c6b35e3a7ee8cb7dea83bc4ee5babc59f413ab0067a0d20ca5c4ab3450b1e
EXPECTED_SMALL_REFERENCE_SHA256=cd86559a6ef0289ae006b04f365f623531ef8ebb9badb61e12538bd8ae886bc9
CANONICAL_SMALL_MODEL_COMMIT=607a30d783dfa663caf39e06633721c8d4cfcd7e

if [[ ${MODEL} == gpt2-small ]]; then
    if [[ -n ${EXPECTED_MODEL_COMMIT:-} ]] &&
            [[ ${EXPECTED_MODEL_COMMIT} != "${CANONICAL_SMALL_MODEL_COMMIT}" ]]; then
        fail "Canonical gpt2-small requires EXPECTED_MODEL_COMMIT=${CANONICAL_SMALL_MODEL_COMMIT}."
    fi
    EFFECTIVE_EXPECTED_MODEL_COMMIT=${CANONICAL_SMALL_MODEL_COMMIT}
else
    EFFECTIVE_EXPECTED_MODEL_COMMIT=${EXPECTED_MODEL_COMMIT:-}
fi

file_sha256() {
    local digest remainder
    read -r digest remainder < <(sha256sum -- "$1")
    printf '%s\n' "${digest}"
}

verify_sha256() {
    local fname=$1
    local expected=$2
    local label=$3
    local actual
    actual=$(file_sha256 "${fname}")
    [[ ${actual} == "${expected}" ]] ||
        fail "${label} SHA-256 mismatch: expected ${expected}, got ${actual}: ${fname}"
}

TRACKED_CHECKPOINT_DIR=${REPO_ROOT}/checkpoints/rt
STAGED_INPUTS=()
stage_tracked_input() {
    local relative=$1
    local label=$2
    local expected_sha=${3:-}
    local source=${TRACKED_CHECKPOINT_DIR}/${relative}
    local destination=${INPUT_CHECKPOINT_DIR}/${relative}
    local repository_relative=checkpoints/rt/${relative}
    local temporary

    git -C "${REPO_ROOT}" ls-files --error-unmatch \
        -- "${repository_relative}" >/dev/null 2>&1 ||
        fail "Required ${label} is not tracked by Git: ${source}"
    git -C "${REPO_ROOT}" diff --quiet -- "${repository_relative}" ||
        fail "Tracked ${label} has unstaged local changes: ${source}"
    git -C "${REPO_ROOT}" diff --cached --quiet -- "${repository_relative}" ||
        fail "Tracked ${label} has staged local changes: ${source}"
    [[ -s ${source} ]] ||
        fail "Tracked ${label} is missing or empty: ${source}"
    if [[ -n ${expected_sha} ]]; then
        verify_sha256 "${source}" "${expected_sha}" "Tracked ${label}"
    fi

    if [[ ${source} == "${destination}" ]]; then
        echo "Verified tracked ${label}: ${source}"
    elif [[ -e ${destination} ]]; then
        [[ -f ${destination} ]] ||
            fail "Scratch ${label} is not a regular file: ${destination}"
        cmp -s -- "${source}" "${destination}" ||
            fail "Scratch ${label} differs from tracked repository input: ${destination}"
        echo "Verified scratch ${label} equals tracked input: ${destination}"
    else
        mkdir -p -- "$(dirname -- "${destination}")"
        temporary=${destination}.tmp.$$
        if ! cp -- "${source}" "${temporary}"; then
            rm -f -- "${temporary}"
            fail "Unable to stage ${label}: ${destination}"
        fi
        mv -- "${temporary}" "${destination}"
        cmp -s -- "${source}" "${destination}" ||
            fail "Staged ${label} failed byte-for-byte verification: ${destination}"
        echo "Staged tracked ${label}: ${destination}"
    fi
    STAGED_INPUTS+=("${destination}")
}

stage_tracked_input \
    text_rt_data/natural_stories.txt \
    "definitive Natural Stories text" \
    "${EXPECTED_TEXT_SHA256}"
stage_tracked_input \
    joint_full/${DEFINITIVE_TAG}/joint-data.tsv \
    "definitive GPT-2-small N+C joint" \
    "${EXPECTED_DEFINITIVE_JOINT_SHA256}"
stage_tracked_input \
    joint_full/${MODEL_TAG}/context-limited.tsv \
    "${MODEL} context-limited predictors"
if [[ ${REFERENCE_POLICY} == tracked ]]; then
    stage_tracked_input \
        surprisals_rt_data/suprisal-natural_stories-gpt2-small.tsv \
        "GPT-2-small ordinary-surprisal reference" \
        "${EXPECTED_SMALL_REFERENCE_SHA256}"
fi

FULL_TEXT_FILE=${INPUT_CHECKPOINT_DIR}/text_rt_data/natural_stories.txt
DEFINITIVE_JOINT_FILE=${INPUT_CHECKPOINT_DIR}/joint_full/${DEFINITIVE_TAG}/joint-data.tsv
if [[ ${REFERENCE_POLICY} == tracked ]]; then
    MODEL_REFERENCE_FILE=${INPUT_CHECKPOINT_DIR}/surprisals_rt_data/suprisal-natural_stories-gpt2-small.tsv
    MODEL_JOINT_FILE=${DEFINITIVE_JOINT_FILE}
elif [[ ${REFERENCE_POLICY} == fresh ]]; then
    MODEL_REFERENCE_FILE=${CHECKPOINT_DIR}/layer_full/natural_stories-${MODEL}-ordinary-reference/surprisal-natural_stories-${MODEL}.tsv
    MODEL_JOINT_FILE=${CHECKPOINT_DIR}/joint_full/${MODEL_TAG}/joint-data.tsv
else
    fail "Unsupported registry reference policy for ${MODEL}: ${REFERENCE_POLICY}"
fi

preflight_args=(
    --text-fname "${FULL_TEXT_FILE}"
    --joint-data-fname "${MODEL_JOINT_FILE}"
    --reference-surprisal-fname "${MODEL_REFERENCE_FILE}"
    --model "${MODEL}"
    --expected-final-layer "${INTERNAL_FINAL_LAYER}"
    --expected-rows 10256
    --expected-passages 10
    --expected-text-sha256 "${EXPECTED_TEXT_SHA256}"
)
if [[ ${REFERENCE_POLICY} == tracked ]]; then
    preflight_args+=(
        --expected-joint-sha256 "${EXPECTED_DEFINITIVE_JOINT_SHA256}"
        --expected-reference-sha256 "${EXPECTED_SMALL_REFERENCE_SHA256}"
    )
fi

make_common_args=(
    -f MakefileLayerFull
    "PYTHON=${PYTHON_BIN}"
    "RSCRIPT=${RSCRIPT_BIN}"
    "MODEL=${MODEL}"
    "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
    "INPUT_CHECKPOINT_DIR=${INPUT_CHECKPOINT_DIR}"
    "RESULTS_DIR=${RESULTS_DIR}"
    "LAYER_FOLDS=${LAYER_FOLDS}"
    "LAYER_SEED=${LAYER_SEED}"
    "FINAL_LAYER_ANCHOR_TOLERANCE=${FINAL_LAYER_ANCHOR_TOLERANCE}"
)

if [[ ${DRY_RUN} == 1 ]]; then
    echo "Dry run: model=${MODEL} hf_name=${MODEL_HF_NAME} family=${MODEL_FAMILY}"
    echo "Dry run: layers=1-${INTERNAL_FINAL_LAYER} layer_ids=${INTERNAL_LAYER_IDS}"
    echo "Dry run: reference_policy=${REFERENCE_POLICY} anchor_tolerance=${FINAL_LAYER_ANCHOR_TOLERANCE}"
    echo "Dry run phase order: fresh reference (when required), model joint, generic preflight, final-layer anchor smoke, full layer run"
    printf 'Dry run generic preflight: %q' "${PYTHON_BIN}"
    printf ' %q' scripts/preflight_layer_full.py "${preflight_args[@]}"
    printf '\n'
    make -n "${make_common_args[@]}" layer_full
    exit 0
fi

[[ -n ${HF_HOME:-} ]] ||
    fail "Set HF_HOME to the populated cluster Hugging Face cache."
command -v "${RSCRIPT_BIN}" >/dev/null 2>&1 ||
    [[ -x ${RSCRIPT_BIN} ]] ||
    fail "Rscript is required; activate the project environment."
command -v "${WORDSPROBABILITY_BIN}" >/dev/null 2>&1 ||
    [[ -x ${WORDSPROBABILITY_BIN} ]] ||
    fail "wordsprobability is required; activate the project environment."

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
[[ ${HF_HUB_OFFLINE} == 1 && ${TRANSFORMERS_OFFLINE} == 1 ]] ||
    fail "Canonical runs require HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1."

[[ -n ${CUDA_VISIBLE_DEVICES:-} ]] ||
    fail "Set CUDA_VISIBLE_DEVICES to exactly one free GPU."
[[ ${CUDA_VISIBLE_DEVICES} != *,* ]] ||
    fail "CUDA_VISIBLE_DEVICES must expose exactly one GPU, not: ${CUDA_VISIBLE_DEVICES}"

mkdir -p -- "${LOG_DIR}"
environment_log=${LOG_DIR}/environment.log
run_log=${LOG_DIR}/${MODEL}-layers_1-${INTERNAL_FINAL_LAYER}-folds_${LAYER_FOLDS}-seed_${LAYER_SEED}.log

{
    echo "run_start=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "repo_root=${REPO_ROOT}"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_status_begin"
    git status --short
    echo "git_status_end"
    echo "MODEL=${MODEL}"
    echo "MODEL_HF_NAME=${MODEL_HF_NAME}"
    echo "MODEL_FAMILY=${MODEL_FAMILY}"
    echo "INTERNAL_FINAL_LAYER=${INTERNAL_FINAL_LAYER}"
    echo "REFERENCE_POLICY=${REFERENCE_POLICY}"
    echo "FINAL_LAYER_ANCHOR_TOLERANCE=${FINAL_LAYER_ANCHOR_TOLERANCE}"
    echo "RUN_ROOT=${RUN_ROOT}"
    echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
    echo "INPUT_CHECKPOINT_DIR=${INPUT_CHECKPOINT_DIR}"
    echo "RESULTS_DIR=${RESULTS_DIR}"
    echo "HF_HOME=${HF_HOME}"
    echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
    echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "EXPECTED_MODEL_COMMIT=${EFFECTIVE_EXPECTED_MODEL_COMMIT:-unconstrained}"
    echo "staged_input_sha256_begin"
    sha256sum -- "${STAGED_INPUTS[@]}"
    echo "staged_input_sha256_end"
    "${PYTHON_BIN}" --version
    "${RSCRIPT_BIN}" --version
    "${PYTHON_BIN}" -c 'from importlib.metadata import version; import torch; print("torch={} transformers={} wordsprobability={}".format(torch.__version__, version("transformers"), version("wordsprobability"))); print("cuda_available={} visible_devices={} gpu={}".format(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"))'
} 2>&1 | tee -a "${environment_log}"

if ! "${PYTHON_BIN}" -c \
        'import sys, torch; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() == 1 else 1)'; then
    fail "PyTorch must see exactly one CUDA GPU."
fi

# Mandatory offline load validates weights, tokenizer, boundary masks,
# layer count, device placement, and the selected snapshot before scoring.
if ! MODEL_ALIAS="${MODEL}" \
        EXPECTED_HF_NAME="${MODEL_HF_NAME}" \
        EXPECTED_MODEL_COMMIT="${EFFECTIVE_EXPECTED_MODEL_COMMIT}" \
        "${PYTHON_BIN}" -c \
        'import os; from src.h01_data.get_context_limited_surprisals import load_wordsprobability_model; from src.h01_data.get_internal_layer_surprisals import validate_registered_model_layer_count; alias=os.environ["MODEL_ALIAS"]; wrapper=load_wordsprobability_model(alias); model=wrapper.model; validate_registered_model_layer_count(alias, model); parameter=next(model.parameters()); name=getattr(model.config, "_name_or_path", None); commit=getattr(model.config, "_commit_hash", None); expected_name=os.environ["EXPECTED_HF_NAME"]; expected_commit=os.environ.get("EXPECTED_MODEL_COMMIT", ""); print("offline_cache_preflight=alias:{} name:{} commit:{} dtype:{} device:{}".format(alias, name, commit, parameter.dtype, parameter.device)); raise SystemExit(0 if name == expected_name and (not expected_commit or commit == expected_commit) else "cached model provenance mismatch: expected name={} commit={} got name={} commit={}".format(expected_name, expected_commit or "unconstrained", name, commit))' \
        2>&1 | tee -a "${environment_log}"; then
    fail "Offline ${MODEL} cache/revision preflight failed under HF_HOME=${HF_HOME}."
fi

echo "Phase 1/5: ordinary reference; policy=${REFERENCE_POLICY}" |
    tee -a "${run_log}"
if [[ ${REFERENCE_POLICY} == fresh ]]; then
    # Force a same-runtime reference even on a resumed invocation.
    if ! make -B "${make_common_args[@]}" layer_full_reference 2>&1 |
            tee -a "${run_log}"; then
        echo "Fresh ordinary-reference generation failed." >&2
        exit 1
    fi
else
    if ! make "${make_common_args[@]}" layer_full_reference 2>&1 |
            tee -a "${run_log}"; then
        echo "Tracked ordinary-reference verification failed." >&2
        exit 1
    fi
fi

echo "Phase 2/5: model-specific N+C joint" | tee -a "${run_log}"
if [[ ${REFERENCE_POLICY} == fresh ]]; then
    if ! make -W "${MODEL_REFERENCE_FILE}" \
            "${make_common_args[@]}" layer_full_model_joint 2>&1 |
            tee -a "${run_log}"; then
        echo "Fresh model-specific joint construction failed." >&2
        exit 1
    fi
else
    if ! make "${make_common_args[@]}" layer_full_model_joint 2>&1 |
            tee -a "${run_log}"; then
        echo "Tracked model-specific joint verification failed." >&2
        exit 1
    fi
fi
[[ -s ${MODEL_REFERENCE_FILE} ]] ||
    fail "Produced reference is missing or empty: ${MODEL_REFERENCE_FILE}"
[[ -s ${MODEL_JOINT_FILE} ]] ||
    fail "Produced model joint is missing or empty: ${MODEL_JOINT_FILE}"

echo "Phase 3/5: generic produced-input preflight" | tee -a "${run_log}"
if ! "${PYTHON_BIN}" scripts/preflight_layer_full.py \
        "${preflight_args[@]}" 2>&1 | tee -a "${run_log}"; then
    echo "Generic full-layer input preflight failed." >&2
    exit 1
fi
sha256sum -- "${MODEL_REFERENCE_FILE}" "${MODEL_JOINT_FILE}" |
    tee -a "${run_log}"

echo "Phase 4/5: full-corpus final-layer anchor smoke" |
    tee -a "${run_log}"
if ! make "${make_common_args[@]}" layer_full_anchor_smoke 2>&1 |
        tee -a "${run_log}"; then
    echo "Anchor smoke failed; full all-layer scoring was not started." >&2
    exit 1
fi

echo "Phase 5/5: full layers 1-${INTERNAL_FINAL_LAYER}" |
    tee -a "${run_log}"
if ! make "${make_common_args[@]}" layer_full 2>&1 |
        tee -a "${run_log}"; then
    echo "Layer run failed; rerun identically to reuse passage checkpoints." >&2
    exit 1
fi
echo "run_end=$(date --iso-8601=seconds)" | tee -a "${run_log}"
