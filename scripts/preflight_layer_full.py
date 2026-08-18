#!/usr/bin/env python3

"""Fail-fast validation for canonical full internal-layer inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_ROWS = 10_256
EXPECTED_PASSAGES = 10
NGRAM_CONTEXTS = tuple(range(5))
CONTEXT_LIMITED_CONTEXTS = tuple(range(1, 5))
SPILLOVER_PREFIXES = ("", "prev_", "prev2_", "prev3_")

MODEL_FINAL_LAYERS = {
    "gpt2-small": 12,
    "gpt2-medium": 24,
    "gpt2-large": 36,
    "gpt2-xl": 48,
    "pythia-70m": 6,
    "pythia-160m": 12,
    "pythia-410m": 24,
    "pythia-14b": 24,
    "pythia-28b": 32,
    "pythia-69b": 32,
    "pythia-120b": 36,
}

NGRAM_PATTERN = re.compile(r"ngram_surprisal_context_(\d+)$")
CONTEXT_PATTERN = re.compile(r"context_limited_surprisal_context_(\d+)$")
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}$")


class ValidationError(ValueError):
    """An input artifact does not satisfy the canonical contract."""


def sha256_file(fname: str | Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with open(fname, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_expected_sha256(actual: str, expected: str | None,
                             label: str) -> None:
    """Validate one optional expected digest."""

    if expected is None:
        return
    if not SHA256_PATTERN.fullmatch(expected):
        raise ValidationError(
            f"expected {label} SHA-256 must contain exactly 64 hex digits"
        )
    if actual != expected.lower():
        raise ValidationError(
            f"{label} SHA-256 mismatch: expected {expected.lower()}, got {actual}"
        )


def read_canonical_text(fname: str | Path,
                        expected_passages: int = EXPECTED_PASSAGES) -> list[list[str]]:
    """Read one nonempty whitespace-tokenized passage per line."""

    with open(fname, "r", encoding="utf8") as input_file:
        passages = [line.strip().split() for line in input_file]
    if len(passages) != expected_passages:
        raise ValidationError(
            f"full text has {len(passages)} passages; expected {expected_passages}"
        )
    empty = [index for index, words in enumerate(passages) if not words]
    if empty:
        raise ValidationError(
            "full text contains empty passages at zero-based IDs: "
            + ", ".join(map(str, empty))
        )
    return passages


def expected_word_rows(passages: Iterable[Iterable[str]],
                       text_id_offset: int) -> list[tuple[int, int, str]]:
    """Build ordered canonical ``(text_id, word_id, word)`` rows."""

    return [
        (text_id + text_id_offset, word_id, word)
        for text_id, words in enumerate(passages)
        for word_id, word in enumerate(words)
    ]


def _integer_values(series: pd.Series, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValidationError(f"{label} must contain finite integers")
    return values.astype(np.int64)


def validate_exact_key_word_rows(dataframe: pd.DataFrame,
                                 expected: list[tuple[int, int, str]],
                                 word_column: str, label: str) -> None:
    """Require exact ordered key/word coverage with no duplicates."""

    required = {"text_id", "word_id", word_column}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValidationError(
            f"{label} is missing columns: {', '.join(sorted(missing))}"
        )
    if len(dataframe) != len(expected):
        raise ValidationError(
            f"{label} has {len(dataframe)} rows; expected {len(expected)}"
        )
    text_ids = _integer_values(dataframe["text_id"], f"{label} text_id")
    word_ids = _integer_values(dataframe["word_id"], f"{label} word_id")
    keys = pd.DataFrame({"text_id": text_ids, "word_id": word_ids})
    if keys.duplicated(["text_id", "word_id"], keep=False).any():
        raise ValidationError(f"{label} contains duplicate keys")

    observed = list(zip(
        text_ids.tolist(),
        word_ids.tolist(),
        dataframe[word_column].astype(str).tolist(),
    ))
    if observed != expected:
        mismatch = next(
            index
            for index, (actual, wanted) in enumerate(zip(observed, expected))
            if actual != wanted
        )
        raise ValidationError(
            f"{label} key/word mismatch at row {mismatch}: "
            f"got {observed[mismatch]!r}, expected {expected[mismatch]!r}"
        )


def _direct_contexts(columns: Iterable[str], pattern: re.Pattern[str]) -> tuple[int, ...]:
    return tuple(sorted(
        int(match.group(1))
        for column in columns
        if (match := pattern.fullmatch(column)) is not None
    ))


def _require_finite(dataframe: pd.DataFrame, columns: Iterable[str],
                    label: str) -> None:
    for column in columns:
        values = pd.to_numeric(dataframe[column], errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            raise ValidationError(f"{label} column {column} is not finite")


def _require_numeric_allow_na(dataframe: pd.DataFrame,
                              columns: Iterable[str], label: str) -> None:
    """Require numeric values while allowing genuine missing observations."""

    for column in columns:
        original = dataframe[column]
        values = pd.to_numeric(original, errors="coerce")
        invalid = values.isna() & ~original.isna()
        if invalid.any():
            raise ValidationError(f"{label} column {column} is not numeric")
        array = values.to_numpy(dtype=float)
        if not (np.isfinite(array) | np.isnan(array)).all():
            raise ValidationError(
                f"{label} column {column} contains a non-finite value"
            )


def validate_joint_predictor_schema(joint: pd.DataFrame) -> None:
    """Require the canonical N0-4/C1-4 predictors and spillovers."""

    ngram_contexts = _direct_contexts(joint.columns, NGRAM_PATTERN)
    context_contexts = _direct_contexts(joint.columns, CONTEXT_PATTERN)
    if ngram_contexts != NGRAM_CONTEXTS:
        raise ValidationError(
            f"joint n-gram contexts are {ngram_contexts}; expected {NGRAM_CONTEXTS}"
        )
    if context_contexts != CONTEXT_LIMITED_CONTEXTS:
        raise ValidationError(
            "joint context-limited contexts are "
            f"{context_contexts}; expected {CONTEXT_LIMITED_CONTEXTS}"
        )

    ngram_columns = [
        f"ngram_surprisal_context_{context}" for context in NGRAM_CONTEXTS
    ]
    context_columns = [
        f"context_limited_surprisal_context_{context}"
        for context in CONTEXT_LIMITED_CONTEXTS
    ]
    direct = ngram_columns + context_columns
    spillovers = [
        f"{prefix}{column}"
        for column in direct
        for prefix in SPILLOVER_PREFIXES[1:]
    ]
    controls = [
        "time", "word_len", "freq", "surprisal",
        "prev_word_len", "prev_freq",
        "prev2_word_len", "prev2_freq",
        "prev3_word_len", "prev3_freq",
    ]
    missing = set(direct + spillovers + controls) - set(joint.columns)
    if missing:
        raise ValidationError(
            "joint table is missing canonical analysis columns: "
            + ", ".join(sorted(missing))
        )
    _require_finite(joint, direct + ["time", "word_len", "surprisal"], "joint")
    _require_numeric_allow_na(
        joint,
        spillovers + [
            "freq", "prev_word_len", "prev_freq",
            "prev2_word_len", "prev2_freq",
            "prev3_word_len", "prev3_freq",
        ],
        "joint",
    )


def validate_model_final_layer(model: str, final_layer: int) -> None:
    """Reject a final-layer ID that does not match the selected model."""

    if model not in MODEL_FINAL_LAYERS:
        raise ValidationError(
            f"unknown model {model!r}; expected one of "
            + ", ".join(sorted(MODEL_FINAL_LAYERS))
        )
    expected = MODEL_FINAL_LAYERS[model]
    if final_layer != expected:
        raise ValidationError(
            f"model {model} has final layer {expected}, not {final_layer}"
        )


def run_preflight(text_fname: str | Path, joint_data_fname: str | Path,
                  reference_surprisal_fname: str | Path, model: str,
                  expected_final_layer: int, expected_rows: int = EXPECTED_ROWS,
                  expected_passages: int = EXPECTED_PASSAGES,
                  expected_text_sha256: str | None = None,
                  expected_joint_sha256: str | None = None,
                  expected_reference_sha256: str | None = None) -> dict:
    """Validate all immutable inputs and return an auditable report."""

    validate_model_final_layer(model, expected_final_layer)
    text_sha = sha256_file(text_fname)
    joint_sha = sha256_file(joint_data_fname)
    reference_sha = sha256_file(reference_surprisal_fname)
    validate_expected_sha256(text_sha, expected_text_sha256, "text")
    validate_expected_sha256(joint_sha, expected_joint_sha256, "joint")
    validate_expected_sha256(
        reference_sha, expected_reference_sha256, "reference"
    )

    passages = read_canonical_text(text_fname, expected_passages)
    zero_based = expected_word_rows(passages, 0)
    one_based = expected_word_rows(passages, 1)
    if len(zero_based) != expected_rows:
        raise ValidationError(
            f"full text has {len(zero_based)} words; expected {expected_rows}"
        )

    joint = pd.read_csv(joint_data_fname, sep="\t", low_memory=False)
    reference = pd.read_csv(
        reference_surprisal_fname, sep="\t", low_memory=False
    )
    validate_exact_key_word_rows(joint, one_based, "ref_token", "joint table")
    validate_exact_key_word_rows(
        reference, zero_based, "word", "reference surprisal"
    )
    validate_joint_predictor_schema(joint)
    if "surprisal" not in reference.columns:
        raise ValidationError("reference surprisal lacks column surprisal")
    _require_finite(reference, ["surprisal"], "reference surprisal")

    return {
        "validated": True,
        "rows": expected_rows,
        "passages": expected_passages,
        "passage_word_counts": [len(words) for words in passages],
        "model": model,
        "final_layer": expected_final_layer,
        "ngram_contexts": list(NGRAM_CONTEXTS),
        "context_limited_contexts": list(CONTEXT_LIMITED_CONTEXTS),
        "artifacts": {
            "text": {
                "fname": str(Path(text_fname).resolve()),
                "sha256": text_sha,
            },
            "joint": {
                "fname": str(Path(joint_data_fname).resolve()),
                "sha256": joint_sha,
            },
            "reference": {
                "fname": str(Path(reference_surprisal_fname).resolve()),
                "sha256": reference_sha,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate canonical inputs for the full layer experiment"
    )
    parser.add_argument("--text-fname", required=True)
    parser.add_argument("--joint-data-fname", required=True)
    parser.add_argument("--reference-surprisal-fname", required=True)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_FINAL_LAYERS))
    parser.add_argument("--expected-final-layer", required=True, type=int)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument(
        "--expected-passages", type=int, default=EXPECTED_PASSAGES
    )
    parser.add_argument("--expected-text-sha256")
    parser.add_argument("--expected-joint-sha256")
    parser.add_argument("--expected-reference-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_preflight(
        args.text_fname,
        args.joint_data_fname,
        args.reference_surprisal_fname,
        args.model,
        args.expected_final_layer,
        expected_rows=args.expected_rows,
        expected_passages=args.expected_passages,
        expected_text_sha256=args.expected_text_sha256,
        expected_joint_sha256=args.expected_joint_sha256,
        expected_reference_sha256=args.expected_reference_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
