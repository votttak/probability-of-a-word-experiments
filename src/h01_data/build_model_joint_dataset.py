#!/usr/bin/env python3

"""Build a model-specific canonical N+C+RT table from trusted artifacts.

The definitive GPT-2-small joint table supplies the Natural Stories reading
times, controls, and common n-gram predictors.  This builder replaces every
model-dependent column with scores from one model's context-limited and
ordinary full-context checkpoints.  Predictor inputs use zero-based passage
IDs, whereas the canonical joint table uses one-based passage IDs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_ROWS = 10_256
KEY_COLUMNS = ["text_id", "word_id"]
CONTEXT_PATTERN = re.compile(r"context_limited_surprisal_context_(\d+)$")
SPILLOVER_PREFIXES = ("prev_", "prev2_", "prev3_")


def _integer_values(series: pd.Series, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label} must contain finite integers")
    return values.astype(np.int64)


def _validate_words_per_text(words_per_text: int | None) -> None:
    if words_per_text is not None and (
        isinstance(words_per_text, bool) or words_per_text < 1
    ):
        raise ValueError("words per text must be a positive integer")


def _validate_keys(dataframe: pd.DataFrame, label: str, expected_rows: int,
                   *, text_id_offset: int = 0,
                   words_per_text: int | None = None) -> pd.DataFrame:
    required = set(KEY_COLUMNS + ["word"])
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"{label} is missing columns: {', '.join(sorted(missing))}"
        )
    validated = dataframe.copy()
    validated["text_id"] = (
        _integer_values(validated["text_id"], f"{label} text_id")
        + text_id_offset
    )
    validated["word_id"] = _integer_values(
        validated["word_id"], f"{label} word_id"
    )
    if (validated["text_id"] < 1).any():
        raise ValueError(f"{label} passage IDs must be one-based after alignment")
    if (validated["word_id"] < 0).any():
        raise ValueError(f"{label} word IDs must be nonnegative")
    if validated.duplicated(KEY_COLUMNS, keep=False).any():
        raise ValueError(f"{label} contains duplicate keys")
    if validated["word"].isna().any():
        raise ValueError(f"{label} contains a missing word")
    if words_per_text is not None:
        validated = validated.loc[
            validated["word_id"] < words_per_text
        ].copy()
    if len(validated) != expected_rows:
        raise ValueError(
            f"{label} has {len(validated)} rows after filtering; "
            f"expected {expected_rows}"
        )
    return validated


def _validate_canonical_keys(canonical: pd.DataFrame,
                             expected_rows: int,
                             words_per_text: int | None = None
                             ) -> tuple[pd.DataFrame, str]:
    required = set(KEY_COLUMNS)
    missing = required - set(canonical.columns)
    if missing:
        raise ValueError(
            "canonical joint is missing columns: "
            + ", ".join(sorted(missing))
        )
    word_column = "ref_token" if "ref_token" in canonical.columns else "word"
    if word_column not in canonical.columns:
        raise ValueError("canonical joint has neither ref_token nor word")

    validated = canonical.copy()
    validated["text_id"] = _integer_values(
        validated["text_id"], "canonical joint text_id"
    )
    validated["word_id"] = _integer_values(
        validated["word_id"], "canonical joint word_id"
    )
    if (validated["text_id"] < 1).any():
        raise ValueError("canonical joint passage IDs must be one-based")
    if (validated["word_id"] < 0).any():
        raise ValueError("canonical joint word IDs must be nonnegative")
    if validated.duplicated(KEY_COLUMNS, keep=False).any():
        raise ValueError("canonical joint contains duplicate keys")
    if validated[word_column].isna().any():
        raise ValueError("canonical joint contains a missing reference word")
    if words_per_text is not None:
        validated = validated.loc[
            validated["word_id"] < words_per_text
        ].copy()
    if len(validated) != expected_rows:
        raise ValueError(
            f"canonical joint has {len(validated)} rows after filtering; "
            f"expected {expected_rows}"
        )

    # Spillovers have a well-defined meaning only for a complete ordered word
    # sequence.  The output still preserves the canonical row order below.
    for text_id, group in validated.groupby("text_id", sort=False):
        observed = np.sort(group["word_id"].to_numpy(dtype=np.int64))
        expected = np.arange(len(group), dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise ValueError(
                f"canonical joint word IDs are not contiguous in text {text_id}"
            )
    return validated, word_column


def _key_tuples(dataframe: pd.DataFrame) -> list[tuple[int, int]]:
    return list(map(tuple, dataframe[KEY_COLUMNS].to_numpy(dtype=np.int64)))


def _align_source(canonical: pd.DataFrame, source: pd.DataFrame,
                  canonical_word_column: str, label: str) -> pd.DataFrame:
    canonical_keys = _key_tuples(canonical)
    source_keys = _key_tuples(source)
    canonical_set = set(canonical_keys)
    source_set = set(source_keys)
    if canonical_set != source_set:
        raise ValueError(
            f"{label} key coverage differs from canonical joint: "
            f"{len(canonical_set - source_set)} missing and "
            f"{len(source_set - canonical_set)} extra"
        )

    source_by_key = source.set_index(KEY_COLUMNS, verify_integrity=True)
    canonical_index = pd.MultiIndex.from_tuples(
        canonical_keys, names=KEY_COLUMNS
    )
    aligned = source_by_key.reindex(canonical_index).reset_index()
    canonical_words = canonical[canonical_word_column].astype(str).to_numpy()
    source_words = aligned["word"].astype(str).to_numpy()
    mismatch = np.flatnonzero(canonical_words != source_words)
    if len(mismatch):
        row = int(mismatch[0])
        raise ValueError(
            f"{label} word mismatch at canonical row {row}: "
            f"{source_words[row]!r} versus {canonical_words[row]!r}"
        )
    return aligned


def _direct_context_columns(columns: Iterable[str]) -> list[str]:
    matches = [
        (int(match.group(1)), column)
        for column in columns
        if (match := CONTEXT_PATTERN.fullmatch(column)) is not None
    ]
    return [column for _, column in sorted(matches)]


def _numeric_values(dataframe: pd.DataFrame, columns: Iterable[str],
                    label: str) -> dict[str, np.ndarray]:
    values = {}
    for column in columns:
        numeric = pd.to_numeric(dataframe[column], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(numeric).all():
            raise ValueError(f"{label} column {column} is not finite numeric")
        if (numeric < 0).any():
            raise ValueError(f"{label} column {column} contains negative values")
        values[column] = numeric
    return values


def _require_spillover_columns(dataframe: pd.DataFrame,
                               variables: Iterable[str]) -> None:
    missing = [
        f"{prefix}{variable}"
        for variable in variables
        for prefix in SPILLOVER_PREFIXES
        if f"{prefix}{variable}" not in dataframe.columns
    ]
    if missing:
        raise ValueError(
            "canonical joint is missing model-specific spillovers: "
            + ", ".join(missing)
        )


def _replace_direct_and_spillovers(output: pd.DataFrame,
                                   values: dict[str, np.ndarray]) -> None:
    for column, column_values in values.items():
        output[column] = column_values

    # Compute lags in key order while retaining the canonical row order.
    ordered_indices = output.sort_values(
        KEY_COLUMNS, kind="stable"
    ).index
    ordered_text_ids = output.loc[ordered_indices, "text_id"]
    for column in values:
        ordered_values = output.loc[ordered_indices, column]
        grouped = ordered_values.groupby(ordered_text_ids, sort=False)
        for lag, prefix in enumerate(SPILLOVER_PREFIXES, start=1):
            shifted = grouped.shift(lag)
            output.loc[ordered_indices, f"{prefix}{column}"] = shifted.to_numpy()


def build_model_joint(canonical: pd.DataFrame, context: pd.DataFrame,
                      reference: pd.DataFrame,
                      expected_rows: int = EXPECTED_ROWS,
                      words_per_text: int | None = None) -> pd.DataFrame:
    """Return one validated model-specific joint table."""

    if expected_rows < 1:
        raise ValueError("expected rows must be positive")
    _validate_words_per_text(words_per_text)
    original_columns = canonical.columns.tolist()
    canonical, canonical_word_column = _validate_canonical_keys(
        canonical, expected_rows, words_per_text=words_per_text
    )
    context = _validate_keys(
        context,
        "context-limited",
        expected_rows,
        text_id_offset=1,
        words_per_text=words_per_text,
    )
    reference = _validate_keys(
        reference,
        "reference surprisal",
        expected_rows,
        text_id_offset=1,
        words_per_text=words_per_text,
    )
    context = _align_source(
        canonical, context, canonical_word_column, "context-limited"
    )
    reference = _align_source(
        canonical, reference, canonical_word_column, "reference surprisal"
    )

    canonical_context_columns = _direct_context_columns(canonical.columns)
    source_context_columns = _direct_context_columns(context.columns)
    if not canonical_context_columns:
        raise ValueError("canonical joint contains no context-limited predictors")
    if source_context_columns != canonical_context_columns:
        raise ValueError(
            "context-limited predictor columns differ from canonical joint: "
            f"{source_context_columns} versus {canonical_context_columns}"
        )

    reference_columns = ["surprisal"]
    if "surprisal" not in reference.columns:
        raise ValueError("reference surprisal is missing column surprisal")
    canonical_has_buggy = "surprisal_buggy" in canonical.columns
    reference_has_buggy = "surprisal_buggy" in reference.columns
    if canonical_has_buggy != reference_has_buggy:
        raise ValueError(
            "canonical joint and reference surprisal must agree on "
            "surprisal_buggy presence"
        )
    if reference_has_buggy:
        reference_columns.append("surprisal_buggy")

    replaced_columns = canonical_context_columns + reference_columns
    _require_spillover_columns(canonical, replaced_columns)
    replacements = _numeric_values(
        context, canonical_context_columns, "context-limited"
    )
    replacements.update(_numeric_values(
        reference, reference_columns, "reference surprisal"
    ))

    output = canonical.copy()
    _replace_direct_and_spillovers(output, replacements)
    if output.columns.tolist() != original_columns:
        raise AssertionError("model-joint construction changed the canonical schema")
    return output


def read_tsv(fname: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        fname, sep="\t", keep_default_na=False, na_values=[""], low_memory=False
    )


def write_tsv_atomic(dataframe: pd.DataFrame, output_fname: str | Path) -> None:
    output_path = Path(output_fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_fname = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf8", newline="") as stream:
            dataframe.to_csv(stream, sep="\t", index=False)
        os.replace(temporary_fname, output_path)
    except Exception:
        if os.path.exists(temporary_fname):
            os.unlink(temporary_fname)
        raise


def build_model_joint_dataset(canonical_joint_fname: str | Path,
                              context_limited_surprisal_fname: str | Path,
                              reference_surprisal_fname: str | Path,
                              output_fname: str | Path,
                              expected_rows: int = EXPECTED_ROWS,
                              words_per_text: int | None = None) -> pd.DataFrame:
    """Read, build, and atomically publish one model-specific joint table."""

    input_paths = [
        Path(canonical_joint_fname).resolve(),
        Path(context_limited_surprisal_fname).resolve(),
        Path(reference_surprisal_fname).resolve(),
    ]
    output_path = Path(output_fname).resolve()
    if output_path in input_paths:
        raise ValueError("output file must not overwrite an input artifact")
    output = build_model_joint(
        read_tsv(canonical_joint_fname),
        read_tsv(context_limited_surprisal_fname),
        read_tsv(reference_surprisal_fname),
        expected_rows=expected_rows,
        words_per_text=words_per_text,
    )
    write_tsv_atomic(output, output_fname)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a model-specific canonical N+C+RT table from the definitive "
            "GPT-2-small joint artifact"
        )
    )
    parser.add_argument("--canonical-joint-fname", required=True)
    parser.add_argument("--context-limited-surprisal-fname", required=True)
    parser.add_argument("--reference-surprisal-fname", required=True)
    parser.add_argument("--output-fname", required=True)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument(
        "--words-per-text",
        type=int,
        help="optional per-passage word prefix used for a matched pilot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_model_joint_dataset(
        args.canonical_joint_fname,
        args.context_limited_surprisal_fname,
        args.reference_surprisal_fname,
        args.output_fname,
        expected_rows=args.expected_rows,
        words_per_text=args.words_per_text,
    )


if __name__ == "__main__":
    main()
