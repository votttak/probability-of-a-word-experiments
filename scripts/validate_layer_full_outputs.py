#!/usr/bin/env python3

"""Validate every canonical full-layer artifact and publish a sentinel."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from .preflight_layer_full import (
        CONTEXT_LIMITED_CONTEXTS,
        EXPECTED_ROWS,
        NGRAM_CONTEXTS,
        SHA256_PATTERN,
        ValidationError,
        sha256_file,
        validate_exact_key_word_rows,
    )
except ImportError:  # Support direct execution from scripts/.
    from preflight_layer_full import (
        CONTEXT_LIMITED_CONTEXTS,
        EXPECTED_ROWS,
        NGRAM_CONTEXTS,
        SHA256_PATTERN,
        ValidationError,
        sha256_file,
        validate_exact_key_word_rows,
    )


EXPECTED_COMPLETE_ROWS = 10_023
EXPECTED_EXCLUDED_ROWS = 233
EXPECTED_FOLDS = 10
EXPECTED_SEED = 42
EXPECTED_FINAL_LAYER = 12
EXPECTED_MODEL = "gpt2-small"
EXPECTED_LAYERS = tuple(range(1, EXPECTED_FINAL_LAYER + 1))

LAYER_PATTERN = re.compile(r"internal_layer_surprisal_layer_(\d+)$")
SPILLOVER_PREFIXES = ("prev_", "prev2_", "prev3_")

FOLD_SCORE_COLUMNS = (
    "ll_m0_mean", "ll_predictor_mean", "ll_layer_mean", "ll_joint_mean",
    "delta_predictor_given_layer_mean", "delta_layer_given_predictor_mean",
    "ll_m0_sum", "ll_predictor_sum", "ll_layer_sum", "ll_joint_sum",
    "delta_predictor_given_layer_sum", "delta_layer_given_predictor_sum",
)

DIRECTIONAL_FOLD_COLUMNS = (
    "delta_n_given_l_mean", "delta_l_given_n_mean",
    "delta_c_given_l_mean", "delta_l_given_c_mean",
    "delta_n_given_l_sum", "delta_l_given_n_sum",
    "delta_c_given_l_sum", "delta_l_given_c_sum",
)

DELTA_GENERIC_COLUMNS = (
    "delta_predictor_given_layer_mean",
    "delta_predictor_given_layer_se",
    "delta_layer_given_predictor_mean",
    "delta_layer_given_predictor_se",
)

DIRECTIONAL_DELTA_COLUMNS = (
    "delta_n_given_l_mean", "delta_n_given_l_se",
    "delta_l_given_n_mean", "delta_l_given_n_se",
    "delta_c_given_l_mean", "delta_c_given_l_se",
    "delta_l_given_c_mean", "delta_l_given_c_se",
)


def _read_tsv(fname: str | Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(fname, sep="\t", low_memory=False)
    except Exception as error:
        raise ValidationError(f"unable to read {label}: {fname}") from error


def _integer_series(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValidationError(f"{label} must contain finite integers")
    return values.astype(np.int64)


def _numeric_columns(dataframe: pd.DataFrame, columns: Iterable[str],
                     label: str, finite: bool = True) -> None:
    for column in columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
        if finite and not np.isfinite(dataframe[column].to_numpy()).all():
            raise ValidationError(f"{label} column {column} is not finite")


def _layer_columns(dataframe: pd.DataFrame, label: str,
                   expected_layers: tuple[int, ...]) -> list[str]:
    pairs = sorted(
        (int(match.group(1)), column)
        for column in dataframe.columns
        if (match := LAYER_PATTERN.fullmatch(column)) is not None
    )
    layers = tuple(layer for layer, _ in pairs)
    if layers != expected_layers:
        raise ValidationError(
            f"{label} layers are {layers}; expected {expected_layers}"
        )
    return [column for _, column in pairs]


def _assert_close(actual, expected, label: str, *, rtol: float = 1e-10,
                  atol: float = 1e-10) -> None:
    if not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True):
        actual_array = np.asarray(actual, dtype=float)
        expected_array = np.asarray(expected, dtype=float)
        differences = np.abs(actual_array - expected_array)
        finite = np.isfinite(differences)
        maximum = float(differences[finite].max()) if finite.any() else math.inf
        raise ValidationError(f"{label} mismatch; maximum absolute error {maximum}")


def _validate_canonical_preserved(canonical: pd.DataFrame,
                                  merged: pd.DataFrame) -> None:
    missing = set(canonical.columns) - set(merged.columns)
    if missing:
        raise ValidationError(
            "merged table lost canonical columns: " + ", ".join(sorted(missing))
        )
    for column in canonical.columns:
        left = canonical[column]
        right = merged[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            _assert_close(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                f"merged canonical column {column}",
                rtol=1e-12,
                atol=1e-12,
            )
        else:
            left_values = left.fillna("<NA>").astype(str).tolist()
            right_values = right.fillna("<NA>").astype(str).tolist()
            if left_values != right_values:
                raise ValidationError(
                    f"merged canonical column {column} changed"
                )


def validate_layer_tables(canonical: pd.DataFrame, internal: pd.DataFrame,
                          merged: pd.DataFrame, expected_rows: int,
                          expected_layers: tuple[int, ...]) -> tuple[list[str], list[tuple[int, int, str]]]:
    """Validate exact L coverage, values, merge preservation, and spillovers."""

    if len(canonical) != expected_rows:
        raise ValidationError(
            f"canonical joint has {len(canonical)} rows; expected {expected_rows}"
        )
    required_joint = {"text_id", "word_id", "ref_token", "surprisal"}
    missing = required_joint - set(canonical.columns)
    if missing:
        raise ValidationError(
            "canonical joint is missing columns: " + ", ".join(sorted(missing))
        )
    canonical_text_ids = _integer_series(
        canonical["text_id"], "canonical joint text_id"
    )
    canonical_word_ids = _integer_series(
        canonical["word_id"], "canonical joint word_id"
    )
    canonical_keys = pd.DataFrame({
        "text_id": canonical_text_ids,
        "word_id": canonical_word_ids,
    })
    if canonical_keys.duplicated(["text_id", "word_id"], keep=False).any():
        raise ValidationError("canonical joint contains duplicate keys")
    expected_one_based = list(zip(
        canonical_text_ids.tolist(),
        canonical_word_ids.tolist(),
        canonical["ref_token"].astype(str).tolist(),
    ))
    if expected_one_based != sorted(expected_one_based, key=lambda row: row[:2]):
        raise ValidationError("canonical joint keys are not in stable sorted order")
    expected_zero_based = [
        (text_id - 1, word_id, word)
        for text_id, word_id, word in expected_one_based
    ]

    validate_exact_key_word_rows(
        internal, expected_zero_based, "word", "internal-layer table"
    )
    validate_exact_key_word_rows(
        merged, expected_one_based, "ref_token", "merged table"
    )
    layer_columns = _layer_columns(
        internal, "internal-layer table", expected_layers
    )
    merged_layer_columns = _layer_columns(
        merged, "merged table", expected_layers
    )
    if merged_layer_columns != layer_columns:
        raise ValidationError("internal and merged layer columns differ")
    _numeric_columns(internal, layer_columns, "internal-layer")
    _numeric_columns(merged, layer_columns, "merged layer")
    if (internal[layer_columns] < 0).any().any():
        raise ValidationError("internal-layer table contains negative values")
    if (merged[layer_columns] < 0).any().any():
        raise ValidationError("merged table contains negative layer values")
    _assert_close(
        internal[layer_columns].to_numpy(),
        merged[layer_columns].to_numpy(),
        "merged direct layer values",
        rtol=1e-12,
        atol=1e-12,
    )
    _validate_canonical_preserved(canonical, merged)

    for column in layer_columns:
        grouped = merged.groupby("text_id", sort=False)[column]
        for shift, prefix in enumerate(SPILLOVER_PREFIXES, start=1):
            spillover = f"{prefix}{column}"
            if spillover not in merged.columns:
                raise ValidationError(
                    f"merged table is missing layer spillover {spillover}"
                )
            actual = pd.to_numeric(merged[spillover], errors="coerce").to_numpy()
            expected = grouped.shift(shift).to_numpy(dtype=float)
            _assert_close(
                actual, expected, f"layer spillover {spillover}",
                rtol=1e-12, atol=1e-12,
            )
    return layer_columns, expected_zero_based


def _summary_mapping(summary: pd.DataFrame) -> dict[str, str]:
    if not {"key", "value"}.issubset(summary.columns):
        raise ValidationError("summary must contain key and value columns")
    if summary["key"].duplicated(keep=False).any():
        raise ValidationError("summary contains duplicate keys")
    return {
        str(key): str(value)
        for key, value in zip(summary["key"], summary["value"])
    }


def _parse_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"summary {label} is not an integer") from error
    return parsed


def _parse_csv_ints(value: str, label: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as error:
        raise ValidationError(f"summary {label} is not an integer list") from error


def validate_summary(summary_fname: str | Path, expected_rows: int,
                     expected_complete_rows: int, expected_excluded_rows: int,
                     expected_folds: int, expected_seed: int,
                     expected_layers: tuple[int, ...],
                     expected_model: str) -> dict[str, str]:
    summary = pd.read_csv(
        summary_fname, sep="\t", dtype=str, keep_default_na=False
    )
    values = _summary_mapping(summary)
    required = {
        "input_rows", "complete_case_rows", "excluded_rows", "folds", "seed",
        "ngram_contexts", "context_limited_contexts", "internal_layers",
        "layer_decoder", "model",
    }
    missing = required - set(values)
    if missing:
        raise ValidationError(
            "summary is missing keys: " + ", ".join(sorted(missing))
        )
    observed = {
        "input_rows": _parse_int(values["input_rows"], "input_rows"),
        "complete_case_rows": _parse_int(
            values["complete_case_rows"], "complete_case_rows"
        ),
        "excluded_rows": _parse_int(values["excluded_rows"], "excluded_rows"),
        "folds": _parse_int(values["folds"], "folds"),
        "seed": _parse_int(values["seed"], "seed"),
    }
    expected = {
        "input_rows": expected_rows,
        "complete_case_rows": expected_complete_rows,
        "excluded_rows": expected_excluded_rows,
        "folds": expected_folds,
        "seed": expected_seed,
    }
    if observed != expected:
        raise ValidationError(
            f"summary counts/settings are {observed}; expected {expected}"
        )
    if observed["complete_case_rows"] + observed["excluded_rows"] != expected_rows:
        raise ValidationError("summary complete and excluded rows do not add to input")
    if _parse_csv_ints(values["ngram_contexts"], "ngram_contexts") != NGRAM_CONTEXTS:
        raise ValidationError("summary n-gram contexts are not 0..4")
    if (
        _parse_csv_ints(
            values["context_limited_contexts"], "context_limited_contexts"
        ) != CONTEXT_LIMITED_CONTEXTS
    ):
        raise ValidationError("summary context-limited contexts are not 1..4")
    if _parse_csv_ints(values["internal_layers"], "internal_layers") != expected_layers:
        raise ValidationError("summary internal layers do not match expected layers")
    if values["layer_decoder"] != "logit lens":
        raise ValidationError("summary layer_decoder must be 'logit lens'")
    if values["model"] != expected_model:
        raise ValidationError(
            f"summary model is {values['model']!r}; expected {expected_model!r}"
        )
    return values


def _expected_combinations(expected_layers: tuple[int, ...],
                           folds: Iterable[int] | None = None) -> set[tuple]:
    records = set()
    fold_values = tuple(folds) if folds is not None else (None,)
    for family, comparison, contexts in (
        ("ngram", "ngram_vs_internal_layer", NGRAM_CONTEXTS),
        (
            "context_limited",
            "context_limited_vs_internal_layer",
            CONTEXT_LIMITED_CONTEXTS,
        ),
    ):
        for context in contexts:
            for layer in expected_layers:
                for fold in fold_values:
                    base = (comparison, family, context, layer)
                    records.add(base if fold is None else base + (fold,))
    return records


def _validate_family_context_columns(dataframe: pd.DataFrame, label: str) -> None:
    ngram = dataframe["predictor_family"] == "ngram"
    context = dataframe["predictor_family"] == "context_limited"
    if not (ngram | context).all():
        raise ValidationError(f"{label} contains an unknown predictor family")
    if not (
        dataframe.loc[ngram, "ngram_context"].to_numpy()
        == dataframe.loc[ngram, "predictor_context"].to_numpy()
    ).all():
        raise ValidationError(f"{label} ngram_context does not match predictor_context")
    if not dataframe.loc[context, "ngram_context"].isna().all():
        raise ValidationError(f"{label} context rows must have NA ngram_context")
    if not (
        dataframe.loc[context, "context_limited_context"].to_numpy()
        == dataframe.loc[context, "predictor_context"].to_numpy()
    ).all():
        raise ValidationError(
            f"{label} context_limited_context does not match predictor_context"
        )
    if not dataframe.loc[ngram, "context_limited_context"].isna().all():
        raise ValidationError(
            f"{label} n-gram rows must have NA context_limited_context"
        )


def _validate_directional_values(dataframe: pd.DataFrame, label: str,
                                 suffixes: tuple[str, ...]) -> None:
    ngram = dataframe["predictor_family"] == "ngram"
    context = dataframe["predictor_family"] == "context_limited"
    for suffix in suffixes:
        n_columns = [f"delta_n_given_l_{suffix}", f"delta_l_given_n_{suffix}"]
        c_columns = [f"delta_c_given_l_{suffix}", f"delta_l_given_c_{suffix}"]
        for column in n_columns:
            if not np.isfinite(dataframe.loc[ngram, column].to_numpy()).all():
                raise ValidationError(f"{label} {column} is not finite on n-gram rows")
            if not dataframe.loc[context, column].isna().all():
                raise ValidationError(f"{label} {column} must be NA on context rows")
        for column in c_columns:
            if not np.isfinite(dataframe.loc[context, column].to_numpy()).all():
                raise ValidationError(f"{label} {column} is not finite on context rows")
            if not dataframe.loc[ngram, column].isna().all():
                raise ValidationError(f"{label} {column} must be NA on n-gram rows")


def validate_fold_results(fold_fname: str | Path, expected_complete_rows: int,
                          expected_folds: int,
                          expected_layers: tuple[int, ...],
                          expected_model: str) -> pd.DataFrame:
    folds = _read_tsv(fold_fname, "fold results")
    required = {
        "model", "comparison", "predictor_family", "predictor_context",
        "ngram_context", "context_limited_context", "layer", "fold",
        "n_train", "n_test", *FOLD_SCORE_COLUMNS, *DIRECTIONAL_FOLD_COLUMNS,
    }
    missing = required - set(folds.columns)
    if missing:
        raise ValidationError(
            "fold results are missing columns: " + ", ".join(sorted(missing))
        )
    if not (folds["model"] == expected_model).all():
        raise ValidationError(
            f"fold results model values must all be {expected_model!r}"
        )
    integer_columns = ("predictor_context", "layer", "fold", "n_train", "n_test")
    for column in integer_columns:
        folds[column] = _integer_series(folds[column], f"fold results {column}")
    for column in ("ngram_context", "context_limited_context"):
        folds[column] = pd.to_numeric(folds[column], errors="coerce")
    _numeric_columns(folds, FOLD_SCORE_COLUMNS, "fold results")
    _numeric_columns(folds, DIRECTIONAL_FOLD_COLUMNS, "fold results", finite=False)

    expected_keys = _expected_combinations(
        expected_layers, range(1, expected_folds + 1)
    )
    actual_keys = set(map(tuple, folds[[
        "comparison", "predictor_family", "predictor_context", "layer", "fold"
    ]].itertuples(index=False, name=None)))
    if len(folds) != len(expected_keys) or actual_keys != expected_keys:
        raise ValidationError(
            f"fold result combinations are incomplete: got {len(folds)} rows "
            f"and {len(actual_keys)} unique keys; expected {len(expected_keys)}"
        )
    if folds.duplicated([
        "comparison", "predictor_family", "predictor_context", "layer", "fold"
    ], keep=False).any():
        raise ValidationError("fold results contain duplicate comparison keys")
    _validate_family_context_columns(folds, "fold results")
    _validate_directional_values(folds, "fold results", ("mean", "sum"))

    sizes = folds[["fold", "n_train", "n_test"]].drop_duplicates()
    if len(sizes) != expected_folds or sizes["fold"].nunique() != expected_folds:
        raise ValidationError("fold train/test sizes are inconsistent across models")
    if not (sizes["n_train"] + sizes["n_test"] == expected_complete_rows).all():
        raise ValidationError("fold n_train + n_test does not equal complete rows")
    if int(sizes["n_test"].sum()) != expected_complete_rows:
        raise ValidationError("fold test sets do not partition complete rows")

    mean_pairs = (
        ("ll_m0_mean", "ll_m0_sum"),
        ("ll_predictor_mean", "ll_predictor_sum"),
        ("ll_layer_mean", "ll_layer_sum"),
        ("ll_joint_mean", "ll_joint_sum"),
        ("delta_predictor_given_layer_mean", "delta_predictor_given_layer_sum"),
        ("delta_layer_given_predictor_mean", "delta_layer_given_predictor_sum"),
    )
    for mean_column, sum_column in mean_pairs:
        _assert_close(
            folds[sum_column].to_numpy(),
            folds[mean_column].to_numpy() * folds["n_test"].to_numpy(),
            f"fold {sum_column} versus {mean_column} * n_test",
            rtol=1e-10,
            atol=1e-8,
        )
    _assert_close(
        folds["delta_predictor_given_layer_mean"],
        folds["ll_joint_mean"] - folds["ll_layer_mean"],
        "fold predictor-given-layer mean identity",
    )
    _assert_close(
        folds["delta_layer_given_predictor_mean"],
        folds["ll_joint_mean"] - folds["ll_predictor_mean"],
        "fold layer-given-predictor mean identity",
    )
    _assert_close(
        folds["delta_predictor_given_layer_sum"],
        folds["ll_joint_sum"] - folds["ll_layer_sum"],
        "fold predictor-given-layer sum identity",
        atol=1e-8,
    )
    _assert_close(
        folds["delta_layer_given_predictor_sum"],
        folds["ll_joint_sum"] - folds["ll_predictor_sum"],
        "fold layer-given-predictor sum identity",
        atol=1e-8,
    )

    if folds.groupby("fold")["ll_m0_mean"].nunique().max() != 1:
        raise ValidationError("M0 score varies within a fold")
    if folds.groupby([
        "predictor_family", "predictor_context", "fold"
    ])["ll_predictor_mean"].nunique().max() != 1:
        raise ValidationError("single-predictor score varies across layers")
    if folds.groupby(["layer", "fold"])["ll_layer_mean"].nunique().max() != 1:
        raise ValidationError("single-layer score varies across comparisons")
    return folds


def validate_conditional_deltas(delta_fname: str | Path, folds: pd.DataFrame,
                                expected_folds: int,
                                expected_layers: tuple[int, ...],
                                expected_model: str) -> pd.DataFrame:
    deltas = _read_tsv(delta_fname, "conditional deltas")
    required = {
        "model", "comparison", "predictor_family", "predictor_context",
        "ngram_context", "context_limited_context", "layer", "folds",
        *DELTA_GENERIC_COLUMNS, *DIRECTIONAL_DELTA_COLUMNS,
    }
    missing = required - set(deltas.columns)
    if missing:
        raise ValidationError(
            "conditional deltas are missing columns: "
            + ", ".join(sorted(missing))
        )
    if not (deltas["model"] == expected_model).all():
        raise ValidationError(
            f"conditional deltas model values must all be {expected_model!r}"
        )
    for column in ("predictor_context", "layer", "folds"):
        deltas[column] = _integer_series(
            deltas[column], f"conditional deltas {column}"
        )
    for column in ("ngram_context", "context_limited_context"):
        deltas[column] = pd.to_numeric(deltas[column], errors="coerce")
    _numeric_columns(deltas, DELTA_GENERIC_COLUMNS, "conditional deltas")
    _numeric_columns(
        deltas, DIRECTIONAL_DELTA_COLUMNS, "conditional deltas", finite=False
    )

    expected_keys = _expected_combinations(expected_layers)
    actual_keys = set(map(tuple, deltas[[
        "comparison", "predictor_family", "predictor_context", "layer"
    ]].itertuples(index=False, name=None)))
    if len(deltas) != len(expected_keys) or actual_keys != expected_keys:
        raise ValidationError(
            f"conditional delta combinations are incomplete: got {len(deltas)} "
            f"rows and {len(actual_keys)} unique keys; expected {len(expected_keys)}"
        )
    if not (deltas["folds"] == expected_folds).all():
        raise ValidationError("conditional delta fold counts are incorrect")
    _validate_family_context_columns(deltas, "conditional deltas")
    _validate_directional_values(deltas, "conditional deltas", ("mean", "se"))

    grouped = folds.groupby(
        ["comparison", "predictor_family", "predictor_context", "layer"],
        sort=False,
    )
    expected_rows = {}
    for key, group in grouped:
        predictor_values = group["delta_predictor_given_layer_mean"]
        layer_values = group["delta_layer_given_predictor_mean"]
        expected_rows[key] = (
            float(predictor_values.mean()),
            float(predictor_values.std(ddof=1) / math.sqrt(len(group))),
            float(layer_values.mean()),
            float(layer_values.std(ddof=1) / math.sqrt(len(group))),
        )
    for row in deltas.itertuples(index=False):
        key = (
            row.comparison,
            row.predictor_family,
            int(row.predictor_context),
            int(row.layer),
        )
        actual = (
            row.delta_predictor_given_layer_mean,
            row.delta_predictor_given_layer_se,
            row.delta_layer_given_predictor_mean,
            row.delta_layer_given_predictor_se,
        )
        _assert_close(actual, expected_rows[key], f"conditional aggregate {key}")
    return deltas


def _load_anchor(anchor_fname: str | Path) -> dict:
    try:
        with open(anchor_fname, "r", encoding="utf8") as input_file:
            report = json.load(input_file)
    except Exception as error:
        raise ValidationError(f"unable to read anchor JSON: {anchor_fname}") from error
    required = {
        "validated", "reference_fname", "reference_sha256", "final_layer",
        "rows", "max_abs_difference", "mean_abs_difference", "tolerance",
    }
    missing = required - set(report)
    if missing:
        raise ValidationError(
            "anchor JSON is missing keys: " + ", ".join(sorted(missing))
        )
    return report


def _scalar_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValidationError(
            f"{label} is {actual}; recomputed value is {expected}"
        )


def validate_anchor(anchor_fname: str | Path, internal: pd.DataFrame,
                    merged: pd.DataFrame,
                    expected_zero_based: list[tuple[int, int, str]],
                    expected_rows: int, expected_final_layer: int) -> dict:
    """Validate scorer-reference and independently recomputed merged anchors."""

    report = _load_anchor(anchor_fname)
    if report["validated"] is not True:
        raise ValidationError("anchor JSON is not marked validated")
    try:
        final_layer = int(report["final_layer"])
        rows = int(report["rows"])
        tolerance = float(report["tolerance"])
        reported_max = float(report["max_abs_difference"])
        reported_mean = float(report["mean_abs_difference"])
        reported_p99 = (
            float(report["p99_abs_difference"])
            if "p99_abs_difference" in report else None
        )
    except (TypeError, ValueError) as error:
        raise ValidationError("anchor JSON contains invalid numeric values") from error
    if final_layer != expected_final_layer:
        raise ValidationError(
            f"anchor final layer is {final_layer}; expected {expected_final_layer}"
        )
    if rows != expected_rows:
        raise ValidationError(f"anchor has {rows} rows; expected {expected_rows}")
    values = (tolerance, reported_max, reported_mean)
    if reported_p99 is not None:
        values += (reported_p99,)
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValidationError("anchor tolerance/differences must be finite and nonnegative")
    if reported_max > tolerance or reported_mean > reported_max + 1e-15:
        raise ValidationError("anchor JSON reports an invalid or failed tolerance check")
    if reported_p99 is not None and reported_p99 > reported_max + 1e-15:
        raise ValidationError("anchor p99_abs_difference exceeds max_abs_difference")
    reference_sha = str(report["reference_sha256"]).lower()
    if not SHA256_PATTERN.fullmatch(reference_sha):
        raise ValidationError("anchor reference_sha256 is not a SHA-256 digest")
    reference_fname = Path(str(report["reference_fname"]))
    if not reference_fname.is_file():
        raise ValidationError(f"anchor reference file is unavailable: {reference_fname}")
    actual_reference_sha = sha256_file(reference_fname)
    if actual_reference_sha != reference_sha:
        raise ValidationError(
            "anchor reference SHA-256 does not match its reference file"
        )

    reference = _read_tsv(reference_fname, "anchor reference surprisal")
    validate_exact_key_word_rows(
        reference,
        expected_zero_based,
        "word",
        "anchor reference surprisal",
    )
    if "surprisal" not in reference.columns:
        raise ValidationError("anchor reference lacks column surprisal")
    reference_values = pd.to_numeric(
        reference["surprisal"], errors="coerce"
    ).to_numpy()
    if not np.isfinite(reference_values).all():
        raise ValidationError("anchor reference surprisal is not finite")
    final_column = f"internal_layer_surprisal_layer_{expected_final_layer}"
    internal_values = pd.to_numeric(
        internal[final_column], errors="coerce"
    ).to_numpy()
    reference_differences = np.abs(internal_values - reference_values)
    recomputed_reference_max = float(reference_differences.max())
    recomputed_reference_mean = float(reference_differences.mean())
    # Match the scorer's nearest-rank definition (ceil(0.99 * n)).
    recomputed_reference_p99 = float(
        np.quantile(reference_differences, 0.99, method="higher")
    )
    _scalar_close(reported_max, recomputed_reference_max, "anchor max_abs_difference")
    _scalar_close(
        reported_mean, recomputed_reference_mean, "anchor mean_abs_difference"
    )
    if reported_p99 is not None:
        _scalar_close(
            reported_p99,
            recomputed_reference_p99,
            "anchor p99_abs_difference",
        )
    if recomputed_reference_max > tolerance:
        raise ValidationError("recomputed scorer-reference anchor exceeds tolerance")

    merged_reference = pd.to_numeric(
        merged["surprisal"], errors="coerce"
    ).to_numpy()
    merged_values = pd.to_numeric(
        merged[final_column], errors="coerce"
    ).to_numpy()
    if not np.isfinite(merged_reference).all():
        raise ValidationError("merged ordinary surprisal is not finite")
    merged_differences = np.abs(merged_values - merged_reference)
    merged_max = float(merged_differences.max())
    merged_mean = float(merged_differences.mean())
    merged_p99 = float(
        np.quantile(merged_differences, 0.99, method="higher")
    )
    if merged_max > tolerance:
        raise ValidationError(
            f"merged final-layer anchor {merged_max} exceeds tolerance {tolerance}"
        )
    return {
        "final_layer": expected_final_layer,
        "rows": expected_rows,
        "tolerance": tolerance,
        "scorer_reference_max_abs_difference": recomputed_reference_max,
        "scorer_reference_mean_abs_difference": recomputed_reference_mean,
        "scorer_reference_p99_abs_difference": recomputed_reference_p99,
        "merged_max_abs_difference": merged_max,
        "merged_mean_abs_difference": merged_mean,
        "merged_p99_abs_difference": merged_p99,
        "reference_fname": str(reference_fname.resolve()),
        "reference_sha256": actual_reference_sha,
    }


def _artifact_record(fname: str | Path) -> dict:
    path = Path(fname)
    return {
        "fname": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json_atomic_if_changed(data: dict, fname: str | Path) -> None:
    """Atomically publish deterministic JSON without changing an equal file."""

    output_path = Path(fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf8")
    if output_path.is_file() and output_path.read_bytes() == serialized:
        return
    descriptor, temporary_fname = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output_file:
            output_file.write(serialized)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_fname, output_path)
    except Exception:
        if os.path.exists(temporary_fname):
            os.unlink(temporary_fname)
        raise


def validate_outputs(canonical_joint_fname: str | Path,
                     internal_layer_fname: str | Path,
                     merged_data_fname: str | Path,
                     summary_fname: str | Path,
                     fold_results_fname: str | Path,
                     conditional_deltas_fname: str | Path,
                     anchor_json_fname: str | Path,
                     completion_json_fname: str | Path,
                     expected_rows: int = EXPECTED_ROWS,
                     expected_complete_rows: int = EXPECTED_COMPLETE_ROWS,
                     expected_excluded_rows: int = EXPECTED_EXCLUDED_ROWS,
                     expected_folds: int = EXPECTED_FOLDS,
                     expected_seed: int = EXPECTED_SEED,
                     expected_final_layer: int = EXPECTED_FINAL_LAYER,
                     expected_model: str = EXPECTED_MODEL) -> dict:
    """Validate the full output set, then atomically publish its manifest."""

    expected_layers = tuple(range(1, expected_final_layer + 1))
    canonical = _read_tsv(canonical_joint_fname, "canonical joint")
    internal = _read_tsv(internal_layer_fname, "internal layers")
    merged = _read_tsv(merged_data_fname, "merged layer data")
    _, expected_zero_based = validate_layer_tables(
        canonical, internal, merged, expected_rows, expected_layers
    )
    validate_summary(
        summary_fname,
        expected_rows,
        expected_complete_rows,
        expected_excluded_rows,
        expected_folds,
        expected_seed,
        expected_layers,
        expected_model,
    )
    folds = validate_fold_results(
        fold_results_fname,
        expected_complete_rows,
        expected_folds,
        expected_layers,
        expected_model,
    )
    deltas = validate_conditional_deltas(
        conditional_deltas_fname,
        folds,
        expected_folds,
        expected_layers,
        expected_model,
    )
    anchor = validate_anchor(
        anchor_json_fname,
        internal,
        merged,
        expected_zero_based,
        expected_rows,
        expected_final_layer,
    )

    artifacts = {
        "canonical_joint": _artifact_record(canonical_joint_fname),
        "internal_layer": _artifact_record(internal_layer_fname),
        "merged_data": _artifact_record(merged_data_fname),
        "summary": _artifact_record(summary_fname),
        "fold_results": _artifact_record(fold_results_fname),
        "conditional_deltas": _artifact_record(conditional_deltas_fname),
        "anchor_json": _artifact_record(anchor_json_fname),
        "anchor_reference": _artifact_record(anchor["reference_fname"]),
    }
    completion = {
        "schema_version": 1,
        "validated": True,
        "model": expected_model,
        "counts": {
            "input_rows": expected_rows,
            "complete_case_rows": expected_complete_rows,
            "excluded_rows": expected_excluded_rows,
            "folds": expected_folds,
            "fold_result_rows": len(folds),
            "conditional_delta_rows": len(deltas),
            "internal_layers": list(expected_layers),
        },
        "anchor": anchor,
        "artifacts": artifacts,
    }
    write_json_atomic_if_changed(completion, completion_json_fname)
    return completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a canonical full internal-layer result set"
    )
    parser.add_argument("--canonical-joint-fname", required=True)
    parser.add_argument("--internal-layer-fname", required=True)
    parser.add_argument("--merged-data-fname", required=True)
    parser.add_argument("--summary-fname", required=True)
    parser.add_argument("--fold-results-fname", required=True)
    parser.add_argument("--conditional-deltas-fname", required=True)
    parser.add_argument("--anchor-json-fname", required=True)
    parser.add_argument("--completion-json-fname", required=True)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument(
        "--expected-complete-rows", type=int, default=EXPECTED_COMPLETE_ROWS
    )
    parser.add_argument(
        "--expected-excluded-rows", type=int, default=EXPECTED_EXCLUDED_ROWS
    )
    parser.add_argument("--expected-folds", type=int, default=EXPECTED_FOLDS)
    parser.add_argument("--expected-seed", type=int, default=EXPECTED_SEED)
    parser.add_argument(
        "--expected-final-layer", type=int, default=EXPECTED_FINAL_LAYER
    )
    parser.add_argument("--expected-model", default=EXPECTED_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    completion = validate_outputs(
        args.canonical_joint_fname,
        args.internal_layer_fname,
        args.merged_data_fname,
        args.summary_fname,
        args.fold_results_fname,
        args.conditional_deltas_fname,
        args.anchor_json_fname,
        args.completion_json_fname,
        expected_rows=args.expected_rows,
        expected_complete_rows=args.expected_complete_rows,
        expected_excluded_rows=args.expected_excluded_rows,
        expected_folds=args.expected_folds,
        expected_seed=args.expected_seed,
        expected_final_layer=args.expected_final_layer,
        expected_model=args.expected_model,
    )
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
