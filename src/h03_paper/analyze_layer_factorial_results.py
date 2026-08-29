#!/usr/bin/env python3

"""Combine and report the context by score by decoder layer factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from h01_data.layer_factorial_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_layer_factorial_config,
)


CONTEXT_ORDER = ("passage", "sentence")
LENS_ORDER = ("logit-lens", "tuned-lens")
SCORE_ORDER = ("corrected", "buggy")
DEFAULT_EARLY_THRESHOLD = 0.2
KEY_COLUMNS = (
    "response_column",
    "context_unit",
    "lens_method",
    "score_kind",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_atomic(text, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf8") as handle:
            handle.write(text)
        os.replace(temporary, output_path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def write_tsv_atomic(dataframe, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf8", newline=""
        ) as handle:
            dataframe.to_csv(handle, sep="\t", index=False)
        os.replace(temporary, output_path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def read_results(paths):
    frames = []
    for path in paths:
        frame = pd.read_csv(path, sep="\t", keep_default_na=False)
        frame["_source_path"] = str(Path(path).resolve())
        frame["_source_sha256"] = sha256_file(path)
        frames.append(frame)
    if not frames:
        raise ValueError("at least one layer-results file is required")
    return pd.concat(frames, ignore_index=True, sort=False)


def _ordered(
    dataframe,
    *,
    contexts=CONTEXT_ORDER,
    lenses=LENS_ORDER,
    score_kinds=SCORE_ORDER,
    response_columns=None,
):
    output = dataframe.copy()
    responses = (
        tuple(response_columns)
        if response_columns is not None
        else tuple(sorted(output["response_column"].drop_duplicates()))
    )
    orders = {
        "response_column": responses,
        "context_unit": tuple(contexts),
        "lens_method": tuple(lenses),
        "score_kind": tuple(score_kinds),
    }
    for column, values in orders.items():
        output[f"_{column}_order"] = output[column].map(
            {value: index for index, value in enumerate(values)}
        )
    output.sort_values(
        [
            "_response_column_order",
            "_context_unit_order",
            "_lens_method_order",
            "_score_kind_order",
            "layer",
        ],
        kind="stable",
        inplace=True,
    )
    return output.drop(
        columns=[f"_{column}_order" for column in orders]
    ).reset_index(drop=True)


def _require_single_value(dataframe, column, scope):
    values = dataframe[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"inconsistent {column} for {scope}")
    return values.iloc[0]


def _coerce_nonnegative_integers(dataframe, columns):
    for column in columns:
        values = pd.to_numeric(dataframe[column], errors="raise")
        if not values.map(math.isfinite).all():
            raise ValueError(f"{column} contains non-finite values")
        if not values.map(lambda value: value == math.floor(value)).all():
            raise ValueError(f"{column} must contain integers")
        if (values < 0).any():
            raise ValueError(f"{column} must contain non-negative integers")
        dataframe[column] = values.map(int)


def validate_and_select_best(
    layer_results,
    *,
    contexts=CONTEXT_ORDER,
    lenses=LENS_ORDER,
    score_kinds=SCORE_ORDER,
    response_columns=None,
    early_threshold=DEFAULT_EARLY_THRESHOLD,
):
    contexts = tuple(contexts)
    lenses = tuple(lenses)
    score_kinds = tuple(score_kinds)
    if not contexts or not lenses or not score_kinds:
        raise ValueError("factor selections must be nonempty")
    if not math.isfinite(early_threshold) or not 0 < early_threshold <= 1:
        raise ValueError("early_threshold must be finite and in (0, 1]")
    required = {
        "response_column",
        "context_unit",
        "lens_method",
        "score_kind",
        "model",
        "analysis_mode",
        "layer",
        "min_layer",
        "max_layer",
        "delta_ll",
        "ppp_x1000",
        "input_rows",
        "analysis_rows",
    }
    missing = required - set(layer_results.columns)
    if missing:
        raise ValueError(
            "layer results lack columns: " + ", ".join(sorted(missing))
        )
    if layer_results.empty:
        raise ValueError("layer results contain no rows")
    layer_results = layer_results.copy()

    for column in ("response_column", "model", "analysis_mode"):
        invalid = layer_results[column].map(
            lambda value: pd.isna(value) or not str(value).strip()
        )
        if invalid.any():
            raise ValueError(f"{column} contains missing or empty values")
    for column, expected in (
        ("context_unit", contexts),
        ("lens_method", lenses),
        ("score_kind", score_kinds),
    ):
        invalid_rows = layer_results[column].map(
            lambda value: pd.isna(value) or value not in expected
        )
        invalid = sorted(
            {str(value) for value in layer_results.loc[invalid_rows, column]}
        )
        if invalid:
            raise ValueError(
                f"invalid {column} values: {invalid}"
            )

    _coerce_nonnegative_integers(
        layer_results,
        ("layer", "min_layer", "max_layer", "input_rows", "analysis_rows"),
    )
    if (layer_results["max_layer"] == 0).any():
        raise ValueError("max_layer must be positive")
    if (layer_results["analysis_rows"] == 0).any():
        raise ValueError("analysis_rows must be positive")
    if (layer_results["input_rows"] == 0).any():
        raise ValueError("input_rows must be positive")
    if (layer_results["analysis_rows"] > layer_results["input_rows"]).any():
        raise ValueError("analysis_rows cannot exceed input_rows")

    for column in ("delta_ll", "ppp_x1000"):
        layer_results[column] = pd.to_numeric(
            layer_results[column], errors="raise"
        )
        if not layer_results[column].map(math.isfinite).all():
            raise ValueError(f"{column} contains non-finite values")
    duplicate = layer_results.duplicated(
        list(KEY_COLUMNS) + ["layer"], keep=False
    )
    if duplicate.any():
        raise ValueError("duplicate condition/layer rows in results")

    _require_single_value(layer_results, "model", "factorial experiment")
    _require_single_value(
        layer_results, "analysis_mode", "factorial experiment"
    )

    best_rows = []
    reference_layer_range = None
    for key, group in layer_results.groupby(
        list(KEY_COLUMNS), sort=False, dropna=False
    ):
        scope = f"condition {key}"
        minimum = _require_single_value(group, "min_layer", scope)
        maximum = _require_single_value(group, "max_layer", scope)
        if minimum > maximum:
            raise ValueError(f"min_layer exceeds max_layer for {scope}")
        layers = sorted(group["layer"].tolist())
        expected_layers = list(range(minimum, maximum + 1))
        if layers != expected_layers:
            raise ValueError(f"incomplete layer range for {scope}")
        layer_range = (minimum, maximum)
        if reference_layer_range is None:
            reference_layer_range = layer_range
        elif layer_range != reference_layer_range:
            raise ValueError(
                "inconsistent layer ranges across factorial cells"
            )

        # R's which.max selects the first occurrence of the exact maximum.
        # Do not merge merely close floating-point values into a tie.
        best_position = int(group["delta_ll"].to_numpy().argmax())
        best_rows.append(group.iloc[[best_position]])

    for response, group in layer_results.groupby(
        "response_column", sort=False, dropna=False
    ):
        scope = f"response {response!r}"
        _require_single_value(group, "input_rows", scope)
        _require_single_value(group, "analysis_rows", scope)

    best = pd.concat(best_rows, ignore_index=True)
    best["layer_fraction_recomputed"] = (
        best["layer"] / best["max_layer"]
    )
    best["best_in_first_20pct"] = (
        best["layer_fraction_recomputed"] <= early_threshold
    )

    expected_per_response = (
        len(contexts) * len(lenses) * len(score_kinds)
    )
    if response_columns is not None:
        observed_responses = set(best["response_column"])
        if observed_responses != set(response_columns):
            raise ValueError(
                "response columns differ from the selected configuration"
            )
    for response, group in best.groupby("response_column"):
        observed = set(
            zip(
                group["context_unit"],
                group["lens_method"],
                group["score_kind"],
            )
        )
        expected = {
            (context, lens, score)
            for context in contexts
            for lens in lenses
            for score in score_kinds
        }
        if observed != expected or len(group) != expected_per_response:
            raise ValueError(
                f"response {response!r} does not contain the selected "
                f"{expected_per_response}-cell grid"
            )
    return (
        _ordered(
            layer_results,
            contexts=contexts,
            lenses=lenses,
            score_kinds=score_kinds,
            response_columns=response_columns,
        ),
        _ordered(
            best,
            contexts=contexts,
            lenses=lenses,
            score_kinds=score_kinds,
            response_columns=response_columns,
        ),
    )


def format_number(value):
    return f"{float(value):.6g}"


def make_report(
    best,
    *,
    title,
    note,
    early_threshold=DEFAULT_EARLY_THRESHOLD,
):
    models = sorted(set(best["model"]))
    modes = sorted(set(best["analysis_mode"]))
    responses = list(dict.fromkeys(best["response_column"]))
    lines = [
        f"# {title}",
        "",
        (
            f"Model: {', '.join(models)}. Analysis: {', '.join(modes)}. "
            f"Responses: {', '.join(responses)}."
        ),
        "",
    ]
    if note:
        lines.extend([note, ""])

    early_label = f"Early <= {100 * early_threshold:g}% depth"
    for response in responses:
        subset = best.loc[best["response_column"] == response]
        early = int(subset["best_in_first_20pct"].sum())
        lines.extend([
            f"## Response: {response}",
            "",
            (
                f"{early} of {len(subset)} factorial cells select a layer "
                f"at or before {100 * early_threshold:g}% of model depth "
                f"(layer / D <= {early_threshold:g})."
            ),
            "",
            "| Context | Decoder | Score | Best layer | Layer / D | "
            f"Delta LL | PPP x1000 | {early_label} |",
            "|---|---|---|---:|---:|---:|---:|:---:|",
        ])
        for row in subset.itertuples(index=False):
            lines.append(
                "| "
                + " | ".join([
                    str(row.context_unit),
                    str(row.lens_method),
                    str(row.score_kind),
                    str(int(row.layer)),
                    format_number(row.layer_fraction_recomputed),
                    format_number(row.delta_ll),
                    format_number(row.ppp_x1000),
                    "yes" if row.best_in_first_20pct else "no",
                ])
                + " |"
            )
        lines.append("")

    lines.extend([
        "## Interpretation guardrail",
        "",
        (
            "This is a factorial diagnostic. A small pivot identifies code-path "
            "and directional differences; it is not a stable estimate of the "
            "full-corpus best layer. Confirm any apparent reproduction on all "
            "10,256 Natural Stories words and additional compatible models."
        ),
        "",
    ])
    return "\n".join(lines)


def analyze(
    layer_result_paths,
    output_layer_path,
    output_best_path,
    output_report_path,
    output_json_path,
    *,
    title="Internal-layer factorial experiment",
    note="",
    contexts=CONTEXT_ORDER,
    lenses=LENS_ORDER,
    score_kinds=SCORE_ORDER,
    response_columns=None,
    early_threshold=DEFAULT_EARLY_THRESHOLD,
):
    layers = read_results(layer_result_paths)
    layers, best = validate_and_select_best(
        layers,
        contexts=contexts,
        lenses=lenses,
        score_kinds=score_kinds,
        response_columns=response_columns,
        early_threshold=early_threshold,
    )
    write_tsv_atomic(layers, output_layer_path)
    write_tsv_atomic(best, output_best_path)
    write_text_atomic(
        make_report(
            best,
            title=title,
            note=note,
            early_threshold=early_threshold,
        ),
        output_report_path,
    )
    counts = {
        response: {
            "cells": len(group),
            "best_in_first_20pct": int(
                group["best_in_first_20pct"].sum()
            ),
        }
        for response, group in best.groupby("response_column")
    }
    payload = {
        "schema_version": 1,
        "models": sorted(set(best["model"])),
        "analysis_modes": sorted(set(best["analysis_mode"])),
        "layer_rows": len(layers),
        "factorial_cells": len(best),
        "early_layer_threshold": early_threshold,
        "by_response": counts,
        "inputs": [
            {
                "path": str(Path(path).resolve()),
                "sha256": sha256_file(path),
            }
            for path in layer_result_paths
        ],
    }
    write_text_atomic(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        output_json_path,
    )
    return layers, best, payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine and report internal-layer factorial results"
    )
    parser.add_argument(
        "--layer-results-fnames", nargs="+", required=True
    )
    parser.add_argument("--output-layer-results-fname", required=True)
    parser.add_argument("--output-best-layers-fname", required=True)
    parser.add_argument("--output-report-fname", required=True)
    parser.add_argument("--output-summary-json-fname", required=True)
    parser.add_argument(
        "--title", default="Internal-layer factorial experiment"
    )
    parser.add_argument("--note", default="")
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH)
    )
    parser.add_argument("--contexts", nargs="+", choices=CONTEXT_ORDER)
    parser.add_argument("--lens-methods", nargs="+", choices=LENS_ORDER)
    parser.add_argument("--score-kinds", nargs="+", choices=SCORE_ORDER)
    parser.add_argument("--response-columns", nargs="+")
    parser.add_argument("--early-layer-threshold", type=float)
    args = parser.parse_args()
    config = load_layer_factorial_config(args.config)
    args.contexts = list(args.contexts or config.switches.contexts)
    args.lens_methods = list(
        args.lens_methods or config.switches.lens_methods
    )
    args.score_kinds = list(
        args.score_kinds or config.switches.score_kinds
    )
    args.response_columns = list(
        args.response_columns or config.switches.responses
    )
    if args.early_layer_threshold is None:
        args.early_layer_threshold = (
            config.analysis.early_layer_threshold
        )
    return args


def main():
    args = parse_args()
    analyze(
        args.layer_results_fnames,
        args.output_layer_results_fname,
        args.output_best_layers_fname,
        args.output_report_fname,
        args.output_summary_json_fname,
        title=args.title,
        note=args.note,
        contexts=args.contexts,
        lenses=args.lens_methods,
        score_kinds=args.score_kinds,
        response_columns=args.response_columns,
        early_threshold=args.early_layer_threshold,
    )


if __name__ == "__main__":
    main()
