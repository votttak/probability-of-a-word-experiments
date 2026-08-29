#!/usr/bin/env python3
"""Validate selected extraction cells in the layer-factorial experiment."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_ROWS = 10_256
EXPECTED_FINAL_LAYER = 12
EXPECTED_MIN_LAYER = 1
DEFAULT_TOLERANCE = 5e-4
KEY_COLUMNS = ("text_id", "word_id", "word")
CORRECTED_PREFIX = "internal_layer_surprisal_layer_"
BUGGY_PREFIX = "internal_layer_surprisal_buggy_layer_"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CONTEXT_UNITS = ("passage", "sentence")
LENS_METHODS = ("logit-lens", "tuned-lens")
SCORE_KINDS = ("corrected", "buggy")
SCORE_PREFIXES = {
    "corrected": CORRECTED_PREFIX,
    "buggy": BUGGY_PREFIX,
}
CELL_SPECS = {
    "passage_logit": ("passage", "logit-lens"),
    "passage_tuned": ("passage", "tuned-lens"),
    "sentence_logit": ("sentence", "logit-lens"),
    "sentence_tuned": ("sentence", "tuned-lens"),
}
CELL_LABELS = {spec: label for label, spec in CELL_SPECS.items()}


class ValidationError(ValueError):
    """A factorial extraction artifact violates a required invariant."""


def sha256_file(fname: str | Path) -> str:
    """Return a streaming SHA-256 digest."""
    path = Path(fname)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            for block in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValidationError(f"unable to hash artifact: {path}") from error
    return digest.hexdigest()


def _positive_int(value: Any, label: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValidationError(f"{label} must be a {qualifier} integer")
    return value


def _expected_layers(min_layer: int, final_layer: int) -> tuple[int, ...]:
    _positive_int(min_layer, "expected minimum layer", allow_zero=True)
    _positive_int(final_layer, "expected final layer")
    if final_layer <= min_layer:
        raise ValidationError(
            "expected final layer must exceed the minimum layer so an "
            "intermediate decoder comparison is possible"
        )
    return tuple(range(min_layer, final_layer + 1))


def _read_tsv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path, sep="\t", keep_default_na=False, low_memory=False
        )
    except Exception as error:
        raise ValidationError(f"unable to read {label} TSV: {path}") from error


def _integer_column(series: pd.Series, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValidationError(f"{label} must contain finite integers")
    return values.astype(np.int64)


def _columns(prefix: str, layers: tuple[int, ...]) -> list[str]:
    return [f"{prefix}{layer}" for layer in layers]


def _normalize_score_kinds(score_kinds: Sequence[str]) -> tuple[str, ...]:
    if isinstance(score_kinds, (str, bytes)) or not score_kinds:
        raise ValidationError("score_kinds must be a nonempty sequence")
    observed = list(score_kinds)
    invalid = sorted(set(observed) - set(SCORE_KINDS))
    if invalid:
        raise ValidationError(
            "unsupported score kinds: " + ", ".join(invalid)
        )
    if len(observed) != len(set(observed)):
        raise ValidationError("score_kinds must not contain duplicates")
    return tuple(kind for kind in SCORE_KINDS if kind in observed)


def _normalize_cell_paths(
    cell_paths: Mapping[tuple[str, str], str | Path],
) -> dict[tuple[str, str], Path]:
    if not isinstance(cell_paths, Mapping) or not cell_paths:
        raise ValidationError("cell_paths must be a nonempty mapping")
    normalized: dict[tuple[str, str], Path] = {}
    for key, path in cell_paths.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValidationError(
                "cell_paths keys must be (context_unit, lens_method) tuples"
            )
        context, lens = key
        if context not in CONTEXT_UNITS:
            raise ValidationError(f"unsupported context unit: {context!r}")
        if lens not in LENS_METHODS:
            raise ValidationError(f"unsupported lens method: {lens!r}")
        normalized[(context, lens)] = Path(path)
    return {
        (context, lens): normalized[(context, lens)]
        for context in CONTEXT_UNITS
        for lens in LENS_METHODS
        if (context, lens) in normalized
    }


def _validate_table(
    path: Path,
    label: str,
    expected_rows: int,
    layers: tuple[int, ...],
    score_kinds: tuple[str, ...],
) -> pd.DataFrame:
    table = _read_tsv(path, label)
    if len(table) != expected_rows:
        raise ValidationError(
            f"{label} has {len(table)} rows; expected {expected_rows}"
        )
    predictor_columns = [
        column
        for score_kind in score_kinds
        for column in _columns(SCORE_PREFIXES[score_kind], layers)
    ]
    expected = set(KEY_COLUMNS) | set(predictor_columns)
    observed = set(table.columns)
    missing, extra = expected - observed, observed - expected
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if extra:
            details.append("extra=" + ",".join(sorted(extra)))
        family_description = (
            "dual" if score_kinds == SCORE_KINDS else "selected"
        )
        raise ValidationError(
            f"{label} does not contain the exact {family_description} "
            "predictor families: " + "; ".join(details)
        )

    table = table.copy().reset_index(drop=True)
    table["text_id"] = _integer_column(table["text_id"], f"{label} text_id")
    table["word_id"] = _integer_column(table["word_id"], f"{label} word_id")
    if (table["text_id"] < 0).any() or (table["word_id"] < 0).any():
        raise ValidationError(f"{label} keys must be nonnegative")
    if table.duplicated(["text_id", "word_id"], keep=False).any():
        raise ValidationError(f"{label} contains duplicate keys")
    table["word"] = table["word"].astype(str)
    if (table["word"].str.len() == 0).any():
        raise ValidationError(f"{label} contains an empty word")
    for column in predictor_columns:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValidationError(f"{label} column {column} is not finite")
        if (values < 0).any():
            raise ValidationError(f"{label} column {column} contains negatives")
        table[column] = values
    return table


def _assert_key_word_identity(tables: dict[str, pd.DataFrame]) -> None:
    reference_label = next(iter(tables))
    reference = tables[reference_label]
    reference_keys = reference[["text_id", "word_id"]].to_numpy(dtype=np.int64)
    reference_words = reference["word"].to_numpy(dtype=str)
    for label, table in tables.items():
        if label == reference_label:
            continue
        keys = table[["text_id", "word_id"]].to_numpy(dtype=np.int64)
        if not np.array_equal(keys, reference_keys):
            mismatch = np.flatnonzero(np.any(keys != reference_keys, axis=1))
            row = int(mismatch[0]) if len(mismatch) else -1
            raise ValidationError(
                f"{label} key order differs from {reference_label} at row {row}"
            )
        words = table["word"].to_numpy(dtype=str)
        if not np.array_equal(words, reference_words):
            mismatch = np.flatnonzero(words != reference_words)
            row = int(mismatch[0]) if len(mismatch) else -1
            raise ValidationError(
                f"{label} word order differs from {reference_label} at row {row}"
            )


def _read_anchor(tsv_path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    anchor_path = Path(f"{tsv_path}.anchor.json")
    try:
        with anchor_path.open("r", encoding="utf8") as input_file:
            anchor = json.load(input_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(
            f"unable to read {label} anchor JSON: {anchor_path}"
        ) from error
    if not isinstance(anchor, dict):
        raise ValidationError(f"{label} anchor must be a JSON object")
    return anchor_path, anchor


def _sha256_identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_anchor_provenance(
    anchors: dict[str, dict[str, Any]],
    cell_specs: dict[str, tuple[str, str]],
    layers: tuple[int, ...],
    score_kinds: tuple[str, ...],
) -> tuple[str, str, dict[str, Any] | None, str | None]:
    models: list[str] = []
    revisions: list[str] = []
    tuned_identities: list[dict[str, Any]] = []
    sentence_hashes: list[str] = []
    for label, anchor in anchors.items():
        experiment = anchor.get("experiment")
        if not isinstance(experiment, dict):
            raise ValidationError(f"{label} anchor lacks experiment provenance")
        expected_context, expected_lens = cell_specs[label]
        if experiment.get("context_unit") != expected_context:
            raise ValidationError(
                f"{label} anchor context_unit is "
                f"{experiment.get('context_unit')!r}; expected {expected_context!r}"
            )
        if experiment.get("lens_method") != expected_lens:
            raise ValidationError(
                f"{label} anchor lens_method is "
                f"{experiment.get('lens_method')!r}; expected {expected_lens!r}"
            )
        model = experiment.get("model")
        revision = experiment.get("model_revision_effective")
        if not isinstance(model, str) or not model.strip():
            raise ValidationError(f"{label} anchor model must be nonempty")
        if not isinstance(revision, str) or not revision.strip():
            raise ValidationError(
                f"{label} anchor must include a nonempty effective model revision"
            )
        models.append(model)
        revisions.append(revision)
        if experiment.get("layers") != list(layers):
            raise ValidationError(
                f"{label} anchor layers are {experiment.get('layers')!r}; "
                f"expected {list(layers)!r}"
            )
        if experiment.get("score_kinds") != list(score_kinds):
            raise ValidationError(
                f"{label} anchor must identify exactly the selected score kinds"
            )
        if experiment.get("include_embedding_layer") is not (layers[0] == 0):
            raise ValidationError(
                f"{label} anchor include_embedding_layer disagrees with min layer"
            )

        identity = experiment.get("lens_identity")
        if expected_lens == "logit-lens":
            if identity is not None:
                raise ValidationError(
                    f"{label} logit-lens anchor unexpectedly has a lens identity"
                )
        else:
            if not isinstance(identity, dict) or not identity:
                raise ValidationError(
                    f"{label} tuned-lens anchor lacks its artifact identity"
                )
            artifact_identity = identity.get("artifact", identity)
            if not isinstance(artifact_identity, dict):
                raise ValidationError(
                    f"{label} tuned-lens artifact identity must be an object"
                )
            _sha256_identity(
                artifact_identity.get("config_sha256"),
                f"{label} tuned-lens config_sha256",
            )
            _sha256_identity(
                artifact_identity.get("params_sha256"),
                f"{label} tuned-lens params_sha256",
            )
            if not isinstance(
                artifact_identity.get("base_model_name_or_path"), str
            ) or not (
                artifact_identity["base_model_name_or_path"].strip()
            ):
                raise ValidationError(
                    f"{label} tuned-lens identity lacks base_model_name_or_path"
                )
            tuned_identities.append(identity)

        manifest_hash = experiment.get("sentence_manifest_sha256")
        if expected_context == "sentence":
            sentence_hashes.append(
                _sha256_identity(
                    manifest_hash, f"{label} sentence_manifest_sha256"
                )
            )
        elif manifest_hash not in (None, ""):
            raise ValidationError(
                f"{label} passage anchor unexpectedly has a sentence manifest hash"
            )

    if len(set(models)) != 1:
        raise ValidationError(f"factor cells use different models: {models}")
    if len(set(revisions)) != 1:
        raise ValidationError(
            f"factor cells use different effective model revisions: {revisions}"
        )
    identities = [
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for value in tuned_identities
    ]
    if len(set(identities)) > 1:
        raise ValidationError(
            "sentence and passage tuned cells use different artifacts"
        )
    if len(set(sentence_hashes)) > 1:
        raise ValidationError("sentence cells use different sentence manifests")
    tuned_identity = tuned_identities[0] if tuned_identities else None
    sentence_hash = sentence_hashes[0] if sentence_hashes else None
    return models[0], revisions[0], tuned_identity, sentence_hash


def _score_column(score_kind: str, layer: int) -> str:
    return f"{SCORE_PREFIXES[score_kind]}{layer}"


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def _validate_comparisons(
    tables: dict[tuple[str, str], pd.DataFrame],
    layers: tuple[int, ...],
    tolerance: float,
    score_kinds: tuple[str, ...],
) -> dict[str, Any]:
    final_layer = layers[-1]
    intermediate_layers = layers[:-1]
    final_diffs: dict[str, dict[str, float]] = {}
    intermediate_diffs: dict[str, dict[str, float]] = {}
    unavailable: list[dict[str, Any]] = []
    selected_contexts = {context for context, _ in tables}
    selected_lenses = {lens for _, lens in tables}
    for context in CONTEXT_UNITS:
        if context not in selected_contexts:
            continue
        required = {
            (context, "logit-lens"),
            (context, "tuned-lens"),
        }
        missing = required - set(tables)
        if missing:
            unavailable.append({
                "comparison": "logit_vs_tuned",
                "context_unit": context,
                "missing_cells": [
                    {"context_unit": cell[0], "lens_method": cell[1]}
                    for cell in sorted(missing)
                ],
                "reason": "requires both logit-lens and tuned-lens",
            })
            continue
        logit = tables[(context, "logit-lens")]
        tuned = tables[(context, "tuned-lens")]
        final_diffs[context] = {}
        intermediate_diffs[context] = {}
        context_has_intermediate_difference = False
        for score_kind in score_kinds:
            final_column = _score_column(score_kind, final_layer)
            final_max = _max_abs(
                logit[final_column].to_numpy(dtype=float),
                tuned[final_column].to_numpy(dtype=float),
            )
            final_diffs[context][score_kind] = final_max
            if final_max > tolerance:
                raise ValidationError(
                    f"{context} {score_kind} final-layer logit/tuned difference "
                    f"{final_max:.9g} exceeds tolerance {tolerance:.9g}"
                )
            columns = [
                _score_column(score_kind, layer) for layer in intermediate_layers
            ]
            logit_values = logit[columns].to_numpy(dtype=float)
            tuned_values = tuned[columns].to_numpy(dtype=float)
            intermediate_max = _max_abs(logit_values, tuned_values)
            intermediate_diffs[context][score_kind] = intermediate_max
            context_has_intermediate_difference |= not np.array_equal(
                logit_values, tuned_values
            )
        if not context_has_intermediate_difference:
            raise ValidationError(
                f"{context} intermediate logit and tuned predictions are all identical"
            )

    context_diffs: dict[str, dict[str, float]] = {}
    for lens_method in LENS_METHODS:
        if lens_method not in selected_lenses:
            continue
        required = {
            ("passage", lens_method),
            ("sentence", lens_method),
        }
        missing = required - set(tables)
        if missing:
            unavailable.append({
                "comparison": "passage_vs_sentence",
                "lens_method": lens_method,
                "missing_cells": [
                    {"context_unit": cell[0], "lens_method": cell[1]}
                    for cell in sorted(missing)
                ],
                "reason": "requires both passage and sentence contexts",
            })
            continue
        lens = lens_method.removesuffix("-lens")
        passage = tables[("passage", lens_method)]
        sentence = tables[("sentence", lens_method)]
        context_diffs[lens] = {}
        for score_kind in score_kinds:
            column = _score_column(score_kind, final_layer)
            context_diffs[lens][score_kind] = _max_abs(
                passage[column].to_numpy(dtype=float),
                sentence[column].to_numpy(dtype=float),
            )
    if context_diffs and not any(
        difference > 0.0
        for lens_differences in context_diffs.values()
        for difference in lens_differences.values()
    ):
        raise ValidationError(
            "passage and sentence final-layer scores are identical across all cells"
        )
    return {
        "final_logit_vs_tuned_max_abs_difference": final_diffs,
        "intermediate_logit_vs_tuned_max_abs_difference": intermediate_diffs,
        "passage_vs_sentence_final_max_abs_difference": context_diffs,
        "unavailable": unavailable,
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json_atomic_if_changed(data: dict[str, Any], fname: str | Path) -> None:
    """Atomically publish deterministic JSON without rewriting equal content."""
    output_path = Path(fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf8")
    if output_path.is_file() and output_path.read_bytes() == serialized:
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output_file:
            output_file.write(serialized)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise


def validate_selected_outputs(
    cell_paths: Mapping[tuple[str, str], str | Path],
    completion_json_fname: str | Path,
    *,
    score_kinds: Sequence[str] = SCORE_KINDS,
    expected_rows: int = EXPECTED_ROWS,
    expected_final_layer: int = EXPECTED_FINAL_LAYER,
    expected_min_layer: int = EXPECTED_MIN_LAYER,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Validate a nonempty selected cell grid and publish its manifest."""
    _positive_int(expected_rows, "expected rows")
    layers = _expected_layers(expected_min_layer, expected_final_layer)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValidationError("tolerance must be finite and nonnegative")

    selected_scores = _normalize_score_kinds(score_kinds)
    selected_paths = _normalize_cell_paths(cell_paths)
    cell_specs = {
        CELL_LABELS[cell]: cell for cell in selected_paths
    }
    paths = {
        CELL_LABELS[cell]: path for cell, path in selected_paths.items()
    }
    tables = {
        label: _validate_table(
            path, label, expected_rows, layers, selected_scores
        )
        for label, path in paths.items()
    }
    _assert_key_word_identity(tables)

    anchor_paths: dict[str, Path] = {}
    anchors: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        anchor_path, anchor = _read_anchor(path, label)
        anchor_paths[label] = anchor_path
        anchors[label] = anchor
    model, revision, tuned_identity, sentence_hash = _validate_anchor_provenance(
        anchors, cell_specs, layers, selected_scores
    )
    comparisons = _validate_comparisons(
        {
            cell_specs[label]: table
            for label, table in tables.items()
        },
        layers,
        tolerance,
        selected_scores,
    )

    artifacts = {
        label: {
            "tsv": _artifact_record(paths[label]),
            "anchor_json": _artifact_record(anchor_paths[label]),
        }
        for label in paths
    }
    selected_contexts = [
        context
        for context in CONTEXT_UNITS
        if any(cell[0] == context for cell in selected_paths)
    ]
    selected_lenses = [
        lens
        for lens in LENS_METHODS
        if any(cell[1] == lens for cell in selected_paths)
    ]
    completion = {
        "schema_version": 1,
        "validated": True,
        "model": model,
        "model_revision_effective": revision,
        "expected": {
            "rows": expected_rows,
            "min_layer": expected_min_layer,
            "final_layer": expected_final_layer,
            "layers": list(layers),
            "final_layer_tolerance": tolerance,
            "contexts": selected_contexts,
            "lens_methods": selected_lenses,
            "score_kinds": list(selected_scores),
            "cells": [
                {
                    "context_unit": context,
                    "lens_method": lens,
                    "artifact": CELL_LABELS[(context, lens)],
                }
                for context, lens in selected_paths
            ],
        },
        "sentence_manifest_sha256": sentence_hash,
        "tuned_lens_identity": tuned_identity,
        "comparisons": comparisons,
        "artifacts": artifacts,
    }
    write_json_atomic_if_changed(completion, completion_json_fname)
    return completion


def validate_outputs(
    passage_logit_fname: str | Path,
    passage_tuned_fname: str | Path,
    sentence_logit_fname: str | Path,
    sentence_tuned_fname: str | Path,
    completion_json_fname: str | Path,
    *,
    expected_rows: int = EXPECTED_ROWS,
    expected_final_layer: int = EXPECTED_FINAL_LAYER,
    expected_min_layer: int = EXPECTED_MIN_LAYER,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Validate the legacy full four-cell, dual-score extraction grid."""
    return validate_selected_outputs(
        {
            ("passage", "logit-lens"): passage_logit_fname,
            ("passage", "tuned-lens"): passage_tuned_fname,
            ("sentence", "logit-lens"): sentence_logit_fname,
            ("sentence", "tuned-lens"): sentence_tuned_fname,
        },
        completion_json_fname,
        score_kinds=SCORE_KINDS,
        expected_rows=expected_rows,
        expected_final_layer=expected_final_layer,
        expected_min_layer=expected_min_layer,
        tolerance=tolerance,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate selected internal-layer factorial extraction cells"
    )
    parser.add_argument("--passage-logit-fname")
    parser.add_argument("--passage-tuned-fname")
    parser.add_argument("--sentence-logit-fname")
    parser.add_argument("--sentence-tuned-fname")
    parser.add_argument(
        "--cell",
        dest="cells",
        action="append",
        nargs=3,
        metavar=("CONTEXT", "LENS", "PATH"),
        help=(
            "selected extraction cell; repeat for each context/lens artifact "
            "instead of using the four legacy filename options"
        ),
    )
    parser.add_argument(
        "--score-kinds",
        nargs="+",
        choices=SCORE_KINDS,
        default=list(SCORE_KINDS),
    )
    parser.add_argument("--completion-json-fname", required=True)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument(
        "--expected-final-layer", type=int, default=EXPECTED_FINAL_LAYER
    )
    parser.add_argument("--expected-min-layer", type=int, default=EXPECTED_MIN_LAYER)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args(argv)

    legacy = {
        ("passage", "logit-lens"): args.passage_logit_fname,
        ("passage", "tuned-lens"): args.passage_tuned_fname,
        ("sentence", "logit-lens"): args.sentence_logit_fname,
        ("sentence", "tuned-lens"): args.sentence_tuned_fname,
    }
    supplied_legacy = [path is not None for path in legacy.values()]
    if args.cells:
        if any(supplied_legacy):
            parser.error(
                "--cell cannot be combined with the legacy cell filename options"
            )
        cell_paths: dict[tuple[str, str], str] = {}
        for context, lens, path in args.cells:
            if context not in CONTEXT_UNITS:
                parser.error(
                    f"--cell context must be one of {', '.join(CONTEXT_UNITS)}"
                )
            if lens not in LENS_METHODS:
                parser.error(
                    f"--cell lens must be one of {', '.join(LENS_METHODS)}"
                )
            key = (context, lens)
            if key in cell_paths:
                parser.error(
                    f"duplicate --cell for context={context}, lens={lens}"
                )
            cell_paths[key] = path
        args.cell_paths = cell_paths
    else:
        if not all(supplied_legacy):
            parser.error(
                "provide at least one --cell, or all four legacy cell "
                "filename options"
            )
        args.cell_paths = legacy
    return args


def main() -> None:
    args = parse_args()
    completion = validate_selected_outputs(
        args.cell_paths,
        args.completion_json_fname,
        score_kinds=args.score_kinds,
        expected_rows=args.expected_rows,
        expected_final_layer=args.expected_final_layer,
        expected_min_layer=args.expected_min_layer,
        tolerance=args.tolerance,
    )
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
