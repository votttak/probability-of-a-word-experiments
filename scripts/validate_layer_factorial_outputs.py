#!/usr/bin/env python3
"""Validate the four extraction cells in the layer-factorial experiment."""

from __future__ import annotations

import argparse
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
CELL_SPECS = {
    "passage_logit": ("passage", "logit-lens"),
    "passage_tuned": ("passage", "tuned-lens"),
    "sentence_logit": ("sentence", "logit-lens"),
    "sentence_tuned": ("sentence", "tuned-lens"),
}


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


def _validate_table(
    path: Path, label: str, expected_rows: int, layers: tuple[int, ...]
) -> pd.DataFrame:
    table = _read_tsv(path, label)
    if len(table) != expected_rows:
        raise ValidationError(
            f"{label} has {len(table)} rows; expected {expected_rows}"
        )
    corrected = _columns(CORRECTED_PREFIX, layers)
    buggy = _columns(BUGGY_PREFIX, layers)
    expected = set(KEY_COLUMNS) | set(corrected) | set(buggy)
    observed = set(table.columns)
    missing, extra = expected - observed, observed - expected
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if extra:
            details.append("extra=" + ",".join(sorted(extra)))
        raise ValidationError(
            f"{label} does not contain the exact dual predictor families: "
            + "; ".join(details)
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
    for column in corrected + buggy:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValidationError(f"{label} column {column} is not finite")
        if (values < 0).any():
            raise ValidationError(f"{label} column {column} contains negatives")
        table[column] = values
    return table


def _assert_key_word_identity(tables: dict[str, pd.DataFrame]) -> None:
    reference_label = "passage_logit"
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
    anchors: dict[str, dict[str, Any]], layers: tuple[int, ...]
) -> tuple[str, str, dict[str, Any], str]:
    models: list[str] = []
    revisions: list[str] = []
    tuned_identities: list[dict[str, Any]] = []
    sentence_hashes: list[str] = []
    for label, anchor in anchors.items():
        experiment = anchor.get("experiment")
        if not isinstance(experiment, dict):
            raise ValidationError(f"{label} anchor lacks experiment provenance")
        expected_context, expected_lens = CELL_SPECS[label]
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
        if experiment.get("score_kinds") != ["corrected", "buggy"]:
            raise ValidationError(
                f"{label} anchor must identify corrected and buggy score kinds"
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
    if len(set(identities)) != 1:
        raise ValidationError(
            "sentence and passage tuned cells use different artifacts"
        )
    if len(set(sentence_hashes)) != 1:
        raise ValidationError("sentence cells use different sentence manifests")
    return models[0], revisions[0], tuned_identities[0], sentence_hashes[0]


def _score_column(score_kind: str, layer: int) -> str:
    prefix = CORRECTED_PREFIX if score_kind == "corrected" else BUGGY_PREFIX
    return f"{prefix}{layer}"


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def _validate_comparisons(
    tables: dict[str, pd.DataFrame],
    layers: tuple[int, ...],
    tolerance: float,
) -> dict[str, Any]:
    final_layer = layers[-1]
    intermediate_layers = layers[:-1]
    final_diffs: dict[str, dict[str, float]] = {}
    intermediate_diffs: dict[str, dict[str, float]] = {}
    for context in ("passage", "sentence"):
        logit = tables[f"{context}_logit"]
        tuned = tables[f"{context}_tuned"]
        final_diffs[context] = {}
        intermediate_diffs[context] = {}
        context_has_intermediate_difference = False
        for score_kind in ("corrected", "buggy"):
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
    for lens in ("logit", "tuned"):
        passage = tables[f"passage_{lens}"]
        sentence = tables[f"sentence_{lens}"]
        context_diffs[lens] = {}
        for score_kind in ("corrected", "buggy"):
            column = _score_column(score_kind, final_layer)
            context_diffs[lens][score_kind] = _max_abs(
                passage[column].to_numpy(dtype=float),
                sentence[column].to_numpy(dtype=float),
            )
    if not any(
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
    """Validate all four cells and atomically publish a completion manifest."""
    _positive_int(expected_rows, "expected rows")
    layers = _expected_layers(expected_min_layer, expected_final_layer)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValidationError("tolerance must be finite and nonnegative")

    paths = {
        "passage_logit": Path(passage_logit_fname),
        "passage_tuned": Path(passage_tuned_fname),
        "sentence_logit": Path(sentence_logit_fname),
        "sentence_tuned": Path(sentence_tuned_fname),
    }
    tables = {
        label: _validate_table(path, label, expected_rows, layers)
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
        anchors, layers
    )
    comparisons = _validate_comparisons(tables, layers, tolerance)

    artifacts = {
        label: {
            "tsv": _artifact_record(paths[label]),
            "anchor_json": _artifact_record(anchor_paths[label]),
        }
        for label in CELL_SPECS
    }
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
        },
        "sentence_manifest_sha256": sentence_hash,
        "tuned_lens_identity": tuned_identity,
        "comparisons": comparisons,
        "artifacts": artifacts,
    }
    write_json_atomic_if_changed(completion, completion_json_fname)
    return completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the four internal-layer factorial extraction cells"
    )
    parser.add_argument("--passage-logit-fname", required=True)
    parser.add_argument("--passage-tuned-fname", required=True)
    parser.add_argument("--sentence-logit-fname", required=True)
    parser.add_argument("--sentence-tuned-fname", required=True)
    parser.add_argument("--completion-json-fname", required=True)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument(
        "--expected-final-layer", type=int, default=EXPECTED_FINAL_LAYER
    )
    parser.add_argument("--expected-min-layer", type=int, default=EXPECTED_MIN_LAYER)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    completion = validate_outputs(
        args.passage_logit_fname,
        args.passage_tuned_fname,
        args.sentence_logit_fname,
        args.sentence_tuned_fname,
        args.completion_json_fname,
        expected_rows=args.expected_rows,
        expected_final_layer=args.expected_final_layer,
        expected_min_layer=args.expected_min_layer,
        tolerance=args.tolerance,
    )
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
