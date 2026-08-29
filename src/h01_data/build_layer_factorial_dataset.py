#!/usr/bin/env python3
"""Build a validated factorial internal-layer reading-time dataset."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from .build_natural_stories_sentence_manifest import (
        SentenceUnit,
        read_sentence_manifest,
    )
except ImportError:  # Support direct execution from src/h01_data.
    from build_natural_stories_sentence_manifest import (
        SentenceUnit,
        read_sentence_manifest,
    )


KEY_COLUMNS = ["text_id", "word_id"]
CORRECTED_PREFIX = "internal_layer_surprisal_layer_"
BUGGY_PREFIX = "internal_layer_surprisal_buggy_layer_"
METADATA_COLUMNS = [
    "model",
    "context_unit",
    "lens_method",
    "first_token_policy",
    "sentence_first_token_policy",
    "include_embedding_layer",
    "lag_boundary",
    "lag_padding",
]
PAPER_CONTROL_COLUMNS = [
    "paper_length",
    "paper_length_prev_1",
    "paper_length_prev_2",
    "paper_log_gmean_freq",
    "paper_log_gmean_freq_prev_1",
    "paper_log_gmean_freq_prev_2",
    "length",
    "length_prev_1",
    "length_prev_2",
    "log_gmean_freq",
    "log_gmean_freq_prev_1",
    "log_gmean_freq_prev_2",
]
CONTEXT_UNITS = ("passage", "sentence")
LENS_METHODS = ("logit-lens", "tuned-lens")
FIRST_TOKEN_POLICIES = ("bos", "bow")
LAG_BOUNDARIES = ("text", "sentence")
LAG_PADDING_MODES = ("missing", "global-mean")
WORDFREQ_VERSION = "3.1.1"
WORDFREQ_EPSILON = 1e-7
PEAKED_PAPER_KEY = (2, 748)


def _integer_series(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label} must contain finite integers")
    return pd.Series(values.astype(np.int64), index=series.index)


def _reference_word_column(dataframe: pd.DataFrame, label: str) -> str:
    if "ref_token" in dataframe.columns:
        return "ref_token"
    if "word" in dataframe.columns:
        return "word"
    raise ValueError(f"{label} has neither ref_token nor word")


def _validate_joint(
    canonical_joint: pd.DataFrame,
) -> tuple[pd.DataFrame, str, list[list[str]]]:
    missing = set(KEY_COLUMNS) - set(canonical_joint.columns)
    if missing:
        raise ValueError(
            "canonical joint is missing columns: " + ", ".join(sorted(missing))
        )
    if canonical_joint.empty:
        raise ValueError("canonical joint contains no rows")

    joint = canonical_joint.copy().reset_index(drop=True)
    word_column = _reference_word_column(joint, "canonical joint")
    joint["text_id"] = _integer_series(
        joint["text_id"], "canonical joint text_id"
    )
    joint["word_id"] = _integer_series(
        joint["word_id"], "canonical joint word_id"
    )
    if (joint["text_id"] < 1).any():
        raise ValueError("canonical joint text_id must be one-based")
    if (joint["word_id"] < 0).any():
        raise ValueError("canonical joint word_id must be nonnegative")
    if joint.duplicated(KEY_COLUMNS, keep=False).any():
        raise ValueError("canonical joint contains duplicate keys")
    if joint[word_column].isna().any() or (
        joint[word_column].astype(str).str.len() == 0
    ).any():
        raise ValueError("canonical joint contains a missing or empty word")

    text_ids = sorted(joint["text_id"].unique().tolist())
    expected_text_ids = list(range(1, len(text_ids) + 1))
    if text_ids != expected_text_ids:
        raise ValueError(
            "canonical joint text_id values must be the complete one-based "
            f"range; observed={text_ids}, expected={expected_text_ids}"
        )

    passages: list[list[str]] = []
    for text_id in expected_text_ids:
        group = joint.loc[joint["text_id"] == text_id].sort_values(
            "word_id", kind="stable"
        )
        word_ids = group["word_id"].tolist()
        expected_word_ids = list(range(len(group)))
        if word_ids != expected_word_ids:
            raise ValueError(
                f"canonical joint word_id values in text {text_id} must be "
                f"exactly 0..N-1; observed={word_ids}"
            )
        passages.append(group[word_column].astype(str).tolist())
    return joint, word_column, passages


def _layer_columns(
    dataframe: pd.DataFrame, prefix: str, family_label: str
) -> tuple[list[int], list[str]]:
    prefixed = [column for column in dataframe.columns if column.startswith(prefix)]
    if not prefixed:
        raise ValueError(f"layer TSV contains no {family_label} predictor columns")

    parsed: list[tuple[int, str]] = []
    pattern = re.compile(re.escape(prefix) + r"([0-9]+)$")
    malformed = []
    for column in prefixed:
        match = pattern.fullmatch(column)
        if match is None:
            malformed.append(column)
        else:
            parsed.append((int(match.group(1)), column))
    if malformed:
        raise ValueError(
            f"layer TSV has malformed {family_label} columns: "
            + ", ".join(malformed)
        )

    parsed.sort()
    layer_ids = [layer_id for layer_id, _ in parsed]
    if len(layer_ids) != len(set(layer_ids)):
        raise ValueError(f"layer TSV contains duplicate {family_label} layer IDs")
    if layer_ids[0] not in (0, 1):
        raise ValueError(
            f"{family_label} layer IDs must start at 0 or 1; "
            f"observed start={layer_ids[0]}"
        )
    expected = list(range(layer_ids[0], layer_ids[-1] + 1))
    if layer_ids != expected:
        raise ValueError(
            f"{family_label} layer IDs must be complete and consecutive; "
            f"observed={layer_ids}, expected={expected}"
        )
    return layer_ids, [column for _, column in parsed]


def _validate_layer_scores(
    layer_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int], list[str], list[str]]:
    required = set(KEY_COLUMNS + ["word"])
    missing = required - set(layer_scores.columns)
    if missing:
        raise ValueError(
            "layer TSV is missing columns: " + ", ".join(sorted(missing))
        )
    if layer_scores.empty:
        raise ValueError("layer TSV contains no rows")

    layer = layer_scores.copy().reset_index(drop=True)
    layer["text_id"] = _integer_series(layer["text_id"], "layer TSV text_id")
    layer["word_id"] = _integer_series(layer["word_id"], "layer TSV word_id")
    if (layer["text_id"] < 0).any():
        raise ValueError("layer TSV text_id must be zero-based and nonnegative")
    if (layer["word_id"] < 0).any():
        raise ValueError("layer TSV word_id must be nonnegative")
    if layer.duplicated(KEY_COLUMNS, keep=False).any():
        raise ValueError("layer TSV contains duplicate keys")
    if layer["word"].isna().any() or (layer["word"].astype(str).str.len() == 0).any():
        raise ValueError("layer TSV contains a missing or empty word")

    corrected_ids, corrected_columns = _layer_columns(
        layer, CORRECTED_PREFIX, "corrected"
    )
    buggy_ids, buggy_columns = _layer_columns(layer, BUGGY_PREFIX, "buggy")
    if corrected_ids != buggy_ids:
        raise ValueError(
            "corrected and buggy layer predictors must have identical layer IDs; "
            f"corrected={corrected_ids}, buggy={buggy_ids}"
        )

    for column in corrected_columns + buggy_columns:
        numeric = pd.to_numeric(layer[column], errors="coerce")
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"layer predictor {column} contains non-finite values")
        if (values < 0).any():
            raise ValueError(f"layer predictor {column} contains negative values")
        layer[column] = values
    return layer, corrected_ids, corrected_columns, buggy_columns


def _key_tuples(dataframe: pd.DataFrame) -> list[tuple[int, int]]:
    return list(
        map(tuple, dataframe[KEY_COLUMNS].to_numpy(dtype=np.int64))
    )


def _align_layer_to_joint(
    joint: pd.DataFrame,
    joint_word_column: str,
    layer: pd.DataFrame,
    predictor_columns: Sequence[str],
) -> pd.DataFrame:
    shifted = layer.copy()
    shifted["text_id"] += 1
    joint_keys = _key_tuples(joint)
    layer_keys = _key_tuples(shifted)
    joint_set = set(joint_keys)
    layer_set = set(layer_keys)
    if joint_set != layer_set:
        raise ValueError(
            "layer TSV key coverage differs from canonical joint: "
            f"{len(joint_set - layer_set)} missing and "
            f"{len(layer_set - joint_set)} extra"
        )

    index = pd.MultiIndex.from_tuples(joint_keys, names=KEY_COLUMNS)
    aligned = (
        shifted.set_index(KEY_COLUMNS, verify_integrity=True)
        .reindex(index)
        .reset_index()
    )
    joint_words = joint[joint_word_column].astype(str).to_numpy()
    layer_words = aligned["word"].astype(str).to_numpy()
    mismatch = np.flatnonzero(joint_words != layer_words)
    if len(mismatch):
        row = int(mismatch[0])
        raise ValueError(
            f"layer TSV word mismatch at canonical row {row}: "
            f"{layer_words[row]!r} versus {joint_words[row]!r}"
        )
    return aligned[KEY_COLUMNS + ["word"] + list(predictor_columns)]


def _validate_sentence_mapping(
    sentence_map: dict[int, list[SentenceUnit]],
    passages: Sequence[Sequence[str]],
) -> dict[tuple[int, int], tuple[int, int]]:
    expected_text_ids = set(range(len(passages)))
    if set(sentence_map) != expected_text_ids:
        raise ValueError(
            "sentence manifest text coverage differs from canonical joint; "
            f"observed={sorted(sentence_map)}, expected={sorted(expected_text_ids)}"
        )

    lookup: dict[tuple[int, int], tuple[int, int]] = {}
    observed: list[tuple[int, int, str]] = []
    for text_id, canonical_words in enumerate(passages):
        units = sentence_map[text_id]
        if not units:
            raise ValueError(f"sentence manifest has no sentences for text {text_id}")
        for expected_sentence_id, unit in enumerate(units):
            if unit.sentence_id != expected_sentence_id:
                raise ValueError(
                    f"sentence manifest sentence IDs in text {text_id} must be "
                    f"contiguous from 0; got {unit.sentence_id}, "
                    f"expected {expected_sentence_id}"
                )
            if not unit.word_ids or len(unit.word_ids) != len(unit.words):
                raise ValueError(
                    f"sentence manifest has an empty or malformed sentence "
                    f"{unit.sentence_id} in text {text_id}"
                )
            for sentence_word_id, (word_id, word) in enumerate(
                zip(unit.word_ids, unit.words)
            ):
                key = (text_id, word_id)
                if key in lookup:
                    raise ValueError(
                        f"sentence manifest contains duplicate key {key}"
                    )
                lookup[key] = (unit.sentence_id, sentence_word_id)
                observed.append((text_id, word_id, word))

    expected = [
        (text_id, word_id, word)
        for text_id, words in enumerate(passages)
        for word_id, word in enumerate(words)
    ]
    if observed != expected:
        raise ValueError(
            "sentence manifest does not flatten exactly to canonical joint words"
        )
    return lookup


def _validate_options(
    model: str,
    context_unit: str,
    lens_method: str,
    first_token_policy: str,
    lag_boundary: str,
    lag_padding: str | None,
) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model metadata must be a nonempty string")
    if context_unit not in CONTEXT_UNITS:
        raise ValueError(f"context_unit must be one of {CONTEXT_UNITS}")
    if lens_method not in LENS_METHODS:
        raise ValueError(f"lens_method must be one of {LENS_METHODS}")
    if first_token_policy not in FIRST_TOKEN_POLICIES:
        raise ValueError(
            f"first_token_policy must be one of {FIRST_TOKEN_POLICIES}"
        )
    if context_unit == "passage" and first_token_policy != "bos":
        raise ValueError("passage context requires first_token_policy=bos")
    if lag_boundary not in LAG_BOUNDARIES:
        raise ValueError(f"lag_boundary must be one of {LAG_BOUNDARIES}")
    resolved_padding = lag_padding
    if resolved_padding is None:
        resolved_padding = (
            "global-mean" if lag_boundary == "sentence" else "missing"
        )
    if resolved_padding not in LAG_PADDING_MODES:
        raise ValueError(f"lag_padding must be one of {LAG_PADDING_MODES}")
    return resolved_padding


def _load_word_frequency() -> Callable[[str, str], float]:
    try:
        installed_version = version("wordfreq")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "Paper frequency controls require wordfreq==3.1.1. Install it or "
            "provide --precomputed-frequency-fname with keyed "
            "paper_log_gmean_freq values."
        ) from error
    if installed_version != WORDFREQ_VERSION:
        raise RuntimeError(
            f"Paper frequency controls require wordfreq=={WORDFREQ_VERSION}; "
            f"found {installed_version}. Use the pinned version or provide "
            "--precomputed-frequency-fname."
        )
    try:
        from wordfreq import word_frequency
    except ImportError as error:
        raise RuntimeError(
            "wordfreq metadata is installed but word_frequency cannot be "
            "imported; reinstall wordfreq==3.1.1 or provide "
            "--precomputed-frequency-fname."
        ) from error
    return word_frequency


def _computed_frequency_values(
    words: Sequence[str],
    word_frequency_fn: Callable[[str, str], float] | None,
) -> np.ndarray:
    provider = word_frequency_fn or _load_word_frequency()
    cache: dict[str, float] = {}
    values: list[float] = []
    for word in words:
        if word not in cache:
            try:
                raw_frequency = float(provider(word, "en"))
            except Exception as error:
                raise RuntimeError(
                    f"wordfreq failed while scoring word {word!r}"
                ) from error
            if not math.isfinite(raw_frequency) or raw_frequency < 0:
                raise ValueError(
                    f"wordfreq returned an invalid frequency for {word!r}: "
                    f"{raw_frequency!r}"
                )
            cache[word] = math.log(raw_frequency + WORDFREQ_EPSILON)
        values.append(cache[word])
    return np.asarray(values, dtype=float)


def _paper_frequency_words(
    joint: pd.DataFrame, joint_word_column: str
) -> list[str]:
    """Return spellings used by the paper before its later tokens.json fix."""
    words = joint[joint_word_column].astype(str).tolist()
    for row, (text_id, word_id) in enumerate(
        joint[KEY_COLUMNS].itertuples(index=False, name=None)
    ):
        if (int(text_id), int(word_id)) == PEAKED_PAPER_KEY:
            if words[row] != "peeked":
                raise ValueError(
                    "paper frequency compatibility key (2, 748) must contain "
                    f"canonical 'peeked'; observed={words[row]!r}"
                )
            words[row] = "peaked"
    return words


def _precomputed_frequency_values(
    frequency_table: pd.DataFrame,
    joint: pd.DataFrame,
    joint_word_column: str,
) -> np.ndarray:
    required = set(KEY_COLUMNS + ["word"])
    missing = required - set(frequency_table.columns)
    if missing:
        raise ValueError(
            "precomputed frequency TSV is missing columns: "
            + ", ".join(sorted(missing))
        )
    candidates = [
        column
        for column in ("paper_log_gmean_freq", "log_gmean_freq")
        if column in frequency_table.columns
    ]
    if len(candidates) != 1:
        raise ValueError(
            "precomputed frequency TSV must contain exactly one of "
            "paper_log_gmean_freq or log_gmean_freq"
        )

    frequency = frequency_table.copy().reset_index(drop=True)
    frequency["text_id"] = _integer_series(
        frequency["text_id"], "precomputed frequency text_id"
    )
    frequency["word_id"] = _integer_series(
        frequency["word_id"], "precomputed frequency word_id"
    )
    if (frequency["text_id"] < 1).any():
        raise ValueError(
            "precomputed frequency text_id must use canonical one-based IDs"
        )
    if (frequency["word_id"] < 0).any():
        raise ValueError("precomputed frequency word_id must be nonnegative")
    if frequency.duplicated(KEY_COLUMNS, keep=False).any():
        raise ValueError("precomputed frequency TSV contains duplicate keys")

    joint_keys = _key_tuples(joint)
    frequency_keys = _key_tuples(frequency)
    if set(joint_keys) != set(frequency_keys):
        raise ValueError(
            "precomputed frequency key coverage differs from canonical joint"
        )
    index = pd.MultiIndex.from_tuples(joint_keys, names=KEY_COLUMNS)
    aligned = (
        frequency.set_index(KEY_COLUMNS, verify_integrity=True)
        .reindex(index)
        .reset_index()
    )
    joint_words = joint[joint_word_column].astype(str).to_numpy()
    frequency_words = aligned["word"].astype(str).to_numpy()
    for row, (joint_word, frequency_word) in enumerate(
        zip(joint_words, frequency_words)
    ):
        compatible = (
            joint_keys[row] == PEAKED_PAPER_KEY
            and joint_word == "peeked"
            and frequency_word == "peaked"
        )
        if frequency_word != joint_word and not compatible:
            raise ValueError(
                f"precomputed frequency word mismatch at canonical row {row}: "
                f"{frequency_word!r} versus {joint_word!r}"
            )
    values = pd.to_numeric(aligned[candidates[0]], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError(
            f"precomputed frequency column {candidates[0]} contains "
            "non-finite values"
        )
    return values


def _paper_time_values(
    paper_rt_table: pd.DataFrame,
    joint: pd.DataFrame,
    joint_word_column: str,
) -> np.ndarray:
    required = {"item", "zone", "word", "meanItemRT"}
    missing = required - set(paper_rt_table.columns)
    if missing:
        raise ValueError(
            "paper RT TSV is missing columns: " + ", ".join(sorted(missing))
        )
    paper = paper_rt_table.loc[
        :, ["item", "zone", "word", "meanItemRT"]
    ].drop_duplicates().reset_index(drop=True)
    paper["item"] = _integer_series(paper["item"], "paper RT item")
    paper["zone"] = _integer_series(paper["zone"], "paper RT zone")
    if (paper["item"] < 1).any() or (paper["zone"] < 1).any():
        raise ValueError("paper RT item and zone must use positive one-based IDs")
    if paper.duplicated(["item", "zone"], keep=False).any():
        raise ValueError(
            "paper RT TSV has conflicting word or meanItemRT values for a key"
        )

    paper["text_id"] = paper["item"]
    paper["word_id"] = paper["zone"] - 1
    joint_keys = _key_tuples(joint)
    paper_keys = _key_tuples(paper)
    missing_keys = set(joint_keys) - set(paper_keys)
    if missing_keys:
        raise ValueError(
            "paper RT does not cover every canonical joint key; "
            f"missing={len(missing_keys)}"
        )
    index = pd.MultiIndex.from_tuples(joint_keys, names=KEY_COLUMNS)
    aligned = (
        paper.set_index(KEY_COLUMNS, verify_integrity=True)
        .reindex(index)
        .reset_index()
    )

    joint_words = joint[joint_word_column].astype(str).to_numpy()
    paper_words = aligned["word"].astype(str).to_numpy()
    for row, (joint_word, paper_word) in enumerate(
        zip(joint_words, paper_words)
    ):
        text_id, word_id = joint_keys[row]
        compatible = (
            (text_id, word_id) == PEAKED_PAPER_KEY
            and joint_word == "peeked"
            and paper_word == "peaked"
        )
        if paper_word != joint_word and not compatible:
            raise ValueError(
                f"paper RT word mismatch at canonical key "
                f"({text_id}, {word_id}): {paper_word!r} versus "
                f"{joint_word!r}"
            )

    values = pd.to_numeric(aligned["meanItemRT"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("paper RT meanItemRT must contain finite positive values")
    return values


def _shifted(
    dataframe: pd.DataFrame,
    source_column: str,
    lag: int,
    boundary_columns: Sequence[str],
    padding: str,
) -> pd.Series:
    ordered = dataframe.sort_values(KEY_COLUMNS, kind="stable")
    shifted = ordered.groupby(
        list(boundary_columns), sort=False, dropna=False
    )[source_column].shift(lag)
    if padding == "global-mean":
        shifted = shifted.fillna(float(dataframe[source_column].mean()))
    result = pd.Series(np.nan, index=dataframe.index, dtype=float)
    result.loc[ordered.index] = shifted.to_numpy(dtype=float)
    return result


def _lag_column_name(source_column: str, lag: int) -> str:
    prefixes = {1: "prev_", 2: "prev2_", 3: "prev3_"}
    return f"{prefixes[lag]}{source_column}"


def build_layer_factorial_dataframe(
    canonical_joint: pd.DataFrame,
    layer_scores: pd.DataFrame,
    sentence_map: dict[int, list[SentenceUnit]],
    *,
    model: str,
    context_unit: str,
    lens_method: str,
    first_token_policy: str,
    lag_boundary: str,
    lag_padding: str | None = None,
    frequency_table: pd.DataFrame | None = None,
    word_frequency_fn: Callable[[str, str], float] | None = None,
    paper_rt_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate, align, and merge one factorial cell in canonical row order."""
    resolved_padding = _validate_options(
        model,
        context_unit,
        lens_method,
        first_token_policy,
        lag_boundary,
        lag_padding,
    )
    joint, joint_word_column, passages = _validate_joint(canonical_joint)
    layer, layer_ids, corrected_columns, buggy_columns = _validate_layer_scores(
        layer_scores
    )
    predictor_columns = corrected_columns + buggy_columns
    aligned_layer = _align_layer_to_joint(
        joint, joint_word_column, layer, predictor_columns
    )
    sentence_lookup = _validate_sentence_mapping(sentence_map, passages)

    if frequency_table is not None and word_frequency_fn is not None:
        raise ValueError(
            "provide either frequency_table or word_frequency_fn, not both"
        )
    if frequency_table is not None:
        frequency_values = _precomputed_frequency_values(
            frequency_table, joint, joint_word_column
        )
    else:
        frequency_values = _computed_frequency_values(
            _paper_frequency_words(joint, joint_word_column),
            word_frequency_fn,
        )
    paper_time_values = (
        _paper_time_values(paper_rt_table, joint, joint_word_column)
        if paper_rt_table is not None
        else None
    )

    generated_columns = (
        ["sentence_id", "sentence_word_id"]
        + METADATA_COLUMNS
        + PAPER_CONTROL_COLUMNS
        + predictor_columns
        + [
            _lag_column_name(column, lag)
            for column in predictor_columns
            for lag in (1, 2, 3)
        ]
        + (["paper_time"] if paper_time_values is not None else [])
    )
    collisions = sorted(set(joint.columns) & set(generated_columns))
    if collisions:
        raise ValueError(
            "canonical joint already contains generated factorial columns: "
            + ", ".join(collisions)
        )

    output = joint.copy()
    positions = [
        sentence_lookup[(int(text_id) - 1, int(word_id))]
        for text_id, word_id in output[KEY_COLUMNS].itertuples(
            index=False, name=None
        )
    ]
    output["sentence_id"] = [position[0] for position in positions]
    output["sentence_word_id"] = [position[1] for position in positions]
    if paper_time_values is not None:
        output["paper_time"] = paper_time_values

    metadata = {
        "model": model.strip(),
        "context_unit": context_unit,
        "lens_method": lens_method,
        "first_token_policy": first_token_policy,
        "sentence_first_token_policy": first_token_policy,
        "include_embedding_layer": layer_ids[0] == 0,
        "lag_boundary": lag_boundary,
        "lag_padding": resolved_padding,
    }
    for column in METADATA_COLUMNS:
        output[column] = metadata[column]

    output["paper_length"] = (
        output[joint_word_column].astype(str).map(len).astype(np.int64)
    )
    output["paper_log_gmean_freq"] = frequency_values
    for source_column in ("paper_length", "paper_log_gmean_freq"):
        for lag in (1, 2):
            output[f"{source_column}_prev_{lag}"] = _shifted(
                output,
                source_column,
                lag,
                ("text_id", "sentence_id"),
                "global-mean",
            )

    output["length"] = output["paper_length"]
    output["length_prev_1"] = output["paper_length_prev_1"]
    output["length_prev_2"] = output["paper_length_prev_2"]
    output["log_gmean_freq"] = output["paper_log_gmean_freq"]
    output["log_gmean_freq_prev_1"] = output[
        "paper_log_gmean_freq_prev_1"
    ]
    output["log_gmean_freq_prev_2"] = output[
        "paper_log_gmean_freq_prev_2"
    ]

    predictor_frame = aligned_layer[predictor_columns].astype(float)
    predictor_frame.index = output.index
    output = pd.concat([output, predictor_frame], axis=1)
    lag_group_columns = (
        ("text_id", "sentence_id")
        if lag_boundary == "sentence"
        else ("text_id",)
    )
    predictor_lags = {}
    for column in predictor_columns:
        for lag in (1, 2, 3):
            predictor_lags[_lag_column_name(column, lag)] = _shifted(
                output,
                column,
                lag,
                lag_group_columns,
                resolved_padding,
            )
    output = pd.concat(
        [output, pd.DataFrame(predictor_lags, index=output.index)],
        axis=1,
    )

    if output.columns[: len(canonical_joint.columns)].tolist() != list(
        canonical_joint.columns
    ):
        raise AssertionError("factorial merge did not preserve joint columns")
    return output


def read_tsv(fname: str | Path, label: str) -> pd.DataFrame:
    path = Path(fname)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"{label} TSV is empty: {path}") from error
    duplicates = sorted(
        column for column in set(header) if header.count(column) > 1
    )
    if duplicates:
        raise ValueError(
            f"{label} TSV contains duplicate columns: " + ", ".join(duplicates)
        )
    return pd.read_csv(
        path,
        sep="\t",
        keep_default_na=False,
        low_memory=False,
        encoding="utf-8-sig",
    )


def write_tsv_atomic(
    dataframe: pd.DataFrame, output_fname: str | Path
) -> None:
    output_path = Path(output_fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_fname = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline=""
        ) as stream:
            dataframe.to_csv(stream, sep="\t", index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_fname, output_path)
    except Exception:
        if os.path.exists(temporary_fname):
            os.unlink(temporary_fname)
        raise


def build_layer_factorial_dataset(
    canonical_joint_fname: str | Path,
    layer_fname: str | Path,
    sentence_manifest_fname: str | Path,
    output_fname: str | Path,
    *,
    model: str,
    context_unit: str,
    lens_method: str,
    first_token_policy: str,
    lag_boundary: str,
    lag_padding: str | None = None,
    precomputed_frequency_fname: str | Path | None = None,
    paper_rt_fname: str | Path | None = None,
    word_frequency_fn: Callable[[str, str], float] | None = None,
) -> pd.DataFrame:
    """Read, build, and atomically publish one factorial merge artifact."""
    input_paths = [
        Path(canonical_joint_fname).resolve(),
        Path(layer_fname).resolve(),
        Path(sentence_manifest_fname).resolve(),
    ]
    if precomputed_frequency_fname is not None:
        input_paths.append(Path(precomputed_frequency_fname).resolve())
    if paper_rt_fname is not None:
        input_paths.append(Path(paper_rt_fname).resolve())
    output_path = Path(output_fname).resolve()
    if output_path in input_paths:
        raise ValueError("output file must not overwrite an input artifact")

    canonical_joint = read_tsv(canonical_joint_fname, "canonical joint")
    _, _, passages = _validate_joint(canonical_joint)
    sentence_map, _ = read_sentence_manifest(
        Path(sentence_manifest_fname), passages
    )
    layer_scores = read_tsv(layer_fname, "layer")
    frequency_table = (
        read_tsv(precomputed_frequency_fname, "precomputed frequency")
        if precomputed_frequency_fname is not None
        else None
    )
    paper_rt_table = (
        read_tsv(paper_rt_fname, "paper RT")
        if paper_rt_fname is not None
        else None
    )
    output = build_layer_factorial_dataframe(
        canonical_joint,
        layer_scores,
        sentence_map,
        model=model,
        context_unit=context_unit,
        lens_method=lens_method,
        first_token_policy=first_token_policy,
        lag_boundary=lag_boundary,
        lag_padding=lag_padding,
        frequency_table=frequency_table,
        word_frequency_fn=word_frequency_fn,
        paper_rt_table=paper_rt_table,
    )
    write_tsv_atomic(output, output_fname)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a canonical joint RT table with corrected and buggy "
            "internal-layer predictors for one factorial experiment cell."
        )
    )
    parser.add_argument("--canonical-joint-fname", required=True)
    parser.add_argument(
        "--layer-fname",
        "--internal-layer-fname",
        dest="layer_fname",
        required=True,
    )
    parser.add_argument(
        "--sentence-manifest-fname",
        "--sentence-map-fname",
        dest="sentence_manifest_fname",
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--context-unit", choices=CONTEXT_UNITS, required=True
    )
    parser.add_argument(
        "--lens-method", choices=LENS_METHODS, required=True
    )
    parser.add_argument(
        "--first-token-policy",
        "--sentence-first-token-policy",
        dest="first_token_policy",
        choices=FIRST_TOKEN_POLICIES,
        required=True,
    )
    parser.add_argument(
        "--lag-boundary",
        choices=LAG_BOUNDARIES,
        required=True,
        help=(
            "boundary for layer t-1/t-2/t-3 columns; explicit so extraction "
            "context and analysis spillover boundary can vary independently"
        ),
    )
    parser.add_argument(
        "--lag-padding",
        choices=LAG_PADDING_MODES,
        default=None,
        help=(
            "boundary padding; defaults to global-mean for sentence boundary "
            "and missing for text boundary"
        ),
    )
    parser.add_argument(
        "--precomputed-frequency-fname",
        help=(
            "optional one-based keyed TSV with word and exactly one of "
            "paper_log_gmean_freq or log_gmean_freq; otherwise pinned "
            "wordfreq 3.1.1 is required"
        ),
    )
    parser.add_argument(
        "--paper-rt-fname",
        help=(
            "optional official processed_RTs.tsv used to attach keyed "
            "meanItemRT as paper_time"
        ),
    )
    parser.add_argument("--output-fname", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_layer_factorial_dataset(
        args.canonical_joint_fname,
        args.layer_fname,
        args.sentence_manifest_fname,
        args.output_fname,
        model=args.model,
        context_unit=args.context_unit,
        lens_method=args.lens_method,
        first_token_policy=args.first_token_policy,
        lag_boundary=args.lag_boundary,
        lag_padding=args.lag_padding,
        precomputed_frequency_fname=args.precomputed_frequency_fname,
        paper_rt_fname=args.paper_rt_fname,
    )


if __name__ == "__main__":
    main()
