#!/usr/bin/env python3

"""Merge matched n-gram/context-limited predictors into an RT pilot table."""

import argparse
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd


KEY_COLUMNS = ["text_id", "word_id"]
NGRAM_PATTERN = re.compile(r"ngram_surprisal_context_(\d+)$")
CONTEXT_PATTERN = re.compile(
    r"context_limited_surprisal_context_(\d+)$"
)


def predictor_columns(dataframe, pattern, family_name):
    """Return predictor columns in numeric context-length order."""

    columns = [column for column in dataframe.columns if pattern.fullmatch(column)]
    if not columns:
        raise ValueError(f"{family_name} table contains no predictor columns")
    return sorted(columns, key=lambda column: int(pattern.fullmatch(column).group(1)))


def validate_predictor_table(dataframe, pattern, family_name):
    """Validate keys, words, and finite numeric predictor values."""

    missing = set(KEY_COLUMNS + ["word"]) - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"{family_name} table is missing columns: {', '.join(sorted(missing))}"
        )
    duplicates = dataframe.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        raise ValueError(f"{family_name} table contains duplicate keys")

    columns = predictor_columns(dataframe, pattern, family_name)
    for column in columns:
        values = pd.to_numeric(dataframe[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{family_name} column {column} contains non-finite values")
        dataframe[column] = values
    return columns


def validate_contiguous_prefixes(dataframe):
    """Reject gaps that would make shift-based spillovers incorrect."""

    for text_id, group in dataframe.groupby("text_id", sort=False):
        word_ids = sorted(group["word_id"].tolist())
        expected = list(range(len(word_ids)))
        if word_ids != expected:
            raise ValueError(
                f"pilot keys for text {text_id} must be a contiguous prefix "
                "starting at word_id 0"
            )


def merge_predictor_families(ngram, context):
    """Require identical pilot coverage and merge the two predictor families."""

    ngram = ngram.copy()
    context = context.copy()
    ngram_columns = validate_predictor_table(
        ngram, NGRAM_PATTERN, "n-gram"
    )
    context_columns = validate_predictor_table(
        context, CONTEXT_PATTERN, "context-limited"
    )

    merged = ngram[KEY_COLUMNS + ["word"] + ngram_columns].merge(
        context[KEY_COLUMNS + ["word"] + context_columns].rename(
            columns={"word": "context_word"}
        ),
        on=KEY_COLUMNS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        raise ValueError("n-gram and context-limited pilot keys do not match")
    if not (merged["word"] == merged["context_word"]).all():
        raise ValueError("n-gram and context-limited pilot words do not match")

    merged = merged.drop(columns=["context_word", "_merge"])
    validate_contiguous_prefixes(merged)
    return merged, ngram_columns, context_columns


def merge_with_base_data(base, predictors, predictor_columns_all):
    """Attach pilot predictors to the one-based, full RT/control table."""

    base = base.copy()
    predictors = predictors.copy()
    base = base.loc[:, ~base.columns.str.match(r"^Unnamed:")]

    required_base = set(KEY_COLUMNS + ["time", "word_len", "freq"])
    missing = required_base - set(base.columns)
    if missing:
        raise ValueError(
            f"base RT table is missing columns: {', '.join(sorted(missing))}"
        )
    if base.duplicated(KEY_COLUMNS, keep=False).any():
        raise ValueError("base RT table contains duplicate keys")

    # Predictor generators use zero-based text IDs; established merged RT files
    # use the source dataset's one-based Natural Stories text IDs.
    predictors["text_id"] = predictors["text_id"] + 1
    predictors = predictors.rename(columns={"word": "pilot_word"})
    merged = predictors.merge(
        base,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("base RT table does not cover every pilot key")

    reference_column = "ref_token" if "ref_token" in merged.columns else "word"
    if reference_column not in merged.columns:
        raise ValueError("base RT table has neither ref_token nor word")
    if not (merged["pilot_word"] == merged[reference_column]).all():
        raise ValueError("pilot and base RT words do not match")

    merged = merged.drop(columns=["pilot_word", "_merge"])
    merged.sort_values(KEY_COLUMNS, kind="stable", inplace=True)
    for variable in predictor_columns_all:
        grouped = merged.groupby("text_id", sort=False)[variable]
        merged[f"prev_{variable}"] = grouped.shift(1)
        merged[f"prev2_{variable}"] = grouped.shift(2)
        merged[f"prev3_{variable}"] = grouped.shift(3)
    return merged


def build_joint_pilot_dataset(base_fname, ngram_fname, context_fname):
    """Read, validate, and merge all joint-pilot inputs."""

    base = pd.read_csv(base_fname, sep="\t", keep_default_na=False, na_values=[""])
    ngram = pd.read_csv(ngram_fname, sep="\t", keep_default_na=False)
    context = pd.read_csv(context_fname, sep="\t", keep_default_na=False)
    predictors, ngram_columns, context_columns = merge_predictor_families(
        ngram, context
    )
    return merge_with_base_data(
        base, predictors, ngram_columns + context_columns
    )


def write_tsv_atomic(dataframe, output_fname):
    """Write the completed table atomically."""

    output_path = Path(output_fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_fname = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    try:
        dataframe.to_csv(temporary_fname, sep="\t", index=False)
        os.replace(temporary_fname, output_path)
    except Exception:
        if os.path.exists(temporary_fname):
            os.unlink(temporary_fname)
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build matched RT data for joint surprisal evaluation"
    )
    parser.add_argument("--base-merged-fname", required=True)
    parser.add_argument("--ngram-surprisal-fname", required=True)
    parser.add_argument("--context-limited-surprisal-fname", required=True)
    parser.add_argument("--output-fname", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dataframe = build_joint_pilot_dataset(
        args.base_merged_fname,
        args.ngram_surprisal_fname,
        args.context_limited_surprisal_fname,
    )
    write_tsv_atomic(dataframe, args.output_fname)


if __name__ == "__main__":
    main()
