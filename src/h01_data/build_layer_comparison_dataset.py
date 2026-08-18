#!/usr/bin/env python3

"""Merge n-gram, context-limited, and internal-layer predictors with RTs."""

import argparse
import re

import numpy as np
import pandas as pd

try:
    from .build_joint_pilot_dataset import (
        KEY_COLUMNS,
        merge_predictor_families,
        merge_with_base_data,
        validate_contiguous_prefixes,
        validate_predictor_table,
        write_tsv_atomic,
    )
except ImportError:  # Support direct execution from src/h01_data.
    from build_joint_pilot_dataset import (
        KEY_COLUMNS,
        merge_predictor_families,
        merge_with_base_data,
        validate_contiguous_prefixes,
        validate_predictor_table,
        write_tsv_atomic,
    )


LAYER_PATTERN = re.compile(r"internal_layer_surprisal_layer_(\d+)$")


def validate_layer_table(layer):
    """Validate L values and exclude the non-contextual embedding stream."""

    layer_columns = validate_predictor_table(
        layer, LAYER_PATTERN, "internal-layer"
    )
    layer_ids = [
        int(LAYER_PATTERN.fullmatch(column).group(1))
        for column in layer_columns
    ]
    if min(layer_ids) < 1:
        raise ValueError("internal-layer indices must start at transformer layer 1")
    if (layer[layer_columns] < 0).any().any():
        raise ValueError("internal-layer table contains negative surprisal values")
    return layer_columns


def merge_layer_predictor_families(ngram, context, layer):
    """Require identical coverage and words across all three families."""

    predictors, ngram_columns, context_columns = merge_predictor_families(
        ngram, context
    )
    layer = layer.copy()
    layer_columns = validate_layer_table(layer)
    merged = predictors.merge(
        layer[KEY_COLUMNS + ["word"] + layer_columns].rename(
            columns={"word": "layer_word"}
        ),
        on=KEY_COLUMNS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        raise ValueError(
            "n-gram/context-limited and internal-layer pilot keys do not match"
        )
    if not (merged["word"] == merged["layer_word"]).all():
        raise ValueError(
            "n-gram/context-limited and internal-layer pilot words do not match"
        )
    merged = merged.drop(columns=["layer_word", "_merge"])
    return merged, ngram_columns, context_columns, layer_columns


def merge_layers_with_joint_data(joint, layer, expected_final_layer=None,
                                 anchor_tolerance=5e-4,
                                 require_exact_joint_coverage=False,
                                 expected_rows=None):
    """Attach a keyed L table to an established full N+C+RT table."""

    joint = joint.copy()
    layer = layer.copy()
    if not np.isfinite(anchor_tolerance) or anchor_tolerance < 0:
        raise ValueError("anchor tolerance must be finite and nonnegative")
    joint = joint.loc[:, ~joint.columns.str.match(r"^Unnamed:")]
    required_joint = set(KEY_COLUMNS + ["time", "word_len", "freq"])
    missing = required_joint - set(joint.columns)
    if missing:
        raise ValueError(
            "joint table is missing columns: " + ", ".join(sorted(missing))
        )
    if joint.duplicated(KEY_COLUMNS, keep=False).any():
        raise ValueError("joint table contains duplicate keys")
    if expected_rows is not None and len(joint) != expected_rows:
        raise ValueError(
            f"joint table has {len(joint)} rows; expected {expected_rows}"
        )

    layer_columns = validate_layer_table(layer)
    validate_contiguous_prefixes(layer)
    if expected_rows is not None and len(layer) != expected_rows:
        raise ValueError(
            f"internal-layer table has {len(layer)} rows; expected {expected_rows}"
        )
    collisions = set(layer_columns).intersection(joint.columns)
    if collisions:
        raise ValueError(
            "joint table already contains internal-layer columns: "
            + ", ".join(sorted(collisions))
        )

    # Layer extraction uses zero-based passage IDs; the canonical Natural
    # Stories joint table uses the dataset's one-based passage IDs.
    layer["text_id"] = layer["text_id"] + 1
    if require_exact_joint_coverage:
        joint_keys = set(map(tuple, joint[KEY_COLUMNS].to_numpy()))
        layer_keys = set(map(tuple, layer[KEY_COLUMNS].to_numpy()))
        missing_from_layers = joint_keys - layer_keys
        missing_from_joint = layer_keys - joint_keys
        if missing_from_layers or missing_from_joint:
            raise ValueError(
                "joint/internal-layer keys are not identical: "
                f"{len(missing_from_layers)} missing from layers, "
                f"{len(missing_from_joint)} missing from joint"
            )
    layer = layer.rename(columns={"word": "layer_word"})
    merged = layer[KEY_COLUMNS + ["layer_word"] + layer_columns].merge(
        joint,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("joint table does not cover every internal-layer key")
    reference_column = "ref_token" if "ref_token" in merged.columns else "word"
    if reference_column not in merged.columns:
        raise ValueError("joint table has neither ref_token nor word")
    if not (merged["layer_word"] == merged[reference_column]).all():
        raise ValueError("internal-layer and joint-table words do not match")
    merged = merged.drop(columns=["layer_word", "_merge"])
    merged.sort_values(KEY_COLUMNS, kind="stable", inplace=True)

    final_column = (
        f"internal_layer_surprisal_layer_{expected_final_layer}"
        if expected_final_layer is not None else None
    )
    if expected_final_layer is not None and final_column not in merged.columns:
        raise ValueError(
            f"internal-layer table must include anchor column {final_column}"
        )
    if expected_final_layer is not None and "surprisal" not in merged.columns:
        raise ValueError(
            "joint table lacks ordinary surprisal for the final-layer anchor"
        )
    if (
        final_column is not None
        and "surprisal" in merged.columns
        and final_column in merged.columns
    ):
        differences = np.abs(
            merged[final_column].to_numpy(dtype=float)
            - pd.to_numeric(merged["surprisal"], errors="coerce").to_numpy()
        )
        finite = np.isfinite(differences)
        if not finite.all():
            raise ValueError("final-layer anchor contains non-finite values")
        if differences.max() > anchor_tolerance:
            raise ValueError(
                f"{final_column} differs from ordinary surprisal by "
                f"{differences.max():.6g}, above tolerance {anchor_tolerance}"
            )

    for variable in layer_columns:
        grouped = merged.groupby("text_id", sort=False)[variable]
        merged[f"prev_{variable}"] = grouped.shift(1)
        merged[f"prev2_{variable}"] = grouped.shift(2)
        merged[f"prev3_{variable}"] = grouped.shift(3)
    return merged


def build_layer_comparison_dataset(base_fname, ngram_fname, context_fname,
                                   layer_fname):
    """Read, validate, merge, and add text-bounded spillovers."""

    base = pd.read_csv(base_fname, sep="\t", keep_default_na=False,
                       na_values=[""])
    ngram = pd.read_csv(ngram_fname, sep="\t", keep_default_na=False)
    context = pd.read_csv(context_fname, sep="\t", keep_default_na=False)
    layer = pd.read_csv(layer_fname, sep="\t", keep_default_na=False)
    predictors, ngram_columns, context_columns, layer_columns = (
        merge_layer_predictor_families(ngram, context, layer)
    )
    return merge_with_base_data(
        base,
        predictors,
        ngram_columns + context_columns + layer_columns,
    )


def build_layer_comparison_from_joint(joint_fname, layer_fname,
                                      expected_final_layer=None,
                                      anchor_tolerance=5e-4,
                                      require_exact_joint_coverage=False,
                                      expected_rows=None):
    """Read a canonical N+C joint table and add a validated L table."""

    joint = pd.read_csv(
        joint_fname, sep="\t", keep_default_na=False, na_values=[""]
    )
    layer = pd.read_csv(layer_fname, sep="\t", keep_default_na=False)
    return merge_layers_with_joint_data(
        joint,
        layer,
        expected_final_layer=expected_final_layer,
        anchor_tolerance=anchor_tolerance,
        require_exact_joint_coverage=require_exact_joint_coverage,
        expected_rows=expected_rows,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build matched RT data for N-vs-L and C-vs-L evaluation"
    )
    parser.add_argument(
        "--joint-data-fname",
        help="existing canonical N+C+RT table to join on the layer keys",
    )
    parser.add_argument("--base-merged-fname")
    parser.add_argument("--ngram-surprisal-fname")
    parser.add_argument("--context-limited-surprisal-fname")
    parser.add_argument("--internal-layer-surprisal-fname", required=True)
    parser.add_argument("--output-fname", required=True)
    parser.add_argument("--anchor-tolerance", type=float, default=5e-4)
    parser.add_argument("--expected-final-layer", type=int)
    parser.add_argument(
        "--require-exact-joint-coverage", action="store_true"
    )
    parser.add_argument("--expected-rows", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.joint_data_fname:
        if any((
            args.base_merged_fname,
            args.ngram_surprisal_fname,
            args.context_limited_surprisal_fname,
        )):
            raise ValueError(
                "--joint-data-fname cannot be combined with separate base/N/C inputs"
            )
        dataframe = build_layer_comparison_from_joint(
            args.joint_data_fname,
            args.internal_layer_surprisal_fname,
            expected_final_layer=args.expected_final_layer,
            anchor_tolerance=args.anchor_tolerance,
            require_exact_joint_coverage=args.require_exact_joint_coverage,
            expected_rows=args.expected_rows,
        )
    else:
        joint_only_options = {
            "--expected-final-layer": args.expected_final_layer,
            "--require-exact-joint-coverage": (
                args.require_exact_joint_coverage
            ),
            "--expected-rows": args.expected_rows,
        }
        invalid = [
            name for name, value in joint_only_options.items()
            if value not in (None, False)
        ]
        if invalid:
            raise ValueError(
                "these options require --joint-data-fname: "
                + ", ".join(invalid)
            )
        required = {
            "--base-merged-fname": args.base_merged_fname,
            "--ngram-surprisal-fname": args.ngram_surprisal_fname,
            "--context-limited-surprisal-fname": (
                args.context_limited_surprisal_fname
            ),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                "separate-input mode requires " + ", ".join(missing)
            )
        dataframe = build_layer_comparison_dataset(
            args.base_merged_fname,
            args.ngram_surprisal_fname,
            args.context_limited_surprisal_fname,
            args.internal_layer_surprisal_fname,
        )
    write_tsv_atomic(dataframe, args.output_fname)


if __name__ == "__main__":
    main()
