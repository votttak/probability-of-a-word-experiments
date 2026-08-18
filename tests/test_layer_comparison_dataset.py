"""Tests for strict N/C/L predictor merging and spillover creation."""

from pathlib import Path
import sys
import unittest

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from h01_data.build_layer_comparison_dataset import (  # noqa: E402
    merge_layer_predictor_families,
    merge_layers_with_joint_data,
)
from h01_data.build_joint_pilot_dataset import merge_with_base_data  # noqa: E402


class LayerComparisonMergeTest(unittest.TestCase):
    def setUp(self):
        keys = {
            "text_id": [0, 0, 0, 0],
            "word_id": [0, 1, 2, 3],
            "word": ["a", "b", "c", "d"],
        }
        self.ngram = pd.DataFrame({
            **keys,
            "ngram_surprisal_context_0": [1.0, 2.0, 3.0, 4.0],
        })
        self.context = pd.DataFrame({
            **keys,
            "context_limited_surprisal_context_1": [5.0, 6.0, 7.0, 8.0],
        })
        self.layer = pd.DataFrame({
            **keys,
            "internal_layer_surprisal_layer_1": [9.0, 10.0, 11.0, 12.0],
            "internal_layer_surprisal_layer_12": [4.0, 3.0, 2.0, 1.0],
        })
        self.base = pd.DataFrame({
            "text_id": [1, 1, 1, 1],
            "word_id": [0, 1, 2, 3],
            "ref_token": ["a", "b", "c", "d"],
            "time": [100.0, 110.0, 120.0, 130.0],
            "word_len": [1.0] * 4,
            "freq": [4.0, 3.0, 2.0, 1.0],
        })

    def test_merge_adds_layer_spillovers(self):
        predictors, ngram_columns, context_columns, layer_columns = (
            merge_layer_predictor_families(
                self.ngram, self.context, self.layer
            )
        )
        merged = merge_with_base_data(
            self.base,
            predictors,
            ngram_columns + context_columns + layer_columns,
        )
        self.assertEqual(layer_columns, [
            "internal_layer_surprisal_layer_1",
            "internal_layer_surprisal_layer_12",
        ])
        self.assertEqual(
            merged.iloc[3]["prev3_internal_layer_surprisal_layer_12"], 4.0
        )

    def test_layer_coverage_and_words_must_match(self):
        with self.assertRaisesRegex(ValueError, "keys do not match"):
            merge_layer_predictor_families(
                self.ngram, self.context, self.layer.iloc[:-1]
            )
        layer = self.layer.copy()
        layer.loc[2, "word"] = "wrong"
        with self.assertRaisesRegex(ValueError, "words do not match"):
            merge_layer_predictor_families(
                self.ngram, self.context, layer
            )

    def test_canonical_joint_subset_and_final_layer_anchor(self):
        joint = self.base.copy()
        joint["surprisal"] = self.layer[
            "internal_layer_surprisal_layer_12"
        ].to_numpy()
        merged = merge_layers_with_joint_data(
            joint,
            self.layer,
            expected_final_layer=12,
            anchor_tolerance=1e-8,
        )
        self.assertEqual(len(merged), 4)
        self.assertEqual(
            merged.iloc[3]["prev3_internal_layer_surprisal_layer_1"], 9.0
        )

        bad_layer = self.layer.copy()
        bad_layer.loc[0, "internal_layer_surprisal_layer_12"] += 0.1
        with self.assertRaisesRegex(ValueError, "above tolerance"):
            merge_layers_with_joint_data(
                joint,
                bad_layer,
                expected_final_layer=12,
                anchor_tolerance=1e-8,
            )

    def test_full_mode_requires_identical_key_coverage_and_row_count(self):
        joint = self.base.copy()
        joint["surprisal"] = self.layer[
            "internal_layer_surprisal_layer_12"
        ].to_numpy()
        with self.assertRaisesRegex(ValueError, "expected 4"):
            merge_layers_with_joint_data(
                joint,
                self.layer.iloc[:-1],
                expected_final_layer=12,
                require_exact_joint_coverage=True,
                expected_rows=4,
            )
        with self.assertRaisesRegex(ValueError, "keys are not identical"):
            merge_layers_with_joint_data(
                joint,
                self.layer.iloc[:-1],
                expected_final_layer=12,
                require_exact_joint_coverage=True,
            )

    def test_embedding_stream_and_negative_values_are_rejected(self):
        layer = self.layer.rename(columns={
            "internal_layer_surprisal_layer_1":
                "internal_layer_surprisal_layer_0"
        })
        with self.assertRaisesRegex(ValueError, "transformer layer 1"):
            merge_layer_predictor_families(
                self.ngram, self.context, layer
            )
        layer = self.layer.copy()
        layer.loc[0, "internal_layer_surprisal_layer_1"] = -0.1
        with self.assertRaisesRegex(ValueError, "negative"):
            merge_layer_predictor_families(
                self.ngram, self.context, layer
            )


if __name__ == "__main__":
    unittest.main()
