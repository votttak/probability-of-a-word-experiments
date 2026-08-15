"""Tests for balanced pilot sampling and strict joint predictor merging."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from h01_data.build_joint_pilot_dataset import (  # noqa: E402
    merge_predictor_families,
    merge_with_base_data,
)
from h01_data.create_joint_pilot_text import select_prefixes  # noqa: E402


class JointPilotSamplingTest(unittest.TestCase):
    def test_balanced_prefixes(self):
        self.assertEqual(
            select_prefixes([["a", "b", "c"], ["d", "e", "f"]], 2),
            [["a", "b"], ["d", "e"]],
        )

    def test_short_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot select"):
            select_prefixes([["a"]], 2)


class JointPilotMergeTest(unittest.TestCase):
    def setUp(self):
        self.ngram = pd.DataFrame({
            "text_id": [0, 0, 0, 0],
            "word_id": [0, 1, 2, 3],
            "word": ["a", "b", "c", "d"],
            "ngram_surprisal_context_0": [1.0, 2.0, 3.0, 4.0],
            "ngram_surprisal_context_2": [1.1, 2.1, 3.1, 4.1],
        })
        self.context = pd.DataFrame({
            "text_id": [0, 0, 0, 0],
            "word_id": [0, 1, 2, 3],
            "word": ["a", "b", "c", "d"],
            "context_limited_surprisal_context_1": [5.0, 6.0, 7.0, 8.0],
        })
        self.base = pd.DataFrame({
            "text_id": [1, 1, 1, 1, 1],
            "word_id": [0, 1, 2, 3, 4],
            "ref_token": ["a", "b", "c", "d", "extra"],
            "word": ["a", "b", "c", "d", "extra"],
            "time": [100.0, 110.0, 120.0, 130.0, 140.0],
            "word_len": [1, 1, 1, 1, 5],
            "freq": [9.0, 8.0, 7.0, 6.0, 5.0],
        })

    def test_merge_adds_text_bounded_spillovers(self):
        predictors, ngram_columns, context_columns = merge_predictor_families(
            self.ngram, self.context
        )
        merged = merge_with_base_data(
            self.base, predictors, ngram_columns + context_columns
        )
        self.assertEqual(len(merged), 4)
        self.assertTrue(pd.isna(merged.iloc[0]["prev_ngram_surprisal_context_0"]))
        self.assertEqual(
            merged.iloc[3]["prev3_context_limited_surprisal_context_1"], 5.0
        )

    def test_mismatched_keys_are_rejected(self):
        context = self.context.iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "keys do not match"):
            merge_predictor_families(self.ngram, context)

    def test_gapped_prefix_is_rejected(self):
        ngram = self.ngram.drop(index=1)
        context = self.context.drop(index=1)
        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            merge_predictor_families(ngram, context)

    def test_word_mismatch_and_nonfinite_values_are_rejected(self):
        context = self.context.copy()
        context.loc[1, "word"] = "wrong"
        with self.assertRaisesRegex(ValueError, "words do not match"):
            merge_predictor_families(self.ngram, context)

        ngram = self.ngram.copy()
        ngram.loc[1, "ngram_surprisal_context_0"] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            merge_predictor_families(ngram, self.context)


if __name__ == "__main__":
    unittest.main()
