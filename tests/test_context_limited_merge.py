"""CONTEXT-LIMITED: Tests for strict keyed merge and bounded spillovers."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


# CONTEXT-LIMITED: Import the merge module without packaging the repository.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from h01_data.get_rt_with_surprisal_dataset import (  # noqa: E402
    get_spillover_vars,
    merge_context_limited_surprisals,
)


class ContextLimitedMergeTest(unittest.TestCase):
    """CONTEXT-LIMITED: Ensure corrupt or misaligned predictors fail loudly."""

    def setUp(self):
        self.lm = pd.DataFrame({
            "text_id": [0, 0],
            "word_id": [0, 1],
            "word": ["alpha", "beta"],
            "surprisal": [1.0, 2.0],
            "surprisal_buggy": [1.1, 2.1],
        })

    @staticmethod
    def write_context_file(directory, words=("alpha", "beta"),
                           values=(3.0, 4.0), word_ids=(0, 1)):
        fname = Path(directory) / "context.tsv"
        pd.DataFrame({
            "text_id": [0, 0],
            "word_id": list(word_ids),
            "word": list(words),
            "context_limited_surprisal_context_1": list(values),
        }).to_csv(fname, sep="\t", index=False)
        return fname

    def test_valid_keyed_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            fname = self.write_context_file(directory)
            merged = merge_context_limited_surprisals(self.lm, fname)
        self.assertEqual(
            list(merged["context_limited_surprisal_context_1"]), [3.0, 4.0])

    def test_word_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fname = self.write_context_file(
                directory, words=("alpha", "wrong"))
            with self.assertRaisesRegex(ValueError, "words do not match"):
                merge_context_limited_surprisals(self.lm, fname)

    def test_missing_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fname = Path(directory) / "context.tsv"
            pd.DataFrame({
                "text_id": [0],
                "word_id": [0],
                "word": ["alpha"],
                "context_limited_surprisal_context_1": [3.0],
            }).to_csv(fname, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "keys do not match"):
                merge_context_limited_surprisals(self.lm, fname)

    def test_duplicate_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fname = self.write_context_file(directory, word_ids=(0, 0))
            with self.assertRaisesRegex(ValueError, "duplicate predictor keys"):
                merge_context_limited_surprisals(self.lm, fname)

    def test_nonfinite_and_negative_values_are_rejected(self):
        for bad_value, message in [(np.inf, "non-finite"), (-0.1, "negative")]:
            with self.subTest(bad_value=bad_value):
                with tempfile.TemporaryDirectory() as directory:
                    fname = self.write_context_file(
                        directory, values=(3.0, bad_value))
                    with self.assertRaisesRegex(ValueError, message):
                        merge_context_limited_surprisals(self.lm, fname)


class ContextLimitedSpilloverTest(unittest.TestCase):
    """CONTEXT-LIMITED: Ensure shifts sort words and reset at each passage."""

    def test_dynamic_context_limited_spillovers(self):
        # CONTEXT-LIMITED: Deliberately unsorted input exercises stable sorting.
        dataframe = pd.DataFrame({
            "text_id": [1, 0, 0],
            "word_id": [0, 1, 0],
            "word": ["gamma", "beta", "alpha"],
            "surprisal": [3.0, 2.0, 1.0],
            "surprisal_buggy": [3.1, 2.1, 1.1],
            "freq": [30.0, 20.0, 10.0],
            "word_len": [5, 4, 5],
            "context_limited_surprisal_context_2": [0.3, 0.2, 0.1],
        })

        get_spillover_vars(dataframe)

        column = "prev_context_limited_surprisal_context_2"
        self.assertEqual(list(dataframe["word"]), ["alpha", "beta", "gamma"])
        self.assertTrue(pd.isna(dataframe.iloc[0][column]))
        self.assertEqual(dataframe.iloc[1][column], 0.1)
        self.assertTrue(pd.isna(dataframe.iloc[2][column]))


if __name__ == "__main__":
    unittest.main()
