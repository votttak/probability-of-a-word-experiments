"""N-GRAM: Tests for dynamic, text-bounded n-gram spillover columns."""

from pathlib import Path
import sys
import unittest

import pandas as pd


# N-GRAM: Import the merge module without installing the repository as a package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from h01_data.get_rt_with_surprisal_dataset import get_spillover_vars  # noqa: E402


class NgramSpilloverTest(unittest.TestCase):
    """Ensure shifts follow word order and never cross text boundaries."""

    def test_dynamic_ngram_spillovers(self):
        # N-GRAM: Input is deliberately unsorted to exercise the explicit sort.
        dataframe = pd.DataFrame({
            "text_id": [1, 0, 0],
            "word_id": [0, 1, 0],
            "word": ["c", "b", "a"],
            "surprisal": [3.0, 2.0, 1.0],
            "surprisal_buggy": [3.1, 2.1, 1.1],
            "freq": [30.0, 20.0, 10.0],
            "word_len": [1, 1, 1],
            "ngram_surprisal_context_2": [0.3, 0.2, 0.1],
        })

        get_spillover_vars(dataframe)

        self.assertEqual(list(dataframe["word"]), ["a", "b", "c"])
        self.assertTrue(pd.isna(dataframe.iloc[0]["prev_ngram_surprisal_context_2"]))
        self.assertEqual(
            dataframe.iloc[1]["prev_ngram_surprisal_context_2"], 0.1)
        self.assertTrue(pd.isna(dataframe.iloc[2]["prev_ngram_surprisal_context_2"]))


if __name__ == "__main__":
    unittest.main()
