"""N-GRAM: Network-free tests for scoring, alignment, and atomic output."""

import csv
import math
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


# N-GRAM: Import project modules without requiring package installation.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from h01_data.get_ngram_surprisals import (  # noqa: E402
    build_rows,
    required_queries,
    score_word,
    write_rows_atomic,
)
from h01_data.get_rt_with_surprisal_dataset import (  # noqa: E402
    merge_ngram_surprisals,
)


class NgramScoringTest(unittest.TestCase):
    """Verify sibling-compatible Stupid-Backoff mathematics."""

    def setUp(self):
        # N-GRAM: Counts are synthetic so tests are deterministic and offline.
        self.counts = {
            "": 1000,
            "a": 100,
            "b": 50,
            "c": 10,
            "d": 0,
            "a b": 20,
            "b c": 0,
            "a b c": 0,
        }

    def test_required_queries_reset_at_each_text(self):
        queries = required_queries([["a", "b", "c"], ["b"]], 2)
        self.assertEqual(
            set(queries),
            {"", "a", "b", "c", "a b", "b c", "a b c"},
        )

    def test_exact_bigram_ratio(self):
        value = score_word(["a", "b"], 1, 2, self.counts, 1000)
        self.assertAlmostEqual(value, -math.log(20 / 100))

    def test_multiple_backoff_steps(self):
        value = score_word(["a", "b", "c"], 2, 2, self.counts, 1000)
        self.assertAlmostEqual(value, -math.log((0.4 ** 2) * 10 / 1000))

    def test_unavailable_text_start_context_has_no_penalty(self):
        value = score_word(["a"], 0, 4, self.counts, 1000)
        self.assertAlmostEqual(value, -math.log(100 / 1000))

    def test_unseen_unigram_uses_pseudocount(self):
        value = score_word(["d"], 0, 0, self.counts, 1000)
        self.assertAlmostEqual(value, -math.log(1 / 1000))

    def test_rows_keep_zero_based_ids_and_context_columns(self):
        rows = build_rows(
            [["a", "b"], ["b"]],
            [0, 1],
            self.counts,
            backoff_alpha=0.4,
            unseen_unigram_count=1,
        )
        self.assertEqual(
            [(row["text_id"], row["word_id"], row["word"]) for row in rows],
            [(0, 0, "a"), (0, 1, "b"), (1, 0, "b")],
        )
        self.assertIn("ngram_surprisal_context_0", rows[0])
        self.assertIn("ngram_surprisal_context_1", rows[0])

    def test_atomic_writer_uses_expected_schema(self):
        rows = build_rows(
            [["a"]], [0], self.counts, 0.4, 1)
        with tempfile.TemporaryDirectory() as directory:
            output_fname = Path(directory) / "ngrams.tsv"
            write_rows_atomic(rows, output_fname, [0])
            with output_fname.open(encoding="utf8", newline="") as input_file:
                written = list(csv.DictReader(input_file, delimiter="\t"))
        self.assertEqual(written[0]["word"], "a")
        self.assertEqual(written[0]["text_id"], "0")


class NgramMergeTest(unittest.TestCase):
    """Verify keyed predictor alignment fails loudly on corrupt input."""

    def setUp(self):
        self.lm = pd.DataFrame({
            "text_id": [0, 0],
            "word_id": [0, 1],
            "word": ["a", "b"],
            "surprisal": [1.0, 2.0],
            "surprisal_buggy": [1.1, 2.1],
        })

    def write_ngram_file(self, directory, words=("a", "b")):
        fname = Path(directory) / "ngrams.tsv"
        pd.DataFrame({
            "text_id": [0, 0],
            "word_id": [0, 1],
            "word": list(words),
            "ngram_surprisal_context_0": [3.0, 4.0],
        }).to_csv(fname, sep="\t", index=False)
        return fname

    def test_valid_keyed_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            fname = self.write_ngram_file(directory)
            merged = merge_ngram_surprisals(self.lm, fname)
        self.assertEqual(list(merged["ngram_surprisal_context_0"]), [3.0, 4.0])

    def test_word_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fname = self.write_ngram_file(directory, words=("a", "wrong"))
            with self.assertRaisesRegex(ValueError, "words do not match"):
                merge_ngram_surprisals(self.lm, fname)


if __name__ == "__main__":
    unittest.main()
