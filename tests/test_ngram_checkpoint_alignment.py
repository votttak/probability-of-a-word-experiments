"""N-GRAM: Optional alignment checks against locally available checkpoints."""

import csv
from pathlib import Path
import sys
import unittest


# N-GRAM: This test skips clean checkouts without generated checkpoints, while
# validating all four English datasets in the experiment workspace.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from h01_data.get_ngram_surprisals import read_texts  # noqa: E402
from h01_data.get_rt_with_surprisal_dataset import gpt2_byte_encode  # noqa: E402


class ExistingCheckpointAlignmentTest(unittest.TestCase):
    """Ensure whitespace word IDs exactly match wordsprobability output."""

    def test_english_dataset_word_alignment(self):
        datasets = ("natural_stories", "provo", "dundee", "brown")
        for dataset in datasets:
            text_fname = (
                REPOSITORY_ROOT / "checkpoints" / "rt" / "text_rt_data" /
                f"{dataset}.txt"
            )
            lm_fname = (
                REPOSITORY_ROOT / "checkpoints" / "rt" /
                "surprisals_rt_data" / f"suprisal-{dataset}-pythia-70m.tsv"
            )
            if not text_fname.exists() or not lm_fname.exists():
                self.skipTest("generated English checkpoints are not available")

            expected = [
                (text_id, word_id, word)
                for text_id, words in enumerate(read_texts(text_fname))
                for word_id, word in enumerate(words)
            ]
            with lm_fname.open(encoding="utf8", newline="") as input_file:
                actual = [
                    (int(row["text_id"]), int(row["word_id"]), row["word"])
                    for row in csv.DictReader(input_file, delimiter="\t")
                ]

            with self.subTest(dataset=dataset):
                # N-GRAM: IDs must match literally. Words may match literally or
                # through wordsprobability's reversible GPT byte alphabet.
                self.assertEqual(
                    [(text_id, word_id) for text_id, word_id, _ in expected],
                    [(text_id, word_id) for text_id, word_id, _ in actual],
                )
                for expected_row, actual_row in zip(expected, actual):
                    raw_word = expected_row[2]
                    lm_word = actual_row[2]
                    self.assertIn(lm_word, {raw_word, gpt2_byte_encode(raw_word)})


if __name__ == "__main__":
    unittest.main()
