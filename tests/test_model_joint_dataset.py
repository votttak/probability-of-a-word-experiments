"""Tests for model-specific reconstruction of the canonical N+C joint table."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from h01_data.build_model_joint_dataset import (  # noqa: E402
    build_model_joint,
    build_model_joint_dataset,
)


class ModelJointDatasetTest(unittest.TestCase):
    def setUp(self):
        self.zero_keys = [
            (0, 0, "a"), (0, 1, "b"), (0, 2, "c"), (0, 3, "d"),
            (1, 0, "e"), (1, 1, "f"),
        ]
        self.canonical = pd.DataFrame({
            "text_id": [text_id + 1 for text_id, _, _ in self.zero_keys],
            "word_id": [word_id for _, word_id, _ in self.zero_keys],
            "ref_token": [word for _, _, word in self.zero_keys],
            "word": [word for _, _, word in self.zero_keys],
            "time": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "freq": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "ngram_surprisal_context_0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "context_limited_surprisal_context_1": [90.0] * 6,
            "context_limited_surprisal_context_2": [91.0] * 6,
            "surprisal": [92.0] * 6,
            "surprisal_buggy": [93.0] * 6,
        })
        for column in (
            "ngram_surprisal_context_0",
            "context_limited_surprisal_context_1",
            "context_limited_surprisal_context_2",
            "surprisal",
            "surprisal_buggy",
        ):
            grouped = self.canonical.groupby("text_id", sort=False)[column]
            self.canonical[f"prev_{column}"] = grouped.shift(1)
            self.canonical[f"prev2_{column}"] = grouped.shift(2)
            self.canonical[f"prev3_{column}"] = grouped.shift(3)

        self.context = pd.DataFrame({
            "text_id": [row[0] for row in self.zero_keys],
            "word_id": [row[1] for row in self.zero_keys],
            "word": [row[2] for row in self.zero_keys],
            "context_limited_surprisal_context_1": np.arange(6) + 10.0,
            "context_limited_surprisal_context_2": np.arange(6) + 20.0,
        }).iloc[[4, 1, 5, 0, 3, 2]].reset_index(drop=True)
        self.reference = pd.DataFrame({
            "text_id": [row[0] for row in self.zero_keys],
            "word_id": [row[1] for row in self.zero_keys],
            "word": [row[2] for row in self.zero_keys],
            "surprisal": np.arange(6) + 30.0,
            "surprisal_buggy": np.arange(6) + 40.0,
        }).iloc[::-1].reset_index(drop=True)

    def test_replaces_model_columns_and_rebuilds_text_bounded_spillovers(self):
        original_columns = self.canonical.columns.tolist()
        result = build_model_joint(
            self.canonical, self.context, self.reference, expected_rows=6
        )

        self.assertEqual(result.columns.tolist(), original_columns)
        self.assertEqual(result["ref_token"].tolist(), list("abcdef"))
        self.assertEqual(
            result["context_limited_surprisal_context_1"].tolist(),
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        )
        self.assertEqual(
            result["surprisal"].tolist(),
            [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
        )
        self.assertEqual(
            result["surprisal_buggy"].tolist(),
            [40.0, 41.0, 42.0, 43.0, 44.0, 45.0],
        )
        self.assertEqual(result.loc[3, "prev3_surprisal"], 30.0)
        self.assertTrue(pd.isna(result.loc[4, "prev_surprisal"]))
        self.assertEqual(
            result.loc[5, "prev_context_limited_surprisal_context_2"], 24.0
        )

        # Model-neutral RT/control and common-N values and lags are untouched.
        for column in (
            "time", "freq", "ngram_surprisal_context_0",
            "prev_ngram_surprisal_context_0",
        ):
            pd.testing.assert_series_equal(
                result[column], self.canonical[column], check_names=True
            )

    def test_requires_exact_key_and_word_coverage(self):
        wrong_keys = self.context.copy()
        wrong_keys.loc[0, "text_id"] = 99
        with self.assertRaisesRegex(ValueError, "key coverage"):
            build_model_joint(
                self.canonical,
                wrong_keys,
                self.reference,
                expected_rows=6,
            )

        reference = self.reference.copy()
        reference.loc[reference["word_id"] == 2, "word"] = "wrong"
        with self.assertRaisesRegex(ValueError, "word mismatch"):
            build_model_joint(
                self.canonical, self.context, reference, expected_rows=6
            )

    def test_rejects_schema_and_numeric_provenance_gaps(self):
        missing_context = self.context.drop(
            columns=["context_limited_surprisal_context_2"]
        )
        with self.assertRaisesRegex(ValueError, "predictor columns differ"):
            build_model_joint(
                self.canonical, missing_context, self.reference, expected_rows=6
            )

        missing_buggy = self.reference.drop(columns=["surprisal_buggy"])
        with self.assertRaisesRegex(ValueError, "surprisal_buggy presence"):
            build_model_joint(
                self.canonical, self.context, missing_buggy, expected_rows=6
            )

        nonfinite = self.context.copy()
        nonfinite.loc[0, "context_limited_surprisal_context_1"] = np.inf
        with self.assertRaisesRegex(ValueError, "not finite numeric"):
            build_model_joint(
                self.canonical, nonfinite, self.reference, expected_rows=6
            )

    def test_file_entry_point_writes_complete_output_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_fname = root / "canonical.tsv"
            context_fname = root / "context.tsv"
            reference_fname = root / "reference.tsv"
            output_fname = root / "model-joint.tsv"
            self.canonical.to_csv(canonical_fname, sep="\t", index=False)
            self.context.to_csv(context_fname, sep="\t", index=False)
            self.reference.to_csv(reference_fname, sep="\t", index=False)

            build_model_joint_dataset(
                canonical_fname,
                context_fname,
                reference_fname,
                output_fname,
                expected_rows=6,
            )
            observed = pd.read_csv(output_fname, sep="\t")
            self.assertEqual(observed["surprisal"].tolist(), [
                30.0, 31.0, 32.0, 33.0, 34.0, 35.0,
            ])
            self.assertEqual(list(root.glob(".model-joint.tsv.*.tmp")), [])
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                build_model_joint_dataset(
                    canonical_fname,
                    context_fname,
                    reference_fname,
                    canonical_fname,
                    expected_rows=6,
                )

    def test_pilot_filters_full_canonical_and_context_with_full_or_pilot_reference(self):
        pilot_reference = self.reference.loc[
            self.reference["word_id"] < 2
        ].copy()
        from_pilot_reference = build_model_joint(
            self.canonical,
            self.context,
            pilot_reference,
            expected_rows=4,
            words_per_text=2,
        )
        from_full_reference = build_model_joint(
            self.canonical,
            self.context,
            self.reference,
            expected_rows=4,
            words_per_text=2,
        )

        self.assertEqual(
            list(zip(
                from_pilot_reference["text_id"],
                from_pilot_reference["word_id"],
            )),
            [(1, 0), (1, 1), (2, 0), (2, 1)],
        )
        pd.testing.assert_frame_equal(
            from_pilot_reference.reset_index(drop=True),
            from_full_reference.reset_index(drop=True),
        )
        self.assertTrue(pd.isna(
            from_pilot_reference.iloc[2]["prev_surprisal"]
        ))

    def test_pilot_requires_positive_words_per_text_and_exact_filtered_rows(self):
        for invalid in (0, -1):
            with self.subTest(words_per_text=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    build_model_joint(
                        self.canonical,
                        self.context,
                        self.reference,
                        expected_rows=4,
                        words_per_text=invalid,
                    )
        with self.assertRaisesRegex(ValueError, "after filtering; expected 5"):
            build_model_joint(
                self.canonical,
                self.context,
                self.reference,
                expected_rows=5,
                words_per_text=2,
            )


if __name__ == "__main__":
    unittest.main()
