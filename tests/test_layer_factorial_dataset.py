import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from h01_data import build_layer_factorial_dataset as factorial
from h01_data import build_natural_stories_sentence_manifest as sentence_manifest


WORDS = ["A", "bb", "CCC", "d", "ee"]
RAW_FREQUENCIES = {
    "A": 0.1,
    "bb": 0.2,
    "CCC": 0.3,
    "d": 0.4,
    "ee": 0.5,
}


def fake_word_frequency(word, language):
    if language != "en":
        raise AssertionError("expected English frequency lookup")
    return RAW_FREQUENCIES[word]


def make_joint():
    return pd.DataFrame(
        {
            "text_id": [1, 1, 1, 1, 2],
            "word_id": [0, 1, 2, 3, 0],
            "ref_token": WORDS,
            "time": [100.0, 110.0, 120.0, 130.0, 140.0],
            "custom_joint_column": ["a", "b", "c", "d", "e"],
        }
    )


def make_layer(start_layer=0):
    layer_ids = [start_layer, start_layer + 1]
    corrected = [
        [1.0, 2.0, 9.0, 4.0, 5.0],
        [10.0, 20.0, 90.0, 40.0, 50.0],
    ]
    buggy = [
        [2.0, 4.0, 18.0, 8.0, 10.0],
        [20.0, 40.0, 180.0, 80.0, 100.0],
    ]
    data = {
        "text_id": [0, 0, 0, 0, 1],
        "word_id": [0, 1, 2, 3, 0],
        "word": WORDS,
    }
    for index, layer_id in enumerate(layer_ids):
        data[f"{factorial.CORRECTED_PREFIX}{layer_id}"] = corrected[index]
    for index, layer_id in enumerate(layer_ids):
        data[f"{factorial.BUGGY_PREFIX}{layer_id}"] = buggy[index]
    return pd.DataFrame(data)


def make_sentence_map():
    return {
        0: [
            factorial.SentenceUnit(0, (0, 1), ("A", "bb")),
            factorial.SentenceUnit(1, (2, 3), ("CCC", "d")),
        ],
        1: [factorial.SentenceUnit(0, (0,), ("ee",))],
    }


def build(layer=None, **overrides):
    options = {
        "model": "fixture-model",
        "context_unit": "sentence",
        "lens_method": "logit-lens",
        "first_token_policy": "bow",
        "lag_boundary": "sentence",
        "word_frequency_fn": fake_word_frequency,
    }
    options.update(overrides)
    return factorial.build_layer_factorial_dataframe(
        make_joint(),
        make_layer() if layer is None else layer,
        make_sentence_map(),
        **options,
    )


class LayerFactorialDatasetTests(unittest.TestCase):
    def test_sentence_global_mean_lags_metadata_controls_and_aliases(self):
        joint = make_joint()
        output = build()
        self.assertEqual(
            output.columns[: len(joint.columns)].tolist(),
            joint.columns.tolist(),
        )
        pd.testing.assert_frame_equal(
            output[joint.columns], joint, check_dtype=False
        )
        self.assertEqual(output["sentence_id"].tolist(), [0, 0, 1, 1, 0])
        self.assertEqual(
            output["sentence_word_id"].tolist(), [0, 1, 0, 1, 0]
        )

        scalar_expected = {
            "model": "fixture-model",
            "context_unit": "sentence",
            "lens_method": "logit-lens",
            "first_token_policy": "bow",
            "sentence_first_token_policy": "bow",
            "include_embedding_layer": True,
            "lag_boundary": "sentence",
            "lag_padding": "global-mean",
        }
        for column, expected in scalar_expected.items():
            self.assertEqual(output[column].nunique(), 1)
            self.assertEqual(output[column].iloc[0], expected)

        predictor = f"{factorial.CORRECTED_PREFIX}0"
        predictor_mean = np.mean([1.0, 2.0, 9.0, 4.0, 5.0])
        np.testing.assert_allclose(
            output[f"prev_{predictor}"],
            [predictor_mean, 1.0, predictor_mean, 9.0, predictor_mean],
        )
        np.testing.assert_allclose(
            output[f"prev2_{predictor}"], [predictor_mean] * 5
        )
        np.testing.assert_allclose(
            output[f"prev3_{predictor}"], [predictor_mean] * 5
        )
        buggy = f"{factorial.BUGGY_PREFIX}1"
        self.assertIn(f"prev_{buggy}", output.columns)
        self.assertIn(f"prev2_{buggy}", output.columns)
        self.assertIn(f"prev3_{buggy}", output.columns)

        np.testing.assert_array_equal(
            output["paper_length"], [1, 2, 3, 1, 2]
        )
        np.testing.assert_allclose(
            output["paper_length_prev_1"], [1.8, 1.0, 1.8, 3.0, 1.8]
        )
        np.testing.assert_allclose(
            output["paper_length_prev_2"], [1.8] * 5
        )
        expected_frequency = np.log(
            np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
            + factorial.WORDFREQ_EPSILON
        )
        np.testing.assert_allclose(
            output["paper_log_gmean_freq"], expected_frequency
        )
        frequency_mean = float(expected_frequency.mean())
        np.testing.assert_allclose(
            output["paper_log_gmean_freq_prev_1"],
            [
                frequency_mean,
                expected_frequency[0],
                frequency_mean,
                expected_frequency[2],
                frequency_mean,
            ],
        )
        np.testing.assert_allclose(
            output["paper_log_gmean_freq_prev_2"], [frequency_mean] * 5
        )
        for canonical, alias in (
            ("paper_length", "length"),
            ("paper_length_prev_1", "length_prev_1"),
            ("paper_length_prev_2", "length_prev_2"),
            ("paper_log_gmean_freq", "log_gmean_freq"),
            ("paper_log_gmean_freq_prev_1", "log_gmean_freq_prev_1"),
            ("paper_log_gmean_freq_prev_2", "log_gmean_freq_prev_2"),
        ):
            np.testing.assert_allclose(output[canonical], output[alias])

    def test_text_boundary_defaults_to_missing_and_crosses_sentences(self):
        output = build(
            context_unit="passage",
            first_token_policy="bos",
            lag_boundary="text",
        )
        predictor = f"{factorial.CORRECTED_PREFIX}0"
        np.testing.assert_allclose(
            output[f"prev_{predictor}"],
            [np.nan, 1.0, 2.0, 9.0, np.nan],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            output[f"prev2_{predictor}"],
            [np.nan, np.nan, 1.0, 2.0, np.nan],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            output[f"prev3_{predictor}"],
            [np.nan, np.nan, np.nan, 1.0, np.nan],
            equal_nan=True,
        )
        self.assertEqual(output["lag_padding"].iloc[0], "missing")
        # Paper controls remain sentence-bounded and globally padded.
        self.assertEqual(output["paper_length_prev_1"].iloc[2], 1.8)

    def test_layer_families_allow_one_based_complete_range(self):
        output = build(layer=make_layer(start_layer=1))
        self.assertFalse(bool(output["include_embedding_layer"].iloc[0]))
        self.assertIn(f"{factorial.CORRECTED_PREFIX}1", output)
        self.assertIn(f"{factorial.BUGGY_PREFIX}2", output)

    def test_file_builder_accepts_corrected_only_extraction_table(self):
        corrected_only = make_layer().drop(
            columns=[
                column
                for column in make_layer().columns
                if column.startswith(factorial.BUGGY_PREFIX)
            ]
        )
        manifest_rows = [
            {
                "text_id": text_id,
                "sentence_id": sentence.sentence_id,
                "sentence_word_id": sentence_word_id,
                "word_id": word_id,
                "word": word,
            }
            for text_id, sentences in make_sentence_map().items()
            for sentence in sentences
            for sentence_word_id, (word_id, word) in enumerate(
                zip(sentence.word_ids, sentence.words)
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            joint_path = directory / "joint.tsv"
            layer_path = directory / "corrected-only-layer.tsv"
            manifest_path = directory / "manifest.tsv"
            output_path = directory / "output.tsv"
            make_joint().to_csv(joint_path, sep="\t", index=False)
            corrected_only.to_csv(layer_path, sep="\t", index=False)
            sentence_manifest.write_manifest(manifest_rows, manifest_path)

            output = factorial.build_layer_factorial_dataset(
                joint_path,
                layer_path,
                manifest_path,
                output_path,
                model="fixture-model",
                context_unit="sentence",
                lens_method="logit-lens",
                first_token_policy="bow",
                lag_boundary="sentence",
                word_frequency_fn=fake_word_frequency,
            )
            self.assertTrue(output_path.is_file())

        corrected_columns = [
            f"{factorial.CORRECTED_PREFIX}{layer}" for layer in (0, 1)
        ]
        for column in corrected_columns:
            self.assertIn(column, output)
            for lag_prefix in ("prev_", "prev2_", "prev3_"):
                self.assertIn(f"{lag_prefix}{column}", output)
        self.assertFalse(
            any(
                factorial.BUGGY_PREFIX in column
                for column in output.columns
            )
        )
        np.testing.assert_allclose(
            output[corrected_columns[0]], corrected_only[corrected_columns[0]]
        )

    def test_rejects_mismatched_gapped_and_invalid_layer_families(self):
        mismatched = make_layer()
        mismatched = mismatched.drop(columns=f"{factorial.BUGGY_PREFIX}1")
        with self.assertRaisesRegex(ValueError, "identical layer IDs"):
            build(layer=mismatched)

        gapped = make_layer().rename(
            columns={
                f"{factorial.CORRECTED_PREFIX}1": (
                    f"{factorial.CORRECTED_PREFIX}2"
                ),
                f"{factorial.BUGGY_PREFIX}1": f"{factorial.BUGGY_PREFIX}2",
            }
        )
        with self.assertRaisesRegex(ValueError, "complete and consecutive"):
            build(layer=gapped)

        negative = make_layer()
        negative.loc[0, f"{factorial.CORRECTED_PREFIX}0"] = -0.1
        with self.assertRaisesRegex(ValueError, "negative"):
            build(layer=negative)

        nonfinite = make_layer()
        nonfinite.loc[0, f"{factorial.BUGGY_PREFIX}0"] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            build(layer=nonfinite)

    def test_rejects_layer_key_word_and_manifest_coverage_mismatches(self):
        missing_key = make_layer().iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "key coverage differs"):
            build(layer=missing_key)

        wrong_word = make_layer()
        wrong_word.loc[1, "word"] = "wrong"
        with self.assertRaisesRegex(ValueError, "word mismatch"):
            build(layer=wrong_word)

        wrong_map = make_sentence_map()
        wrong_map[0] = [
            factorial.SentenceUnit(0, (0, 1), ("A", "WRONG")),
            factorial.SentenceUnit(1, (2, 3), ("CCC", "d")),
        ]
        with self.assertRaisesRegex(ValueError, "does not flatten exactly"):
            factorial.build_layer_factorial_dataframe(
                make_joint(),
                make_layer(),
                wrong_map,
                model="fixture",
                context_unit="sentence",
                lens_method="logit-lens",
                first_token_policy="bow",
                lag_boundary="sentence",
                word_frequency_fn=fake_word_frequency,
            )

    def test_precomputed_frequency_bypasses_wordfreq_and_validates_alignment(self):
        frequency = pd.DataFrame(
            {
                "text_id": [1, 1, 1, 1, 2],
                "word_id": [0, 1, 2, 3, 0],
                "word": WORDS,
                "paper_log_gmean_freq": [-1.0, -2.0, -3.0, -4.0, -5.0],
            }
        )
        with patch.object(
            factorial,
            "_load_word_frequency",
            side_effect=AssertionError("wordfreq must not load"),
        ):
            output = factorial.build_layer_factorial_dataframe(
                make_joint(),
                make_layer(),
                make_sentence_map(),
                model="fixture",
                context_unit="sentence",
                lens_method="logit-lens",
                first_token_policy="bow",
                lag_boundary="sentence",
                frequency_table=frequency,
            )
        np.testing.assert_allclose(
            output["paper_log_gmean_freq"], [-1, -2, -3, -4, -5]
        )

        superset = pd.concat([
            frequency,
            pd.DataFrame({
                "text_id": [3],
                "word_id": [0],
                "word": ["unused"],
                "paper_log_gmean_freq": [-9.0],
            }),
        ], ignore_index=True)
        superset_output = factorial.build_layer_factorial_dataframe(
            make_joint(),
            make_layer(),
            make_sentence_map(),
            model="fixture",
            context_unit="sentence",
            lens_method="logit-lens",
            first_token_policy="bow",
            lag_boundary="sentence",
            frequency_table=superset,
        )
        np.testing.assert_allclose(
            superset_output["paper_log_gmean_freq"],
            [-1, -2, -3, -4, -5],
        )

        bad = frequency.iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "does not cover every"):
            factorial.build_layer_factorial_dataframe(
                make_joint(),
                make_layer(),
                make_sentence_map(),
                model="fixture",
                context_unit="sentence",
                lens_method="logit-lens",
                first_token_policy="bow",
                lag_boundary="sentence",
                frequency_table=bad,
            )

    def test_missing_or_wrong_wordfreq_has_actionable_error(self):
        with patch.object(
            factorial,
            "version",
            side_effect=factorial.PackageNotFoundError,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "wordfreq==3.1.1.*precomputed"
            ):
                build(word_frequency_fn=None)

        with patch.object(factorial, "version", return_value="3.0.0"):
            with self.assertRaisesRegex(
                RuntimeError, "require wordfreq==3.1.1.*found 3.0.0"
            ):
                build(word_frequency_fn=None)

    def test_paper_time_is_keyed_deduplicated_and_peeked_scoped(self):
        joint = pd.DataFrame(
            {"text_id": [2], "word_id": [748], "ref_token": ["peeked"]}
        )
        paper = pd.DataFrame(
            {
                "item": [2, 2, 3],
                "zone": [749, 749, 1],
                "word": ["peaked", "peaked", "extra"],
                "meanItemRT": [321.5, 321.5, 999.0],
            }
        )
        np.testing.assert_allclose(
            factorial._paper_time_values(paper, joint, "ref_token"), [321.5]
        )
        paper = paper.iloc[:2].copy()
        paper.loc[:, "item"] = 1
        joint.loc[:, "text_id"] = 1
        with self.assertRaisesRegex(ValueError, "word mismatch"):
            factorial._paper_time_values(paper, joint, "ref_token")

    def test_paper_frequency_uses_rt_peaked_spelling_only_at_known_key(self):
        joint = pd.DataFrame(
            {
                "text_id": [1, 2],
                "word_id": [0, 748],
                "ref_token": ["ordinary", "peeked"],
            }
        )
        self.assertEqual(
            factorial._paper_frequency_words(joint, "ref_token"),
            ["ordinary", "peaked"],
        )
        frequency = pd.DataFrame(
            {
                "text_id": [1, 2],
                "word_id": [0, 748],
                "word": ["ordinary", "peaked"],
                "paper_log_gmean_freq": [-1.0, -2.0],
            }
        )
        np.testing.assert_allclose(
            factorial._precomputed_frequency_values(
                frequency, joint, "ref_token"
            ),
            [-1.0, -2.0],
        )

    def test_cli_file_api_writes_atomically_with_manifest_and_paper_time(self):
        frequency = pd.DataFrame(
            {
                "text_id": [1, 1, 1, 1, 2],
                "word_id": [0, 1, 2, 3, 0],
                "word": WORDS,
                "log_gmean_freq": [-1.0, -2.0, -3.0, -4.0, -5.0],
            }
        )
        paper = pd.DataFrame(
            {
                "item": [1, 1, 1, 1, 2],
                "zone": [1, 2, 3, 4, 1],
                "word": WORDS,
                "meanItemRT": [101.0, 102.0, 103.0, 104.0, 105.0],
            }
        )
        manifest_rows = [
            {
                "text_id": text_id,
                "sentence_id": sentence.sentence_id,
                "sentence_word_id": sentence_word_id,
                "word_id": word_id,
                "word": word,
            }
            for text_id, sentences in make_sentence_map().items()
            for sentence in sentences
            for sentence_word_id, (word_id, word) in enumerate(
                zip(sentence.word_ids, sentence.words)
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            joint_path = directory / "joint.tsv"
            layer_path = directory / "layer.tsv"
            manifest_path = directory / "manifest.tsv"
            frequency_path = directory / "frequency.tsv"
            paper_path = directory / "processed_RTs.tsv"
            output_path = directory / "output.tsv"
            make_joint().to_csv(joint_path, sep="\t", index=False)
            make_layer().to_csv(layer_path, sep="\t", index=False)
            sentence_manifest.write_manifest(manifest_rows, manifest_path)
            frequency.to_csv(frequency_path, sep="\t", index=False)
            paper.to_csv(paper_path, sep="\t", index=False)
            original_joint = joint_path.read_bytes()

            argv = [
                "build_layer_factorial_dataset.py",
                "--canonical-joint-fname",
                str(joint_path),
                "--layer-fname",
                str(layer_path),
                "--sentence-manifest-fname",
                str(manifest_path),
                "--model",
                "fixture-model",
                "--context-unit",
                "sentence",
                "--lens-method",
                "logit-lens",
                "--first-token-policy",
                "bow",
                "--lag-boundary",
                "sentence",
                "--precomputed-frequency-fname",
                str(frequency_path),
                "--paper-rt-fname",
                str(paper_path),
                "--output-fname",
                str(output_path),
            ]
            with patch.object(sys, "argv", argv):
                factorial.main()

            self.assertTrue(output_path.is_file())
            self.assertEqual(joint_path.read_bytes(), original_joint)
            output = pd.read_csv(output_path, sep="\t")
            np.testing.assert_allclose(
                output["paper_time"], [101, 102, 103, 104, 105]
            )
            self.assertEqual(output["lag_boundary"].unique().tolist(), ["sentence"])
            self.assertEqual(
                output["lag_padding"].unique().tolist(), ["global-mean"]
            )

    def test_rejects_generated_column_collision_and_output_overwrite(self):
        joint = make_joint()
        joint["model"] = "old"
        with self.assertRaisesRegex(ValueError, "already contains generated"):
            factorial.build_layer_factorial_dataframe(
                joint,
                make_layer(),
                make_sentence_map(),
                model="fixture",
                context_unit="sentence",
                lens_method="logit-lens",
                first_token_policy="bow",
                lag_boundary="sentence",
                word_frequency_fn=fake_word_frequency,
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "same.tsv"
            make_joint().to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                factorial.build_layer_factorial_dataset(
                    path,
                    path,
                    path,
                    path,
                    model="fixture",
                    context_unit="sentence",
                    lens_method="logit-lens",
                    first_token_policy="bow",
                    lag_boundary="sentence",
                    word_frequency_fn=fake_word_frequency,
                )


if __name__ == "__main__":
    unittest.main()
