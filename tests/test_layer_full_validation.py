"""Focused tests for canonical full-layer preflight and output validation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.preflight_layer_full import (
    MODEL_FINAL_LAYERS,
    ValidationError,
    run_preflight,
    sha256_file,
)
from scripts.validate_layer_full_outputs import validate_outputs
from src.h01_data.internal_layer_models import MODEL_SPECS


class LayerFullValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.text_fname = self.root / "text.txt"
        self.joint_fname = self.root / "canonical-joint.tsv"
        self.reference_fname = self.root / "reference.tsv"
        self.internal_fname = self.root / "internal-layer.tsv"
        self.merged_fname = self.root / "merged.tsv"
        self.summary_fname = self.root / "summary.tsv"
        self.fold_fname = self.root / "fold-results.tsv"
        self.delta_fname = self.root / "conditional-deltas.tsv"
        self.anchor_fname = self.root / "internal-layer.tsv.anchor.json"
        self.completion_fname = self.root / "complete.json"
        self._write_fixture()

    def tearDown(self):
        self.temporary.cleanup()

    def test_preflight_model_layers_match_canonical_registry(self):
        self.assertEqual(
            MODEL_FINAL_LAYERS,
            {spec.alias: spec.final_layer for spec in MODEL_SPECS},
        )
        self.assertNotIn("pythia-120b", MODEL_FINAL_LAYERS)

    @staticmethod
    def _with_shifts(dataframe, columns):
        dataframe = dataframe.copy()
        for column in columns:
            grouped = dataframe.groupby("text_id", sort=False)[column]
            dataframe[f"prev_{column}"] = grouped.shift(1)
            dataframe[f"prev2_{column}"] = grouped.shift(2)
            dataframe[f"prev3_{column}"] = grouped.shift(3)
        return dataframe

    def _write_fixture(self):
        passages = [["a", "b"], ["c", "d"]]
        self.text_fname.write_text("a b\nc d\n", encoding="utf8")
        zero_keys = [(0, 0, "a"), (0, 1, "b"), (1, 0, "c"), (1, 1, "d")]
        one_keys = [(text + 1, word, token) for text, word, token in zero_keys]

        joint = pd.DataFrame({
            "text_id": [row[0] for row in one_keys],
            "word_id": [row[1] for row in one_keys],
            "ref_token": [row[2] for row in one_keys],
            "word": [row[2] for row in one_keys],
            "time": [100.0, 110.0, 120.0, 130.0],
            "word_len": [1.0] * 4,
            "freq": [4.0, 3.0, 2.0, 1.0],
            "surprisal": [1.0, 2.0, 3.0, 4.0],
        })
        joint["prev_word_len"] = joint.groupby("text_id")["word_len"].shift(1)
        joint["prev2_word_len"] = joint.groupby("text_id")["word_len"].shift(2)
        joint["prev3_word_len"] = joint.groupby("text_id")["word_len"].shift(3)
        joint["prev_freq"] = joint.groupby("text_id")["freq"].shift(1)
        joint["prev2_freq"] = joint.groupby("text_id")["freq"].shift(2)
        joint["prev3_freq"] = joint.groupby("text_id")["freq"].shift(3)
        predictor_columns = []
        for context in range(5):
            column = f"ngram_surprisal_context_{context}"
            joint[column] = np.arange(4, dtype=float) + context + 1
            predictor_columns.append(column)
        for context in range(1, 5):
            column = f"context_limited_surprisal_context_{context}"
            joint[column] = np.arange(4, dtype=float) + context + 2
            predictor_columns.append(column)
        joint = self._with_shifts(joint, predictor_columns)
        joint.to_csv(self.joint_fname, sep="\t", index=False)

        reference = pd.DataFrame({
            "text_id": [row[0] for row in zero_keys],
            "word_id": [row[1] for row in zero_keys],
            "word": [row[2] for row in zero_keys],
            "surprisal": [1.0, 2.0, 3.0, 4.0],
        })
        reference.to_csv(self.reference_fname, sep="\t", index=False)

        internal = reference[["text_id", "word_id", "word"]].copy()
        layer_columns = []
        for layer in range(1, 13):
            column = f"internal_layer_surprisal_layer_{layer}"
            if layer == 12:
                internal[column] = (
                    reference["surprisal"]
                    + np.array([0.0001, 0.0002, 0.0003, 0.0004])
                )
            else:
                internal[column] = layer + np.arange(4, dtype=float) / 10
            layer_columns.append(column)
        internal.to_csv(self.internal_fname, sep="\t", index=False)

        merged = joint.copy()
        for column in layer_columns:
            merged[column] = internal[column].to_numpy()
        merged = self._with_shifts(merged, layer_columns)
        merged.to_csv(self.merged_fname, sep="\t", index=False)

        summary = pd.DataFrame({
            "key": [
                "input_rows", "complete_case_rows", "excluded_rows", "folds",
                "seed", "ngram_contexts", "context_limited_contexts",
                "internal_layers", "layer_decoder", "model",
            ],
            "value": [
                "4", "3", "1", "2", "42", "0,1,2,3,4", "1,2,3,4",
                ",".join(map(str, range(1, 13))), "logit lens", "gpt2-small",
            ],
        })
        summary.to_csv(self.summary_fname, sep="\t", index=False)
        folds = self._make_fold_results()
        folds.to_csv(self.fold_fname, sep="\t", index=False)
        self._make_conditional_deltas(folds).to_csv(
            self.delta_fname, sep="\t", index=False
        )
        anchor = {
            "validated": True,
            "reference_fname": str(self.reference_fname.resolve()),
            "reference_sha256": sha256_file(self.reference_fname),
            "final_layer": 12,
            "rows": 4,
            "max_abs_difference": 0.0004,
            "mean_abs_difference": 0.00025,
            "p99_abs_difference": 0.0004,
            "tolerance": 0.0005,
        }
        self.anchor_fname.write_text(
            json.dumps(anchor, indent=2) + "\n", encoding="utf8"
        )

    @staticmethod
    def _make_fold_results():
        rows = []
        for family, comparison, contexts in (
            ("ngram", "ngram_vs_internal_layer", range(5)),
            (
                "context_limited",
                "context_limited_vs_internal_layer",
                range(1, 5),
            ),
        ):
            for context in contexts:
                for layer in range(1, 13):
                    for fold in range(1, 3):
                        n_test = 2 if fold == 1 else 1
                        n_train = 3 - n_test
                        m0 = -5.0 - fold / 10
                        predictor = m0 + 0.5 + context / 100
                        layer_score = m0 + 0.3 + layer / 1000
                        joint = predictor + 0.2
                        predictor_delta = joint - layer_score
                        layer_delta = joint - predictor
                        is_ngram = family == "ngram"
                        row = {
                            "model": "gpt2-small",
                            "comparison": comparison,
                            "predictor_family": family,
                            "predictor_context": context,
                            "ngram_context": context if is_ngram else np.nan,
                            "context_limited_context": (
                                np.nan if is_ngram else context
                            ),
                            "layer": layer,
                            "fold": fold,
                            "n_train": n_train,
                            "n_test": n_test,
                            "ll_m0_mean": m0,
                            "ll_predictor_mean": predictor,
                            "ll_layer_mean": layer_score,
                            "ll_joint_mean": joint,
                            "delta_predictor_given_layer_mean": predictor_delta,
                            "delta_layer_given_predictor_mean": layer_delta,
                            "delta_n_given_l_mean": (
                                predictor_delta if is_ngram else np.nan
                            ),
                            "delta_l_given_n_mean": (
                                layer_delta if is_ngram else np.nan
                            ),
                            "delta_c_given_l_mean": (
                                np.nan if is_ngram else predictor_delta
                            ),
                            "delta_l_given_c_mean": (
                                np.nan if is_ngram else layer_delta
                            ),
                        }
                        for prefix in (
                            "ll_m0", "ll_predictor", "ll_layer", "ll_joint",
                            "delta_predictor_given_layer",
                            "delta_layer_given_predictor",
                            "delta_n_given_l", "delta_l_given_n",
                            "delta_c_given_l", "delta_l_given_c",
                        ):
                            row[f"{prefix}_sum"] = row[f"{prefix}_mean"] * n_test
                        rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _make_conditional_deltas(folds):
        rows = []
        grouped = folds.groupby([
            "comparison", "predictor_family", "predictor_context", "layer"
        ], sort=False)
        for (comparison, family, context, layer), group in grouped:
            predictor = group["delta_predictor_given_layer_mean"]
            layer_values = group["delta_layer_given_predictor_mean"]
            is_ngram = family == "ngram"
            predictor_mean = predictor.mean()
            predictor_se = predictor.std(ddof=1) / np.sqrt(len(group))
            layer_mean = layer_values.mean()
            layer_se = layer_values.std(ddof=1) / np.sqrt(len(group))
            rows.append({
                "model": "gpt2-small",
                "comparison": comparison,
                "predictor_family": family,
                "predictor_context": context,
                "ngram_context": context if is_ngram else np.nan,
                "context_limited_context": np.nan if is_ngram else context,
                "layer": layer,
                "folds": len(group),
                "delta_predictor_given_layer_mean": predictor_mean,
                "delta_predictor_given_layer_se": predictor_se,
                "delta_layer_given_predictor_mean": layer_mean,
                "delta_layer_given_predictor_se": layer_se,
                "delta_n_given_l_mean": predictor_mean if is_ngram else np.nan,
                "delta_n_given_l_se": predictor_se if is_ngram else np.nan,
                "delta_l_given_n_mean": layer_mean if is_ngram else np.nan,
                "delta_l_given_n_se": layer_se if is_ngram else np.nan,
                "delta_c_given_l_mean": np.nan if is_ngram else predictor_mean,
                "delta_c_given_l_se": np.nan if is_ngram else predictor_se,
                "delta_l_given_c_mean": np.nan if is_ngram else layer_mean,
                "delta_l_given_c_se": np.nan if is_ngram else layer_se,
            })
        return pd.DataFrame(rows)

    def _validate_outputs(self, expected_anchor_tolerance=0.0005):
        return validate_outputs(
            self.joint_fname,
            self.internal_fname,
            self.merged_fname,
            self.summary_fname,
            self.fold_fname,
            self.delta_fname,
            self.anchor_fname,
            self.completion_fname,
            expected_rows=4,
            expected_complete_rows=3,
            expected_excluded_rows=1,
            expected_folds=2,
            expected_seed=42,
            expected_final_layer=12,
            expected_anchor_tolerance=expected_anchor_tolerance,
        )

    def test_preflight_accepts_exact_inputs_and_hashes(self):
        joint = pd.read_csv(self.joint_fname, sep="\t")
        joint.loc[1, "freq"] = np.nan
        joint.to_csv(self.joint_fname, sep="\t", index=False)
        report = run_preflight(
            self.text_fname,
            self.joint_fname,
            self.reference_fname,
            "gpt2-small",
            12,
            expected_rows=4,
            expected_passages=2,
            expected_text_sha256=sha256_file(self.text_fname),
            expected_joint_sha256=sha256_file(self.joint_fname),
            expected_reference_sha256=sha256_file(self.reference_fname),
        )
        self.assertTrue(report["validated"])
        self.assertEqual(report["passage_word_counts"], [2, 2])

    def test_preflight_rejects_bad_coverage_schema_hash_and_mapping(self):
        joint = pd.read_csv(self.joint_fname, sep="\t")
        joint.iloc[:-1].to_csv(self.joint_fname, sep="\t", index=False)
        with self.assertRaisesRegex(ValidationError, "has 3 rows"):
            run_preflight(
                self.text_fname, self.joint_fname, self.reference_fname,
                "gpt2-small", 12, expected_rows=4, expected_passages=2,
            )
        self._write_fixture()
        joint = pd.read_csv(self.joint_fname, sep="\t").drop(
            columns="ngram_surprisal_context_4"
        )
        joint.to_csv(self.joint_fname, sep="\t", index=False)
        with self.assertRaisesRegex(ValidationError, "n-gram contexts"):
            run_preflight(
                self.text_fname, self.joint_fname, self.reference_fname,
                "gpt2-small", 12, expected_rows=4, expected_passages=2,
            )
        self._write_fixture()
        with self.assertRaisesRegex(ValidationError, "SHA-256 mismatch"):
            run_preflight(
                self.text_fname, self.joint_fname, self.reference_fname,
                "gpt2-small", 12, expected_rows=4, expected_passages=2,
                expected_text_sha256="0" * 64,
            )
        with self.assertRaisesRegex(ValidationError, "final layer 12"):
            run_preflight(
                self.text_fname, self.joint_fname, self.reference_fname,
                "gpt2-small", 11, expected_rows=4, expected_passages=2,
            )

    def test_output_validation_writes_deterministic_hashed_completion(self):
        completion = self._validate_outputs()
        self.assertTrue(completion["validated"])
        self.assertEqual(completion["counts"]["fold_result_rows"], 216)
        self.assertEqual(completion["counts"]["conditional_delta_rows"], 108)
        self.assertEqual(completion["model"], "gpt2-small")
        self.assertAlmostEqual(
            completion["anchor"]["scorer_reference_p99_abs_difference"], 0.0004
        )
        self.assertAlmostEqual(
            completion["anchor"]["merged_p99_abs_difference"], 0.0004
        )
        self.assertEqual(completion["anchor"]["tolerance"], 0.0005)
        written = self.completion_fname.read_bytes()
        second = self._validate_outputs()
        self.assertEqual(completion, second)
        self.assertEqual(written, self.completion_fname.read_bytes())
        for artifact in completion["artifacts"].values():
            self.assertRegex(artifact["sha256"], r"^[0-9a-f]{64}$")

    def test_output_validation_rejects_missing_layer_without_sentinel(self):
        internal = pd.read_csv(self.internal_fname, sep="\t").drop(
            columns="internal_layer_surprisal_layer_6"
        )
        internal.to_csv(self.internal_fname, sep="\t", index=False)
        with self.assertRaisesRegex(ValidationError, "layers are"):
            self._validate_outputs()
        self.assertFalse(self.completion_fname.exists())

    def test_output_validation_rejects_bad_combination_and_anchor(self):
        folds = pd.read_csv(self.fold_fname, sep="\t").iloc[:-1]
        folds.to_csv(self.fold_fname, sep="\t", index=False)
        with self.assertRaisesRegex(ValidationError, "combinations are incomplete"):
            self._validate_outputs()
        self.assertFalse(self.completion_fname.exists())

    def test_output_validation_accepts_pythia_tolerance_and_rejects_mismatch(self):
        anchor = json.loads(self.anchor_fname.read_text(encoding="utf8"))
        anchor["tolerance"] = 0.01
        self.anchor_fname.write_text(json.dumps(anchor), encoding="utf8")

        completion = self._validate_outputs(expected_anchor_tolerance=0.01)
        self.assertEqual(completion["anchor"]["tolerance"], 0.01)

        self.completion_fname.unlink()
        with self.assertRaisesRegex(
            ValidationError,
            "anchor tolerance is 0.01; expected 0.0005",
        ):
            self._validate_outputs(expected_anchor_tolerance=0.0005)
        self.assertFalse(self.completion_fname.exists())

        self._write_fixture()
        anchor = json.loads(self.anchor_fname.read_text(encoding="utf8"))
        anchor["max_abs_difference"] = 0.00045
        self.anchor_fname.write_text(json.dumps(anchor), encoding="utf8")
        with self.assertRaisesRegex(ValidationError, "recomputed value"):
            self._validate_outputs()
        self.assertFalse(self.completion_fname.exists())

    def test_output_validation_rejects_wrong_model_provenance(self):
        summary = pd.read_csv(self.summary_fname, sep="\t", dtype=str)
        summary.loc[summary["key"] == "model", "value"] = "wrong-model"
        summary.to_csv(self.summary_fname, sep="\t", index=False)
        with self.assertRaisesRegex(ValidationError, "summary model"):
            self._validate_outputs()

        self._write_fixture()
        folds = pd.read_csv(self.fold_fname, sep="\t")
        folds.loc[0, "model"] = "wrong-model"
        folds.to_csv(self.fold_fname, sep="\t", index=False)
        with self.assertRaisesRegex(ValidationError, "fold results model"):
            self._validate_outputs()

        self._write_fixture()
        deltas = pd.read_csv(self.delta_fname, sep="\t")
        deltas.loc[0, "model"] = "wrong-model"
        deltas.to_csv(self.delta_fname, sep="\t", index=False)
        with self.assertRaisesRegex(ValidationError, "conditional deltas model"):
            self._validate_outputs()
        self.assertFalse(self.completion_fname.exists())


if __name__ == "__main__":
    unittest.main()
