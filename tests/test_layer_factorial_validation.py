"""Focused tests for the four-cell layer-factorial output validator."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.validate_layer_factorial_outputs import (
    BUGGY_PREFIX,
    CORRECTED_PREFIX,
    ValidationError,
    sha256_file,
    validate_outputs,
)


class LayerFactorialValidationTest(unittest.TestCase):
    ROWS = 4
    MIN_LAYER = 1
    FINAL_LAYER = 3
    TOLERANCE = 1e-5

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = {
            "passage_logit": self.root / "passage-logit.tsv",
            "passage_tuned": self.root / "passage-tuned.tsv",
            "sentence_logit": self.root / "sentence-logit.tsv",
            "sentence_tuned": self.root / "sentence-tuned.tsv",
        }
        self.completion = self.root / "validation-complete.json"
        self._write_grid()

    def tearDown(self):
        self.temporary.cleanup()

    @property
    def layers(self):
        return range(self.MIN_LAYER, self.FINAL_LAYER + 1)

    @staticmethod
    def _keys():
        return pd.DataFrame({
            "text_id": [0, 0, 1, 1],
            "word_id": [0, 1, 0, 1],
            "word": ["a", "b", "c", "d"],
        })

    def _table(self, context: str, lens: str) -> pd.DataFrame:
        table = self._keys()
        row_offset = np.arange(self.ROWS, dtype=float) / 10
        context_offset = np.array([0.0, 0.4, 0.0, 0.7])
        for prefix, score_offset in (
            (CORRECTED_PREFIX, 0.0),
            (BUGGY_PREFIX, 0.5),
        ):
            for layer in self.layers:
                values = layer + row_offset + score_offset
                if context == "sentence" and layer == self.FINAL_LAYER:
                    values = values + context_offset
                if lens == "tuned":
                    if layer == self.FINAL_LAYER:
                        values = values + 1e-7
                    else:
                        values = values + 0.25
                table[f"{prefix}{layer}"] = values
        return table

    def _anchor(self, context: str, lens: str) -> dict:
        identity = None
        if lens == "tuned":
            identity = {
                "decoder": "tuned-lens",
                "final_layer_policy": "ordinary-logits-bypass",
                "artifact": {
                    "resource_kind": "local-directory",
                    "resource_path": "/artifacts/gpt2",
                    "config_sha256": "a" * 64,
                    "params_sha256": "b" * 64,
                    "base_model_name_or_path": "openai-community/gpt2",
                    "base_model_revision": "revision-abc",
                    "d_model": 768,
                    "num_hidden_layers": 3,
                },
            }
        return {
            "validated": False,
            "reason": "factorial fixture",
            "experiment": {
                "model": "gpt2-small",
                "model_revision_requested": "revision-abc",
                "model_revision_effective": "revision-abc",
                "context_unit": context,
                "sentence_first_token_policy": (
                    "bow" if context == "sentence" else "bos"
                ),
                "sentence_manifest_sha256": (
                    "c" * 64 if context == "sentence" else None
                ),
                "lens_method": f"{lens}-lens",
                "lens_identity": identity,
                "score_kinds": ["corrected", "buggy"],
                "include_embedding_layer": False,
                "layers": list(self.layers),
            },
        }

    def _write_grid(self):
        for label, path in self.paths.items():
            context, lens = label.split("_")
            self._table(context, lens).to_csv(path, sep="\t", index=False)
            Path(f"{path}.anchor.json").write_text(
                json.dumps(self._anchor(context, lens), indent=2) + "\n",
                encoding="utf8",
            )

    def _validate(self):
        return validate_outputs(
            self.paths["passage_logit"],
            self.paths["passage_tuned"],
            self.paths["sentence_logit"],
            self.paths["sentence_tuned"],
            self.completion,
            expected_rows=self.ROWS,
            expected_final_layer=self.FINAL_LAYER,
            expected_min_layer=self.MIN_LAYER,
            tolerance=self.TOLERANCE,
        )

    def _mutate_anchor(self, label, mutate):
        path = Path(f"{self.paths[label]}.anchor.json")
        anchor = json.loads(path.read_text(encoding="utf8"))
        mutate(anchor["experiment"])
        path.write_text(json.dumps(anchor, indent=2) + "\n", encoding="utf8")

    def test_valid_grid_writes_auditable_atomic_manifest(self):
        result = self._validate()
        self.assertTrue(result["validated"])
        self.assertEqual(result["expected"]["layers"], [1, 2, 3])
        self.assertEqual(result["model_revision_effective"], "revision-abc")
        self.assertEqual(result["sentence_manifest_sha256"], "c" * 64)
        self.assertAlmostEqual(
            result["comparisons"][
                "final_logit_vs_tuned_max_abs_difference"
            ]["passage"]["corrected"],
            1e-7,
        )
        artifact = result["artifacts"]["sentence_tuned"]
        self.assertEqual(
            artifact["tsv"]["sha256"],
            sha256_file(self.paths["sentence_tuned"]),
        )
        self.assertEqual(
            artifact["anchor_json"]["sha256"],
            sha256_file(f"{self.paths['sentence_tuned']}.anchor.json"),
        )
        published = json.loads(self.completion.read_text(encoding="utf8"))
        self.assertEqual(published, result)
        inode = self.completion.stat().st_ino
        self.assertEqual(self._validate(), result)
        self.assertEqual(self.completion.stat().st_ino, inode)

    def test_rejects_key_or_word_mismatch(self):
        for column, replacement, message in (
            ("word_id", 99, "key order differs"),
            ("word", "wrong", "word order differs"),
        ):
            with self.subTest(column=column):
                self._write_grid()
                table = pd.read_csv(self.paths["sentence_tuned"], sep="\t")
                table.loc[2, column] = replacement
                table.to_csv(self.paths["sentence_tuned"], sep="\t", index=False)
                with self.assertRaisesRegex(ValidationError, message):
                    self._validate()

    def test_rejects_incomplete_or_extra_predictor_family(self):
        for mutation, message in (
            ("drop", "exact dual predictor families"),
            ("extra", "exact dual predictor families"),
        ):
            with self.subTest(mutation=mutation):
                self._write_grid()
                path = self.paths["passage_logit"]
                table = pd.read_csv(path, sep="\t")
                if mutation == "drop":
                    table = table.drop(columns=[f"{BUGGY_PREFIX}2"])
                else:
                    table[f"{CORRECTED_PREFIX}4"] = 1.0
                table.to_csv(path, sep="\t", index=False)
                with self.assertRaisesRegex(ValidationError, message):
                    self._validate()

    def test_rejects_nonfinite_or_negative_predictors(self):
        for value, message in ((np.nan, "not finite"), (-0.1, "negatives")):
            with self.subTest(value=value):
                self._write_grid()
                path = self.paths["sentence_logit"]
                table = pd.read_csv(path, sep="\t")
                table.loc[0, f"{BUGGY_PREFIX}1"] = value
                table.to_csv(path, sep="\t", index=False)
                with self.assertRaisesRegex(ValidationError, message):
                    self._validate()

    def test_rejects_final_decoder_disagreement(self):
        path = self.paths["passage_tuned"]
        table = pd.read_csv(path, sep="\t")
        table.loc[0, f"{CORRECTED_PREFIX}{self.FINAL_LAYER}"] += 0.01
        table.to_csv(path, sep="\t", index=False)
        with self.assertRaisesRegex(ValidationError, "exceeds tolerance"):
            self._validate()

    def test_rejects_identical_intermediate_decoder_outputs(self):
        path = self.paths["sentence_tuned"]
        tuned = pd.read_csv(path, sep="\t")
        logit = pd.read_csv(self.paths["sentence_logit"], sep="\t")
        for layer in range(self.MIN_LAYER, self.FINAL_LAYER):
            for prefix in (CORRECTED_PREFIX, BUGGY_PREFIX):
                column = f"{prefix}{layer}"
                tuned[column] = logit[column]
        tuned.to_csv(path, sep="\t", index=False)
        with self.assertRaisesRegex(ValidationError, "all identical"):
            self._validate()

    def test_rejects_wrong_or_inconsistent_provenance(self):
        cases = (
            (
                "passage_logit",
                lambda experiment: experiment.update(context_unit="sentence"),
                "context_unit",
            ),
            (
                "sentence_logit",
                lambda experiment: experiment.update(
                    model_revision_effective="different-revision"
                ),
                "different effective model revisions",
            ),
            (
                "sentence_tuned",
                lambda experiment: experiment["lens_identity"]["artifact"].update(
                    params_sha256="d" * 64
                ),
                "different artifacts",
            ),
            (
                "sentence_tuned",
                lambda experiment: experiment.update(
                    sentence_manifest_sha256="e" * 64
                ),
                "different sentence manifests",
            ),
        )
        for label, mutation, message in cases:
            with self.subTest(message=message):
                self._write_grid()
                self._mutate_anchor(label, mutation)
                with self.assertRaisesRegex(ValidationError, message):
                    self._validate()

    def test_rejects_missing_revision_or_tuned_identity(self):
        cases = (
            (
                "passage_logit",
                lambda experiment: experiment.update(
                    model_revision_effective=None
                ),
                "effective model revision",
            ),
            (
                "passage_tuned",
                lambda experiment: experiment.update(lens_identity=None),
                "lacks its artifact identity",
            ),
        )
        for label, mutation, message in cases:
            with self.subTest(message=message):
                self._write_grid()
                self._mutate_anchor(label, mutation)
                with self.assertRaisesRegex(ValidationError, message):
                    self._validate()

    def test_rejects_identical_passage_and_sentence_final_scores(self):
        for lens in ("logit", "tuned"):
            passage = pd.read_csv(self.paths[f"passage_{lens}"], sep="\t")
            sentence_path = self.paths[f"sentence_{lens}"]
            sentence = pd.read_csv(sentence_path, sep="\t")
            for prefix in (CORRECTED_PREFIX, BUGGY_PREFIX):
                column = f"{prefix}{self.FINAL_LAYER}"
                sentence[column] = passage[column]
            sentence.to_csv(sentence_path, sep="\t", index=False)
        with self.assertRaisesRegex(ValidationError, "scores are identical"):
            self._validate()

    def test_rejects_invalid_expected_range_or_tolerance(self):
        kwargs = {
            "expected_rows": self.ROWS,
            "expected_final_layer": self.FINAL_LAYER,
            "expected_min_layer": self.MIN_LAYER,
            "tolerance": self.TOLERANCE,
        }
        positional = (
            self.paths["passage_logit"],
            self.paths["passage_tuned"],
            self.paths["sentence_logit"],
            self.paths["sentence_tuned"],
            self.completion,
        )
        with self.assertRaisesRegex(ValidationError, "must exceed"):
            validate_outputs(
                *positional,
                **{**kwargs, "expected_min_layer": self.FINAL_LAYER},
            )
        with self.assertRaisesRegex(ValidationError, "finite and nonnegative"):
            validate_outputs(*positional, **{**kwargs, "tolerance": np.nan})


if __name__ == "__main__":
    unittest.main()
