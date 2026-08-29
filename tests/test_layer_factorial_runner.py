"""Focused tests for factorial-runner path and cache provenance checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_layer_factorial as runner


class LayerFactorialRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.text = self.root / "text.txt"
        self.text.write_text("A bb\nCCC\n", encoding="utf8")
        self.manifest = self.root / "sentences.tsv"
        self.manifest.write_text("manifest\n", encoding="utf8")
        self.lens = self.root / "lens"
        self.lens.mkdir()
        (self.lens / "config.json").write_text("{}\n", encoding="utf8")
        (self.lens / "params.pt").write_bytes(b"parameters")
        self.paths = {}
        for context in ("passage", "sentence"):
            for lens in ("logit-lens", "tuned-lens"):
                path = self.root / f"{context}-{lens}.tsv"
                with path.open("w", encoding="utf8", newline="") as output:
                    writer = csv.DictWriter(
                        output,
                        delimiter="\t",
                        fieldnames=("text_id", "word_id", "word"),
                    )
                    writer.writeheader()
                    writer.writerows(
                        [
                            {"text_id": 0, "word_id": 0, "word": "A"},
                            {"text_id": 0, "word_id": 1, "word": "bb"},
                            {"text_id": 1, "word_id": 0, "word": "CCC"},
                        ]
                    )
                experiment = {
                    "model": "gpt2-small",
                    "model_revision_effective": "revision",
                    "sentence_first_token_policy": "bow",
                }
                Path(f"{path}.anchor.json").write_text(
                    json.dumps({"experiment": experiment}), encoding="utf8"
                )
                self.paths[(context, lens)] = path
        self.validation = self.root / "validation.json"
        self._write_validation()
        self.args = SimpleNamespace(
            model="gpt2-small",
            include_embedding_layer=True,
            sentence_manifest_fname=self.manifest,
            tuned_lens_path=self.lens,
            sentence_first_token_policy="bow",
            text_fname=self.text,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _write_validation(self):
        payload = {
            "validated": True,
            "model": "gpt2-small",
            "model_revision_effective": "revision",
            "expected": {"rows": 3, "final_layer": 12, "min_layer": 0},
            "sentence_manifest_sha256": runner.sha256_file(self.manifest),
            "tuned_lens_identity": {
                "artifact": {
                    "config_sha256": runner.sha256_file(
                        self.lens / "config.json"
                    ),
                    "params_sha256": runner.sha256_file(
                        self.lens / "params.pt"
                    ),
                }
            },
        }
        self.validation.write_text(json.dumps(payload), encoding="utf8")

    def _validate(self):
        runner.validate_extraction_matches_run(
            self.args,
            self.paths,
            self.validation,
            "revision",
            12,
            runner.read_expected_word_rows(self.text),
        )

    def test_current_extraction_provenance_passes(self):
        self._validate()

    def test_stale_model_is_rejected(self):
        self.args.model = "gpt2-large"
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            self._validate()

    def test_stale_sentence_policy_is_rejected(self):
        self.args.sentence_first_token_policy = "bos"
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            self._validate()

    def test_changed_lens_artifact_is_rejected(self):
        (self.lens / "params.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            self._validate()

    def test_changed_input_text_is_rejected(self):
        self.text.write_text("A XX\nCCC\n", encoding="utf8")
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            self._validate()

    def test_relative_paths_are_rooted_at_repository(self):
        with patch.object(runner, "REPOSITORY_ROOT", self.root):
            self.assertEqual(
                runner.resolve_repo_path("relative/file.tsv"),
                (self.root / "relative/file.tsv").resolve(),
            )

    def test_precomputed_frequency_path_is_resolved(self):
        args = SimpleNamespace(
            text_fname="text",
            sentence_manifest_fname="manifest",
            joint_data_fname="joint",
            paper_rt_fname="paper",
            precomputed_frequency_fname="frequency",
            tuned_lens_path="lens",
            tuned_lens_pythonpath=None,
            wordfreq_pythonpath=None,
            checkpoint_root="checkpoints",
            results_root="results",
        )
        with patch.object(runner, "REPOSITORY_ROOT", self.root):
            runner.resolve_path_arguments(args)
        self.assertEqual(
            args.precomputed_frequency_fname,
            (self.root / "frequency").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
