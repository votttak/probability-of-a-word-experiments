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
        (self.lens / "config.json").write_text(
            json.dumps({
                "base_model_name_or_path": "gpt2",
                "base_model_revision": "revision",
                "num_hidden_layers": 12,
            }) + "\n",
            encoding="utf8",
        )
        (self.lens / "params.pt").write_bytes(b"parameters")
        self.spec = SimpleNamespace(
            alias="gpt2-small",
            hf_name="gpt2",
            base_model_revision="revision",
            final_layer=12,
            lens_artifact="gpt2",
            lens_base_model_revision="revision",
        )
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
                    "hf_model_name_effective": "gpt2",
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
            contexts=["passage", "sentence"],
            lens_methods=["logit-lens", "tuned-lens"],
            score_kinds=["corrected", "buggy"],
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _write_validation(self):
        payload = {
            "validated": True,
            "model": "gpt2-small",
            "model_revision_effective": "revision",
            "expected": {
                "rows": 3,
                "final_layer": 12,
                "min_layer": 0,
                "contexts": ["passage", "sentence"],
                "lens_methods": ["logit-lens", "tuned-lens"],
                "score_kinds": ["corrected", "buggy"],
            },
            "sentence_manifest_sha256": runner.sha256_file(self.manifest),
            "tuned_lens_identity": {
                "artifact": {
                    "config_sha256": runner.sha256_file(
                        self.lens / "config.json"
                    ),
                    "params_sha256": runner.sha256_file(
                        self.lens / "params.pt"
                    ),
                    "base_model_name_or_path": "gpt2",
                    "base_model_revision": "revision",
                }
            },
        }
        self.validation.write_text(json.dumps(payload), encoding="utf8")

    def _validate(self):
        runner.validate_extraction_matches_run(
            self.args,
            self.paths,
            self.validation,
            self.spec,
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

    def test_stale_hugging_face_model_is_rejected(self):
        anchor_path = Path(
            f"{self.paths[('sentence', 'tuned-lens')]}.anchor.json"
        )
        anchor = json.loads(anchor_path.read_text(encoding="utf8"))
        anchor["experiment"]["hf_model_name_effective"] = (
            "EleutherAI/pythia-70m-deduped"
        )
        anchor_path.write_text(json.dumps(anchor), encoding="utf8")
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

    def test_extraction_command_passes_pinned_hf_override(self):
        args = SimpleNamespace(
            python="python",
            model="pythia-70m",
            text_fname=self.text,
            include_embedding_layer=True,
            sentence_manifest_fname=self.manifest,
            sentence_first_token_policy="bow",
            tuned_lens_path=self.lens,
            score_kinds=["corrected", "buggy"],
        )
        spec = SimpleNamespace(
            hf_name="EleutherAI/pythia-70m-deduped",
            base_model_revision="base-revision",
        )
        command = runner.extraction_command(
            args, "sentence", "tuned-lens", spec, self.root / "cell"
        )
        self.assertEqual(
            command[command.index("--hf-model-name") + 1], spec.hf_name
        )
        self.assertEqual(
            command[command.index("--model-revision") + 1],
            spec.base_model_revision,
        )

    def test_config_defaults_are_replaced_by_explicit_cli_switches(self):
        args = runner.parse_args([
            "--config",
            str(runner.DEFAULT_CONFIG_PATH),
            "--no-include-embedding-layer",
            "--contexts",
            "sentence",
            "--response-columns",
            "paper_time",
        ])
        self.assertFalse(args.include_embedding_layer)
        self.assertEqual(args.contexts, ["sentence"])
        self.assertEqual(args.response_columns, ["paper_time"])
        self.assertEqual(args.score_kinds, ["corrected", "buggy"])
        self.assertIn("--contexts", args.cli_overrides)

    def test_cli_can_enable_embedding_disabled_by_config(self):
        payload = json.loads(
            runner.DEFAULT_CONFIG_PATH.read_text(encoding="utf8")
        )
        payload["switches"]["include_embedding_layer"] = False
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(payload), encoding="utf8")
        args = runner.parse_args([
            "--config",
            str(config_path),
            "--include-embedding-layer",
        ])
        self.assertTrue(args.include_embedding_layer)

    def test_cli_can_clear_configured_report_note(self):
        args = runner.parse_args([
            "--config",
            str(runner.DEFAULT_CONFIG_PATH),
            "--report-note",
            "",
        ])
        self.assertEqual(args.report_note, "")
        self.assertIn("--report-note", args.cli_overrides)

    def test_empty_config_report_note_remains_empty(self):
        payload = json.loads(
            runner.DEFAULT_CONFIG_PATH.read_text(encoding="utf8")
        )
        payload["report_note"] = ""
        config_path = self.root / "empty-note-config.json"
        config_path.write_text(json.dumps(payload), encoding="utf8")
        args = runner.parse_args(["--config", str(config_path)])
        self.assertEqual(args.report_note, "")

    def test_corrected_only_extraction_omits_buggy_output_flag(self):
        args = SimpleNamespace(
            python="python",
            model="gpt2-small",
            text_fname=self.text,
            include_embedding_layer=True,
            sentence_manifest_fname=self.manifest,
            sentence_first_token_policy="bow",
            tuned_lens_path=self.lens,
            score_kinds=["corrected"],
        )
        command = runner.extraction_command(
            args, "passage", "logit-lens", self.spec, self.root / "cell"
        )
        self.assertNotIn("--return-buggy-surprisals", command)

    def test_lens_config_accepts_registry_expected_null_revision(self):
        spec = SimpleNamespace(
            hf_name="EleutherAI/pythia-70m-deduped",
            lens_base_model_revision=None,
            final_layer=6,
        )
        config_path = self.lens / "config.json"
        config_path.write_text(
            json.dumps({
                "base_model_name_or_path": spec.hf_name,
                "base_model_revision": None,
                "num_hidden_layers": spec.final_layer,
            }),
            encoding="utf8",
        )
        observed = runner.read_lens_config(self.lens, spec)
        self.assertIsNone(observed["base_model_revision"])

        config_path.write_text(
            json.dumps({
                "base_model_name_or_path": spec.hf_name,
                "base_model_revision": "unexpected",
                "num_hidden_layers": spec.final_layer,
            }),
            encoding="utf8",
        )
        with self.assertRaisesRegex(ValueError, "base revision mismatch"):
            runner.read_lens_config(self.lens, spec)


if __name__ == "__main__":
    unittest.main()
