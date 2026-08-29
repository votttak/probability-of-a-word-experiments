"""Tests for the central layer-factorial experiment configuration."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from src.h01_data import layer_factorial_config as config_module
from src.h01_data.layer_factorial_models import model_aliases


class LayerFactorialConfigTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.default_payload = json.loads(
            config_module.DEFAULT_CONFIG_PATH.read_text(encoding="utf8")
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _write(self, payload=None):
        path = self.root / "layer-factorial.json"
        path.write_text(
            json.dumps(
                self.default_payload if payload is None else payload
            ),
            encoding="utf8",
        )
        return path

    def _assert_invalid(self, mutate, message):
        payload = json.loads(json.dumps(self.default_payload))
        mutate(payload)
        with self.assertRaisesRegex(
            config_module.LayerFactorialConfigError, message
        ):
            config_module.load_layer_factorial_config(
                self._write(payload)
            )

    def test_default_config_has_complete_factorial_and_provenance(self):
        config = config_module.load_layer_factorial_config()
        self.assertEqual(config.models, model_aliases())
        self.assertEqual(config.switches.responses, ("time", "paper_time"))
        self.assertEqual(config.switches.contexts, ("passage", "sentence"))
        self.assertEqual(
            config.switches.lens_methods,
            ("logit-lens", "tuned-lens"),
        )
        self.assertEqual(
            config.switches.score_kinds, ("corrected", "buggy")
        )
        self.assertTrue(config.switches.include_embedding_layer)
        self.assertEqual(config.analysis.mode, "paper-exact")
        self.assertEqual(config.analysis.early_layer_threshold, 0.2)
        self.assertEqual(
            config.source_sha256,
            hashlib.sha256(config.source_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(config.to_dict(), self.default_payload)

    def test_unknown_and_missing_keys_are_rejected_at_every_level(self):
        cases = (
            (lambda value: value.update({"typo": 1}), "unknown keys: typo"),
            (lambda value: value.pop("runtime"), "missing keys: runtime"),
            (
                lambda value: value["switches"].update({"typo": True}),
                "switches has unknown keys: typo",
            ),
            (
                lambda value: value["analysis"].pop("mode"),
                "analysis is missing keys: mode",
            ),
            (
                lambda value: value["paths"].update({"extra": "x"}),
                "paths has unknown keys: extra",
            ),
        )
        for mutate, message in cases:
            with self.subTest(message=message):
                self._assert_invalid(mutate, message)

    def test_switch_arrays_must_be_nonempty_unique_and_supported(self):
        cases = (
            (
                lambda value: value["switches"].update({"contexts": []}),
                "nonempty array",
            ),
            (
                lambda value: value["switches"].update({
                    "responses": ["time", "time"]
                }),
                "duplicate values",
            ),
            (
                lambda value: value["switches"].update({
                    "score_kinds": ["legacy"]
                }),
                "must be one of",
            ),
        )
        for mutate, message in cases:
            with self.subTest(message=message):
                self._assert_invalid(mutate, message)

    def test_models_must_be_enabled_registry_aliases_without_duplicates(self):
        self._assert_invalid(
            lambda value: value.update({"models": ["gpt2-medium"]}),
            "must be one of",
        )
        self._assert_invalid(
            lambda value: value.update({
                "models": ["gpt2-small", "gpt2-small"]
            }),
            "duplicate values",
        )

    def test_bool_is_not_accepted_as_an_integer_or_number(self):
        cases = (
            lambda value: value["runtime"].update({"jobs": True}),
            lambda value: value["analysis"].update({
                "early_layer_threshold": True
            }),
            lambda value: value["extraction"].update({
                "final_layer_tolerance": False
            }),
        )
        for mutate in cases:
            with self.subTest(mutate=mutate):
                self._assert_invalid(mutate, "must be")

    def test_report_note_may_be_empty_but_must_be_a_clean_string(self):
        payload = json.loads(json.dumps(self.default_payload))
        payload["report_note"] = ""
        config = config_module.load_layer_factorial_config(
            self._write(payload)
        )
        self.assertEqual(config.report_note, "")

        for value in (" padded ", " ", None, 1):
            with self.subTest(value=value):
                self._assert_invalid(
                    lambda candidate, value=value: candidate.update({
                        "report_note": value
                    }),
                    "report_note must be a string without surrounding",
                )

    def test_runtime_bounds_are_validated(self):
        cases = (
            (
                lambda value: value["runtime"].update({"jobs": 5}),
                "between 1 and 4",
            ),
            (
                lambda value: value["runtime"].update({
                    "threads_per_job": 0
                }),
                "at least 1",
            ),
            (
                lambda value: value["runtime"].update({
                    "pivot_sentences_per_text": -1
                }),
                "at least 1",
            ),
        )
        for mutate, message in cases:
            with self.subTest(message=message):
                self._assert_invalid(mutate, message)

    def test_paper_exact_requires_its_actual_lag_policy(self):
        for key, value in (
            ("lag_boundary", "text"),
            ("lag_padding", "missing"),
        ):
            with self.subTest(key=key):
                self._assert_invalid(
                    lambda payload, key=key, value=value: payload[
                        "analysis"
                    ].update({key: value}),
                    "paper-exact requires",
                )

    def test_paths_are_portable_and_cannot_escape_repository(self):
        cases = (
            (
                lambda value: value["paths"].update({"text": "/tmp/x"}),
                "remain inside",
            ),
            (
                lambda value: value["paths"].update({"text": "../x"}),
                "remain inside",
            ),
            (
                lambda value: value["paths"].update({"text": "C:/tmp/x"}),
                "remain inside",
            ),
            (
                lambda value: value["paths"].update({
                    "joint_template": "joint/model.tsv"
                }),
                r"one \{model\} placeholder",
            ),
            (
                lambda value: value["paths"].update({
                    "joint_template": "joint/{name}.tsv"
                }),
                r"one \{model\} placeholder",
            ),
            (
                lambda value: value["paths"].update({
                    "paper_rt": r"checkpoints\paper.tsv"
                }),
                "portable repository-relative",
            ),
        )
        for mutate, message in cases:
            with self.subTest(message=message):
                self._assert_invalid(mutate, message)

    def test_duplicate_json_keys_and_nonfinite_numbers_are_rejected(self):
        duplicate = self.root / "duplicate.json"
        duplicate.write_text(
            '{"schema_version": 1, "schema_version": 1}',
            encoding="utf8",
        )
        with self.assertRaisesRegex(
            config_module.LayerFactorialConfigError,
            "duplicate JSON object key",
        ):
            config_module.load_layer_factorial_config(duplicate)

        nonfinite_payload = json.dumps(self.default_payload).replace(
            '"early_layer_threshold": 0.2',
            '"early_layer_threshold": NaN',
        )
        nonfinite = self.root / "nonfinite.json"
        nonfinite.write_text(nonfinite_payload, encoding="utf8")
        with self.assertRaisesRegex(
            config_module.LayerFactorialConfigError,
            "non-finite JSON number",
        ):
            config_module.load_layer_factorial_config(nonfinite)

    def test_malformed_and_nonobject_json_are_rejected(self):
        malformed = self.root / "malformed.json"
        malformed.write_text("{", encoding="utf8")
        with self.assertRaisesRegex(
            config_module.LayerFactorialConfigError, "unable to read"
        ):
            config_module.load_layer_factorial_config(malformed)

        nonobject = self.root / "array.json"
        nonobject.write_text("[]", encoding="utf8")
        with self.assertRaisesRegex(
            config_module.LayerFactorialConfigError,
            "configuration must be an object",
        ):
            config_module.load_layer_factorial_config(nonobject)

    def test_resolve_path_formats_model_and_roots_at_repository(self):
        config = config_module.load_layer_factorial_config(self._write())
        expected = (
            self.root
            / "checkpoints/rt/merged_data/natural_stories-gpt2-small.tsv"
        ).resolve()
        self.assertEqual(
            config_module.resolve_config_path(
                config,
                "paths.joint_template",
                model="gpt2-small",
                repository_root=self.root,
            ),
            expected,
        )
        with self.assertRaisesRegex(
            config_module.LayerFactorialConfigError, "requires --model"
        ):
            config_module.resolve_config_path(
                config,
                "joint_template",
                repository_root=self.root,
            )
        with self.assertRaisesRegex(
            config_module.LayerFactorialConfigError, "not enabled"
        ):
            config_module.resolve_config_path(
                config,
                "joint_template",
                model="gpt2-medium",
                repository_root=self.root,
            )

    def test_relative_config_path_is_independent_of_caller_cwd(self):
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            config = config_module.load_layer_factorial_config(
                "configs/layer_factorial.json"
            )
        finally:
            os.chdir(previous)
        self.assertEqual(
            config.source_path, config_module.DEFAULT_CONFIG_PATH.resolve()
        )

    def test_dotted_queries_and_cli_are_shell_friendly(self):
        config_path = self._write()
        config = config_module.load_layer_factorial_config(config_path)
        self.assertEqual(
            config_module.get_config_value(config, "runtime.jobs"), 1
        )
        with self.assertRaisesRegex(
            config_module.LayerFactorialConfigError, "unknown"
        ):
            config_module.get_config_value(config, "runtime.typo")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = config_module.main([
                "--config",
                str(config_path),
                "--get",
                "switches.responses",
            ])
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue(), "time\npaper_time\n")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = config_module.main([
                "--config",
                str(config_path),
                "--get",
                "switches.include_embedding_layer",
            ])
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue(), "true\n")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = config_module.main([
                "--config",
                str(config_path),
                "--list-models",
            ])
        self.assertEqual(status, 0)
        self.assertEqual(
            output.getvalue(), "\n".join(model_aliases()) + "\n"
        )


if __name__ == "__main__":
    unittest.main()
