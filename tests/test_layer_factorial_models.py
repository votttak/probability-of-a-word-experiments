"""Tests for immutable model and tuned-lens identities."""

from __future__ import annotations

import contextlib
import io
import unittest

from src.h01_data import layer_factorial_models as models


class LayerFactorialModelsTest(unittest.TestCase):
    def test_only_scientifically_compatible_models_are_registered(self):
        self.assertEqual(
            models.model_aliases(),
            ("gpt2-small", "gpt2-large", "gpt2-xl"),
        )
        with self.assertRaisesRegex(KeyError, "unsupported"):
            models.get_model_spec("gpt2-medium")
        with self.assertRaisesRegex(KeyError, "unsupported"):
            models.get_model_spec("pythia-70m")

    def test_registry_pins_revisions_layers_and_artifact_hashes(self):
        expected = {
            "gpt2-small": (
                "gpt2",
                12,
                "e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
                "model.safetensors",
                "gpt2",
            ),
            "gpt2-large": (
                "gpt2-large",
                36,
                "212095d5832abbf9926672e1c1e8d14312a3be20",
                "model.safetensors",
                "gpt2-large",
            ),
            "gpt2-xl": (
                "gpt2-xl",
                48,
                "33cdb5c0db5423c1879b1b9f16c352988e8754a8",
                "pytorch_model.bin",
                "gpt2-xl",
            ),
        }
        for alias, values in expected.items():
            spec = models.get_model_spec(alias)
            self.assertEqual(
                (
                    spec.hf_name,
                    spec.final_layer,
                    spec.base_model_revision,
                    spec.base_weight_file,
                    spec.lens_artifact,
                ),
                values,
            )
            self.assertEqual(len(spec.lens_config_sha256), 64)
            self.assertEqual(len(spec.lens_params_sha256), 64)
            self.assertGreater(spec.lens_params_size, 1_000_000)

    def test_cli_is_shell_queryable(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = models.main([
                "--model",
                "gpt2-xl",
                "--field",
                "final_layer",
            ])
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue(), "48\n")


if __name__ == "__main__":
    unittest.main()
