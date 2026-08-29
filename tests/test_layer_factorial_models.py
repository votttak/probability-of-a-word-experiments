"""Tests for immutable model and tuned-lens identities."""

from __future__ import annotations

import contextlib
import io
import unittest

from src.h01_data import layer_factorial_models as models


class LayerFactorialModelsTest(unittest.TestCase):
    def test_scientifically_compatible_models_are_registered(self):
        self.assertEqual(
            models.model_aliases(),
            (
                "gpt2-small",
                "gpt2-large",
                "gpt2-xl",
                "pythia-70m",
                "pythia-160m",
                "pythia-410m",
                "pythia-14b",
                "pythia-28b",
                "pythia-69b",
            ),
        )
        with self.assertRaisesRegex(KeyError, "unsupported"):
            models.get_model_spec("gpt2-medium")

    def test_registry_pins_revisions_layers_and_artifact_hashes(self):
        expected = {
            "gpt2-small": (
                "gpt2",
                12,
                "e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
                ("model.safetensors",),
                "gpt2",
                "e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
            ),
            "gpt2-large": (
                "gpt2-large",
                36,
                "212095d5832abbf9926672e1c1e8d14312a3be20",
                ("model.safetensors",),
                "gpt2-large",
                "212095d5832abbf9926672e1c1e8d14312a3be20",
            ),
            "gpt2-xl": (
                "gpt2-xl",
                48,
                "33cdb5c0db5423c1879b1b9f16c352988e8754a8",
                ("pytorch_model.bin",),
                "gpt2-xl",
                "33cdb5c0db5423c1879b1b9f16c352988e8754a8",
            ),
            "pythia-70m": (
                "EleutherAI/pythia-70m-deduped",
                6,
                "9a7c847e93250c8f24d4b7e7134dbf369e8fc9cb",
                ("model.safetensors",),
                "EleutherAI/pythia-70m-deduped",
                None,
            ),
            "pythia-160m": (
                "EleutherAI/pythia-160m-deduped",
                12,
                "c54a0e0b28cc667b6f278803024438d57f847b5d",
                ("model.safetensors",),
                "EleutherAI/pythia-160m-deduped",
                None,
            ),
            "pythia-410m": (
                "EleutherAI/pythia-410m-deduped",
                24,
                "c66f7467608ffee8fca0d28cf1f46a7574b53cec",
                ("model.safetensors",),
                "EleutherAI/pythia-410m-deduped",
                None,
            ),
            "pythia-14b": (
                "EleutherAI/pythia-1.4b-deduped",
                24,
                "6d1288ca1da05b700367a229ed2000de0eab8c4d",
                ("model.safetensors",),
                "EleutherAI/pythia-1.4b-deduped",
                None,
            ),
            "pythia-28b": (
                "EleutherAI/pythia-2.8b-deduped",
                32,
                "346f515745789fe4b4acbc74b105707cc9d5a36d",
                ("pytorch_model.bin",),
                "EleutherAI/pythia-2.8b-deduped",
                None,
            ),
            "pythia-69b": (
                "EleutherAI/pythia-6.9b-deduped",
                32,
                "f9a3856d3d568fc12eb0c68d5b0cce1be9013642",
                (
                    "pytorch_model.bin.index.json",
                    "pytorch_model-00001-of-00002.bin",
                    "pytorch_model-00002-of-00002.bin",
                ),
                "EleutherAI/pythia-6.9b-deduped",
                None,
            ),
        }
        for alias, values in expected.items():
            spec = models.get_model_spec(alias)
            self.assertEqual(
                (
                    spec.hf_name,
                    spec.final_layer,
                    spec.base_model_revision,
                    spec.base_weight_files,
                    spec.lens_artifact,
                    spec.lens_base_model_revision,
                ),
                values,
            )
            self.assertTrue(spec.base_tokenizer_files)
            self.assertEqual(len(spec.lens_config_sha256), 64)
            self.assertEqual(len(spec.lens_params_sha256), 64)
            self.assertGreater(spec.lens_params_size, 1_000_000)

    def test_pythia_lens_artifacts_and_hashes_are_exact(self):
        expected = {
            "pythia-70m": (
                "7d8fbce626f53f8eef6470e47c3352f956fc8240d3f964f8af06ce6ee2cb0389",
                "c363b641564af68e9f9b73af56015b6b5c5b08caea5880787744eaabfab01343",
                177,
                6_306_803,
            ),
            "pythia-160m": (
                "cde1c23fcd004678209150528f2f6e6938357af0f4d7ca59c58161271b704500",
                "1d16b68baa1eb903bbf74eaf859cee5c34361f3e8dff331b17f55b8ce52a2dbe",
                179,
                28_354_051,
            ),
            "pythia-410m": (
                "c66232b38ed6e1682dd3fc6f4d8ec89d5e978aff62baf33a2052c0894bce124f",
                "166ea259b35481e1eb2feba50b5ac4d9a8faed47b0937ede0d7bd6d9830dbc95",
                180,
                100_773_155,
            ),
            "pythia-14b": (
                "a9c41eafcb120512f4569327b17831d78e789bfbd8ab4d2d7bed2a8f5a1ce033",
                "b56db530d2c0df1bc5916bae58b241cfd4389dd4b1aa29e7210395df97164824",
                180,
                402_861_347,
            ),
            "pythia-28b": (
                "ca5f7e2395d78885fc1ed654714677cb601efb3e5a55c32fd7efe96eeba61376",
                "314d403f2e1b1bf575ab2e851419d3a687c54138bfae5feace3ff00f1a96fd60",
                180,
                839_204_003,
            ),
            "pythia-69b": (
                "82bb2ea0da41dd20de51479728974642c3e14605b6f5ab4991437dae164ccce8",
                "e2d00af6f64631b932b089fac5ca031061e207d51132a0f7433bd7a34fb06da0",
                180,
                2_148_023_459,
            ),
        }
        for alias, values in expected.items():
            spec = models.get_model_spec(alias)
            self.assertEqual(
                (
                    spec.lens_config_sha256,
                    spec.lens_params_sha256,
                    spec.lens_config_size,
                    spec.lens_params_size,
                ),
                values,
            )

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
