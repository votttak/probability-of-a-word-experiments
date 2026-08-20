"""Tests for the canonical multi-model internal-layer registry."""

import contextlib
import io
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from h01_data.internal_layer_models import (  # noqa: E402
    MODEL_SPECS,
    get_model_spec,
    main,
    model_aliases,
)


EXPECTED_MODELS = {
    "gpt2-small": ("gpt2", "gpt2", 12, "tracked", 5e-4),
    "gpt2-medium": ("gpt2-medium", "gpt2", 24, "fresh", 5e-4),
    "gpt2-large": ("gpt2-large", "gpt2", 36, "fresh", 5e-4),
    "gpt2-xl": ("gpt2-xl", "gpt2", 48, "fresh", 5e-4),
    "pythia-70m": (
        "EleutherAI/pythia-70m", "pythia", 6, "fresh", 1e-2
    ),
    "pythia-160m": (
        "EleutherAI/pythia-160m", "pythia", 12, "fresh", 1e-2
    ),
    "pythia-410m": (
        "EleutherAI/pythia-410m", "pythia", 24, "fresh", 1e-2
    ),
    "pythia-14b": (
        "EleutherAI/pythia-1.4b", "pythia", 24, "fresh", 1e-2
    ),
    "pythia-28b": (
        "EleutherAI/pythia-2.8b", "pythia", 32, "fresh", 1e-2
    ),
    "pythia-69b": (
        "EleutherAI/pythia-6.9b", "pythia", 32, "fresh", 1e-2
    ),
}


class InternalLayerModelRegistryTest(unittest.TestCase):
    def test_registry_is_exactly_the_ten_completed_c_models(self):
        self.assertEqual(set(model_aliases()), set(EXPECTED_MODELS))
        self.assertEqual(len(MODEL_SPECS), 10)
        self.assertNotIn("pythia-120b", model_aliases())

        for alias, (
            hf_name, family, final_layer, reference_policy, tolerance
        ) in EXPECTED_MODELS.items():
            spec = get_model_spec(alias)
            self.assertEqual(spec.hf_name, hf_name)
            self.assertEqual(spec.family, family)
            self.assertEqual(spec.final_layer, final_layer)
            self.assertEqual(spec.reference_policy, reference_policy)
            self.assertEqual(
                spec.layer_ids, tuple(range(1, final_layer + 1))
            )
            self.assertEqual(spec.default_anchor_tolerance, tolerance)

    def test_only_validated_gpt2_small_uses_tracked_reference(self):
        tracked = [
            spec.alias
            for spec in MODEL_SPECS
            if spec.reference_policy == "tracked"
        ]
        self.assertEqual(tracked, ["gpt2-small"])

    def test_unknown_alias_has_actionable_error(self):
        with self.assertRaisesRegex(
            KeyError, "unknown internal-layer model 'pythia-120b'"
        ):
            get_model_spec("pythia-120b")

    def test_cli_fields_are_shell_friendly(self):
        cases = {
            "hf_name": "EleutherAI/pythia-1.4b",
            "family": "pythia",
            "final_layer": "24",
            "layer_ids": " ".join(str(layer) for layer in range(1, 25)),
            "reference_policy": "fresh",
            "default_anchor_tolerance": "0.01",
        }
        for field, expected in cases.items():
            with self.subTest(field=field):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    status = main([
                        "--model", "pythia-14b", "--field", field
                    ])
                self.assertEqual(status, 0)
                self.assertEqual(output.getvalue().strip(), expected)

    def test_cli_list_preserves_registry_order(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["--list"])
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue().splitlines(), list(model_aliases()))


if __name__ == "__main__":
    unittest.main()
