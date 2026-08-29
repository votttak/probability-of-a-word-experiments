"""Fake-only tests for the standalone tuned-lens decoder adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from h01_data.tuned_lens_decoder import (  # noqa: E402
    KuribayashiTunedLensDecoder,
    TunedLensArtifactError,
    TunedLensCompatibilityError,
    inspect_local_tuned_lens_artifact,
    load_local_tuned_lens_decoder,
    sha256_file,
)


class FakeLens:
    """Minimal TunedLens test double with observable loader and calls."""

    translator_count = 3
    load_calls = []

    def __init__(self):
        self.calls = []
        self.to_device = None
        self.eval_called = False

    @classmethod
    def from_model_and_pretrained(cls, model, resource_id, **kwargs):
        lens = cls()
        cls.load_calls.append((model, resource_id, kwargs, lens))
        return lens

    def __len__(self):
        return self.translator_count

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, hidden, idx):
        self.calls.append((hidden.copy(), idx))
        return hidden + idx


def fake_model(
    *,
    name="gpt2",
    layers=3,
    hidden_size=4,
    revision="model-revision",
    device="fake-device",
):
    return SimpleNamespace(
        config=SimpleNamespace(
            name_or_path=name,
            _name_or_path=name,
            num_hidden_layers=layers,
            hidden_size=hidden_size,
            _commit_hash=revision,
        ),
        device=device,
    )


class TunedLensDecoderTest(unittest.TestCase):
    def setUp(self):
        FakeLens.load_calls.clear()
        FakeLens.translator_count = 3
        self.temporary = tempfile.TemporaryDirectory()
        self.artifact_dir = Path(self.temporary.name) / "lens"
        self.artifact_dir.mkdir()
        self.params = b"fake tuned-lens parameters\x00\x01"
        self.config = {
            "base_model_name_or_path": "gpt2",
            "base_model_revision": "model-revision",
            "d_model": 4,
            "num_hidden_layers": 3,
            "bias": True,
            "lens_type": "linear_tuned_lens",
        }
        self._write_artifact()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_artifact(self):
        (self.artifact_dir / "config.json").write_text(
            json.dumps(self.config), encoding="utf8"
        )
        (self.artifact_dir / "params.pt").write_bytes(self.params)

    def _load(self, **kwargs):
        return load_local_tuned_lens_decoder(
            fake_model(),
            self.artifact_dir,
            tuned_lens_class=FakeLens,
            package_version="0.2.0-test",
            **kwargs,
        )

    def test_local_artifact_hashes_and_provenance_are_explicit(self):
        decoder = self._load()
        provenance = decoder.provenance()
        artifact = provenance["artifact"]

        self.assertEqual(provenance["decoder"], "tuned-lens")
        self.assertEqual(provenance["tuned_lens_package_version"], "0.2.0-test")
        self.assertEqual(
            provenance["layer_indexing"],
            "hidden_states[layer_id], idx=layer_id",
        )
        self.assertTrue(provenance["embedding_hidden_state_supported"])
        self.assertEqual(
            provenance["final_layer_policy"], "ordinary-logits-bypass"
        )
        self.assertEqual(artifact["resource_path"], str(self.artifact_dir.resolve()))
        self.assertEqual(
            artifact["params_sha256"], hashlib.sha256(self.params).hexdigest()
        )
        self.assertEqual(
            artifact["config_sha256"],
            sha256_file(self.artifact_dir / "config.json"),
        )
        self.assertTrue(
            provenance["validation"]["base_model_name_validated"]
        )
        self.assertTrue(
            provenance["validation"]["base_model_revision_validated"]
        )

        _, resource_id, load_kwargs, lens = FakeLens.load_calls[-1]
        self.assertEqual(resource_id, str(self.artifact_dir.resolve()))
        self.assertEqual(load_kwargs, {"map_location": "fake-device"})
        self.assertEqual(lens.to_device, "fake-device")
        self.assertTrue(lens.eval_called)

    def test_decoder_uses_hidden_state_index_as_translator_index(self):
        decoder = self._load()
        hidden_states = [
            np.full((1, 5, 4), fill_value=index, dtype=float)
            for index in range(4)
        ]
        ordinary_logits = np.full((1, 5, 11), 99.0)

        decoded = decoder.layer_logits(
            1, hidden_states, ordinary_logits, position_offset=2
        )

        self.assertEqual(len(decoder.lens.calls), 1)
        observed_hidden, observed_index = decoder.lens.calls[0]
        self.assertEqual(observed_index, 1)
        np.testing.assert_array_equal(observed_hidden, hidden_states[1][:, 2:, :])
        np.testing.assert_array_equal(decoded, hidden_states[1][:, 2:, :] + 1)

    def test_final_layer_bypasses_lens_and_returns_ordinary_logits(self):
        decoder = self._load()
        hidden_states = [np.zeros((1, 5, 4)) for _ in range(4)]
        ordinary_logits = np.arange(55).reshape(1, 5, 11)

        decoded = decoder.layer_logits(
            3, hidden_states, ordinary_logits, position_offset=1
        )

        self.assertEqual(decoder.lens.calls, [])
        np.testing.assert_array_equal(decoded, ordinary_logits[:, 1:, :])

    def test_embedding_is_decoded_and_out_of_range_layers_are_rejected(self):
        decoder = self._load()
        hidden_states = [np.zeros((1, 2, 4)) for _ in range(4)]
        ordinary_logits = np.zeros((1, 2, 9))
        decoded = decoder.layer_logits(0, hidden_states, ordinary_logits)
        self.assertEqual(decoder.lens.calls[-1][1], 0)
        np.testing.assert_array_equal(decoded, hidden_states[0])
        for layer in (-1, 4):
            with self.subTest(layer=layer):
                with self.assertRaisesRegex(ValueError, "between 0 and 3"):
                    decoder.layer_logits(layer, hidden_states, ordinary_logits)

    def test_artifact_must_be_local_complete_and_well_formed(self):
        with self.assertRaisesRegex(TunedLensArtifactError, "local directory"):
            inspect_local_tuned_lens_artifact(self.artifact_dir / "missing")

        (self.artifact_dir / "params.pt").unlink()
        with self.assertRaisesRegex(TunedLensArtifactError, "parameter checkpoint"):
            inspect_local_tuned_lens_artifact(self.artifact_dir)

        (self.artifact_dir / "params.pt").write_bytes(self.params)
        (self.artifact_dir / "config.json").write_text("[]", encoding="utf8")
        with self.assertRaisesRegex(TunedLensArtifactError, "JSON object"):
            inspect_local_tuned_lens_artifact(self.artifact_dir)

    def test_model_name_layer_dimension_and_revision_mismatches_fail(self):
        cases = (
            ("base_model_name_or_path", "gpt2-large", "base-model mismatch"),
            ("num_hidden_layers", 2, "layer-count mismatch"),
            ("d_model", 8, "hidden-size mismatch"),
            ("base_model_revision", "other-revision", "base-revision mismatch"),
        )
        for key, value, message in cases:
            with self.subTest(key=key):
                self.config[key] = value
                self._write_artifact()
                with self.assertRaisesRegex(TunedLensCompatibilityError, message):
                    self._load()
                self.config = {
                    "base_model_name_or_path": "gpt2",
                    "base_model_revision": "model-revision",
                    "d_model": 4,
                    "num_hidden_layers": 3,
                    "bias": True,
                    "lens_type": "linear_tuned_lens",
                }

    def test_loaded_translator_count_is_validated(self):
        FakeLens.translator_count = 2
        with self.assertRaisesRegex(
            TunedLensCompatibilityError, "translator-count mismatch"
        ):
            self._load()

    def test_explicit_expected_base_name_supports_local_model_paths(self):
        model = fake_model(name=str(Path(self.temporary.name) / "snapshot"))
        decoder = load_local_tuned_lens_decoder(
            model,
            self.artifact_dir,
            expected_base_model_name="gpt2",
            tuned_lens_class=FakeLens,
            package_version="0.2.0-test",
        )
        self.assertEqual(
            decoder.validation["base_model_name_validation_source"], "explicit"
        )

    def test_tuned_lens_import_is_lazy_and_actionable(self):
        with patch(
            "h01_data.tuned_lens_decoder.import_module",
            side_effect=ImportError("not installed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "tuned-lens==0.2.0"):
                KuribayashiTunedLensDecoder.from_local_artifacts(
                    fake_model(), self.artifact_dir
                )


if __name__ == "__main__":
    unittest.main()
