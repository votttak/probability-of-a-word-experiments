"""Static tests for the cluster launch and provenance preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest

from scripts import preflight_layer_factorial as preflight


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class LayerFactorialClusterDeploymentTest(unittest.TestCase):
    def test_shell_launchers_are_valid_and_sequential(self):
        launcher = REPOSITORY_ROOT / "scripts/run_layer_factorial_cluster.sh"
        all_launcher = (
            REPOSITORY_ROOT / "scripts/run_all_layer_factorial_cluster.sh"
        )
        for path in (launcher, all_launcher):
            subprocess.run(["bash", "-n", str(path)], check=True)
        text = launcher.read_text(encoding="utf8")
        self.assertIn("--jobs 1", text)
        self.assertIn("--precomputed-frequency-fname", text)
        self.assertIn("--response-columns time paper_time", text)
        self.assertIn("HF_HUB_OFFLINE=1", text)
        self.assertIn("--smoke-load", text)
        self.assertIn("flock -n 9", text)
        self.assertNotIn("--jobs 4", text)
        all_text = all_launcher.read_text(encoding="utf8")
        self.assertIn(
            'mapfile -t MODELS < <("$PYTHON_BIN" "$REGISTRY" --list)',
            all_text,
        )
        self.assertIn('for model in "${MODELS[@]}"', all_text)

    def test_makefile_uses_tracked_portable_inputs_and_one_job(self):
        text = (
            REPOSITORY_ROOT / "MakefileLayerFactorial"
        ).read_text(encoding="utf8")
        self.assertIn("FACTORIAL_JOBS ?= 1", text)
        self.assertIn(
            "checkpoints/rt/merged_data/natural_stories-$(MODEL).tsv",
            text,
        )
        self.assertIn("--precomputed-frequency-fname", text)
        self.assertNotIn(
            "layer_factorial_full: $(FULL_SENTENCE_MANIFEST)", text
        )

    def test_resource_staging_defaults_to_non_xet_downloads(self):
        text = (
            REPOSITORY_ROOT / "scripts/stage_layer_factorial_resources.py"
        ).read_text(encoding="utf8")
        self.assertIn(
            'os.environ.setdefault("HF_HUB_DISABLE_XET", "1")', text
        )
        self.assertIn("*spec.base_weight_files", text)
        self.assertIn("*spec.base_tokenizer_files", text)

    def test_preflight_validates_lens_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            lens = Path(directory)
            config = {
                "base_model_name_or_path": "fixture/model",
                "base_model_revision": "revision",
                "num_hidden_layers": 2,
            }
            config_bytes = json.dumps(config).encode("utf8")
            params_bytes = b"fixture parameters"
            (lens / "config.json").write_bytes(config_bytes)
            (lens / "params.pt").write_bytes(params_bytes)
            spec = SimpleNamespace(
                alias="fixture",
                hf_name="fixture/model",
                final_layer=2,
                base_model_revision="revision",
                lens_base_model_revision="revision",
                lens_config_sha256=hashlib.sha256(
                    config_bytes
                ).hexdigest(),
                lens_config_size=len(config_bytes),
                lens_params_sha256=hashlib.sha256(
                    params_bytes
                ).hexdigest(),
                lens_params_size=len(params_bytes),
            )
            observed = preflight.validate_lens(lens, spec)
            self.assertEqual(
                observed["params_sha256"], spec.lens_params_sha256
            )
            (lens / "params.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                preflight.validate_lens(lens, spec)


if __name__ == "__main__":
    unittest.main()
