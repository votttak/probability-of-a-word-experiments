"""Static tests for the cluster launch and provenance preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import preflight_layer_factorial as preflight
from scripts import stage_layer_factorial_resources as staging


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class LayerFactorialClusterDeploymentTest(unittest.TestCase):
    def test_shell_launchers_are_valid_and_config_driven(self):
        launcher = REPOSITORY_ROOT / "scripts/run_layer_factorial_cluster.sh"
        all_launcher = (
            REPOSITORY_ROOT / "scripts/run_all_layer_factorial_cluster.sh"
        )
        for path in (launcher, all_launcher):
            subprocess.run(["bash", "-n", str(path)], check=True)
        text = launcher.read_text(encoding="utf8")
        self.assertIn("layer_factorial_config.py", text)
        self.assertIn("--config", text)
        self.assertIn("--get runtime.jobs", text)
        self.assertIn('--jobs "$FACTORIAL_JOBS"', text)
        self.assertIn("--precomputed-frequency-fname", text)
        self.assertNotIn("--response-columns time paper_time", text)
        self.assertIn("HF_HUB_OFFLINE=1", text)
        self.assertIn("--smoke-load", text)
        self.assertIn("flock -n 9", text)
        all_text = all_launcher.read_text(encoding="utf8")
        self.assertIn("layer_factorial_config.py", all_text)
        self.assertIn("--list-models", all_text)
        self.assertIn('for model in "${MODELS[@]}"', all_text)

    def test_makefile_uses_the_central_configuration(self):
        text = (
            REPOSITORY_ROOT / "MakefileLayerFactorial"
        ).read_text(encoding="utf8")
        self.assertIn(
            "LAYER_FACTORIAL_CONFIG ?= configs/layer_factorial.json",
            text,
        )
        self.assertIn("--get runtime.jobs", text)
        self.assertIn("--resolve-path joint_template", text)
        self.assertIn('--config "$(LAYER_FACTORIAL_CONFIG)"', text)
        self.assertNotIn("--response-columns time paper_time", text)
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
        self.assertNotIn("snapshot_download(", text)

    def test_model_staging_downloads_only_exact_files_in_stable_order(self):
        spec = SimpleNamespace(
            alias="fixture",
            hf_name="fixture/model",
            base_model_revision="a" * 40,
            base_tokenizer_files=("tokenizer.json", "config.json"),
            base_weight_files=("weights.bin", "tokenizer.json"),
        )
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshots" / spec.base_model_revision

            def download(**kwargs):
                calls.append(kwargs)
                return snapshot / kwargs["filename"]

            observed = staging.stage_model_snapshot(
                spec, Path(directory) / "hub", False, download
            )

        self.assertEqual(observed, snapshot)
        self.assertEqual(
            [call["filename"] for call in calls],
            ["config.json", "tokenizer.json", "weights.bin"],
        )
        self.assertTrue(all(not call["local_files_only"] for call in calls))

    def test_model_verify_only_uses_offline_exact_file_lookups(self):
        spec = SimpleNamespace(
            alias="fixture",
            hf_name="fixture/model",
            base_model_revision="b" * 40,
            base_tokenizer_files=("tokenizer.json",),
            base_weight_files=("weights.bin",),
        )
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshots" / spec.base_model_revision

            def download(**kwargs):
                calls.append(kwargs)
                return snapshot / kwargs["filename"]

            staging.stage_model_snapshot(
                spec, Path(directory) / "hub", True, download
            )

        self.assertEqual(
            [call["filename"] for call in calls],
            ["config.json", "tokenizer.json", "weights.bin"],
        )
        self.assertTrue(all(call["local_files_only"] for call in calls))

    def test_logit_only_runtime_does_not_require_tuned_lens(self):
        calls = []

        def package_version(name):
            calls.append(name)
            if name == "wordsprobability":
                return preflight.WORSPROBABILITY_VERSION
            if name == "tuned-lens":
                return preflight.TUNED_LENS_PACKAGE_VERSION
            return "fixture"

        torch = ModuleType("torch")
        torch.cuda = SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
        )
        with (
            patch.object(
                preflight, "package_version", side_effect=package_version
            ),
            patch.dict(sys.modules, {"torch": torch}),
        ):
            observed = preflight.validate_runtime(
                False, require_tuned_lens=False
            )
        self.assertNotIn("tuned-lens", calls)
        self.assertNotIn("tuned-lens", observed["packages"])

        calls.clear()
        with (
            patch.object(
                preflight, "package_version", side_effect=package_version
            ),
            patch.dict(sys.modules, {"torch": torch}),
        ):
            observed = preflight.validate_runtime(
                False, require_tuned_lens=True
            )
        self.assertIn("tuned-lens", calls)
        self.assertEqual(
            observed["packages"]["tuned-lens"],
            preflight.TUNED_LENS_PACKAGE_VERSION,
        )

    def test_logit_only_staging_touches_only_base_model_files(self):
        spec = SimpleNamespace(
            alias="fixture",
            hf_name="fixture/model",
            base_model_revision="c" * 40,
            final_layer=2,
            base_tokenizer_files=("tokenizer.json",),
            base_weight_files=("weights.bin",),
        )
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshots" / spec.base_model_revision

            def download(**kwargs):
                calls.append(kwargs)
                path = snapshot / kwargs["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                if kwargs["filename"] == "config.json":
                    path.write_text(
                        json.dumps({"n_layer": spec.final_layer}),
                        encoding="utf8",
                    )
                else:
                    path.write_bytes(b"fixture")
                return path

            huggingface_hub = ModuleType("huggingface_hub")
            huggingface_hub.hf_hub_download = download
            with patch.dict(
                sys.modules, {"huggingface_hub": huggingface_hub}
            ):
                staging.stage_one(
                    spec,
                    None,
                    root / "hub",
                    False,
                    include_tuned_lens=False,
                )

        self.assertEqual(
            [call["filename"] for call in calls],
            ["config.json", "tokenizer.json", "weights.bin"],
        )
        self.assertTrue(
            all(call["repo_id"] == spec.hf_name for call in calls)
        )

    def test_logit_only_staging_cli_does_not_require_lens_root(self):
        payload = json.loads(
            staging.DEFAULT_CONFIG_PATH.read_text(encoding="utf8")
        )
        payload["models"] = ["gpt2-small"]
        payload["switches"]["lens_methods"] = ["logit-lens"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf8")
            with patch.object(staging, "stage_one") as stage_one:
                status = staging.main([
                    "--all",
                    "--config",
                    str(config_path),
                    "--hf-home",
                    str(root / "hf"),
                ])
        self.assertEqual(status, 0)
        stage_one.assert_called_once()
        self.assertIsNone(stage_one.call_args.args[1])
        self.assertFalse(
            stage_one.call_args.kwargs["include_tuned_lens"]
        )

    def test_tuned_lens_staging_still_requires_lens_root(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError, "--lens-root is required"
            ):
                staging.main([
                    "--all",
                    "--config",
                    str(staging.DEFAULT_CONFIG_PATH),
                    "--hf-home",
                    str(Path(directory) / "hf"),
                ])

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
