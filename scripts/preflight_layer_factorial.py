#!/usr/bin/env python3

"""Fail-fast validation for a full layer-factorial cluster run."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from h01_data.layer_factorial_models import (  # noqa: E402
    TUNED_LENS_PACKAGE_VERSION,
    get_model_spec,
    model_aliases,
)


EXPECTED_ROWS = 10_256
EXPECTED_TEXT_SHA256 = (
    "04578a7187ec7edb779362f912df97befc74f7945c4d554902e2049041579da4"
)
EXPECTED_MANIFEST_SHA256 = (
    "f9e14f6ac9d1d7624dba51c9d658721a11c1a596560ade6f33fba6767e4f8263"
)
EXPECTED_PAPER_RT_SHA256 = (
    "cef406dfb4eaef3fdd12b4f94f7f20418fc394dcceb42981a7171370cfc6c145"
)
EXPECTED_FREQUENCY_SHA256 = (
    "208f9749acec1894e9e6b46f56ca889621fdbc8e4a5bb9e879743920f448c47a"
)
EXPECTED_JOINT_SHA256 = {
    "gpt2-small": (
        "a66f7ae10a5f1b342a2d55c7acc1360d9ba119c37db2d29a4be1573f1b84ad84"
    ),
    "gpt2-large": (
        "be507c3bae1f3ffcd150545af0266b92bb98952f3614a16bba671a4d21fc9973"
    ),
    "gpt2-xl": (
        "eee2222c927b46dc40abb8c25867216099afd66fe651c572c8dba4bfc9ad1ec7"
    ),
}
WORSPROBABILITY_VERSION = "0.17"
MODEL_METADATA_PATTERNS = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "vocab.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: observed {observed}, expected {expected}"
        )
    return observed


def read_expected_words(text_path: Path) -> list[tuple[int, int, str]]:
    rows = []
    with text_path.open("r", encoding="utf8") as handle:
        passages = [line.split() for line in handle]
    if len(passages) != 10:
        raise ValueError(f"canonical text has {len(passages)} passages; expected 10")
    for text_id, words in enumerate(passages):
        rows.extend(
            (text_id, word_id, word) for word_id, word in enumerate(words)
        )
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"canonical text has {len(rows)} words; expected {EXPECTED_ROWS}"
        )
    return rows


def read_keyed_words(
    path: Path,
    *,
    text_column: str,
    word_id_column: str,
    word_column: str,
    text_offset: int,
) -> list[tuple[int, int, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {text_column, word_id_column, word_column}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} is missing columns: " + ", ".join(sorted(missing))
            )
        try:
            return [
                (
                    int(row[text_column]) + text_offset,
                    int(row[word_id_column]),
                    row[word_column],
                )
                for row in reader
            ]
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path} contains invalid keyed word rows") from error


def validate_inputs(args, spec) -> dict:
    paths = {
        "text": resolve_path(args.text_fname),
        "manifest": resolve_path(args.sentence_manifest_fname),
        "joint": resolve_path(args.joint_data_fname),
        "paper_rt": resolve_path(args.paper_rt_fname),
        "frequency": resolve_path(args.precomputed_frequency_fname),
    }
    hashes = {
        "text": require_hash(
            paths["text"], EXPECTED_TEXT_SHA256, "canonical text"
        ),
        "manifest": require_hash(
            paths["manifest"], EXPECTED_MANIFEST_SHA256, "sentence manifest"
        ),
        "joint": require_hash(
            paths["joint"], EXPECTED_JOINT_SHA256[spec.alias], "canonical joint"
        ),
        "paper_rt": require_hash(
            paths["paper_rt"], EXPECTED_PAPER_RT_SHA256, "portable paper RT"
        ),
        "frequency": require_hash(
            paths["frequency"],
            EXPECTED_FREQUENCY_SHA256,
            "portable frequency",
        ),
    }

    expected = read_expected_words(paths["text"])
    joint = read_keyed_words(
        paths["joint"],
        text_column="text_id",
        word_id_column="word_id",
        word_column="ref_token",
        text_offset=-1,
    )
    manifest = read_keyed_words(
        paths["manifest"],
        text_column="text_id",
        word_id_column="word_id",
        word_column="word",
        text_offset=0,
    )
    frequency = read_keyed_words(
        paths["frequency"],
        text_column="text_id",
        word_id_column="word_id",
        word_column="word",
        text_offset=-1,
    )
    if joint != expected:
        raise ValueError("canonical joint keys/words do not match canonical text")
    if manifest != expected:
        raise ValueError("sentence manifest keys/words do not match canonical text")
    if len(frequency) != len(expected):
        raise ValueError("portable frequency has incomplete key coverage")
    for expected_row, frequency_row in zip(expected, frequency):
        compatible = (
            expected_row[:2] == (1, 748)
            and expected_row[2] == "peeked"
            and frequency_row[2] == "peaked"
        )
        if frequency_row[:2] != expected_row[:2] or (
            frequency_row[2] != expected_row[2] and not compatible
        ):
            raise ValueError(
                "portable frequency keys/words do not match canonical text"
            )
    with paths["paper_rt"].open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        paper_rows = sum(1 for _ in csv.DictReader(handle, delimiter="\t"))
    if paper_rows != EXPECTED_ROWS:
        raise ValueError(
            f"portable paper RT has {paper_rows} rows; expected {EXPECTED_ROWS}"
        )
    return {
        name: {"path": str(paths[name]), "sha256": hashes[name]}
        for name in paths
    }


def validate_lens(path: Path, spec) -> dict:
    config_path = path / "config.json"
    params_path = path / "params.pt"
    config_hash = require_hash(
        config_path, spec.lens_config_sha256, "tuned-lens config"
    )
    params_hash = require_hash(
        params_path, spec.lens_params_sha256, "tuned-lens parameters"
    )
    if config_path.stat().st_size != spec.lens_config_size:
        raise ValueError("tuned-lens config size mismatch")
    if params_path.stat().st_size != spec.lens_params_size:
        raise ValueError("tuned-lens parameter size mismatch")
    config = json.loads(config_path.read_text(encoding="utf8"))
    expected = {
        "base_model_name_or_path": spec.hf_name,
        "base_model_revision": spec.base_model_revision,
        "num_hidden_layers": spec.final_layer,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"tuned-lens config {key} mismatch: "
                f"observed {config.get(key)!r}, expected {value!r}"
            )
    return {
        "path": str(path),
        "config_sha256": config_hash,
        "params_sha256": params_hash,
    }


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as error:
        raise RuntimeError(f"required package is not installed: {name}") from error


def validate_runtime(require_cuda: bool) -> dict:
    packages = {
        name: package_version(name)
        for name in (
            "wordsprobability",
            "tuned-lens",
            "torch",
            "transformers",
            "huggingface-hub",
            "pandas",
            "numpy",
        )
    }
    expected = {
        "wordsprobability": WORSPROBABILITY_VERSION,
        "tuned-lens": TUNED_LENS_PACKAGE_VERSION,
    }
    for name, wanted in expected.items():
        if packages[name] != wanted:
            raise RuntimeError(
                f"{name}=={wanted} is required; found {packages[name]}"
            )
    import torch

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda = {
        "CUDA_VISIBLE_DEVICES": cuda_visible,
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    if require_cuda:
        if not cuda_visible or "," in cuda_visible:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES must name exactly one physical GPU"
            )
        if not cuda["available"] or cuda["device_count"] != 1:
            raise RuntimeError(
                "the scoring process must see exactly one available CUDA GPU"
            )
        cuda["device_name"] = torch.cuda.get_device_name(0)
    return {"packages": packages, "cuda": cuda}


def validate_model_cache(spec, hf_home: Path) -> dict:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required for cache preflight") from error
    snapshot = Path(snapshot_download(
        repo_id=spec.hf_name,
        revision=spec.base_model_revision,
        cache_dir=hf_home / "hub",
        allow_patterns=(*MODEL_METADATA_PATTERNS, spec.base_weight_file),
        local_files_only=True,
    ))
    config = json.loads((snapshot / "config.json").read_text(encoding="utf8"))
    layers = config.get(
        "n_layer", config.get("num_hidden_layers", config.get("num_layers"))
    )
    if layers != spec.final_layer:
        raise ValueError(
            f"cached model has {layers!r} layers; expected {spec.final_layer}"
        )
    for filename in ("vocab.json", "merges.txt", spec.base_weight_file):
        path = snapshot / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"cached model resource is missing: {path}")
    return {
        "repository": spec.hf_name,
        "revision": spec.base_model_revision,
        "weight_file": spec.base_weight_file,
        "snapshot_path": str(snapshot.resolve()),
    }


def validate_offline_smoke(spec, lens_path: Path) -> dict:
    """Actually load and execute the pinned model and tuned lens offline."""

    true_values = {"1", "true", "yes", "on"}
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(variable, "").lower() not in true_values:
            raise RuntimeError(
                f"{variable}=1 is required for the offline load smoke"
            )
    from h01_data.get_context_limited_surprisals import (
        load_wordsprobability_model,
    )
    from h01_data.tuned_lens_decoder import (
        load_local_tuned_lens_decoder,
    )
    import torch

    wrapper = load_wordsprobability_model(
        spec.alias, revision=spec.base_model_revision
    )
    decoder = load_local_tuned_lens_decoder(
        wrapper.model,
        lens_path,
        expected_base_model_name=spec.hf_name,
    )
    token_ids = wrapper.tokenizer.encode(
        " preflight", add_special_tokens=False
    )
    if not token_ids:
        raise RuntimeError("tokenizer returned no IDs for preflight input")
    bos_token_id = wrapper.tokenizer.bos_token_id
    if bos_token_id is None:
        raise RuntimeError("tokenizer has no BOS token for preflight")
    input_ids = torch.tensor(
        [[bos_token_id, *token_ids]],
        dtype=torch.long,
        device=wrapper.model.device,
    )
    with torch.inference_mode():
        output = wrapper.model(
            input_ids, output_hidden_states=True, use_cache=False
        )
        decoded = decoder.layer_logits(
            0, output.hidden_states, output.logits
        )
    if decoded.shape != output.logits.shape:
        raise RuntimeError(
            "tuned-lens smoke output shape differs from ordinary logits"
        )
    if not bool(torch.isfinite(decoded).all().item()):
        raise RuntimeError("tuned-lens smoke produced non-finite logits")
    parameter = next(wrapper.model.parameters())
    return {
        "validated": True,
        "input_tokens": int(input_ids.shape[1]),
        "logit_shape": list(decoded.shape),
        "device": str(parameter.device),
        "dtype": str(parameter.dtype),
        "decoder_package_version": decoder.package_version,
        "artifact_config_sha256": decoder.artifact.config_sha256,
        "artifact_params_sha256": decoder.artifact.params_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate inputs, artifacts, cache, and runtime before a run"
    )
    parser.add_argument("--model", choices=model_aliases(), required=True)
    parser.add_argument(
        "--text-fname",
        default="checkpoints/rt/text_rt_data/natural_stories.txt",
    )
    parser.add_argument(
        "--sentence-manifest-fname",
        default=(
            "checkpoints/rt/layer_factorial/manifests/"
            "natural-stories-sentences.tsv"
        ),
    )
    parser.add_argument("--joint-data-fname")
    parser.add_argument(
        "--paper-rt-fname",
        default=(
            "checkpoints/rt/layer_factorial/inputs/"
            "natural-stories-paper-time.tsv"
        ),
    )
    parser.add_argument(
        "--precomputed-frequency-fname",
        default=(
            "checkpoints/rt/layer_factorial/inputs/"
            "natural-stories-paper-frequency.tsv"
        ),
    )
    parser.add_argument("--tuned-lens-path", required=True)
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--check-model-cache", action="store_true")
    parser.add_argument("--smoke-load", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.require_cuda and not args.check_runtime:
        raise ValueError("--require-cuda requires --check-runtime")
    if args.smoke_load and (
        not args.check_runtime or not args.check_model_cache
    ):
        raise ValueError(
            "--smoke-load requires --check-runtime and --check-model-cache"
        )
    spec = get_model_spec(args.model)
    if args.joint_data_fname is None:
        args.joint_data_fname = (
            f"checkpoints/rt/merged_data/natural_stories-{spec.alias}.tsv"
        )
    payload = {
        "schema_version": 1,
        "validated": True,
        "model": spec.alias,
        "base_model_revision": spec.base_model_revision,
        "inputs": validate_inputs(args, spec),
        "tuned_lens": validate_lens(
            resolve_path(args.tuned_lens_path), spec
        ),
    }
    if args.check_runtime:
        payload["runtime"] = validate_runtime(args.require_cuda)
    if args.check_model_cache:
        if not args.hf_home:
            raise ValueError("--hf-home or HF_HOME is required for cache checks")
        payload["model_cache"] = validate_model_cache(
            spec, Path(args.hf_home).expanduser().resolve()
        )
    if args.smoke_load:
        payload["offline_smoke"] = validate_offline_smoke(
            spec, resolve_path(args.tuned_lens_path)
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
