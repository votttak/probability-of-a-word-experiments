#!/usr/bin/env python3

"""Download or offline-verify pinned models and tuned-lens artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import sys
import tempfile


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from h01_data.layer_factorial_models import (  # noqa: E402
    MODEL_SPECS,
    TUNED_LENS_REPOSITORY,
    TUNED_LENS_REPOSITORY_TYPE,
    TUNED_LENS_REVISION,
    get_model_spec,
    model_aliases,
)

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(
    path: Path, *, expected_hash: str, expected_size: int, label: str
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    size = path.stat().st_size
    if size != expected_size:
        raise ValueError(
            f"{label} size mismatch: observed {size}, expected {expected_size}"
        )
    digest = sha256_file(path)
    if digest != expected_hash:
        raise ValueError(
            f"{label} SHA-256 mismatch: observed {digest}, "
            f"expected {expected_hash}"
        )


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def write_json_atomic(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def verify_lens_config(path: Path, spec) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read tuned-lens config: {path}") from error
    expected = {
        "base_model_name_or_path": spec.hf_name,
        "base_model_revision": spec.lens_base_model_revision,
        "num_hidden_layers": spec.final_layer,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"tuned-lens config {key} mismatch for {spec.alias}: "
                f"observed {payload.get(key)!r}, expected {value!r}"
            )
    return payload


def verify_model_snapshot(snapshot_path: Path, spec) -> None:
    config_path = snapshot_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"unable to read cached model config: {config_path}"
        ) from error
    layer_count = config.get(
        "n_layer", config.get("num_hidden_layers", config.get("num_layers"))
    )
    if layer_count != spec.final_layer:
        raise ValueError(
            f"cached {spec.alias} has {layer_count!r} layers; "
            f"expected {spec.final_layer}"
        )
    for filename in required_model_files(spec):
        path = snapshot_path / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"cached {spec.alias} resource is missing: {path}"
            )


def required_model_files(spec) -> tuple[str, ...]:
    """Return the exact pinned model files once, in deterministic order."""

    return tuple(dict.fromkeys((
        "config.json",
        *spec.base_tokenizer_files,
        *spec.base_weight_files,
    )))


def snapshot_parent(downloaded_path: Path, filename: str) -> Path:
    """Derive the snapshot directory from an exact Hub download path."""

    parts = PurePosixPath(filename).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"invalid Hugging Face model filename: {filename!r}")
    parent = downloaded_path
    for expected_part in reversed(parts):
        if parent.name != expected_part:
            raise ValueError(
                f"downloaded path {downloaded_path} does not end with "
                f"the requested filename {filename!r}"
            )
        parent = parent.parent
    return parent


def stage_model_snapshot(spec, hub_cache: Path, verify_only: bool, download) -> Path:
    """Download or locate only the registry-pinned files for one model."""

    snapshot_path = None
    for filename in required_model_files(spec):
        downloaded_path = Path(download(
            repo_id=spec.hf_name,
            filename=filename,
            revision=spec.base_model_revision,
            cache_dir=hub_cache,
            local_files_only=verify_only,
        ))
        observed_snapshot = snapshot_parent(downloaded_path, filename)
        if snapshot_path is None:
            snapshot_path = observed_snapshot
        elif observed_snapshot != snapshot_path:
            raise ValueError(
                f"cached {spec.alias} files span multiple snapshots: "
                f"{snapshot_path} and {observed_snapshot}"
            )
    if snapshot_path is None:  # pragma: no cover - registry invariants forbid it
        raise ValueError(f"no required model files registered for {spec.alias}")
    return snapshot_path


def stage_one(spec, lens_root: Path, hub_cache: Path, verify_only: bool) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required to stage layer-factorial resources"
        ) from error

    lens_path = lens_root / spec.lens_artifact
    expected_files = {
        "config.json": (spec.lens_config_sha256, spec.lens_config_size),
        "params.pt": (spec.lens_params_sha256, spec.lens_params_size),
    }
    if not verify_only:
        for filename, (digest, size) in expected_files.items():
            source = Path(hf_hub_download(
                repo_id=TUNED_LENS_REPOSITORY,
                filename=f"lens/{spec.lens_artifact}/{filename}",
                repo_type=TUNED_LENS_REPOSITORY_TYPE,
                revision=TUNED_LENS_REVISION,
                cache_dir=hub_cache,
            ))
            verify_file(
                source,
                expected_hash=digest,
                expected_size=size,
                label=f"downloaded {spec.alias} {filename}",
            )
            copy_atomic(source, lens_path / filename)

    for filename, (digest, size) in expected_files.items():
        verify_file(
            lens_path / filename,
            expected_hash=digest,
            expected_size=size,
            label=f"{spec.alias} {filename}",
        )
    verify_lens_config(lens_path / "config.json", spec)

    snapshot_path = stage_model_snapshot(
        spec, hub_cache, verify_only, hf_hub_download
    )
    verify_model_snapshot(snapshot_path, spec)
    if not verify_only:
        write_json_atomic(
            {
                "schema_version": 2,
                "model": spec.alias,
                "base_model": {
                    "repository": spec.hf_name,
                    "revision": spec.base_model_revision,
                    "weight_files": list(spec.base_weight_files),
                    "tokenizer_files": list(spec.base_tokenizer_files),
                    "snapshot_path": str(snapshot_path.resolve()),
                },
                "tuned_lens": {
                    "repository": TUNED_LENS_REPOSITORY,
                    "repository_type": TUNED_LENS_REPOSITORY_TYPE,
                    "revision": TUNED_LENS_REVISION,
                    "artifact": spec.lens_artifact,
                    "base_model_revision": spec.lens_base_model_revision,
                    "config_sha256": spec.lens_config_sha256,
                    "params_sha256": spec.lens_params_sha256,
                },
            },
            lens_path / "resource-manifest.json",
        )
    print(
        f"{'Verified' if verify_only else 'Staged'} {spec.alias}: "
        f"{lens_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage exact model and tuned-lens snapshots for the cluster"
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model", choices=model_aliases())
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--lens-root", required=True)
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME"),
        help="Hugging Face home; its hub subdirectory stores snapshots",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.hf_home:
        raise ValueError("--hf-home or HF_HOME is required")
    hf_home = Path(args.hf_home).expanduser().resolve()
    lens_root = Path(args.lens_root).expanduser().resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    lens_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    specs = MODEL_SPECS if args.all else (get_model_spec(args.model),)
    for spec in specs:
        stage_one(spec, lens_root, hf_home / "hub", args.verify_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
