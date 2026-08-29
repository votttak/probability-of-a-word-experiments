"""Standalone, provenance-aware tuned-lens decoding support.

This module intentionally does not import :mod:`tuned_lens` at import time. A
canonical caller must provide a local directory containing ``config.json`` and
``params.pt``; Hugging Face resource names are never forwarded to the tuned-lens
loader. This makes offline execution and artifact identity explicit.

Layer IDs follow Kuribayashi et al.'s implementation:
``hidden_states[layer_id]`` is decoded with translator
``idx=layer_id``, including hidden state zero (the embedding stream), and
the final hidden state bypasses the tuned lens in favour of ordinary logits.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any


CONFIG_FILENAME = "config.json"
PARAMS_FILENAME = "params.pt"
DECODER_NAME = "tuned-lens"


class TunedLensArtifactError(ValueError):
    """A local tuned-lens artifact is missing or malformed."""


class TunedLensCompatibilityError(ValueError):
    """A tuned lens is incompatible with its proposed base model."""


def sha256_file(fname: str | Path, block_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for an artifact file."""

    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise ValueError("block_size must be a positive integer")
    if block_size < 1:
        raise ValueError("block_size must be a positive integer")
    path = Path(fname)
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def tuned_lens_package_version() -> str:
    """Return the installed distribution version without importing it."""

    try:
        return version("tuned-lens")
    except PackageNotFoundError:
        return "unavailable"


def _required_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TunedLensArtifactError(
            f"tuned-lens config {label} must be a positive integer"
        )
    return value


@dataclass(frozen=True)
class LocalTunedLensArtifact:
    """Validated paths, configuration, and content hashes for one local lens."""

    directory: Path
    config_path: Path
    params_path: Path
    config: Mapping[str, Any]
    config_sha256: str
    params_sha256: str
    config_size_bytes: int
    params_size_bytes: int

    def provenance(self) -> dict[str, Any]:
        """Return deterministic artifact identity suitable for a run manifest."""

        return {
            "resource_kind": "local-directory",
            "resource_path": str(self.directory),
            "config_filename": self.config_path.name,
            "config_sha256": self.config_sha256,
            "config_size_bytes": self.config_size_bytes,
            "params_filename": self.params_path.name,
            "params_sha256": self.params_sha256,
            "params_size_bytes": self.params_size_bytes,
            "base_model_name_or_path": self.config["base_model_name_or_path"],
            "base_model_revision": self.config.get("base_model_revision"),
            "d_model": self.config["d_model"],
            "num_hidden_layers": self.config["num_hidden_layers"],
            "lens_type": self.config.get("lens_type"),
        }


def inspect_local_tuned_lens_artifact(
    directory: str | Path,
) -> LocalTunedLensArtifact:
    """Validate and hash an explicit local tuned-lens directory.

    Performing these checks before importing tuned-lens prevents a misspelled
    local path from being interpreted as a remote resource identifier.
    """

    raw_path = Path(directory).expanduser()
    if not raw_path.is_dir():
        raise TunedLensArtifactError(
            f"tuned-lens resource must be an existing local directory: {raw_path}"
        )
    artifact_dir = raw_path.resolve()
    config_path = artifact_dir / CONFIG_FILENAME
    params_path = artifact_dir / PARAMS_FILENAME
    for path, label in (
        (config_path, "configuration"),
        (params_path, "parameter checkpoint"),
    ):
        if not path.is_file():
            raise TunedLensArtifactError(
                f"tuned-lens {label} is missing: {path}"
            )

    try:
        with config_path.open("r", encoding="utf8") as input_file:
            config = json.load(input_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TunedLensArtifactError(
            f"unable to read tuned-lens configuration: {config_path}"
        ) from error
    if not isinstance(config, dict):
        raise TunedLensArtifactError("tuned-lens configuration must be a JSON object")

    base_name = config.get("base_model_name_or_path")
    if not isinstance(base_name, str) or not base_name.strip():
        raise TunedLensArtifactError(
            "tuned-lens config base_model_name_or_path must be a nonempty string"
        )
    _required_positive_int(config.get("d_model"), "d_model")
    _required_positive_int(
        config.get("num_hidden_layers"), "num_hidden_layers"
    )
    lens_type = config.get("lens_type")
    if lens_type not in (None, "linear_tuned_lens"):
        raise TunedLensArtifactError(
            f"unsupported tuned-lens config lens_type: {lens_type!r}"
        )

    return LocalTunedLensArtifact(
        directory=artifact_dir,
        config_path=config_path,
        params_path=params_path,
        config=MappingProxyType(dict(config)),
        config_sha256=sha256_file(config_path),
        params_sha256=sha256_file(params_path),
        config_size_bytes=config_path.stat().st_size,
        params_size_bytes=params_path.stat().st_size,
    )


def _model_config(model: Any) -> Any:
    config = getattr(model, "config", None)
    if config is None:
        raise TunedLensCompatibilityError("base model has no configuration")
    return config


def _model_layer_count(model: Any) -> int:
    config = _model_config(model)
    for attribute in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise TunedLensCompatibilityError(
        "unable to determine the base model's transformer-layer count"
    )


def _model_hidden_size(model: Any) -> int | None:
    config = _model_config(model)
    for attribute in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _model_name(model: Any) -> str | None:
    config = _model_config(model)
    for attribute in ("name_or_path", "_name_or_path"):
        value = getattr(config, attribute, None)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _model_revision(model: Any) -> str | None:
    config = _model_config(model)
    for attribute in ("_commit_hash", "revision"):
        value = getattr(config, attribute, None)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _is_usable_hub_name(value: str | None) -> bool:
    """Return whether a model name is usable for artifact-name validation."""

    if value is None:
        return False
    if value.startswith((".", "~", "/", "\\")):
        return False
    if len(value) >= 2 and value[1] == ":":  # Windows drive path.
        return False
    return not Path(value).exists()


def validate_artifact_for_model(
    artifact: LocalTunedLensArtifact,
    model: Any,
    *,
    expected_base_model_name: str | None = None,
) -> dict[str, Any]:
    """Validate artifact dimensions and base identity wherever observable."""

    model_layers = _model_layer_count(model)
    lens_layers = int(artifact.config["num_hidden_layers"])
    if lens_layers != model_layers:
        raise TunedLensCompatibilityError(
            "tuned-lens layer-count mismatch: "
            f"artifact has {lens_layers}, base model has {model_layers}"
        )

    model_hidden_size = _model_hidden_size(model)
    lens_hidden_size = int(artifact.config["d_model"])
    if model_hidden_size is not None and lens_hidden_size != model_hidden_size:
        raise TunedLensCompatibilityError(
            "tuned-lens hidden-size mismatch: "
            f"artifact has {lens_hidden_size}, base model has {model_hidden_size}"
        )

    lens_base_name = str(artifact.config["base_model_name_or_path"])
    model_name = _model_name(model)
    if expected_base_model_name is not None:
        if not isinstance(expected_base_model_name, str) or not (
            expected_base_model_name.strip()
        ):
            raise ValueError("expected_base_model_name must be a nonempty string")
        expected_name = expected_base_model_name
        validation_source = "explicit"
    elif _is_usable_hub_name(model_name):
        expected_name = model_name
        validation_source = "model-config"
    else:
        expected_name = None
        validation_source = "unavailable"

    if expected_name is not None and lens_base_name != expected_name:
        raise TunedLensCompatibilityError(
            "tuned-lens base-model mismatch: "
            f"artifact names {lens_base_name!r}, expected {expected_name!r}"
        )
    if (
        expected_base_model_name is not None
        and _is_usable_hub_name(model_name)
        and model_name != expected_base_model_name
    ):
        raise TunedLensCompatibilityError(
            "loaded base-model mismatch: "
            f"model config names {model_name!r}, expected "
            f"{expected_base_model_name!r}"
        )

    lens_revision = artifact.config.get("base_model_revision")
    model_revision = _model_revision(model)
    revision_validated = bool(lens_revision and model_revision)
    if revision_validated and lens_revision != model_revision:
        raise TunedLensCompatibilityError(
            "tuned-lens base-revision mismatch: "
            f"artifact names {lens_revision!r}, loaded model is "
            f"{model_revision!r}"
        )

    return {
        "model_layers": model_layers,
        "model_hidden_size": model_hidden_size,
        "model_name_or_path": model_name,
        "model_revision": model_revision,
        "base_model_name_validated": expected_name is not None,
        "base_model_name_validation_source": validation_source,
        "base_model_revision_validated": revision_validated,
    }


def _import_tuned_lens_class() -> type:
    try:
        module = import_module("tuned_lens")
        return module.TunedLens
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "tuned-lens is required for tuned decoding; install "
            "tuned-lens==0.2.0 in the scoring environment"
        ) from error


def _model_device(model: Any) -> Any | None:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return None


class KuribayashiTunedLensDecoder:
    """Decode model hidden states using Kuribayashi et al.'s convention."""

    def __init__(
        self,
        lens: Any,
        artifact: LocalTunedLensArtifact,
        validation: Mapping[str, Any],
        *,
        package_version: str,
    ) -> None:
        self.lens = lens
        self.artifact = artifact
        self.validation = MappingProxyType(dict(validation))
        self.package_version = package_version
        self.final_layer = int(artifact.config["num_hidden_layers"])
        self.hidden_size = int(artifact.config["d_model"])

    @classmethod
    def from_local_artifacts(
        cls,
        model: Any,
        directory: str | Path,
        *,
        expected_base_model_name: str | None = None,
        device: Any | None = None,
        tuned_lens_class: type | None = None,
        package_version: str | None = None,
    ) -> "KuribayashiTunedLensDecoder":
        """Load a compatible lens from an explicit local directory."""

        artifact = inspect_local_tuned_lens_artifact(directory)
        validation = validate_artifact_for_model(
            artifact,
            model,
            expected_base_model_name=expected_base_model_name,
        )
        lens_class = tuned_lens_class or _import_tuned_lens_class()
        target_device = device if device is not None else _model_device(model)
        load_kwargs = {}
        if target_device is not None:
            load_kwargs["map_location"] = target_device
        lens = lens_class.from_model_and_pretrained(
            model,
            str(artifact.directory),
            **load_kwargs,
        )
        try:
            translator_count = len(lens)
        except (AttributeError, TypeError) as error:
            raise TunedLensCompatibilityError(
                "loaded tuned lens does not expose its translator count"
            ) from error
        if translator_count != validation["model_layers"]:
            raise TunedLensCompatibilityError(
                "loaded tuned-lens translator-count mismatch: "
                f"lens has {translator_count}, base model has "
                f"{validation['model_layers']}"
            )
        if target_device is not None:
            lens = lens.to(target_device)
        if hasattr(lens, "eval"):
            lens.eval()
        return cls(
            lens,
            artifact,
            validation,
            package_version=(
                tuned_lens_package_version()
                if package_version is None else package_version
            ),
        )

    def provenance(self) -> dict[str, Any]:
        """Return decoder, artifact, and compatibility identity for manifests."""

        return {
            "decoder": DECODER_NAME,
            "tuned_lens_package_version": self.package_version,
            "layer_indexing": "hidden_states[layer_id], idx=layer_id",
            "embedding_hidden_state_supported": True,
            "final_layer_policy": "ordinary-logits-bypass",
            "artifact": self.artifact.provenance(),
            "validation": dict(self.validation),
        }

    def layer_logits(
        self,
        layer_id: int,
        hidden_states: Any,
        ordinary_logits: Any,
        *,
        position_offset: int = 0,
    ) -> Any:
        """Decode one block output, bypassing the lens at the final layer.

        ``layer_id`` is the hidden-state tuple index, matching the official
        Kuribayashi implementation. State zero is the embedding stream,
        transformer block outputs use IDs 1 through D, and state D bypasses
        the lens.
        """

        if isinstance(layer_id, bool) or not isinstance(layer_id, int):
            raise ValueError("layer_id must be an integer")
        if layer_id < 0 or layer_id > self.final_layer:
            raise ValueError(
                f"layer_id must be between 0 and {self.final_layer}"
            )
        if isinstance(position_offset, bool) or not isinstance(position_offset, int):
            raise ValueError("position_offset must be a nonnegative integer")
        if position_offset < 0:
            raise ValueError("position_offset must be a nonnegative integer")
        if len(hidden_states) != self.final_layer + 1:
            raise TunedLensCompatibilityError(
                "causal LM returned an unexpected number of hidden states: "
                f"got {len(hidden_states)}, expected {self.final_layer + 1}"
            )

        if layer_id == self.final_layer:
            return ordinary_logits[:, position_offset:, :]

        hidden = hidden_states[layer_id][:, position_offset:, :]
        shape = getattr(hidden, "shape", None)
        if shape is not None and len(shape) > 0 and shape[-1] != self.hidden_size:
            raise TunedLensCompatibilityError(
                "hidden-state width does not match tuned lens: "
                f"got {shape[-1]}, expected {self.hidden_size}"
            )
        decoded = self.lens(hidden, idx=layer_id)
        target_device = getattr(ordinary_logits, "device", None)
        decoded_device = getattr(decoded, "device", None)
        if (
            target_device is not None
            and decoded_device != target_device
            and hasattr(decoded, "to")
        ):
            decoded = decoded.to(target_device)
        return decoded


def load_local_tuned_lens_decoder(
    model: Any,
    directory: str | Path,
    **kwargs: Any,
) -> KuribayashiTunedLensDecoder:
    """Convenience wrapper for :meth:`from_local_artifacts`."""

    return KuribayashiTunedLensDecoder.from_local_artifacts(
        model, directory, **kwargs
    )
