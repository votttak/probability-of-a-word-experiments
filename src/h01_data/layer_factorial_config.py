#!/usr/bin/env python3

"""Load and query the central internal-layer factorial configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from string import Formatter
from typing import Any, Mapping

try:
    from .layer_factorial_models import model_aliases
except ImportError:  # Support direct execution from src/h01_data.
    from layer_factorial_models import model_aliases


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs/layer_factorial.json"

SCHEMA_VERSION = 1
RESPONSES = ("time", "paper_time")
CONTEXTS = ("passage", "sentence")
LENS_METHODS = ("logit-lens", "tuned-lens")
SCORE_KINDS = ("corrected", "buggy")
SENTENCE_FIRST_TOKEN_POLICIES = ("bos", "bow")
ANALYSIS_MODES = ("paper-exact", "project-bridge")
LAG_BOUNDARIES = ("text", "sentence")
LAG_PADDING_MODES = ("missing", "global-mean")
PATH_KEYS = (
    "text",
    "sentence_manifest",
    "paper_rt",
    "precomputed_frequency",
    "joint_template",
    "local_checkpoint_root",
    "local_results_root",
)

ROOT_KEYS = {
    "schema_version",
    "models",
    "switches",
    "extraction",
    "analysis",
    "runtime",
    "paths",
    "report_note",
}
SWITCH_KEYS = {
    "responses",
    "contexts",
    "lens_methods",
    "score_kinds",
    "include_embedding_layer",
}
EXTRACTION_KEYS = {
    "sentence_first_token_policy",
    "final_layer_tolerance",
}
ANALYSIS_KEYS = {
    "mode",
    "lag_boundary",
    "lag_padding",
    "early_layer_threshold",
    "transformer_only_sensitivity",
}
RUNTIME_KEYS = {"jobs", "threads_per_job", "pivot_sentences_per_text"}


class LayerFactorialConfigError(ValueError):
    """Raised when a layer-factorial configuration is invalid."""


@dataclass(frozen=True)
class LayerFactorialSwitches:
    responses: tuple[str, ...]
    contexts: tuple[str, ...]
    lens_methods: tuple[str, ...]
    score_kinds: tuple[str, ...]
    include_embedding_layer: bool


@dataclass(frozen=True)
class LayerFactorialExtraction:
    sentence_first_token_policy: str
    final_layer_tolerance: float


@dataclass(frozen=True)
class LayerFactorialAnalysis:
    mode: str
    lag_boundary: str
    lag_padding: str
    early_layer_threshold: float
    transformer_only_sensitivity: bool


@dataclass(frozen=True)
class LayerFactorialRuntime:
    jobs: int
    threads_per_job: int
    pivot_sentences_per_text: int


@dataclass(frozen=True)
class LayerFactorialPaths:
    text: str
    sentence_manifest: str
    paper_rt: str
    precomputed_frequency: str
    joint_template: str
    local_checkpoint_root: str
    local_results_root: str


@dataclass(frozen=True)
class LayerFactorialConfig:
    schema_version: int
    models: tuple[str, ...]
    switches: LayerFactorialSwitches
    extraction: LayerFactorialExtraction
    analysis: LayerFactorialAnalysis
    runtime: LayerFactorialRuntime
    paths: LayerFactorialPaths
    report_note: str
    source_path: Path
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return only the portable, persisted configuration payload."""

        return {
            "schema_version": self.schema_version,
            "models": list(self.models),
            "switches": {
                "responses": list(self.switches.responses),
                "contexts": list(self.switches.contexts),
                "lens_methods": list(self.switches.lens_methods),
                "score_kinds": list(self.switches.score_kinds),
                "include_embedding_layer": (
                    self.switches.include_embedding_layer
                ),
            },
            "extraction": {
                "sentence_first_token_policy": (
                    self.extraction.sentence_first_token_policy
                ),
                "final_layer_tolerance": (
                    self.extraction.final_layer_tolerance
                ),
            },
            "analysis": {
                "mode": self.analysis.mode,
                "lag_boundary": self.analysis.lag_boundary,
                "lag_padding": self.analysis.lag_padding,
                "early_layer_threshold": (
                    self.analysis.early_layer_threshold
                ),
                "transformer_only_sensitivity": (
                    self.analysis.transformer_only_sensitivity
                ),
            },
            "runtime": {
                "jobs": self.runtime.jobs,
                "threads_per_job": self.runtime.threads_per_job,
                "pivot_sentences_per_text": (
                    self.runtime.pivot_sentences_per_text
                ),
            },
            "paths": {
                key: getattr(self.paths, key)
                for key in PATH_KEYS
            },
            "report_note": self.report_note,
        }


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LayerFactorialConfigError(
                f"duplicate JSON object key: {key!r}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise LayerFactorialConfigError(
        f"non-finite JSON number is not allowed: {value}"
    )


def _require_object(
    value: Any, expected_keys: set[str], label: str
) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise LayerFactorialConfigError(f"{label} must be an object")
    observed = set(value)
    missing = expected_keys - observed
    unknown = observed - expected_keys
    if missing:
        raise LayerFactorialConfigError(
            f"{label} is missing keys: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise LayerFactorialConfigError(
            f"{label} has unknown keys: {', '.join(sorted(unknown))}"
        )
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise LayerFactorialConfigError(f"{label} must be a boolean")
    return value


def _require_int(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise LayerFactorialConfigError(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        interval = (
            f"between {minimum} and {maximum}"
            if maximum is not None
            else f"at least {minimum}"
        )
        raise LayerFactorialConfigError(f"{label} must be {interval}")
    return value


def _require_number(
    value: Any,
    label: str,
    *,
    minimum_exclusive: float,
    maximum_inclusive: float | None = None,
) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise LayerFactorialConfigError(
            f"{label} must be a finite number"
        )
    numeric = float(value)
    if numeric <= minimum_exclusive:
        raise LayerFactorialConfigError(
            f"{label} must be greater than {minimum_exclusive}"
        )
    if maximum_inclusive is not None and numeric > maximum_inclusive:
        raise LayerFactorialConfigError(
            f"{label} must be at most {maximum_inclusive}"
        )
    return numeric


def _require_string(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
    ):
        raise LayerFactorialConfigError(
            f"{label} must be a nonempty string without surrounding whitespace"
        )
    return value


def _require_report_note(value: Any) -> str:
    if type(value) is not str or value != value.strip():
        raise LayerFactorialConfigError(
            "report_note must be a string without surrounding whitespace"
        )
    return value


def _require_choice(
    value: Any, allowed: tuple[str, ...], label: str
) -> str:
    observed = _require_string(value, label)
    if observed not in allowed:
        raise LayerFactorialConfigError(
            f"{label} must be one of: {', '.join(allowed)}"
        )
    return observed


def _require_choice_list(
    value: Any, allowed: tuple[str, ...], label: str
) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise LayerFactorialConfigError(
            f"{label} must be a nonempty array"
        )
    result = tuple(
        _require_choice(item, allowed, f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise LayerFactorialConfigError(
            f"{label} must not contain duplicate values"
        )
    return result


def _require_relative_path(
    value: Any, label: str, *, model_template: bool = False
) -> str:
    path_value = _require_string(value, label)
    if chr(0) in path_value:
        raise LayerFactorialConfigError(
            f"{label} must be a portable repository-relative path"
        )
    if "\\" in path_value or "\\x00" in path_value:
        raise LayerFactorialConfigError(
            f"{label} must be a portable repository-relative path"
        )

    try:
        parsed = list(Formatter().parse(path_value))
    except ValueError as error:
        raise LayerFactorialConfigError(
            f"{label} contains an invalid format placeholder"
        ) from error
    fields = []
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if format_spec or conversion:
            raise LayerFactorialConfigError(
                f"{label} placeholders cannot use conversions or formats"
            )
        fields.append(field_name)
    expected_fields = ["model"] if model_template else []
    if fields != expected_fields:
        if model_template:
            requirement = "exactly one {model} placeholder"
        else:
            requirement = "no placeholders"
        raise LayerFactorialConfigError(
            f"{label} must contain {requirement}"
        )

    rendered = path_value.format(model="MODEL")
    posix_path = PurePosixPath(rendered)
    windows_path = PureWindowsPath(rendered)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or rendered.startswith("~")
        or ".." in posix_path.parts
        or posix_path == PurePosixPath(".")
    ):
        raise LayerFactorialConfigError(
            f"{label} must remain inside the repository"
        )
    return path_value


def _parse_config(
    payload: Any, source_path: Path, source_sha256: str
) -> LayerFactorialConfig:
    root = _require_object(payload, ROOT_KEYS, "configuration")
    schema_version = _require_int(
        root["schema_version"], "schema_version", minimum=1
    )
    if schema_version != SCHEMA_VERSION:
        raise LayerFactorialConfigError(
            "unsupported schema_version "
            f"{schema_version}; expected {SCHEMA_VERSION}"
        )

    models = _require_choice_list(
        root["models"], model_aliases(), "models"
    )

    switch_values = _require_object(
        root["switches"], SWITCH_KEYS, "switches"
    )
    switches = LayerFactorialSwitches(
        responses=_require_choice_list(
            switch_values["responses"], RESPONSES, "switches.responses"
        ),
        contexts=_require_choice_list(
            switch_values["contexts"], CONTEXTS, "switches.contexts"
        ),
        lens_methods=_require_choice_list(
            switch_values["lens_methods"],
            LENS_METHODS,
            "switches.lens_methods",
        ),
        score_kinds=_require_choice_list(
            switch_values["score_kinds"],
            SCORE_KINDS,
            "switches.score_kinds",
        ),
        include_embedding_layer=_require_bool(
            switch_values["include_embedding_layer"],
            "switches.include_embedding_layer",
        ),
    )

    extraction_values = _require_object(
        root["extraction"], EXTRACTION_KEYS, "extraction"
    )
    extraction = LayerFactorialExtraction(
        sentence_first_token_policy=_require_choice(
            extraction_values["sentence_first_token_policy"],
            SENTENCE_FIRST_TOKEN_POLICIES,
            "extraction.sentence_first_token_policy",
        ),
        final_layer_tolerance=_require_number(
            extraction_values["final_layer_tolerance"],
            "extraction.final_layer_tolerance",
            minimum_exclusive=0.0,
        ),
    )

    analysis_values = _require_object(
        root["analysis"], ANALYSIS_KEYS, "analysis"
    )
    analysis = LayerFactorialAnalysis(
        mode=_require_choice(
            analysis_values["mode"], ANALYSIS_MODES, "analysis.mode"
        ),
        lag_boundary=_require_choice(
            analysis_values["lag_boundary"],
            LAG_BOUNDARIES,
            "analysis.lag_boundary",
        ),
        lag_padding=_require_choice(
            analysis_values["lag_padding"],
            LAG_PADDING_MODES,
            "analysis.lag_padding",
        ),
        early_layer_threshold=_require_number(
            analysis_values["early_layer_threshold"],
            "analysis.early_layer_threshold",
            minimum_exclusive=0.0,
            maximum_inclusive=1.0,
        ),
        transformer_only_sensitivity=_require_bool(
            analysis_values["transformer_only_sensitivity"],
            "analysis.transformer_only_sensitivity",
        ),
    )
    if (
        analysis.mode == "paper-exact"
        and (
            analysis.lag_boundary != "sentence"
            or analysis.lag_padding != "global-mean"
        )
    ):
        raise LayerFactorialConfigError(
            "analysis.mode paper-exact requires lag_boundary=sentence "
            "and lag_padding=global-mean"
        )

    runtime_values = _require_object(
        root["runtime"], RUNTIME_KEYS, "runtime"
    )
    runtime = LayerFactorialRuntime(
        jobs=_require_int(
            runtime_values["jobs"], "runtime.jobs", minimum=1, maximum=4
        ),
        threads_per_job=_require_int(
            runtime_values["threads_per_job"],
            "runtime.threads_per_job",
            minimum=1,
        ),
        pivot_sentences_per_text=_require_int(
            runtime_values["pivot_sentences_per_text"],
            "runtime.pivot_sentences_per_text",
            minimum=1,
        ),
    )

    path_values = _require_object(root["paths"], set(PATH_KEYS), "paths")
    paths = LayerFactorialPaths(
        **{
            key: _require_relative_path(
                path_values[key],
                f"paths.{key}",
                model_template=(key == "joint_template"),
            )
            for key in PATH_KEYS
        }
    )

    return LayerFactorialConfig(
        schema_version=schema_version,
        models=models,
        switches=switches,
        extraction=extraction,
        analysis=analysis,
        runtime=runtime,
        paths=paths,
        report_note=_require_report_note(root["report_note"]),
        source_path=source_path,
        source_sha256=source_sha256,
    )


def load_layer_factorial_config(
    path: str | Path | None = None,
) -> LayerFactorialConfig:
    """Load, fully validate, and freeze one JSON configuration."""

    source_path = Path(path or DEFAULT_CONFIG_PATH).expanduser()
    if not source_path.is_absolute():
        source_path = REPOSITORY_ROOT / source_path
    source_path = source_path.resolve()
    try:
        encoded = source_path.read_bytes()
        text = encoded.decode("utf8")
        payload = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except LayerFactorialConfigError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LayerFactorialConfigError(
            f"unable to read layer-factorial config: {source_path}"
        ) from error
    return _parse_config(
        payload, source_path, hashlib.sha256(encoded).hexdigest()
    )


def get_config_value(
    config: LayerFactorialConfig, dotted_key: str
) -> Any:
    """Return a persisted config value selected by a dotted key."""

    if not dotted_key:
        raise LayerFactorialConfigError("--get requires a nonempty key")
    value: Any = config.to_dict()
    traversed = []
    for part in dotted_key.split("."):
        traversed.append(part)
        if type(value) is not dict or part not in value:
            raise LayerFactorialConfigError(
                "unknown configuration key: " + ".".join(traversed)
            )
        value = value[part]
    return value


def resolve_config_path(
    config: LayerFactorialConfig,
    key: str,
    *,
    model: str | None = None,
    repository_root: str | Path | None = None,
) -> Path:
    """Resolve one configured repository-relative path."""

    normalized_key = key.removeprefix("paths.")
    if normalized_key not in PATH_KEYS:
        raise LayerFactorialConfigError(
            f"unknown configuration path key: {key}"
        )
    value = getattr(config.paths, normalized_key)
    if normalized_key == "joint_template":
        if model is None:
            raise LayerFactorialConfigError(
                "paths.joint_template requires --model"
            )
        if model not in config.models:
            raise LayerFactorialConfigError(
                f"model {model!r} is not enabled in this configuration"
            )
        value = value.format(model=model)

    root = Path(repository_root or REPOSITORY_ROOT).expanduser().resolve()
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise LayerFactorialConfigError(
            f"configured path escapes the repository: {key}"
        ) from error
    return resolved


def _print_query_value(value: Any) -> None:
    if type(value) is list:
        print("\n".join(str(item) for item in value))
    elif type(value) is bool:
        print("true" if value else "false")
    elif type(value) is dict:
        print(json.dumps(value, sort_keys=True))
    else:
        print(value)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or query the layer-factorial configuration"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--get", metavar="DOTTED_KEY")
    selection.add_argument("--list-models", action="store_true")
    selection.add_argument("--resolve-path", metavar="KEY")
    parser.add_argument("--model", choices=model_aliases())
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        config = load_layer_factorial_config(args.config)
        if args.list_models:
            if args.model:
                raise LayerFactorialConfigError(
                    "--model is only valid with --resolve-path"
                )
            print("\n".join(config.models))
        elif args.resolve_path:
            print(resolve_config_path(
                config, args.resolve_path, model=args.model
            ))
        else:
            if args.model:
                raise LayerFactorialConfigError(
                    "--model is only valid with --resolve-path"
                )
            _print_query_value(get_config_value(config, args.get))
    except LayerFactorialConfigError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
