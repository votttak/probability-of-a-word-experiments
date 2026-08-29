#!/usr/bin/env python3

"""Run the isolated 2 x 2 x 2 internal-layer factorial experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from h01_data.layer_factorial_models import (  # noqa: E402
    get_model_spec as get_factorial_model_spec,
    model_aliases as factorial_model_aliases,
)
from h01_data.layer_factorial_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_layer_factorial_config,
    resolve_config_path,
)

EXTRACTOR = REPOSITORY_ROOT / "src/h01_data/get_internal_layer_surprisals.py"
MERGER = REPOSITORY_ROOT / "src/h01_data/build_layer_factorial_dataset.py"
EVALUATOR = (
    REPOSITORY_ROOT
    / "src/h02_rt_model/rt_vs_internal_layer_factorial_kuribayashi.R"
)
ANALYZER = (
    REPOSITORY_ROOT
    / "src/h03_paper/analyze_layer_factorial_results.py"
)
VALIDATOR = REPOSITORY_ROOT / "scripts/validate_layer_factorial_outputs.py"
CONTEXTS = ("passage", "sentence")
LENSES = ("logit-lens", "tuned-lens")
SCORE_KINDS = ("corrected", "buggy")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_repo_path(value):
    """Resolve CLI paths consistently from the repository root."""

    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def resolve_path_arguments(args):
    for attribute in (
        "text_fname",
        "sentence_manifest_fname",
        "joint_data_fname",
        "paper_rt_fname",
        "precomputed_frequency_fname",
        "tuned_lens_path",
        "tuned_lens_pythonpath",
        "wordfreq_pythonpath",
        "checkpoint_root",
        "results_root",
    ):
        value = getattr(args, attribute)
        if value is not None:
            setattr(args, attribute, resolve_repo_path(value))
    return args


def read_json_object(path, label):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def read_expected_word_rows(text_path):
    """Return the exact keyed whitespace words consumed by the extractor."""

    rows = []
    try:
        with Path(text_path).open("r", encoding="utf8") as input_file:
            for text_id, line in enumerate(input_file):
                rows.extend(
                    (text_id, word_id, word)
                    for word_id, word in enumerate(line.split())
                )
    except (OSError, UnicodeError) as error:
        raise ValueError(f"unable to read input text: {text_path}") from error
    if not rows:
        raise ValueError("--text-fname must contain at least one word")
    return rows


def _require_equal(label, observed, expected):
    if observed != expected:
        raise ValueError(
            "extraction provenance mismatch for "
            f"{label}: observed {observed!r}, expected {expected!r}"
        )


def validate_extraction_matches_run(
    args,
    extraction_paths,
    validation_path,
    model_spec,
    expected_word_rows,
):
    """Prevent stale extraction cells from being relabeled by this run."""

    validation = read_json_object(
        validation_path, "extraction validation manifest"
    )
    _require_equal("validated", validation.get("validated"), True)
    _require_equal("model", validation.get("model"), args.model)
    _require_equal(
        "model revision",
        validation.get("model_revision_effective"),
        model_spec.base_model_revision,
    )
    expected = validation.get("expected")
    if not isinstance(expected, dict):
        raise ValueError(
            "extraction validation manifest lacks expected dimensions"
        )
    _require_equal("row count", expected.get("rows"), len(expected_word_rows))
    _require_equal(
        "final layer", expected.get("final_layer"), model_spec.final_layer
    )
    _require_equal(
        "minimum layer",
        expected.get("min_layer"),
        0 if args.include_embedding_layer else 1,
    )
    _require_equal(
        "contexts",
        set(expected.get("contexts", [])),
        set(args.contexts),
    )
    _require_equal(
        "lens methods",
        set(expected.get("lens_methods", [])),
        set(args.lens_methods),
    )
    extraction_score_kinds = ["corrected"]
    if "buggy" in args.score_kinds:
        extraction_score_kinds.append("buggy")
    _require_equal(
        "extraction score kinds",
        expected.get("score_kinds"),
        extraction_score_kinds,
    )
    expected_sentence_hash = (
        sha256_file(args.sentence_manifest_fname)
        if "sentence" in args.contexts
        else None
    )
    _require_equal(
        "sentence manifest hash",
        validation.get("sentence_manifest_sha256"),
        expected_sentence_hash,
    )

    lens_identity = validation.get("tuned_lens_identity")
    if "tuned-lens" in args.lens_methods:
        if not isinstance(lens_identity, dict):
            raise ValueError(
                "extraction validation manifest lacks tuned-lens identity"
            )
        artifact = lens_identity.get("artifact", lens_identity)
        if not isinstance(artifact, dict):
            raise ValueError(
                "extraction validation tuned-lens artifact must be an object"
            )
        _require_equal(
            "tuned-lens config hash",
            artifact.get("config_sha256"),
            sha256_file(Path(args.tuned_lens_path) / "config.json"),
        )
        _require_equal(
            "tuned-lens parameter hash",
            artifact.get("params_sha256"),
            sha256_file(Path(args.tuned_lens_path) / "params.pt"),
        )
        _require_equal(
            "tuned-lens base model",
            artifact.get("base_model_name_or_path"),
            model_spec.hf_name,
        )
        _require_equal(
            "tuned-lens base revision",
            artifact.get("base_model_revision"),
            model_spec.lens_base_model_revision,
        )
    elif lens_identity is not None:
        raise ValueError(
            "extraction validation unexpectedly contains a tuned-lens identity"
        )

    for (context, lens), extraction_path in extraction_paths.items():
        anchor = read_json_object(
            f"{extraction_path}.anchor.json",
            f"{context}/{lens} extraction anchor",
        )
        experiment = anchor.get("experiment")
        if not isinstance(experiment, dict):
            raise ValueError(
                f"{context}/{lens} extraction anchor lacks experiment metadata"
            )
        _require_equal(
            f"{context}/{lens} model",
            experiment.get("model"),
            args.model,
        )
        _require_equal(
            f"{context}/{lens} revision",
            experiment.get("model_revision_effective"),
            model_spec.base_model_revision,
        )
        _require_equal(
            f"{context}/{lens} Hugging Face model",
            experiment.get("hf_model_name_effective"),
            model_spec.hf_name,
        )
        if context == "sentence":
            _require_equal(
                f"{context}/{lens} first-token policy",
                experiment.get("sentence_first_token_policy"),
                args.sentence_first_token_policy,
            )

    reference_path = next(iter(extraction_paths.values()))
    try:
        with Path(reference_path).open(
            "r", encoding="utf8", newline=""
        ) as input_file:
            reader = csv.DictReader(input_file, delimiter="\t")
            observed_word_rows = [
                (int(row["text_id"]), int(row["word_id"]), row["word"])
                for row in reader
            ]
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"unable to verify extraction words against {args.text_fname}"
        ) from error
    if observed_word_rows != expected_word_rows:
        mismatch = next(
            (
                index
                for index, pair in enumerate(
                    zip(observed_word_rows, expected_word_rows)
                )
                if pair[0] != pair[1]
            ),
            min(len(observed_word_rows), len(expected_word_rows)),
        )
        raise ValueError(
            "extraction provenance mismatch for input text at row "
            f"{mismatch}"
        )


def write_json_atomic(payload, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output_path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def prepend_pythonpath(environment, paths):
    values = [str(Path(path).resolve()) for path in paths if path]
    existing = environment.get("PYTHONPATH")
    if existing:
        values.append(existing)
    if values:
        environment["PYTHONPATH"] = os.pathsep.join(values)
    return environment


def run_command(command, environment):
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(
        [str(part) for part in command],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )


def read_lens_config(lens_path, model_spec):
    config_path = Path(lens_path) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"unable to read tuned-lens config: {config_path}"
        ) from error
    base_model_name = config.get("base_model_name_or_path")
    if base_model_name != model_spec.hf_name:
        raise ValueError(
            "tuned-lens config base model mismatch: "
            f"observed {base_model_name!r}, expected {model_spec.hf_name!r}"
        )
    revision = config.get("base_model_revision")
    if revision != model_spec.lens_base_model_revision:
        raise ValueError(
            "tuned-lens config base revision mismatch: "
            f"observed {revision!r}, expected "
            f"{model_spec.lens_base_model_revision!r}"
        )
    final_layer = config.get("num_hidden_layers")
    if (
        isinstance(final_layer, bool)
        or not isinstance(final_layer, int)
        or final_layer < 1
    ):
        raise ValueError(
            "tuned-lens config must contain a positive integer "
            "num_hidden_layers"
        )
    if final_layer != model_spec.final_layer:
        raise ValueError(
            "tuned-lens config layer-count mismatch: "
            f"observed {final_layer}, expected {model_spec.final_layer}"
        )
    return config


def cell_name(context, lens):
    return f"context_{context}-lens_{lens}"


def extraction_command(args, context, lens, model_spec, cell_dir):
    command = [
        args.python,
        EXTRACTOR,
        "--input-fname",
        args.text_fname,
        "--output-fname",
        cell_dir / "internal-layer.tsv",
        "--model",
        args.model,
        "--hf-model-name",
        model_spec.hf_name,
        "--model-revision",
        model_spec.base_model_revision,
        "--context-unit",
        context,
        "--lens-method",
        lens,
        "--passage-checkpoint-dir",
        cell_dir / "passage-checkpoints",
    ]
    if "buggy" in args.score_kinds:
        command.append("--return-buggy-surprisals")
    if args.include_embedding_layer:
        command.append("--include-embedding-layer")
    if context == "sentence":
        command.extend([
            "--sentence-map-fname",
            args.sentence_manifest_fname,
            "--sentence-first-token-policy",
            args.sentence_first_token_policy,
        ])
    if lens == "tuned-lens":
        command.extend(["--tuned-lens-path", args.tuned_lens_path])
    return command


def _explicit_cli_options(argv):
    return sorted({
        token.split("=", 1)[0]
        for token in argv
        if token.startswith("--")
    })


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH)
    )
    config_args, _ = config_parser.parse_known_args(argv)
    config = load_layer_factorial_config(config_args.config)

    parser = argparse.ArgumentParser(
        description=(
            "Run a configured context by score by decoder internal-layer "
            "evaluation in an isolated output root"
        )
    )
    parser.add_argument("--config", default=str(config.source_path))
    parser.add_argument(
        "--model", choices=factorial_model_aliases(), default=config.models[0]
    )
    parser.add_argument("--text-fname")
    parser.add_argument("--sentence-manifest-fname")
    parser.add_argument("--joint-data-fname")
    parser.add_argument("--paper-rt-fname")
    parser.add_argument(
        "--precomputed-frequency-fname",
        help=(
            "keyed paper frequency controls; avoids a runtime wordfreq "
            "dependency and is recorded in the run manifest"
        ),
    )
    parser.add_argument("--tuned-lens-path")
    parser.add_argument("--tuned-lens-pythonpath")
    parser.add_argument("--wordfreq-pythonpath")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--results-root")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument(
        "--include-embedding-layer",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--contexts", nargs="+", choices=CONTEXTS)
    parser.add_argument("--lens-methods", nargs="+", choices=LENSES)
    parser.add_argument("--score-kinds", nargs="+", choices=SCORE_KINDS)
    parser.add_argument(
        "--sentence-first-token-policy",
        choices=("bos", "bow"),
        default=None,
    )
    parser.add_argument(
        "--analysis-mode",
        choices=("paper-exact", "project-bridge"),
        default=None,
    )
    parser.add_argument(
        "--lag-boundary", choices=("text", "sentence")
    )
    parser.add_argument(
        "--lag-padding", choices=("missing", "global-mean")
    )
    parser.add_argument(
        "--response-columns",
        nargs="+",
        choices=("time", "paper_time"),
    )
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--threads-per-job", type=int)
    parser.add_argument(
        "--final-layer-tolerance", type=float
    )
    parser.add_argument("--early-layer-threshold", type=float)
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument(
        "--report-note",
        default=None,
    )
    args = parser.parse_args(argv)
    if args.model not in config.models:
        parser.error(
            f"model {args.model!r} is not enabled by {config.source_path}"
        )

    switches = config.switches
    extraction = config.extraction
    analysis = config.analysis
    runtime = config.runtime
    args.contexts = list(args.contexts or switches.contexts)
    args.lens_methods = list(
        args.lens_methods or switches.lens_methods
    )
    args.score_kinds = list(
        args.score_kinds or switches.score_kinds
    )
    args.response_columns = list(
        args.response_columns or switches.responses
    )
    if args.include_embedding_layer is None:
        args.include_embedding_layer = (
            switches.include_embedding_layer
        )
    args.sentence_first_token_policy = (
        args.sentence_first_token_policy
        or extraction.sentence_first_token_policy
    )
    args.final_layer_tolerance = (
        args.final_layer_tolerance
        if args.final_layer_tolerance is not None
        else extraction.final_layer_tolerance
    )
    args.analysis_mode = args.analysis_mode or analysis.mode
    args.lag_boundary = args.lag_boundary or analysis.lag_boundary
    args.lag_padding = args.lag_padding or analysis.lag_padding
    args.early_layer_threshold = (
        args.early_layer_threshold
        if args.early_layer_threshold is not None
        else analysis.early_layer_threshold
    )
    args.jobs = args.jobs if args.jobs is not None else runtime.jobs
    args.threads_per_job = (
        args.threads_per_job
        if args.threads_per_job is not None
        else runtime.threads_per_job
    )
    if args.report_note is None:
        args.report_note = config.report_note

    configured_paths = {
        "text_fname": resolve_config_path(config, "text"),
        "sentence_manifest_fname": resolve_config_path(
            config, "sentence_manifest"
        ),
        "joint_data_fname": resolve_config_path(
            config, "joint_template", model=args.model
        ),
        "paper_rt_fname": resolve_config_path(config, "paper_rt"),
        "precomputed_frequency_fname": resolve_config_path(
            config, "precomputed_frequency"
        ),
        "checkpoint_root": (
            resolve_config_path(config, "local_checkpoint_root")
            / f"full-{args.model}"
        ),
        "results_root": (
            resolve_config_path(config, "local_results_root")
            / f"full-{args.model}"
        ),
    }
    for attribute, value in configured_paths.items():
        if getattr(args, attribute) is None:
            setattr(args, attribute, value)
    args.layer_factorial_config = config
    args.cli_overrides = _explicit_cli_options(argv)
    return args


def main():
    args = resolve_path_arguments(parse_args())
    for label, values in (
        ("contexts", args.contexts),
        ("lens methods", args.lens_methods),
        ("score kinds", args.score_kinds),
        ("response columns", args.response_columns),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"{label} must not contain duplicates")
    if args.jobs < 1 or args.jobs > 4:
        raise ValueError("--jobs must be between 1 and 4")
    if args.threads_per_job < 1:
        raise ValueError("--threads-per-job must be positive")
    if "paper_time" in args.response_columns and not args.paper_rt_fname:
        raise ValueError(
            "--response-columns paper_time requires --paper-rt-fname"
        )
    if (
        args.analysis_mode == "paper-exact"
        and (
            args.lag_boundary != "sentence"
            or args.lag_padding != "global-mean"
        )
    ):
        raise ValueError(
            "paper-exact analysis requires sentence lag boundaries with "
            "global-mean padding"
        )
    if (
        not 0 < args.early_layer_threshold <= 1
        or not math.isfinite(args.early_layer_threshold)
    ):
        raise ValueError(
            "--early-layer-threshold must be finite and in (0, 1]"
        )
    if "tuned-lens" in args.lens_methods and not args.tuned_lens_path:
        raise ValueError(
            "--lens-methods tuned-lens requires --tuned-lens-path"
        )
    checkpoint_root = Path(args.checkpoint_root)
    results_root = Path(args.results_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    model_spec = get_factorial_model_spec(args.model)
    if "tuned-lens" in args.lens_methods:
        read_lens_config(args.tuned_lens_path, model_spec)
    revision = model_spec.base_model_revision
    final_layer = model_spec.final_layer
    expected_word_rows = read_expected_word_rows(args.text_fname)
    expected_rows = len(expected_word_rows)

    base_environment = os.environ.copy()
    base_environment.update({
        "OMP_NUM_THREADS": str(args.threads_per_job),
        "MKL_NUM_THREADS": str(args.threads_per_job),
        "OPENBLAS_NUM_THREADS": str(args.threads_per_job),
        "TOKENIZERS_PARALLELISM": "false",
    })
    tuned_environment = prepend_pythonpath(
        base_environment.copy(), [args.tuned_lens_pythonpath]
    )
    merge_environment = prepend_pythonpath(
        base_environment.copy(), [args.wordfreq_pythonpath]
    )

    cells = []
    for context in args.contexts:
        for lens in args.lens_methods:
            name = cell_name(context, lens)
            cell_checkpoint = checkpoint_root / "cells" / name
            cell_checkpoint.mkdir(parents=True, exist_ok=True)
            cells.append((context, lens, name, cell_checkpoint))

    if not args.skip_extraction:
        jobs = []
        for context, lens, _, cell_checkpoint in cells:
            command = extraction_command(
                args, context, lens, model_spec, cell_checkpoint
            )
            environment = (
                tuned_environment if lens == "tuned-lens"
                else base_environment
            )
            jobs.append((command, environment))
        if args.jobs == 1:
            for command, environment in jobs:
                run_command(command, environment)
        else:
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                futures = [
                    executor.submit(run_command, command, environment)
                    for command, environment in jobs
                ]
                for future in futures:
                    future.result()

    extraction_paths = {
        (context, lens): cell_checkpoint / "internal-layer.tsv"
        for context, lens, _, cell_checkpoint in cells
    }
    extraction_validation_path = (
        checkpoint_root / "extraction-validation.json"
    )
    extraction_score_kinds = ["corrected"]
    if "buggy" in args.score_kinds:
        extraction_score_kinds.append("buggy")
    validator_command = [
        args.python,
        VALIDATOR,
        "--completion-json-fname",
        extraction_validation_path,
        "--score-kinds",
        *extraction_score_kinds,
        "--expected-rows",
        expected_rows,
        "--expected-final-layer",
        final_layer,
        "--expected-min-layer",
        0 if args.include_embedding_layer else 1,
        "--tolerance",
        args.final_layer_tolerance,
    ]
    for (context, lens), path in extraction_paths.items():
        validator_command.extend(["--cell", context, lens, path])
    run_command(validator_command, base_environment)
    validate_extraction_matches_run(
        args,
        extraction_paths,
        extraction_validation_path,
        model_spec,
        expected_word_rows,
    )

    layer_result_paths = []
    for context, lens, name, cell_checkpoint in cells:
        first_policy = (
            args.sentence_first_token_policy
            if context == "sentence" else "bos"
        )
        merged_path = cell_checkpoint / "joint-data.tsv"
        merge_command = [
            args.python,
            MERGER,
            "--canonical-joint-fname",
            args.joint_data_fname,
            "--layer-fname",
            cell_checkpoint / "internal-layer.tsv",
            "--sentence-manifest-fname",
            args.sentence_manifest_fname,
            "--model",
            args.model,
            "--context-unit",
            context,
            "--lens-method",
            lens,
            "--first-token-policy",
            first_policy,
            "--lag-boundary",
            args.lag_boundary,
            "--lag-padding",
            args.lag_padding,
            "--output-fname",
            merged_path,
        ]
        if args.paper_rt_fname:
            merge_command.extend([
                "--paper-rt-fname", args.paper_rt_fname
            ])
        if args.precomputed_frequency_fname:
            merge_command.extend([
                "--precomputed-frequency-fname",
                args.precomputed_frequency_fname,
            ])
        run_command(merge_command, merge_environment)

        for response in args.response_columns:
            response_dir = results_root / name / f"response_{response}"
            response_dir.mkdir(parents=True, exist_ok=True)
            layer_results = response_dir / "layer-results.tsv"
            run_command(
                [
                    args.rscript,
                    EVALUATOR,
                    merged_path,
                    layer_results,
                    response_dir / "best-layers.tsv",
                    response_dir / "summary.tsv",
                    "--analysis-mode",
                    args.analysis_mode,
                    "--response-column",
                    response,
                    "--score-kinds",
                    ",".join(args.score_kinds),
                ],
                base_environment,
            )
            layer_result_paths.append(layer_results)

    combined_dir = results_root / "combined"
    run_command(
        [
            args.python,
            ANALYZER,
            "--layer-results-fnames",
            *layer_result_paths,
            "--output-layer-results-fname",
            combined_dir / "layer-results.tsv",
            "--output-best-layers-fname",
            combined_dir / "best-layers.tsv",
            "--output-report-fname",
            combined_dir / "REPORT.md",
            "--output-summary-json-fname",
            combined_dir / "summary.json",
            "--title",
            f"Internal-layer factorial: {args.model}",
            "--note",
            args.report_note,
            "--config",
            args.layer_factorial_config.source_path,
            "--contexts",
            *args.contexts,
            "--lens-methods",
            *args.lens_methods,
            "--score-kinds",
            *args.score_kinds,
            "--response-columns",
            *args.response_columns,
            "--early-layer-threshold",
            args.early_layer_threshold,
        ],
        base_environment,
    )

    effective_config = {
        "model": args.model,
        "switches": {
            "responses": list(args.response_columns),
            "contexts": list(args.contexts),
            "lens_methods": list(args.lens_methods),
            "score_kinds": list(args.score_kinds),
            "include_embedding_layer": args.include_embedding_layer,
        },
        "extraction": {
            "sentence_first_token_policy": (
                args.sentence_first_token_policy
            ),
            "final_layer_tolerance": args.final_layer_tolerance,
        },
        "analysis": {
            "mode": args.analysis_mode,
            "lag_boundary": args.lag_boundary,
            "lag_padding": args.lag_padding,
            "early_layer_threshold": args.early_layer_threshold,
            "transformer_only_sensitivity": (
                args.layer_factorial_config.analysis
                .transformer_only_sensitivity
            ),
        },
        "runtime": {
            "jobs": args.jobs,
            "threads_per_job": args.threads_per_job,
        },
        "paths": {
            "text": str(Path(args.text_fname).resolve()),
            "sentence_manifest": str(
                Path(args.sentence_manifest_fname).resolve()
            ),
            "joint": str(Path(args.joint_data_fname).resolve()),
            "paper_rt": (
                str(Path(args.paper_rt_fname).resolve())
                if args.paper_rt_fname else None
            ),
            "precomputed_frequency": (
                str(Path(args.precomputed_frequency_fname).resolve())
                if args.precomputed_frequency_fname else None
            ),
            "tuned_lens": (
                str(Path(args.tuned_lens_path).resolve())
                if args.tuned_lens_path else None
            ),
            "checkpoint_root": str(checkpoint_root.resolve()),
            "results_root": str(results_root.resolve()),
        },
        "report_note": args.report_note,
    }
    payload = {
        "schema_version": 3,
        "model": args.model,
        "hf_model_name": model_spec.hf_name,
        "base_model_revision": revision,
        "tuned_lens_artifact": model_spec.lens_artifact,
        "tuned_lens_base_model_revision": (
            model_spec.lens_base_model_revision
        ),
        "include_embedding_layer": args.include_embedding_layer,
        "contexts": list(args.contexts),
        "score_kinds": list(args.score_kinds),
        "lens_methods": list(args.lens_methods),
        "analysis_mode": args.analysis_mode,
        "response_columns": args.response_columns,
        "analysis_lag_boundary": args.lag_boundary,
        "analysis_lag_padding": args.lag_padding,
        "early_layer_threshold": args.early_layer_threshold,
        "configuration": {
            "path": str(
                args.layer_factorial_config.source_path.resolve()
            ),
            "sha256": args.layer_factorial_config.source_sha256,
            "effective_sha256": sha256_json(effective_config),
            "effective": effective_config,
            "cli_overrides": args.cli_overrides,
        },
        "extraction_validation": {
            "path": str(extraction_validation_path.resolve()),
            "sha256": sha256_file(extraction_validation_path),
        },
        "inputs": {
            "text": {
                "path": str(Path(args.text_fname).resolve()),
                "sha256": sha256_file(args.text_fname),
            },
            "sentence_manifest": {
                "path": str(Path(args.sentence_manifest_fname).resolve()),
                "sha256": sha256_file(args.sentence_manifest_fname),
            },
            "joint": {
                "path": str(Path(args.joint_data_fname).resolve()),
                "sha256": sha256_file(args.joint_data_fname),
            },
        },
        "combined_report": str(
            (combined_dir / "REPORT.md").resolve()
        ),
    }
    if "tuned-lens" in args.lens_methods:
        for name, filename in (
            ("tuned_lens_config", "config.json"),
            ("tuned_lens_params", "params.pt"),
        ):
            artifact_path = Path(args.tuned_lens_path) / filename
            payload["inputs"][name] = {
                "path": str(artifact_path.resolve()),
                "sha256": sha256_file(artifact_path),
            }
    if args.paper_rt_fname:
        payload["inputs"]["paper_rt"] = {
            "path": str(Path(args.paper_rt_fname).resolve()),
            "sha256": sha256_file(args.paper_rt_fname),
        }
    if args.precomputed_frequency_fname:
        payload["inputs"]["precomputed_frequency"] = {
            "path": str(Path(args.precomputed_frequency_fname).resolve()),
            "sha256": sha256_file(args.precomputed_frequency_fname),
        }
    write_json_atomic(payload, checkpoint_root / "run-manifest.json")
    print(f"Factorial experiment complete: {combined_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
