#!/usr/bin/env python3

'''Validate and synthesize the full cross-model layer-factorial experiment.'''

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from h01_data.layer_factorial_models import (  # noqa: E402
    LayerFactorialModel,
    get_model_spec,
    model_aliases,
)
from h03_paper.analyze_layer_factorial_results import (  # noqa: E402
    CONTEXT_ORDER,
    KEY_COLUMNS,
    LENS_ORDER,
    SCORE_ORDER,
    sha256_file,
    validate_and_select_best,
    write_text_atomic,
    write_tsv_atomic,
)


RESPONSE_ORDER = ('paper_time', 'time')
SCOPE_ORDER = ('including-embedding', 'transformer-only')
EXPECTED_ANALYSIS = 'kuribayashi_paper_exact_L_nesting'
EXPECTED_ANALYSIS_MODE = 'paper-exact'
EXPECTED_INPUT_ROWS = 10_256
EXPECTED_ANALYSIS_ROWS = 9_771
EXPECTED_EXCLUDED_ROWS = 485
EARLY_THRESHOLD = 0.2
SHA256_PATTERN = re.compile(r'[0-9a-f]{64}')
SOURCE_KEYS = ('response_column', 'context_unit', 'lens_method')
GROUP_KEYS = ('model',) + KEY_COLUMNS
SHARED_INPUTS = (
    'paper_rt',
    'precomputed_frequency',
    'sentence_manifest',
    'text',
)
FACTOR_SPECS = (
    ('response_column', 'time', 'paper_time'),
    ('context_unit', 'passage', 'sentence'),
    ('lens_method', 'logit-lens', 'tuned-lens'),
    ('score_kind', 'corrected', 'buggy'),
)


@dataclass(frozen=True)
class ModelRun:
    alias: str
    spec: LayerFactorialModel
    layers: pd.DataFrame
    best: pd.DataFrame
    manifest: dict
    validation: dict
    shared_input_hashes: dict[str, str]
    integrity: dict
    input_artifacts: tuple[dict, ...]


def _read_json(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding='utf8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f'unable to read {label}: {path}') from error
    if not isinstance(payload, dict):
        raise ValueError(f'{label} must be a JSON object: {path}')
    return payload


def _read_tsv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path, sep='\t', keep_default_na=False, low_memory=False
        )
    except Exception as error:
        raise ValueError(f'unable to read {label}: {path}') from error


def _digest(value, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f'{label} must be a lowercase SHA-256 digest')
    return value


def _nonempty(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be a nonempty string')
    return value


def _require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f'{label} mismatch: observed {actual!r}; expected {expected!r}'
        )


def _require_values(
    frame: pd.DataFrame, column: str, expected, label: str
) -> None:
    if column not in frame:
        raise ValueError(f'{label} lacks column {column}')
    observed = set(frame[column].drop_duplicates().tolist())
    expected_set = set(expected)
    if observed != expected_set:
        raise ValueError(
            f'{label} {column} values differ: '
            f'observed {sorted(map(str, observed))}; '
            f'expected {sorted(map(str, expected_set))}'
        )


def _require_single(
    frame: pd.DataFrame, column: str, expected, label: str
) -> None:
    _require_values(frame, column, (expected,), label)


def _boolean_series(series: pd.Series, label: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        'True': True,
        'False': False,
        'true': True,
        'false': False,
    }
    converted = series.map(mapping)
    if converted.isna().any():
        raise ValueError(f'{label} must contain booleans')
    return converted.astype(bool)


def _json_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient='records'))


def _model_paths(run_root: Path, alias: str) -> dict[str, Path]:
    result_dir = run_root / 'results/layer-factorial' / alias / 'combined'
    checkpoint_dir = run_root / 'checkpoints/layer-factorial' / alias
    paths = {
        'layers': result_dir / 'layer-results.tsv',
        'best': result_dir / 'best-layers.tsv',
        'summary': result_dir / 'summary.json',
        'manifest': checkpoint_dir / 'run-manifest.json',
        'validation': checkpoint_dir / 'extraction-validation.json',
    }
    missing = [
        f'{name}: {path}'
        for name, path in paths.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f'missing artifacts for {alias}: ' + '; '.join(missing)
        )
    return paths


def _validate_table_metadata(
    layers: pd.DataFrame,
    best: pd.DataFrame,
    alias: str,
    spec: LayerFactorialModel,
) -> None:
    label = f'{alias} combined layer results'
    _require_single(layers, 'model', alias, label)
    _require_single(layers, 'analysis', EXPECTED_ANALYSIS, label)
    _require_single(
        layers, 'analysis_mode', EXPECTED_ANALYSIS_MODE, label
    )
    _require_values(layers, 'response_column', RESPONSE_ORDER, label)
    _require_single(layers, 'min_layer', 0, label)
    _require_single(layers, 'max_layer', spec.final_layer, label)
    _require_single(layers, 'input_rows', EXPECTED_INPUT_ROWS, label)
    _require_single(layers, 'analysis_rows', EXPECTED_ANALYSIS_ROWS, label)
    _require_single(layers, 'excluded_rows', EXPECTED_EXCLUDED_ROWS, label)
    _require_single(layers, 'include_embedding_layer', True, label)
    _require_single(layers, 'lag_boundary', 'sentence', label)
    _require_single(layers, 'lag_padding', 'global-mean', label)
    _require_single(layers, 'analysis_lag_boundary', 'sentence', label)
    _require_single(
        layers, 'analysis_lag_padding', 'global-mean', label
    )

    expected_rows = (
        len(RESPONSE_ORDER)
        * len(CONTEXT_ORDER)
        * len(LENS_ORDER)
        * len(SCORE_ORDER)
        * (spec.final_layer + 1)
    )
    if len(layers) != expected_rows:
        raise ValueError(
            f'{alias} has {len(layers)} curve rows; expected {expected_rows}'
        )
    if len(best) != 16:
        raise ValueError(f'{alias} has {len(best)} cells; expected 16')

    for context, policy in (('passage', 'bos'), ('sentence', 'bow')):
        subset = layers.loc[layers['context_unit'] == context]
        _require_single(
            subset, 'first_token_policy', policy, f'{alias} {context}'
        )
        _require_single(
            subset,
            'sentence_first_token_policy',
            policy,
            f'{alias} {context}',
        )
    for score, prefix in (
        ('corrected', 'internal_layer_surprisal_layer_'),
        ('buggy', 'internal_layer_surprisal_buggy_layer_'),
    ):
        _require_single(
            layers.loc[layers['score_kind'] == score],
            'predictor_prefix',
            prefix,
            f'{alias} {score}',
        )

    expected_fraction = (
        layers['layer'].to_numpy(dtype=float) / spec.final_layer
    )
    for column in ('relative_depth_block', 'layer_fraction'):
        values = pd.to_numeric(
            layers[column], errors='raise'
        ).to_numpy()
        if not np.allclose(
            values, expected_fraction, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f'{alias} {column} disagrees with layer / D')

    expected_final = layers['layer'].to_numpy() == spec.final_layer
    expected_embedding = layers['layer'].to_numpy() == 0
    for column, expected in (
        ('is_final_layer', expected_final),
        ('is_embedding_layer', expected_embedding),
    ):
        observed = _boolean_series(layers[column], f'{alias} {column}')
        if not np.array_equal(observed.to_numpy(), expected):
            raise ValueError(f'{alias} {column} flags are incorrect')

    selected = set(
        best[list(GROUP_KEYS) + ['layer']].itertuples(
            index=False, name=None
        )
    )
    expected_best = np.array([
        tuple(row) in selected
        for row in layers[list(GROUP_KEYS) + ['layer']].itertuples(
            index=False, name=None
        )
    ])
    observed_best = _boolean_series(
        layers['is_best_layer'], f'{alias} is_best_layer'
    )
    if not np.array_equal(observed_best.to_numpy(), expected_best):
        raise ValueError(f'{alias} is_best_layer flags are incorrect')


def _validate_stored_best(
    stored: pd.DataFrame, recomputed: pd.DataFrame, alias: str
) -> None:
    key_columns = list(GROUP_KEYS) + ['layer']
    if len(stored) != 16 or stored.duplicated(list(GROUP_KEYS)).any():
        raise ValueError(
            f'{alias} stored best-layers cardinality is invalid'
        )
    observed = set(
        stored[key_columns].itertuples(index=False, name=None)
    )
    expected = set(
        recomputed[key_columns].itertuples(index=False, name=None)
    )
    if observed != expected:
        raise ValueError(f'{alias} stored best layers differ from argmax')

    merged = recomputed.merge(
        stored,
        on=list(GROUP_KEYS),
        suffixes=('_expected', '_stored'),
        validate='one_to_one',
    )
    for column in ('delta_ll', 'ppp_x1000'):
        expected_values = pd.to_numeric(
            merged[f'{column}_expected'], errors='raise'
        )
        observed_values = pd.to_numeric(
            merged[f'{column}_stored'], errors='raise'
        )
        if not np.allclose(
            expected_values, observed_values, rtol=0.0, atol=1e-10
        ):
            raise ValueError(f'{alias} stored {column} values differ')
    expected_early = (
        merged['layer_expected'] / merged['max_layer_expected']
        <= EARLY_THRESHOLD
    )
    observed_early = _boolean_series(
        merged['best_in_first_20pct_stored'],
        f'{alias} stored best_in_first_20pct',
    )
    if not np.array_equal(
        expected_early.to_numpy(), observed_early.to_numpy()
    ):
        raise ValueError(f'{alias} stored early-layer flags differ')


def _source_records(
    layers: pd.DataFrame, alias: str
) -> set[tuple[str, str]]:
    for column in ('_source_path', '_source_sha256'):
        if column not in layers:
            raise ValueError(f'{alias} layer results lack {column}')
    records = set()
    for key, group in layers.groupby(list(SOURCE_KEYS), sort=False):
        paths = group['_source_path'].drop_duplicates().tolist()
        digests = group['_source_sha256'].drop_duplicates().tolist()
        if len(paths) != 1 or len(digests) != 1:
            raise ValueError(
                f'{alias} source provenance varies within cell {key}'
            )
        path = _nonempty(paths[0], f'{alias} source path')
        digest = _digest(digests[0], f'{alias} source SHA-256')
        records.add((path, digest))
    if len(records) != 8:
        raise ValueError(
            f'{alias} has {len(records)} source files; expected 8'
        )
    return records


def _validate_combined_summary(
    summary: dict,
    layers: pd.DataFrame,
    best: pd.DataFrame,
    alias: str,
    source_records: set[tuple[str, str]],
) -> None:
    _require_equal(
        summary.get('schema_version'), 1, f'{alias} summary schema'
    )
    _require_equal(
        summary.get('models'), [alias], f'{alias} summary models'
    )
    _require_equal(
        summary.get('analysis_modes'),
        [EXPECTED_ANALYSIS_MODE],
        f'{alias} summary analysis modes',
    )
    _require_equal(
        summary.get('layer_rows'),
        len(layers),
        f'{alias} summary layer rows',
    )
    _require_equal(
        summary.get('factorial_cells'), 16, f'{alias} summary cells'
    )
    expected_by_response = {}
    for response in RESPONSE_ORDER:
        subset = best.loc[best['response_column'] == response]
        expected_by_response[response] = {
            'cells': 8,
            'best_in_first_20pct': int(
                subset['best_in_first_20pct'].sum()
            ),
        }
    _require_equal(
        summary.get('by_response'),
        expected_by_response,
        f'{alias} summary response counts',
    )
    inputs = summary.get('inputs')
    if not isinstance(inputs, list):
        raise ValueError(f'{alias} summary inputs must be a list')
    recorded = {
        (
            _nonempty(item.get('path'), f'{alias} summary input path'),
            _digest(
                item.get('sha256'),
                f'{alias} summary input SHA-256',
            ),
        )
        for item in inputs
        if isinstance(item, dict)
    }
    if len(recorded) != len(inputs) or recorded != source_records:
        raise ValueError(f'{alias} summary source records differ')


def _manifest_input_hashes(
    manifest: dict, alias: str, spec: LayerFactorialModel
) -> dict[str, str]:
    inputs = manifest.get('inputs')
    if not isinstance(inputs, dict):
        raise ValueError(f'{alias} manifest inputs must be an object')
    required = set(SHARED_INPUTS) | {
        'joint',
        'tuned_lens_config',
        'tuned_lens_params',
    }
    if not required <= set(inputs):
        raise ValueError(f'{alias} manifest lacks required input records')
    hashes = {}
    for name in required:
        record = inputs[name]
        if not isinstance(record, dict):
            raise ValueError(
                f'{alias} manifest input {name} is invalid'
            )
        _nonempty(record.get('path'), f'{alias} {name} path')
        hashes[name] = _digest(
            record.get('sha256'), f'{alias} {name} SHA-256'
        )
    _require_equal(
        hashes['tuned_lens_config'],
        spec.lens_config_sha256,
        f'{alias} tuned-lens config hash',
    )
    _require_equal(
        hashes['tuned_lens_params'],
        spec.lens_params_sha256,
        f'{alias} tuned-lens params hash',
    )
    return hashes


def _comparison_values(node, alias: str, label: str) -> list[float]:
    if not isinstance(node, dict):
        raise ValueError(
            f'{alias} {label} comparison must be an object'
        )
    values = []
    for context in CONTEXT_ORDER:
        scores = node.get(context)
        if (
            not isinstance(scores, dict)
            or set(scores) != set(SCORE_ORDER)
        ):
            raise ValueError(
                f'{alias} {label} comparison grid is incomplete'
            )
        for score in SCORE_ORDER:
            value = float(scores[score])
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f'{alias} {label} comparison contains invalid values'
                )
            values.append(value)
    return values


def _validate_manifest_and_extraction(
    manifest: dict,
    validation: dict,
    validation_path: Path,
    alias: str,
    spec: LayerFactorialModel,
) -> tuple[dict[str, str], float, float, bool]:
    _require_equal(
        manifest.get('schema_version'), 3, f'{alias} manifest schema'
    )
    _require_equal(
        manifest.get('model'), alias, f'{alias} manifest model'
    )
    _require_equal(
        manifest.get('hf_model_name'),
        spec.hf_name,
        f'{alias} HF model',
    )
    _require_equal(
        manifest.get('base_model_revision'),
        spec.base_model_revision,
        f'{alias} base revision',
    )
    _require_equal(
        manifest.get('tuned_lens_artifact'),
        spec.lens_artifact,
        f'{alias} tuned-lens artifact',
    )
    _require_equal(
        manifest.get('tuned_lens_base_model_revision'),
        spec.lens_base_model_revision,
        f'{alias} tuned-lens base revision',
    )
    _require_equal(
        manifest.get('contexts'),
        list(CONTEXT_ORDER),
        f'{alias} contexts',
    )
    _require_equal(
        manifest.get('lens_methods'),
        list(LENS_ORDER),
        f'{alias} lenses',
    )
    _require_equal(
        manifest.get('score_kinds'),
        list(SCORE_ORDER),
        f'{alias} scores',
    )
    _require_equal(
        manifest.get('response_columns'),
        ['time', 'paper_time'],
        f'{alias} responses',
    )
    _require_equal(
        manifest.get('include_embedding_layer'),
        True,
        f'{alias} embedding',
    )
    _require_equal(
        manifest.get('analysis_mode'),
        EXPECTED_ANALYSIS_MODE,
        f'{alias} analysis mode',
    )
    _require_equal(
        manifest.get('analysis_lag_boundary'),
        'sentence',
        f'{alias} analysis lag boundary',
    )
    _require_equal(
        manifest.get('analysis_lag_padding'),
        'global-mean',
        f'{alias} analysis lag padding',
    )

    extraction_record = manifest.get('extraction_validation')
    if not isinstance(extraction_record, dict):
        raise ValueError(
            f'{alias} manifest lacks extraction validation'
        )
    recorded_hash = _digest(
        extraction_record.get('sha256'),
        f'{alias} extraction validation SHA-256',
    )
    _require_equal(
        sha256_file(validation_path),
        recorded_hash,
        f'{alias} extraction validation hash',
    )

    _require_equal(
        validation.get('schema_version'),
        1,
        f'{alias} validation schema',
    )
    _require_equal(
        validation.get('validated'),
        True,
        f'{alias} validated flag',
    )
    _require_equal(
        validation.get('model'),
        alias,
        f'{alias} validation model',
    )
    _require_equal(
        validation.get('model_revision_effective'),
        spec.base_model_revision,
        f'{alias} effective revision',
    )
    expected = validation.get('expected')
    if not isinstance(expected, dict):
        raise ValueError(
            f'{alias} validation lacks expected dimensions'
        )
    _require_equal(
        expected.get('rows'), EXPECTED_INPUT_ROWS, f'{alias} rows'
    )
    _require_equal(
        expected.get('min_layer'), 0, f'{alias} minimum layer'
    )
    _require_equal(
        expected.get('final_layer'),
        spec.final_layer,
        f'{alias} final layer',
    )
    _require_equal(
        expected.get('layers'),
        list(range(spec.final_layer + 1)),
        f'{alias} layer grid',
    )
    tolerance = float(expected.get('final_layer_tolerance'))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(
            f'{alias} final-layer tolerance is invalid'
        )

    comparisons = validation.get('comparisons')
    if not isinstance(comparisons, dict):
        raise ValueError(f'{alias} validation lacks comparisons')
    final_values = _comparison_values(
        comparisons.get(
            'final_logit_vs_tuned_max_abs_difference'
        ),
        alias,
        'final logit/tuned',
    )
    intermediate_values = _comparison_values(
        comparisons.get(
            'intermediate_logit_vs_tuned_max_abs_difference'
        ),
        alias,
        'intermediate logit/tuned',
    )
    if max(final_values) > tolerance:
        raise ValueError(
            f'{alias} final logit/tuned difference is too large'
        )
    if min(intermediate_values) <= tolerance:
        raise ValueError(
            f'{alias} tuned/logit manipulation is not distinct'
        )

    identity = validation.get('tuned_lens_identity')
    if not isinstance(identity, dict):
        raise ValueError(f'{alias} tuned-lens identity is missing')
    artifact = identity.get('artifact')
    identity_validation = identity.get('validation')
    if (
        not isinstance(artifact, dict)
        or not isinstance(identity_validation, dict)
    ):
        raise ValueError(
            f'{alias} tuned-lens identity is malformed'
        )
    _require_equal(
        artifact.get('config_sha256'),
        spec.lens_config_sha256,
        f'{alias} validation lens config hash',
    )
    _require_equal(
        artifact.get('params_sha256'),
        spec.lens_params_sha256,
        f'{alias} validation lens params hash',
    )
    _require_equal(
        artifact.get('num_hidden_layers'),
        spec.final_layer,
        f'{alias} validation lens depth',
    )
    revision_validated = bool(
        identity_validation.get('base_model_revision_validated')
    )
    if spec.lens_base_model_revision is not None and not revision_validated:
        raise ValueError(
            f'{alias} tuned-lens base revision should be validated'
        )

    input_hashes = _manifest_input_hashes(manifest, alias, spec)
    _require_equal(
        validation.get('sentence_manifest_sha256'),
        input_hashes['sentence_manifest'],
        f'{alias} sentence-manifest hash',
    )
    return (
        {name: input_hashes[name] for name in SHARED_INPUTS},
        max(final_values),
        min(intermediate_values),
        revision_validated,
    )


def load_model(run_root: Path, alias: str) -> ModelRun:
    spec = get_model_spec(alias)
    paths = _model_paths(run_root, alias)
    raw_layers = _read_tsv(paths['layers'], f'{alias} layer results')
    layers, best = validate_and_select_best(raw_layers)
    _validate_table_metadata(layers, best, alias, spec)

    source_records = _source_records(layers, alias)
    stored_best = _read_tsv(
        paths['best'], f'{alias} stored best layers'
    )
    _validate_stored_best(stored_best, best, alias)
    summary = _read_json(
        paths['summary'], f'{alias} combined summary'
    )
    _validate_combined_summary(
        summary, layers, best, alias, source_records
    )
    manifest = _read_json(
        paths['manifest'], f'{alias} run manifest'
    )
    validation = _read_json(
        paths['validation'], f'{alias} extraction validation'
    )
    (
        shared_hashes,
        final_difference,
        intermediate_difference,
        lens_revision_validated,
    ) = _validate_manifest_and_extraction(
        manifest, validation, paths['validation'], alias, spec
    )

    relative_layer_path = (
        paths['layers'].relative_to(run_root).as_posix()
    )
    relative_best_path = (
        paths['best'].relative_to(run_root).as_posix()
    )
    layers = layers.copy()
    best = best.copy()
    layers['_combined_source_path'] = relative_layer_path
    layers['_combined_source_sha256'] = sha256_file(paths['layers'])
    best['_combined_source_path'] = relative_layer_path
    best['_combined_source_sha256'] = sha256_file(paths['layers'])

    artifact_records = tuple({
        'model': alias,
        'kind': name,
        'path': path.relative_to(run_root).as_posix(),
        'sha256': sha256_file(path),
    } for name, path in paths.items())
    integrity = {
        'model': alias,
        'max_layer': spec.final_layer,
        'layer_rows': len(layers),
        'factorial_cells': len(best),
        'source_cell_files': len(source_records),
        'input_rows': EXPECTED_INPUT_ROWS,
        'analysis_rows': EXPECTED_ANALYSIS_ROWS,
        'excluded_rows': EXPECTED_EXCLUDED_ROWS,
        'combined_layer_results_path': relative_layer_path,
        'combined_layer_results_sha256': sha256_file(paths['layers']),
        'combined_best_layers_path': relative_best_path,
        'combined_best_layers_sha256': sha256_file(paths['best']),
        'manifest_sha256': sha256_file(paths['manifest']),
        'extraction_validation_sha256': sha256_file(
            paths['validation']
        ),
        'final_logit_tuned_max_abs_difference': final_difference,
        'intermediate_logit_tuned_min_max_abs_difference': (
            intermediate_difference
        ),
        'tuned_lens_base_revision_validated': (
            lens_revision_validated
        ),
    }
    return ModelRun(
        alias=alias,
        spec=spec,
        layers=layers,
        best=best,
        manifest=manifest,
        validation=validation,
        shared_input_hashes=shared_hashes,
        integrity=integrity,
        input_artifacts=artifact_records,
    )


def _order_frame(
    frame: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    output = frame.copy()
    maps = {
        'model': {
            value: index
            for index, value in enumerate(model_aliases())
        },
        'response_column': {
            value: index
            for index, value in enumerate(RESPONSE_ORDER)
        },
        'context_unit': {
            value: index
            for index, value in enumerate(CONTEXT_ORDER)
        },
        'lens_method': {
            value: index
            for index, value in enumerate(LENS_ORDER)
        },
        'score_kind': {
            value: index
            for index, value in enumerate(SCORE_ORDER)
        },
        'layer_scope': {
            value: index
            for index, value in enumerate(SCOPE_ORDER)
        },
        'factor': {
            value: index
            for index, (value, _, _) in enumerate(FACTOR_SPECS)
        },
    }
    temporary = []
    for column in columns:
        order_column = f'__{column}_order'
        output[order_column] = (
            output[column].map(maps[column])
            if column in maps
            else output[column]
        )
        temporary.append(order_column)
    output.sort_values(temporary, kind='stable', inplace=True)
    return output.drop(columns=temporary).reset_index(drop=True)


def _best_for_scope(
    layers: pd.DataFrame, scope: str
) -> pd.DataFrame:
    if scope not in SCOPE_ORDER:
        raise ValueError(f'unknown layer scope: {scope}')
    eligible = (
        layers
        if scope == 'including-embedding'
        else layers.loc[layers['layer'] > 0]
    )
    rows = []
    for _, group in eligible.groupby(
        list(GROUP_KEYS), sort=False, dropna=False
    ):
        values = group['delta_ll'].to_numpy(dtype=float)
        position = int(values.argmax())
        selected = group.iloc[[position]].copy()
        descending = np.sort(values)[::-1]
        selected['peak_margin_delta_ll'] = float(
            descending[0] - descending[1]
        )
        rows.append(selected)
    best = pd.concat(rows, ignore_index=True)
    best['layer_fraction_recomputed'] = (
        best['layer'] / best['max_layer']
    )
    best['best_in_first_20pct'] = (
        best['layer_fraction_recomputed'] <= EARLY_THRESHOLD
    )
    best['layer_scope'] = scope
    best['best_is_embedding'] = best['layer'] == 0
    return _order_frame(
        best,
        [
            'model',
            'response_column',
            'context_unit',
            'lens_method',
            'score_kind',
            'layer_scope',
        ],
    )


def build_condition_summary(best: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        'layer_scope',
        'response_column',
        'context_unit',
        'lens_method',
        'score_kind',
    ]
    rows = []
    model_order = {
        value: index for index, value in enumerate(model_aliases())
    }
    for key, group in best.groupby(group_columns, sort=False):
        ordered_models = sorted(
            group['model'].tolist(), key=model_order.get
        )
        early_models = sorted(
            group.loc[
                group['best_in_first_20pct'], 'model'
            ].tolist(),
            key=model_order.get,
        )
        lookup = group.set_index('model')
        rows.append({
            **dict(zip(group_columns, key)),
            'model_count': len(group),
            'early_model_count': int(
                group['best_in_first_20pct'].sum()
            ),
            'early_model_fraction': float(
                group['best_in_first_20pct'].mean()
            ),
            'median_best_layer_fraction': float(
                group['layer_fraction_recomputed'].median()
            ),
            'mean_best_layer_fraction': float(
                group['layer_fraction_recomputed'].mean()
            ),
            'embedding_best_count': int(
                group['best_is_embedding'].sum()
            ),
            'median_delta_ll': float(group['delta_ll'].median()),
            'median_peak_margin_delta_ll': float(
                group['peak_margin_delta_ll'].median()
            ),
            'early_models': ','.join(early_models),
            'best_layers': ','.join(
                f'{model}:{int(lookup.loc[model, "layer"])}/'
                f'{int(lookup.loc[model, "max_layer"])}'
                for model in ordered_models
            ),
        })
    return _order_frame(
        pd.DataFrame(rows),
        [
            'response_column',
            'context_unit',
            'lens_method',
            'score_kind',
            'layer_scope',
        ],
    )


def build_model_response_summary(
    best: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for key, group in best.groupby(
        ['model', 'response_column', 'layer_scope'], sort=False
    ):
        rows.append({
            'model': key[0],
            'response_column': key[1],
            'layer_scope': key[2],
            'factorial_cells': len(group),
            'early_cell_count': int(
                group['best_in_first_20pct'].sum()
            ),
            'early_cell_fraction': float(
                group['best_in_first_20pct'].mean()
            ),
            'embedding_best_count': int(
                group['best_is_embedding'].sum()
            ),
            'median_best_layer_fraction': float(
                group['layer_fraction_recomputed'].median()
            ),
        })
    return _order_frame(
        pd.DataFrame(rows),
        ['model', 'response_column', 'layer_scope'],
    )


def build_factor_effects(best: pd.DataFrame) -> pd.DataFrame:
    factor_columns = list(KEY_COLUMNS)
    rows = []
    for scope in SCOPE_ORDER:
        scoped = best.loc[best['layer_scope'] == scope]
        for factor, reference_level, target_level in FACTOR_SPECS:
            other = ['model'] + [
                column
                for column in factor_columns
                if column != factor
            ]
            metrics = [
                'layer_fraction_recomputed',
                'best_in_first_20pct',
                'layer',
            ]
            reference = scoped.loc[
                scoped[factor] == reference_level,
                other + metrics,
            ]
            target = scoped.loc[
                scoped[factor] == target_level,
                other + metrics,
            ]
            paired = reference.merge(
                target,
                on=other,
                suffixes=('_reference', '_target'),
                validate='one_to_one',
            )
            expected_pairs = scoped['model'].nunique() * 8
            if len(paired) != expected_pairs:
                raise ValueError(
                    f'{scope} {factor} has {len(paired)} pairs; '
                    f'expected {expected_pairs}'
                )
            shift = (
                paired['layer_fraction_recomputed_target']
                - paired['layer_fraction_recomputed_reference']
            )
            layer_shift = (
                paired['layer_target']
                - paired['layer_reference']
            )
            reference_early = paired[
                'best_in_first_20pct_reference'
            ].astype(bool)
            target_early = paired[
                'best_in_first_20pct_target'
            ].astype(bool)
            tolerance = 1e-15
            rows.append({
                'layer_scope': scope,
                'factor': factor,
                'reference_level': reference_level,
                'target_level': target_level,
                'paired_cells': len(paired),
                'model_count': scoped['model'].nunique(),
                'mean_layer_shift': float(layer_shift.mean()),
                'median_layer_shift': float(
                    layer_shift.median()
                ),
                'mean_depth_shift': float(shift.mean()),
                'median_depth_shift': float(shift.median()),
                'earlier_pair_count': int(
                    (shift < -tolerance).sum()
                ),
                'unchanged_pair_count': int(
                    (shift.abs() <= tolerance).sum()
                ),
                'later_pair_count': int(
                    (shift > tolerance).sum()
                ),
                'reference_early_count': int(
                    reference_early.sum()
                ),
                'target_early_count': int(target_early.sum()),
                'reference_early_fraction': float(
                    reference_early.mean()
                ),
                'target_early_fraction': float(
                    target_early.mean()
                ),
                'early_gain_count': int(
                    ((~reference_early) & target_early).sum()
                ),
                'early_loss_count': int(
                    (reference_early & (~target_early)).sum()
                ),
                'net_early_gain': int(
                    target_early.sum() - reference_early.sum()
                ),
            })
    return _order_frame(
        pd.DataFrame(rows), ['layer_scope', 'factor']
    )


def _condition_lookup(
    summary: pd.DataFrame,
    response: str,
    context: str,
    lens: str,
    score: str,
    scope: str,
) -> pd.Series:
    selected = summary.loc[
        (summary['response_column'] == response)
        & (summary['context_unit'] == context)
        & (summary['lens_method'] == lens)
        & (summary['score_kind'] == score)
        & (summary['layer_scope'] == scope)
    ]
    if len(selected) != 1:
        raise ValueError('condition summary lookup is not unique')
    return selected.iloc[0]


def _format_depth(value) -> str:
    return f'{100 * float(value):.1f}%'


def make_report(
    runs: list[ModelRun],
    best: pd.DataFrame,
    condition_summary: pd.DataFrame,
    factor_effects: pd.DataFrame,
    shared_inputs: dict[str, str],
) -> str:
    model_count = len(runs)
    target = (
        'paper_time',
        'sentence',
        'tuned-lens',
        'buggy',
    )
    baseline = (
        'time',
        'passage',
        'logit-lens',
        'corrected',
    )
    target_all = _condition_lookup(
        condition_summary, *target, 'including-embedding'
    )
    target_blocks = _condition_lookup(
        condition_summary, *target, 'transformer-only'
    )
    baseline_all = _condition_lookup(
        condition_summary, *baseline, 'including-embedding'
    )
    baseline_blocks = _condition_lookup(
        condition_summary, *baseline, 'transformer-only'
    )
    pair_count = model_count * 8
    block_effects = factor_effects.set_index(
        ['layer_scope', 'factor']
    )
    context_blocks = block_effects.loc[
        ('transformer-only', 'context_unit')
    ]
    lens_blocks = block_effects.loc[
        ('transformer-only', 'lens_method')
    ]
    response_blocks = block_effects.loc[
        ('transformer-only', 'response_column')
    ]
    score_blocks = block_effects.loc[
        ('transformer-only', 'score_kind')
    ]
    lines = [
        '# Cross-model internal-layer factorial replication',
        '',
        '## Result',
        '',
        (
            'The full paper-motivated cell (paper_time + sentence context '
            '+ tuned lens + buggy surprisal) selects an early layer in '
            f'{int(target_all.early_model_count)} of {model_count} models. '
            'When layer 0 is excluded and the argmax is recomputed over '
            'transformer layers, the result is '
            f'{int(target_blocks.early_model_count)} of {model_count}.'
        ),
        '',
        (
            'The earlier project-style baseline (time + passage context + '
            'logit lens + corrected surprisal) reaches '
            f'{int(baseline_all.early_model_count)} of {model_count} with '
            'layer 0 eligible and '
            f'{int(baseline_blocks.early_model_count)} of {model_count} '
            'over transformer layers only.'
        ),
        '',
        (
            'Early means layer / D <= 0.2. The transformer-only '
            'sensitivity keeps this architectural-depth definition; it '
            'does not renumber layer 1 as depth zero.'
        ),
        '',
        '## What changed the result',
        '',
    ]
    for scope in SCOPE_ORDER:
        effects = factor_effects.loc[
            factor_effects['layer_scope'] == scope
        ]
        strongest = effects.sort_values(
            'mean_depth_shift', kind='stable'
        ).iloc[0]
        lines.append(
            f'- **{scope}:** the largest mean shift toward earlier depth '
            f'is {strongest.factor} ({strongest.reference_level} -> '
            f'{strongest.target_level}), '
            f'{_format_depth(strongest.mean_depth_shift)} of model depth.'
        )
    lines.extend([
        '',
        (
            f'These are descriptive matched contrasts across {pair_count} '
            'factorial '
            'pairs per factor. They expose interactions but are not '
            'independent observations or inferential tests.'
        ),
        '',
        '| Scope | Factor change | Mean depth shift | Median shift | Earlier / same / later | Early count before -> after |',
        '|---|---|---:|---:|---:|---:|',
    ])
    for row in factor_effects.itertuples(index=False):
        lines.append(
            '| '
            + ' | '.join([
                str(row.layer_scope),
                f'{row.reference_level} -> {row.target_level}',
                _format_depth(row.mean_depth_shift),
                _format_depth(row.median_depth_shift),
                (
                    f'{row.earlier_pair_count} / '
                    f'{row.unchanged_pair_count} / '
                    f'{row.later_pair_count}'
                ),
                (
                    f'{row.reference_early_count} -> '
                    f'{row.target_early_count}'
                ),
            ])
            + ' |'
        )
    lines.extend([
        '',
        (
            'The robust driver is sentence-bounded context: over '
            'transformer layers it moves the optimum earlier in '
            f'{int(context_blocks.earlier_pair_count)} of {pair_count} '
            'matched cells, leaves '
            f'{int(context_blocks.unchanged_pair_count)} unchanged, and '
            f'moves none later. Tuned-lens decoding moves '
            f'{int(lens_blocks.earlier_pair_count)} earlier, while the '
            'buggy score leaves '
            f'{int(score_blocks.unchanged_pair_count)} unchanged. Changing '
            'time to paper_time is embedding-sensitive: over transformer '
            'layers it moves more cells later '
            f'({int(response_blocks.later_pair_count)}) than earlier '
            f'({int(response_blocks.earlier_pair_count)}).'
        ),
    ])

    lines.extend([
        '',
        '## All factorial conditions',
        '',
        '| Response | Context | Decoder | Score | Early incl. layer 0 | Median depth | Embedding wins | Early transformer-only | Median transformer depth |',
        '|---|---|---|---|---:|---:|---:|---:|---:|',
    ])
    for response in RESPONSE_ORDER:
        for context in CONTEXT_ORDER:
            for lens in LENS_ORDER:
                for score in SCORE_ORDER:
                    all_row = _condition_lookup(
                        condition_summary,
                        response,
                        context,
                        lens,
                        score,
                        'including-embedding',
                    )
                    block_row = _condition_lookup(
                        condition_summary,
                        response,
                        context,
                        lens,
                        score,
                        'transformer-only',
                    )
                    lines.append(
                        '| '
                        + ' | '.join([
                            response,
                            context,
                            lens,
                            score,
                            (
                                f'{int(all_row.early_model_count)}/'
                                f'{model_count}'
                            ),
                            _format_depth(
                                all_row.median_best_layer_fraction
                            ),
                            str(int(all_row.embedding_best_count)),
                            (
                                f'{int(block_row.early_model_count)}/'
                                f'{model_count}'
                            ),
                            _format_depth(
                                block_row.median_best_layer_fraction
                            ),
                        ])
                        + ' |'
                    )

    lines.extend([
        '',
        '## Full target by model',
        '',
        '| Model | D | Best incl. layer 0 | Depth | Best transformer | Depth | Early transformer | Delta LL |',
        '|---|---:|---:|---:|---:|---:|:---:|---:|',
    ])
    for run in runs:
        common = (
            (best['model'] == run.alias)
            & (best['response_column'] == target[0])
            & (best['context_unit'] == target[1])
            & (best['lens_method'] == target[2])
            & (best['score_kind'] == target[3])
        )
        all_row = best.loc[
            common
            & (best['layer_scope'] == 'including-embedding')
        ].iloc[0]
        block_row = best.loc[
            common
            & (best['layer_scope'] == 'transformer-only')
        ].iloc[0]
        lines.append(
            '| '
            + ' | '.join([
                run.alias,
                str(run.spec.final_layer),
                str(int(all_row.layer)),
                _format_depth(
                    all_row.layer_fraction_recomputed
                ),
                str(int(block_row.layer)),
                _format_depth(
                    block_row.layer_fraction_recomputed
                ),
                (
                    'yes'
                    if block_row.best_in_first_20pct
                    else 'no'
                ),
                f'{float(block_row.delta_ll):.6g}',
            ])
            + ' |'
        )

    embedding_wins = int(
        best.loc[
            best['layer_scope'] == 'including-embedding',
            'best_is_embedding',
        ].sum()
    )
    pythia_unpinned = [
        run.alias
        for run in runs
        if not run.integrity[
            'tuned_lens_base_revision_validated'
        ]
    ]
    lines.extend([
        '',
        '## Integrity and interpretation guardrails',
        '',
        (
            f'- Validated {sum(len(run.layers) for run in runs):,} '
            f'curve rows, {model_count * 16} factorial cells, complete '
            'layer ranges, stored argmax selections, combined summaries, '
            'run manifests, and extraction-validation hashes.'
        ),
        (
            '- Final-layer logit/tuned predictions agree within the '
            'declared tolerance for every model, while every intermediate '
            'manipulation is nonzero.'
        ),
        (
            f'- Shared input hashes agree across models: '
            f'{len(shared_inputs)} immutable text/RT/control artifacts.'
        ),
        (
            f'- Layer 0 is the optimum in {embedding_wins} of '
            f'{model_count * 16} cells. The transformer-only results are '
            'the safer comparison when the paper means transformer blocks.'
        ),
    ])
    if pythia_unpinned:
        lines.append(
            '- Official Pythia tuned-lens configs do not record the exact '
            'base-model revision. Base models and lens files are pinned '
            'and their name, architecture, and hashes validate, but the '
            'lens-training revision itself cannot be recovered for: '
            + ', '.join(pythia_unpinned)
            + '.'
        )
    lines.extend([
        (
            '- This compact archive omits the eight source cell TSVs and '
            'upstream model/lens binaries per model. Their declared hashes '
            'cross-check internally, but cannot be recomputed from this '
            'archive alone.'
        ),
        (
            '- Run manifests do not contain the Git commit SHA. The '
            'postprocessor therefore does not infer one from directory or '
            'archive names.'
        ),
        '',
        '## Machine-readable outputs',
        '',
        '- all-layer-results.tsv: all validated layer curves.',
        '- best-layers.tsv: exact argmax rows for both layer scopes.',
        '- condition-summary.tsv: cross-model results for all 16 cells.',
        '- model-response-summary.tsv: eight-cell summaries per model/RT response.',
        '- factor-effects.tsv: paired descriptive factorial contrasts.',
        '- integrity-checks.tsv: dimensions, hashes, and decoder checks.',
        '- summary.json: provenance and key headline counts.',
        '',
    ])
    return '\n'.join(lines)


def run_analysis(
    run_root: Path,
    output_dir: Path,
    models: tuple[str, ...] | list[str] | None = None,
) -> dict[str, object]:
    run_root = Path(run_root).resolve()
    output_dir = Path(output_dir).resolve()
    selected_models = tuple(models or model_aliases())
    if (
        not selected_models
        or len(set(selected_models)) != len(selected_models)
    ):
        raise ValueError(
            'models must be a nonempty list without duplicates'
        )
    unsupported = set(selected_models) - set(model_aliases())
    if unsupported:
        raise ValueError(
            'unsupported models: ' + ', '.join(sorted(unsupported))
        )
    selected_models = tuple(
        alias
        for alias in model_aliases()
        if alias in selected_models
    )

    runs = [
        load_model(run_root, alias)
        for alias in selected_models
    ]
    shared_inputs = runs[0].shared_input_hashes
    for run in runs[1:]:
        if run.shared_input_hashes != shared_inputs:
            raise ValueError(
                f'{run.alias} shared input hashes differ across models'
            )

    layers = _order_frame(
        pd.concat(
            [run.layers for run in runs], ignore_index=True
        ),
        [
            'model',
            'response_column',
            'context_unit',
            'lens_method',
            'score_kind',
            'layer',
        ],
    )
    best = pd.concat(
        [
            _best_for_scope(layers, scope)
            for scope in SCOPE_ORDER
        ],
        ignore_index=True,
    )
    best = _order_frame(
        best,
        [
            'model',
            'response_column',
            'context_unit',
            'lens_method',
            'score_kind',
            'layer_scope',
        ],
    )
    condition_summary = build_condition_summary(best)
    model_summary = build_model_response_summary(best)
    factor_effects = build_factor_effects(best)
    integrity = _order_frame(
        pd.DataFrame([run.integrity for run in runs]),
        ['model'],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        'all_layer_results': output_dir / 'all-layer-results.tsv',
        'best_layers': output_dir / 'best-layers.tsv',
        'condition_summary': output_dir / 'condition-summary.tsv',
        'model_response_summary': (
            output_dir / 'model-response-summary.tsv'
        ),
        'factor_effects': output_dir / 'factor-effects.tsv',
        'integrity_checks': output_dir / 'integrity-checks.tsv',
        'report': output_dir / 'REPORT.md',
        'summary': output_dir / 'summary.json',
    }
    for frame, name in (
        (layers, 'all_layer_results'),
        (best, 'best_layers'),
        (condition_summary, 'condition_summary'),
        (model_summary, 'model_response_summary'),
        (factor_effects, 'factor_effects'),
        (integrity, 'integrity_checks'),
    ):
        write_tsv_atomic(frame, outputs[name])
    write_text_atomic(
        make_report(
            runs,
            best,
            condition_summary,
            factor_effects,
            shared_inputs,
        ),
        outputs['report'],
    )

    target = (
        'paper_time',
        'sentence',
        'tuned-lens',
        'buggy',
    )
    baseline = (
        'time',
        'passage',
        'logit-lens',
        'corrected',
    )
    headline = {}
    for label, condition in (
        ('target', target),
        ('baseline', baseline),
    ):
        headline[label] = {}
        for scope in SCOPE_ORDER:
            row = _condition_lookup(
                condition_summary, *condition, scope
            )
            headline[label][scope] = {
                'early_models': int(row.early_model_count),
                'models': int(row.model_count),
                'median_best_layer_fraction': float(
                    row.median_best_layer_fraction
                ),
            }
    payload = {
        'schema_version': 1,
        'run_root': str(run_root),
        'models': list(selected_models),
        'early_definition': {
            'formula': 'layer / D <= threshold',
            'threshold': EARLY_THRESHOLD,
            'transformer_only_policy': (
                'exclude layer 0, recompute argmax, retain layer / D depth'
            ),
        },
        'counts': {
            'layer_rows': len(layers),
            'factorial_cells_per_scope': (
                len(best) // len(SCOPE_ORDER)
            ),
            'best_rows_all_scopes': len(best),
        },
        'headline': headline,
        'shared_input_sha256': shared_inputs,
        'input_artifacts': [
            record
            for run in runs
            for record in run.input_artifacts
        ],
        'factor_effects': _json_records(factor_effects),
        'outputs': {
            name: {
                'path': path.name,
                'sha256': sha256_file(path),
            }
            for name, path in outputs.items()
            if name != 'summary'
        },
    }
    write_text_atomic(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        outputs['summary'],
    )
    print(
        f'Cross-model layer-factorial analysis complete: '
        f'{output_dir}'
    )
    return {
        'layers': layers,
        'best': best,
        'condition_summary': condition_summary,
        'model_response_summary': model_summary,
        'factor_effects': factor_effects,
        'integrity': integrity,
        'summary': payload,
    }


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--run-root',
        type=Path,
        required=True,
        help=(
            'root containing results/layer-factorial and '
            'checkpoints/layer-factorial'
        ),
    )
    parser.add_argument(
        '--output-dir', type=Path, required=True
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=model_aliases(),
        default=list(model_aliases()),
        help=(
            'explicit model subset; defaults to the full pinned registry'
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_analysis(args.run_root, args.output_dir, args.models)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
