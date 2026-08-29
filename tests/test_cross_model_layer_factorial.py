'''Tests for strict cross-model layer-factorial synthesis.'''

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from h01_data.layer_factorial_models import get_model_spec  # noqa: E402
from h01_data.layer_factorial_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_layer_factorial_config,
)
from h03_paper.analyze_cross_model_layer_factorial import (  # noqa: E402
    EXPECTED_ANALYSIS,
    EXPECTED_ANALYSIS_MODE,
    run_analysis,
)
from h03_paper.analyze_layer_factorial_results import (  # noqa: E402
    sha256_file,
    validate_and_select_best,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode('utf8')).hexdigest()


class CrossModelLayerFactorialTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run_root = Path(self.temporary.name) / 'run'
        self.output_dir = Path(self.temporary.name) / 'output'
        self.shared_hashes = {
            'paper_rt': digest('paper-rt'),
            'precomputed_frequency': digest('frequency'),
            'sentence_manifest': digest('sentences'),
            'text': digest('text'),
        }

    def tearDown(self):
        self.temporary.cleanup()

    def _write_config(
        self, mutate=None, name='layer-factorial.json'
    ):
        payload = json.loads(
            DEFAULT_CONFIG_PATH.read_text(encoding='utf8')
        )
        if mutate is not None:
            mutate(payload)
        path = Path(self.temporary.name) / name
        path.write_text(
            json.dumps(payload, indent=2) + '\n',
            encoding='utf8',
        )
        return path

    def _attach_configuration_provenance(
        self,
        alias: str,
        config_path: Path,
        *,
        effective_config_path: Path | None = None,
        cli_overrides: list[str] | None = None,
    ) -> Path:
        source_config = load_layer_factorial_config(config_path)
        config = load_layer_factorial_config(
            effective_config_path or config_path
        )
        checkpoint_dir = (
            self.run_root
            / 'checkpoints/layer-factorial'
            / alias
        )
        manifest_path = checkpoint_dir / 'run-manifest.json'
        manifest = json.loads(
            manifest_path.read_text(encoding='utf8')
        )
        inputs = manifest['inputs']
        effective = {
            'model': alias,
            'switches': {
                'responses': list(config.switches.responses),
                'contexts': list(config.switches.contexts),
                'lens_methods': list(
                    config.switches.lens_methods
                ),
                'score_kinds': list(config.switches.score_kinds),
                'include_embedding_layer': (
                    config.switches.include_embedding_layer
                ),
            },
            'extraction': {
                'sentence_first_token_policy': (
                    config.extraction.sentence_first_token_policy
                ),
                'final_layer_tolerance': (
                    config.extraction.final_layer_tolerance
                ),
            },
            'analysis': {
                'mode': config.analysis.mode,
                'lag_boundary': config.analysis.lag_boundary,
                'lag_padding': config.analysis.lag_padding,
                'early_layer_threshold': (
                    config.analysis.early_layer_threshold
                ),
                'transformer_only_sensitivity': (
                    config.analysis.transformer_only_sensitivity
                ),
            },
            'runtime': {
                'jobs': config.runtime.jobs,
                'threads_per_job': config.runtime.threads_per_job,
            },
            'paths': {
                'text': inputs['text']['path'],
                'sentence_manifest': (
                    inputs['sentence_manifest']['path']
                ),
                'joint': inputs['joint']['path'],
                'paper_rt': inputs['paper_rt']['path'],
                'precomputed_frequency': (
                    inputs['precomputed_frequency']['path']
                ),
                'tuned_lens': str(
                    Path(inputs['tuned_lens_config']['path']).parent
                ),
                'checkpoint_root': '/remote/checkpoints',
                'results_root': '/remote/results',
            },
            'report_note': config.report_note,
        }
        encoded = json.dumps(
            effective,
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        ).encode('utf8')
        manifest['early_layer_threshold'] = (
            config.analysis.early_layer_threshold
        )
        manifest['configuration'] = {
            'path': str(source_config.source_path),
            'sha256': source_config.source_sha256,
            'effective_sha256': hashlib.sha256(encoded).hexdigest(),
            'effective': effective,
            'cli_overrides': (
                cli_overrides
                if cli_overrides is not None
                else ['--config']
            ),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + '\n',
            encoding='utf8',
        )
        return manifest_path

    def _reduce_model_to_config(
        self, alias: str, config_path: Path
    ) -> None:
        config = load_layer_factorial_config(config_path)
        result_dir = (
            self.run_root
            / 'results/layer-factorial'
            / alias
            / 'combined'
        )
        checkpoint_dir = (
            self.run_root
            / 'checkpoints/layer-factorial'
            / alias
        )
        layer_path = result_dir / 'layer-results.tsv'
        layers = pd.read_csv(layer_path, sep='\t')
        layers = layers.loc[
            layers['response_column'].isin(
                config.switches.responses
            )
            & layers['context_unit'].isin(
                config.switches.contexts
            )
            & layers['lens_method'].isin(
                config.switches.lens_methods
            )
            & layers['score_kind'].isin(
                config.switches.score_kinds
            )
        ].copy()
        if not config.switches.include_embedding_layer:
            layers = layers.loc[layers['layer'] > 0].copy()
        minimum_layer = (
            0 if config.switches.include_embedding_layer else 1
        )
        layers['include_embedding_layer'] = (
            config.switches.include_embedding_layer
        )
        layers['min_layer'] = minimum_layer
        layers['lag_boundary'] = config.analysis.lag_boundary
        layers['lag_padding'] = config.analysis.lag_padding
        layers['analysis_lag_boundary'] = (
            config.analysis.lag_boundary
        )
        layers['analysis_lag_padding'] = (
            config.analysis.lag_padding
        )
        layer_path.write_text(
            layers.to_csv(sep='\t', index=False),
            encoding='utf8',
        )
        _, best = validate_and_select_best(
            layers,
            contexts=config.switches.contexts,
            lenses=config.switches.lens_methods,
            score_kinds=config.switches.score_kinds,
            response_columns=config.switches.responses,
            early_threshold=config.analysis.early_layer_threshold,
        )
        best.to_csv(
            result_dir / 'best-layers.tsv',
            sep='\t',
            index=False,
        )
        source_records = (
            layers[['_source_path', '_source_sha256']]
            .drop_duplicates()
            .rename(columns={
                '_source_path': 'path',
                '_source_sha256': 'sha256',
            })
            .to_dict(orient='records')
        )
        cells_per_response = (
            len(config.switches.contexts)
            * len(config.switches.lens_methods)
            * len(config.switches.score_kinds)
        )
        summary = {
            'schema_version': 1,
            'models': [alias],
            'analysis_modes': [config.analysis.mode],
            'layer_rows': len(layers),
            'factorial_cells': len(best),
            'early_layer_threshold': (
                config.analysis.early_layer_threshold
            ),
            'by_response': {
                response: {
                    'cells': cells_per_response,
                    'best_in_first_20pct': int(
                        best.loc[
                            best['response_column'] == response,
                            'best_in_first_20pct',
                        ].sum()
                    ),
                }
                for response in config.switches.responses
            },
            'inputs': source_records,
        }
        (result_dir / 'summary.json').write_text(
            json.dumps(summary, indent=2) + '\n',
            encoding='utf8',
        )

        validation_path = checkpoint_dir / 'extraction-validation.json'
        validation = json.loads(
            validation_path.read_text(encoding='utf8')
        )
        extraction_scores = ['corrected']
        if 'buggy' in config.switches.score_kinds:
            extraction_scores.append('buggy')
        validation['expected'].update({
            'min_layer': minimum_layer,
            'layers': list(
                range(
                    minimum_layer,
                    get_model_spec(alias).final_layer + 1,
                )
            ),
            'contexts': list(config.switches.contexts),
            'lens_methods': list(config.switches.lens_methods),
            'score_kinds': extraction_scores,
        })
        validation['sentence_manifest_sha256'] = (
            self.shared_hashes['sentence_manifest']
            if 'sentence' in config.switches.contexts
            else None
        )
        if 'tuned-lens' not in config.switches.lens_methods:
            validation['tuned_lens_identity'] = None
        validation_path.write_text(
            json.dumps(validation, indent=2) + '\n',
            encoding='utf8',
        )

        manifest_path = checkpoint_dir / 'run-manifest.json'
        manifest = json.loads(
            manifest_path.read_text(encoding='utf8')
        )
        manifest.update({
            'contexts': list(config.switches.contexts),
            'lens_methods': list(config.switches.lens_methods),
            'score_kinds': list(config.switches.score_kinds),
            'response_columns': list(config.switches.responses),
            'include_embedding_layer': (
                config.switches.include_embedding_layer
            ),
            'analysis_mode': config.analysis.mode,
            'analysis_lag_boundary': (
                config.analysis.lag_boundary
            ),
            'analysis_lag_padding': config.analysis.lag_padding,
            'early_layer_threshold': (
                config.analysis.early_layer_threshold
            ),
        })
        manifest['extraction_validation']['sha256'] = sha256_file(
            validation_path
        )
        if 'tuned-lens' not in config.switches.lens_methods:
            manifest['inputs'].pop('tuned_lens_config', None)
            manifest['inputs'].pop('tuned_lens_params', None)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + '\n',
            encoding='utf8',
        )

    def _rows(self, alias: str):
        spec = get_model_spec(alias)
        rows = []
        source_records = []
        for response in ('paper_time', 'time'):
            for context in ('passage', 'sentence'):
                policy = 'bos' if context == 'passage' else 'bow'
                for lens in ('logit-lens', 'tuned-lens'):
                    source_path = (
                        f'/remote/results/{alias}/{response}/'
                        f'{context}/{lens}/layer-results.tsv'
                    )
                    source_hash = digest(source_path)
                    source_records.append({
                        'path': source_path,
                        'sha256': source_hash,
                    })
                    for score in ('corrected', 'buggy'):
                        if context == 'sentence':
                            best_layer = (
                                2 if lens == 'logit-lens' else 1
                            )
                        else:
                            best_layer = spec.final_layer
                        for layer in range(spec.final_layer + 1):
                            delta = 100.0 - abs(layer - best_layer)
                            rows.append({
                                'analysis': EXPECTED_ANALYSIS,
                                'analysis_mode': EXPECTED_ANALYSIS_MODE,
                                'response_column': response,
                                'model': alias,
                                'context_unit': context,
                                'lens_method': lens,
                                'first_token_policy': policy,
                                'sentence_first_token_policy': policy,
                                'include_embedding_layer': True,
                                'lag_boundary': 'sentence',
                                'lag_padding': 'global-mean',
                                'analysis_lag_boundary': 'sentence',
                                'analysis_lag_padding': 'global-mean',
                                'score_kind': score,
                                'predictor_prefix': (
                                    'internal_layer_surprisal_layer_'
                                    if score == 'corrected'
                                    else (
                                        'internal_layer_surprisal_buggy_'
                                        'layer_'
                                    )
                                ),
                                'layer': layer,
                                'min_layer': 0,
                                'max_layer': spec.final_layer,
                                'relative_depth_block': (
                                    layer / spec.final_layer
                                ),
                                'layer_fraction': (
                                    layer / spec.final_layer
                                ),
                                'input_rows': 10_256,
                                'analysis_rows': 9_771,
                                'excluded_rows': 485,
                                'delta_ll': delta,
                                'ppp_x1000': delta / 10,
                                'is_final_layer': (
                                    layer == spec.final_layer
                                ),
                                'is_embedding_layer': layer == 0,
                                'is_best_layer': layer == best_layer,
                                '_source_path': source_path,
                                '_source_sha256': source_hash,
                            })
        return pd.DataFrame(rows), source_records

    def _write_model(
        self,
        alias: str,
        shared_overrides: dict[str, str] | None = None,
    ) -> None:
        spec = get_model_spec(alias)
        result_dir = (
            self.run_root
            / 'results/layer-factorial'
            / alias
            / 'combined'
        )
        checkpoint_dir = (
            self.run_root
            / 'checkpoints/layer-factorial'
            / alias
        )
        result_dir.mkdir(parents=True)
        checkpoint_dir.mkdir(parents=True)

        layers, source_records = self._rows(alias)
        layer_path = result_dir / 'layer-results.tsv'
        layers.to_csv(layer_path, sep='\t', index=False)
        _, best = validate_and_select_best(layers)
        best.to_csv(
            result_dir / 'best-layers.tsv',
            sep='\t',
            index=False,
        )
        by_response = {
            response: {
                'cells': 8,
                'best_in_first_20pct': int(
                    best.loc[
                        best['response_column'] == response,
                        'best_in_first_20pct',
                    ].sum()
                ),
            }
            for response in ('paper_time', 'time')
        }
        summary = {
            'schema_version': 1,
            'models': [alias],
            'analysis_modes': [EXPECTED_ANALYSIS_MODE],
            'layer_rows': len(layers),
            'factorial_cells': 16,
            'by_response': by_response,
            'inputs': source_records,
        }
        (result_dir / 'summary.json').write_text(
            json.dumps(summary, indent=2) + '\n',
            encoding='utf8',
        )

        final_grid = {
            context: {score: 0.0 for score in ('corrected', 'buggy')}
            for context in ('passage', 'sentence')
        }
        intermediate_grid = {
            context: {score: 1.0 for score in ('corrected', 'buggy')}
            for context in ('passage', 'sentence')
        }
        revision_validated = (
            spec.lens_base_model_revision is not None
        )
        validation = {
            'schema_version': 1,
            'validated': True,
            'model': alias,
            'model_revision_effective': spec.base_model_revision,
            'sentence_manifest_sha256': self.shared_hashes[
                'sentence_manifest'
            ],
            'expected': {
                'rows': 10_256,
                'min_layer': 0,
                'final_layer': spec.final_layer,
                'layers': list(range(spec.final_layer + 1)),
                'final_layer_tolerance': 5e-4,
            },
            'comparisons': {
                'final_logit_vs_tuned_max_abs_difference': (
                    final_grid
                ),
                'intermediate_logit_vs_tuned_max_abs_difference': (
                    intermediate_grid
                ),
            },
            'tuned_lens_identity': {
                'artifact': {
                    'config_sha256': spec.lens_config_sha256,
                    'params_sha256': spec.lens_params_sha256,
                    'num_hidden_layers': spec.final_layer,
                },
                'validation': {
                    'base_model_revision_validated': (
                        revision_validated
                    ),
                },
            },
        }
        validation_path = checkpoint_dir / 'extraction-validation.json'
        validation_path.write_text(
            json.dumps(validation, indent=2) + '\n',
            encoding='utf8',
        )

        shared = dict(self.shared_hashes)
        shared.update(shared_overrides or {})
        input_hashes = {
            **shared,
            'joint': digest(f'joint-{alias}'),
            'tuned_lens_config': spec.lens_config_sha256,
            'tuned_lens_params': spec.lens_params_sha256,
        }
        inputs = {
            name: {
                'path': f'/remote/{name}',
                'sha256': value,
            }
            for name, value in input_hashes.items()
        }
        manifest = {
            'schema_version': 3,
            'model': alias,
            'hf_model_name': spec.hf_name,
            'base_model_revision': spec.base_model_revision,
            'tuned_lens_artifact': spec.lens_artifact,
            'tuned_lens_base_model_revision': (
                spec.lens_base_model_revision
            ),
            'contexts': ['passage', 'sentence'],
            'lens_methods': ['logit-lens', 'tuned-lens'],
            'score_kinds': ['corrected', 'buggy'],
            'response_columns': ['time', 'paper_time'],
            'include_embedding_layer': True,
            'analysis_mode': EXPECTED_ANALYSIS_MODE,
            'analysis_lag_boundary': 'sentence',
            'analysis_lag_padding': 'global-mean',
            'inputs': inputs,
            'extraction_validation': {
                'path': f'/remote/{alias}/extraction-validation.json',
                'sha256': sha256_file(validation_path),
            },
        }
        (checkpoint_dir / 'run-manifest.json').write_text(
            json.dumps(manifest, indent=2) + '\n',
            encoding='utf8',
        )

    def test_full_synthetic_run_writes_and_summarizes(self):
        models = ('gpt2-small', 'pythia-70m')
        for alias in models:
            self._write_model(alias)
        outputs = run_analysis(
            self.run_root, self.output_dir, models
        )
        self.assertEqual(len(outputs['layers']), 320)
        self.assertEqual(len(outputs['best']), 64)
        target = outputs['condition_summary'].loc[
            (
                outputs['condition_summary']['response_column']
                == 'paper_time'
            )
            & (
                outputs['condition_summary']['context_unit']
                == 'sentence'
            )
            & (
                outputs['condition_summary']['lens_method']
                == 'tuned-lens'
            )
            & (
                outputs['condition_summary']['score_kind']
                == 'buggy'
            )
        ]
        self.assertEqual(set(target['early_model_count']), {2})
        self.assertEqual(
            outputs['summary']['counts']['layer_rows'], 320
        )
        expected_files = {
            'all-layer-results.tsv',
            'best-layers.tsv',
            'condition-summary.tsv',
            'model-response-summary.tsv',
            'factor-effects.tsv',
            'integrity-checks.tsv',
            'REPORT.md',
            'summary.json',
        }
        self.assertEqual(
            {path.name for path in self.output_dir.iterdir()},
            expected_files,
        )
        report = (self.output_dir / 'REPORT.md').read_text(
            encoding='utf8'
        )
        self.assertIn('2 of 2 models', report)
        self.assertIn('sentence-bounded context', report)
        self.assertIn('## Full target by model', report)
        self.assertIn('## Logit-lens counterpart by model', report)
        self.assertIn(
            'changing only the decoder from tuned lens to logit lens',
            report,
        )
        tuned_section, logit_section = report.split(
            '## Logit-lens counterpart by model',
            maxsplit=1,
        )
        self.assertIn(
            '| gpt2-small | 12 | 1 | 8.3% | 1 | 8.3% | yes | 100 |',
            tuned_section,
        )
        self.assertIn(
            (
                '| gpt2-small | 12 | 2 | 16.7% | 2 | 16.7% | '
                'yes | 100 |'
            ),
            logit_section,
        )

    def test_tampered_stored_best_is_rejected(self):
        alias = 'gpt2-small'
        self._write_model(alias)
        path = (
            self.run_root
            / 'results/layer-factorial'
            / alias
            / 'combined/best-layers.tsv'
        )
        best = pd.read_csv(path, sep='\t')
        best.loc[0, 'layer'] = 0
        best.to_csv(path, sep='\t', index=False)
        with self.assertRaisesRegex(
            ValueError, 'stored best layers differ'
        ):
            run_analysis(
                self.run_root, self.output_dir, (alias,)
            )

    def test_shared_input_mismatch_is_rejected(self):
        models = ('gpt2-small', 'pythia-70m')
        self._write_model(models[0])
        self._write_model(
            models[1],
            shared_overrides={'text': digest('different-text')},
        )
        with self.assertRaisesRegex(
            ValueError, 'shared input hashes differ'
        ):
            run_analysis(
                self.run_root, self.output_dir, models
            )

    def test_summary_source_tampering_is_rejected(self):
        alias = 'gpt2-small'
        self._write_model(alias)
        path = (
            self.run_root
            / 'results/layer-factorial'
            / alias
            / 'combined/summary.json'
        )
        summary = json.loads(path.read_text(encoding='utf8'))
        summary['inputs'][0]['sha256'] = digest('tampered')
        path.write_text(
            json.dumps(summary, indent=2) + '\n',
            encoding='utf8',
        )
        with self.assertRaisesRegex(
            ValueError, 'summary source records differ'
        ):
            run_analysis(
                self.run_root, self.output_dir, (alias,)
            )

    def test_config_models_threshold_and_reduced_grid_are_honored(self):
        alias = 'gpt2-small'
        config_path = self._write_config(
            lambda payload: (
                payload.update({'models': [alias]}),
                payload['switches'].update({
                    'responses': ['paper_time'],
                    'contexts': ['sentence'],
                    'lens_methods': ['tuned-lens'],
                    'score_kinds': ['buggy'],
                    'include_embedding_layer': False,
                }),
                payload['analysis'].update({
                    'early_layer_threshold': 0.05,
                }),
            )
        )
        self._write_model(alias)
        self._reduce_model_to_config(alias, config_path)

        outputs = run_analysis(
            self.run_root,
            self.output_dir,
            config=config_path,
        )
        self.assertEqual(outputs['summary']['models'], [alias])
        self.assertEqual(
            outputs['summary']['early_definition']['threshold'],
            0.05,
        )
        self.assertEqual(
            outputs['summary']['scopes'], ['transformer-only']
        )
        self.assertEqual(len(outputs['layers']), 12)
        self.assertEqual(len(outputs['best']), 1)
        self.assertTrue(outputs['factor_effects'].empty)
        self.assertEqual(
            outputs['summary']['headline']['target']
            ['transformer-only']['early_models'],
            0,
        )
        self.assertNotIn('baseline', outputs['summary']['headline'])
        report = (self.output_dir / 'REPORT.md').read_text(
            encoding='utf8'
        )
        self.assertIn('No paired factor contrast is available', report)

    def test_effective_configuration_provenance_is_validated(self):
        alias = 'gpt2-small'
        config_path = self._write_config(
            lambda payload: payload.update({'models': [alias]})
        )
        self._write_model(alias)
        manifest_path = self._attach_configuration_provenance(
            alias, config_path
        )
        outputs = run_analysis(
            self.run_root,
            self.output_dir,
            config=config_path,
        )
        self.assertEqual(outputs['summary']['models'], [alias])

        manifest = json.loads(
            manifest_path.read_text(encoding='utf8')
        )
        manifest['configuration']['effective']['runtime']['jobs'] = 2
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + '\n',
            encoding='utf8',
        )
        with self.assertRaisesRegex(
            ValueError, 'effective configuration hash mismatch'
        ):
            run_analysis(
                self.run_root,
                self.output_dir,
                config=config_path,
            )

    def test_empty_overridden_report_note_is_valid_provenance(self):
        alias = 'gpt2-small'
        config_path = self._write_config(
            lambda payload: payload.update({'models': [alias]})
        )
        self._write_model(alias)
        manifest_path = self._attach_configuration_provenance(
            alias, config_path, cli_overrides=['--report-note']
        )
        manifest = json.loads(
            manifest_path.read_text(encoding='utf8')
        )
        effective = manifest['configuration']['effective']
        effective['report_note'] = ''
        encoded = json.dumps(
            effective,
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        ).encode('utf8')
        manifest['configuration']['effective_sha256'] = (
            hashlib.sha256(encoded).hexdigest()
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + '\n',
            encoding='utf8',
        )

        outputs = run_analysis(
            self.run_root,
            self.output_dir,
            config=config_path,
        )
        self.assertEqual(outputs['summary']['models'], [alias])

    def test_raw_configuration_hash_mismatch_is_rejected(self):
        alias = 'gpt2-small'
        config_path = self._write_config(
            lambda payload: payload.update({'models': [alias]})
        )
        self._write_model(alias)
        manifest_path = self._attach_configuration_provenance(
            alias, config_path
        )
        manifest = json.loads(
            manifest_path.read_text(encoding='utf8')
        )
        manifest['configuration']['sha256'] = digest(
            'different-configuration'
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + '\n',
            encoding='utf8',
        )
        with self.assertRaisesRegex(
            ValueError, 'configuration source hash mismatch'
        ):
            run_analysis(
                self.run_root,
                self.output_dir,
                config=config_path,
            )

    def test_recorded_scientific_cli_overrides_define_common_grid(self):
        alias = 'gpt2-small'
        raw_config = self._write_config(
            lambda payload: payload.update({'models': [alias]}),
            name='raw-config.json',
        )

        def configure_effective(payload):
            payload.update({'models': [alias]})
            payload['switches'].update({
                'responses': ['paper_time'],
                'contexts': ['sentence'],
                'lens_methods': ['tuned-lens'],
                'score_kinds': ['buggy'],
                'include_embedding_layer': False,
            })
            payload['analysis']['early_layer_threshold'] = 0.05

        effective_config = self._write_config(
            configure_effective,
            name='effective-reference.json',
        )
        self._write_model(alias)
        self._reduce_model_to_config(alias, effective_config)
        manifest_path = self._attach_configuration_provenance(
            alias,
            raw_config,
            effective_config_path=effective_config,
            cli_overrides=[
                '--config',
                '--response-columns',
                '--contexts',
                '--lens-methods',
                '--score-kinds',
                '--no-include-embedding-layer',
                '--early-layer-threshold',
            ],
        )
        outputs = run_analysis(
            self.run_root,
            self.output_dir,
            config=raw_config,
        )
        self.assertEqual(len(outputs['layers']), 12)
        self.assertEqual(
            outputs['summary']['configuration']['selection_source'],
            'manifest-effective',
        )
        self.assertEqual(
            outputs['summary']['early_definition']['threshold'],
            0.05,
        )

        manifest = json.loads(
            manifest_path.read_text(encoding='utf8')
        )
        manifest['configuration']['cli_overrides'].remove(
            '--early-layer-threshold'
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + '\n',
            encoding='utf8',
        )
        with self.assertRaisesRegex(
            ValueError,
            'early-layer threshold.*without a corresponding',
        ):
            run_analysis(
                self.run_root,
                self.output_dir,
                config=raw_config,
            )

    def test_mixed_modern_and_legacy_manifests_are_rejected(self):
        models = ('gpt2-small', 'pythia-70m')
        config_path = self._write_config(
            lambda payload: payload.update({'models': list(models)})
        )
        for alias in models:
            self._write_model(alias)
        self._attach_configuration_provenance(
            models[0], config_path
        )
        with self.assertRaisesRegex(
            ValueError, 'mix modern and legacy'
        ):
            run_analysis(
                self.run_root,
                self.output_dir,
                config=config_path,
            )

    def test_effective_science_must_match_across_models(self):
        models = ('gpt2-small', 'pythia-70m')
        config_path = self._write_config(
            lambda payload: payload.update({'models': list(models)})
        )
        manifest_paths = {}
        for alias in models:
            self._write_model(alias)
            manifest_paths[alias] = (
                self._attach_configuration_provenance(
                    alias, config_path
                )
            )
        manifest_path = manifest_paths[models[1]]
        manifest = json.loads(
            manifest_path.read_text(encoding='utf8')
        )
        record = manifest['configuration']
        record['effective']['analysis'][
            'early_layer_threshold'
        ] = 0.1
        record['cli_overrides'].append('--early-layer-threshold')
        encoded = json.dumps(
            record['effective'],
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        ).encode('utf8')
        record['effective_sha256'] = hashlib.sha256(
            encoded
        ).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + '\n',
            encoding='utf8',
        )
        with self.assertRaisesRegex(
            ValueError, 'settings differ across selected runs'
        ):
            run_analysis(
                self.run_root,
                self.output_dir,
                config=config_path,
            )

    def test_effective_manifest_schema_and_paper_lags_are_hardened(self):
        alias = 'gpt2-small'
        config_path = self._write_config(
            lambda payload: payload.update({'models': [alias]})
        )
        self._write_model(alias)
        manifest_path = self._attach_configuration_provenance(
            alias, config_path
        )
        baseline = json.loads(
            manifest_path.read_text(encoding='utf8')
        )

        cases = (
            (
                lambda effective: effective['switches'].update({
                    'contexts': [['sentence']]
                }),
                '--contexts',
                'effective contexts must contain only strings',
            ),
            (
                lambda effective: effective['analysis'].update({
                    'lag_boundary': 'text'
                }),
                '--lag-boundary',
                'paper-exact requires lag_boundary=sentence',
            ),
        )
        for mutate, option, message in cases:
            with self.subTest(message=message):
                manifest = json.loads(json.dumps(baseline))
                record = manifest['configuration']
                mutate(record['effective'])
                record['cli_overrides'].append(option)
                encoded = json.dumps(
                    record['effective'],
                    sort_keys=True,
                    separators=(',', ':'),
                    allow_nan=False,
                ).encode('utf8')
                record['effective_sha256'] = hashlib.sha256(
                    encoded
                ).hexdigest()
                manifest_path.write_text(
                    json.dumps(manifest, indent=2) + '\n',
                    encoding='utf8',
                )
                with self.assertRaisesRegex(ValueError, message):
                    run_analysis(
                        self.run_root,
                        self.output_dir,
                        config=config_path,
                    )


if __name__ == '__main__':
    unittest.main()
