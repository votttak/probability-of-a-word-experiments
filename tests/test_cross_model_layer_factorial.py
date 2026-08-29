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
                        best_layer = (
                            1 if context == 'sentence'
                            else spec.final_layer
                        )
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


if __name__ == '__main__':
    unittest.main()
