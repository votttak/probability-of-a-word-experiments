"""Pure unit tests for internal-layer chunking, aggregation, and mapping."""

import contextlib
import csv
import io
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from h01_data.get_internal_layer_surprisals import (  # noqa: E402
    aggregate_layer_scores,
    build_passage_chunks,
    logit_lens_modules,
    normalized_texts_sha256,
    parse_args,
    read_passage_checkpoint,
    validate_final_layer_reference,
    validate_layers,
    validate_registered_model_layer_count,
    weighted_boundary_surprisal,
    write_rows_atomic,
)
from h01_data.get_context_limited_surprisals import (  # noqa: E402
    load_wordsprobability_model,
)


class CharacterTokenizer:
    """A deterministic offset tokenizer for overlap tests."""

    def __call__(self, text, max_length, truncation, return_offsets_mapping):
        self.assert_options = (truncation, return_offsets_mapping)
        length = min(len(text), max_length)
        return {
            "input_ids": [ord(character) for character in text[:length]],
            "offset_mapping": [(index, index + 1) for index in range(length)],
        }


class InternalLayerChunkTest(unittest.TestCase):
    def test_overlapping_chunks_keep_every_token_once(self):
        tokenizer = CharacterTokenizer()
        chunks = build_passage_chunks(
            "abcdefghijkl",
            tokenizer,
            bos_token_id=1,
            eos_token_id=2,
            max_encoded_tokens=5,
            stride=2,
        )
        retained = [
            token_id
            for chunk in chunks
            for token_id in chunk.retained_token_ids
        ]
        self.assertEqual(retained, [ord(character) for character in "abcdefghijkl"])
        self.assertEqual([chunk.retained_offset for chunk in chunks], [0, 1, 1])
        self.assertTrue(chunks[-1].is_final)
        self.assertTrue(tokenizer.assert_options[0])
        self.assertTrue(tokenizer.assert_options[1])

    def test_boundary_correction_aggregates_multisubtoken_words(self):
        scores = aggregate_layer_scores(
            raw=[2.0, 3.0, 4.0],
            bow_fix=[0.5, 0.6, 0.7],
            bos_fix=[0.2, 0.2, 0.2],
            final_bow_fix=0.8,
            word_ids=[0, 0, 1],
            is_bow=[False, False, True],
            is_eow=[False, True, True],
            word_count=2,
        )
        self.assertAlmostEqual(scores[0], 5.5)
        self.assertAlmostEqual(scores[1], 4.1)

    def test_only_roundoff_scale_negative_scores_are_clamped(self):
        scores = aggregate_layer_scores(
            raw=[0.0],
            bow_fix=[0.0],
            # Exact one-float32-ULP residue observed for Pythia-160M on CUDA.
            bos_fix=[3.0517578125e-05],
            final_bow_fix=0.0,
            word_ids=[0],
            is_bow=[False],
            is_eow=[True],
            word_count=1,
        )
        self.assertEqual(scores, [0.0])
        with self.assertRaisesRegex(ValueError, "Invalid corrected surprisal"):
            aggregate_layer_scores(
                raw=[0.0],
                bow_fix=[0.0],
                bos_fix=[0.1],
                final_bow_fix=0.0,
                word_ids=[0],
                is_bow=[False],
                is_eow=[True],
                word_count=1,
            )


class InternalLayerMappingTest(unittest.TestCase):
    def test_cli_rejects_context_only_pythia_120b(self):
        argv = [
            "get_internal_layer_surprisals.py",
            "--input-fname", "input.txt",
            "--output-fname", "output.tsv",
            "--model", "pythia-120b",
        ]
        with patch.object(sys, "argv", argv):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    parse_args()
        self.assertEqual(raised.exception.code, 2)

    def test_cli_accepts_explicit_hugging_face_model_name(self):
        argv = [
            "get_internal_layer_surprisals.py",
            "--input-fname", "input.txt",
            "--output-fname", "output.tsv",
            "--model", "pythia-70m",
            "--hf-model-name", "EleutherAI/pythia-70m-deduped",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(
            args.hf_model_name, "EleutherAI/pythia-70m-deduped"
        )

    def test_default_layers_are_block_outputs_only(self):
        model = SimpleNamespace(config=SimpleNamespace(n_layer=12))
        self.assertEqual(validate_layers(model, None), list(range(1, 13)))
        self.assertEqual(
            validate_layers(model, None, include_embedding_layer=True),
            list(range(0, 13)),
        )
        self.assertEqual(
            validate_layers(model, [0, 1, 12], include_embedding_layer=True),
            [0, 1, 12],
        )
        self.assertEqual(validate_layers(model, [12, 1, 6, 6]), [1, 6, 12])
        with self.assertRaisesRegex(ValueError, "include-embedding-layer"):
            validate_layers(model, [0],)
        with self.assertRaisesRegex(ValueError, "between 1 and 12"):
            validate_layers(model, [13])

    def test_loaded_layer_count_must_match_registry(self):
        pythia = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=6)
        )
        self.assertEqual(
            validate_registered_model_layer_count("pythia-70m", pythia),
            6,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "registry expects 12, config advertises 6",
        ):
            validate_registered_model_layer_count("pythia-160m", pythia)

    def test_gpt2_and_pythia_logit_lens_modules(self):
        gpt_norm, gpt_head = object(), object()
        gpt = SimpleNamespace(
            transformer=SimpleNamespace(ln_f=gpt_norm), lm_head=gpt_head
        )
        self.assertEqual(logit_lens_modules(gpt), (gpt_norm, gpt_head))

        pythia_norm, pythia_head = object(), object()
        pythia = SimpleNamespace(
            gpt_neox=SimpleNamespace(final_layer_norm=pythia_norm),
            embed_out=pythia_head,
        )
        self.assertEqual(
            logit_lens_modules(pythia), (pythia_norm, pythia_head)
        )


class WordsProbabilityLoaderTest(unittest.TestCase):
    @staticmethod
    def _fake_modules(get_model, wrapper_class):
        package = ModuleType("wordsprobability")
        models = ModuleType("wordsprobability.models")
        models.MODELS = {"pythia-70m": wrapper_class}
        models.get_model = get_model
        package.models = models
        return {
            "wordsprobability": package,
            "wordsprobability.models": models,
        }

    def test_hf_override_reuses_wrapper_with_exact_revision(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                _name_or_path="EleutherAI/pythia-70m-deduped",
                _commit_hash="immutable-revision",
            ),
            device=torch.device("cpu"),
            eval=Mock(),
        )
        tokenizer = SimpleNamespace(
            bos_token_id=0,
            eos_token_id=1,
            name_or_path="EleutherAI/pythia-70m-deduped",
        )
        model_loader = Mock(return_value=model)
        tokenizer_loader = Mock(return_value=tokenizer)

        class FakeWrapper:
            model_name = "EleutherAI/pythia-70m"
            model_cls = SimpleNamespace(from_pretrained=model_loader)
            tokenizer_cls = SimpleNamespace(from_pretrained=tokenizer_loader)

            def _initialise_vocab_masks(self):
                self.vocab_masks = {"ready": True}

        get_model = Mock()
        modules = self._fake_modules(get_model, FakeWrapper)
        with patch.dict(sys.modules, modules), patch.object(
            torch.cuda, "is_available", return_value=False
        ):
            wrapper = load_wordsprobability_model(
                "pythia-70m",
                revision="immutable-revision",
                hf_model_name="EleutherAI/pythia-70m-deduped",
            )

        get_model.assert_not_called()
        model_loader.assert_called_once_with(
            "EleutherAI/pythia-70m-deduped",
            revision="immutable-revision",
        )
        tokenizer_loader.assert_called_once_with(
            "EleutherAI/pythia-70m-deduped",
            revision="immutable-revision",
        )
        self.assertEqual(
            wrapper.hf_model_name, "EleutherAI/pythia-70m-deduped"
        )
        self.assertEqual(wrapper.hf_model_revision, "immutable-revision")
        self.assertEqual(wrapper.vocab_masks, {"ready": True})

    def _load_override_with_actual_identity(self, name, revision):
        model = SimpleNamespace(
            config=SimpleNamespace(
                _name_or_path=name,
                _commit_hash=revision,
            ),
            device=torch.device("cpu"),
            eval=Mock(),
        )
        tokenizer = SimpleNamespace(
            bos_token_id=0,
            eos_token_id=1,
            name_or_path=name,
        )

        class FakeWrapper:
            model_name = "EleutherAI/pythia-70m"
            model_cls = SimpleNamespace(
                from_pretrained=Mock(return_value=model)
            )
            tokenizer_cls = SimpleNamespace(
                from_pretrained=Mock(return_value=tokenizer)
            )

            def _initialise_vocab_masks(self):
                self.vocab_masks = {"ready": True}

        modules = self._fake_modules(Mock(), FakeWrapper)
        with patch.dict(sys.modules, modules), patch.object(
            torch.cuda, "is_available", return_value=False
        ):
            return load_wordsprobability_model(
                "pythia-70m",
                revision="immutable-revision",
                hf_model_name="EleutherAI/pythia-70m-deduped",
            )

    def test_explicit_hf_name_must_match_loaded_config(self):
        with self.assertRaisesRegex(RuntimeError, "model name mismatch"):
            self._load_override_with_actual_identity(
                "EleutherAI/pythia-70m", "immutable-revision"
            )

    def test_explicit_revision_must_match_loaded_config(self):
        with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
            self._load_override_with_actual_identity(
                "EleutherAI/pythia-70m-deduped", "different-revision"
            )

    def test_default_loader_path_is_unchanged(self):
        legacy = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(_name_or_path="gpt2")),
            tokenizer=object(),
            vocab_masks={},
            model_name="gpt2",
        )

        class FakeWrapper:
            model_name = "gpt2"

        get_model = Mock(return_value=legacy)
        modules = self._fake_modules(get_model, FakeWrapper)
        with patch.dict(sys.modules, modules):
            observed = load_wordsprobability_model("pythia-70m")

        self.assertIs(observed, legacy)
        get_model.assert_called_once_with("pythia-70m")
        self.assertEqual(observed.hf_model_name, "gpt2")
        self.assertIsNone(observed.hf_model_revision)


class InternalLayerPersistenceTest(unittest.TestCase):
    def test_weighted_boundary_uses_stable_weighted_mass(self):
        logits = torch.tensor([[0.2, -0.3, 1.1]], dtype=torch.float32)
        weights = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        observed = weighted_boundary_surprisal(logits, weights, torch)
        probabilities = torch.softmax(logits, dim=-1)
        expected = -torch.log(
            probabilities[:, 1] + 2.0 * probabilities[:, 2]
        )
        self.assertTrue(torch.allclose(observed, expected, atol=1e-6))

    def test_normalized_text_hash_tracks_scored_words(self):
        first = normalized_texts_sha256([["a", "b"], ["c"]])
        self.assertEqual(first, normalized_texts_sha256([["a", "b"], ["c"]]))
        self.assertNotEqual(first, normalized_texts_sha256([["a"], ["b", "c"]]))

    def test_passage_checkpoint_is_strictly_validated(self):
        rows = [
            {
                "text_id": 0,
                "word_id": 0,
                "word": "hello",
                "internal_layer_surprisal_layer_1": 1.25,
            },
            {
                "text_id": 0,
                "word_id": 1,
                "word": "world",
                "internal_layer_surprisal_layer_1": 2.5,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            fname = Path(directory) / "text-00000.tsv"
            write_rows_atomic(rows, fname, [1])
            scores = read_passage_checkpoint(
                fname, ["hello", "world"], 0, [1]
            )
            self.assertEqual(scores[(0, 1, 1)], 2.5)
            with self.assertRaisesRegex(ValueError, "key/word mismatch"):
                read_passage_checkpoint(
                    fname, ["hello", "changed"], 0, [1]
                )

    def test_final_layer_reference_report_and_tolerance(self):
        rows = [
            {
                "text_id": 0,
                "word_id": 0,
                "word": "a",
                "internal_layer_surprisal_layer_2": 1.0001,
            },
            {
                "text_id": 0,
                "word_id": 1,
                "word": "b",
                "internal_layer_surprisal_layer_2": 2.0002,
            },
        ]
        model = SimpleNamespace(config=SimpleNamespace(n_layer=2))
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.tsv"
            with reference.open("w", encoding="utf8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["text_id", "word_id", "word", "surprisal"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows([
                    {
                        "text_id": 0, "word_id": 0,
                        "word": "a", "surprisal": 1.0,
                    },
                    {
                        "text_id": 0, "word_id": 1,
                        "word": "b", "surprisal": 2.0,
                    },
                ])
            report = validate_final_layer_reference(
                rows, [2], model, reference, 5e-4
            )
            self.assertTrue(report["validated"])
            self.assertAlmostEqual(report["max_abs_difference"], 2e-4)
            self.assertAlmostEqual(report["p99_abs_difference"], 2e-4)
            with self.assertRaisesRegex(ValueError, "above tolerance"):
                validate_final_layer_reference(
                    rows, [2], model, reference, 1e-5
                )


if __name__ == "__main__":
    unittest.main()
