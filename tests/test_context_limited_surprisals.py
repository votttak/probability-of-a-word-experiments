"""CONTEXT-LIMITED: Offline tests for windowing and corrected LM scoring."""

import csv
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch


# CONTEXT-LIMITED: Import project code directly without installing the repo.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from h01_data.get_context_limited_surprisals import (  # noqa: E402
    PREDICTOR_PREFIX,
    build_example,
    build_examples,
    build_rows,
    corrected_word_surprisal,
    score_examples,
    validate_options,
    write_rows_atomic,
)


class FakeTokenizer:
    """CONTEXT-LIMITED: Tiny whitespace tokenizer with BOW-aware token IDs."""

    bos_token_id = 0
    eos_token_id = 0
    pad_token_id = 0

    def __init__(self):
        # CONTEXT-LIMITED: Bare IDs model passage-initial tokens; BOW IDs model
        # the same words when introduced by an ordinary leading space.
        self.bare_ids = {"alpha": 1, "beta": 2, "gamma": 3}
        self.bow_ids = {"alpha": 4, "beta": 5, "gamma": 6}

    def encode(self, text, add_special_tokens=False):
        self.last_add_special_tokens = add_special_tokens
        words = text.split()
        if not words:
            return []
        ids = []
        first_has_space = text.startswith(" ")
        for index, word in enumerate(words):
            is_bow = first_has_space or index > 0
            ids.append((self.bow_ids if is_bow else self.bare_ids)[word])
        return ids


class FakeCausalLM(torch.nn.Module):
    """CONTEXT-LIMITED: Deterministic logits indexed by input token position."""

    def __init__(self, vocabulary_size=7):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.vocabulary_size = vocabulary_size
        self.config = types.SimpleNamespace(max_position_embeddings=32)
        self.seen_use_cache = []

    def forward(self, input_ids, attention_mask, use_cache=True):
        self.seen_use_cache.append(use_cache)
        # CONTEXT-LIMITED: At position p, token ID j receives logit j + p/10.
        # The shared positional offset cancels in softmax but makes shapes clear.
        base = torch.arange(
            self.vocabulary_size, dtype=torch.float32,
            device=input_ids.device)
        positions = torch.arange(
            input_ids.shape[1], dtype=torch.float32,
            device=input_ids.device).view(1, -1, 1) / 10
        logits = base.view(1, 1, -1) + positions
        logits = logits.expand(input_ids.shape[0], -1, -1).clone()
        return types.SimpleNamespace(logits=logits)


class FakeWrapper:
    """CONTEXT-LIMITED: Minimal wordsprobability-compatible model wrapper."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.model = FakeCausalLM()
        # CONTEXT-LIMITED: Bare word tokens are BOS-class; leading-space word
        # tokens are BOW-class. EOS belongs to both package-derived masks.
        self.vocab_masks = {
            "bow": torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.float32),
            "mid": torch.tensor([0, 1, 1, 1, 0, 0, 0], dtype=torch.float32),
            "punct": torch.zeros(7),
            "eos": torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
        }


class ContextLimitedWindowTest(unittest.TestCase):
    """CONTEXT-LIMITED: Verify word units, passage boundaries, and BOS policy."""

    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_options_are_sorted_unique_and_positive(self):
        self.assertEqual(validate_options([4, 1, 4, 2], 3), [1, 2, 4])
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_options([0, 1], 3)
        with self.assertRaisesRegex(ValueError, "batch size"):
            validate_options([1], 0)

    def test_first_word_uses_bos_for_every_context_condition(self):
        examples = build_examples(
            ["alpha", "beta"], 7, [1, 4], self.tokenizer, 0)
        first_word_examples = [example for example in examples
                               if example.word_id == 0]
        self.assertTrue(all(example.uses_bos for example in first_word_examples))
        self.assertTrue(all(example.input_ids == (0, 1)
                            for example in first_word_examples))

    def test_nonempty_truncated_context_has_leading_space_but_no_bos(self):
        example = build_example(
            ["alpha", "beta", "gamma"], 0, 2, 1, self.tokenizer, 0)
        self.assertFalse(example.uses_bos)
        self.assertEqual(example.context_word_count, 1)
        self.assertEqual(example.input_ids, (5, 6))
        self.assertEqual((example.target_start, example.target_end), (1, 1))

    def test_context_never_reaches_before_passage_word_zero(self):
        example = build_example(
            ["alpha", "beta"], 1, 1, 20, self.tokenizer, 0)
        self.assertEqual(example.context_word_count, 1)
        self.assertEqual(example.input_ids, (4, 5))


class ContextLimitedScoringTest(unittest.TestCase):
    """CONTEXT-LIMITED: Verify exact raw/BOS/BOW/EOW correction behavior."""

    def setUp(self):
        self.wrapper = FakeWrapper()

    @staticmethod
    def expected_score(target_id, start_ids, end_ids):
        logits = torch.arange(7, dtype=torch.float32)
        log_probs = torch.log_softmax(logits, dim=-1)
        raw = -log_probs[target_id]
        start = -torch.logsumexp(log_probs[start_ids], dim=0)
        end = -torch.logsumexp(log_probs[end_ids], dim=0)
        return float(raw - start + end)

    def test_correction_helper_uses_raw_minus_start_plus_end(self):
        self.assertEqual(corrected_word_surprisal(5.0, 2.0, 3.0), 6.0)

    def test_empty_context_uses_bos_class(self):
        example = build_example(
            ["alpha"], 0, 0, 4, self.wrapper.tokenizer, 0)
        score = score_examples([example], self.wrapper, batch_size=1)[(0, 0, 4)]
        expected = self.expected_score(
            target_id=1,
            start_ids=torch.tensor([0, 1, 2, 3]),
            end_ids=torch.tensor([0, 4, 5, 6]),
        )
        self.assertAlmostEqual(score, expected, places=6)

    def test_nonempty_context_uses_bow_class(self):
        example = build_example(
            ["alpha", "beta"], 0, 1, 1, self.wrapper.tokenizer, 0)
        score = score_examples([example], self.wrapper, batch_size=1)[(0, 1, 1)]
        expected = self.expected_score(
            target_id=5,
            start_ids=torch.tensor([0, 4, 5, 6]),
            end_ids=torch.tensor([0, 4, 5, 6]),
        )
        self.assertAlmostEqual(score, expected, places=6)

    def test_inference_disables_unused_kv_cache(self):
        examples = build_examples(
            ["alpha", "beta"], 0, [1, 2], self.wrapper.tokenizer, 0)
        score_examples(examples, self.wrapper, batch_size=2)
        self.assertTrue(self.wrapper.model.seen_use_cache)
        self.assertTrue(all(value is False
                            for value in self.wrapper.model.seen_use_cache))

    def test_rows_have_zero_based_keys_and_all_context_columns(self):
        rows = build_rows(
            [["alpha", "beta"], ["gamma"]],
            [1, 2],
            self.wrapper,
            batch_size=3,
        )
        self.assertEqual(
            [(row["text_id"], row["word_id"], row["word"]) for row in rows],
            [(0, 0, "alpha"), (0, 1, "beta"), (1, 0, "gamma")],
        )
        self.assertIn(f"{PREDICTOR_PREFIX}1", rows[0])
        self.assertIn(f"{PREDICTOR_PREFIX}2", rows[0])

    def test_atomic_writer_uses_expected_schema(self):
        rows = [{
            "text_id": 0,
            "word_id": 0,
            "word": "alpha",
            f"{PREDICTOR_PREFIX}1": 1.25,
        }]
        with tempfile.TemporaryDirectory() as directory:
            output_fname = Path(directory) / "context.tsv"
            write_rows_atomic(rows, output_fname, [1])
            with output_fname.open(encoding="utf8", newline="") as input_file:
                written = list(csv.DictReader(input_file, delimiter="\t"))
            leftovers = list(Path(directory).glob("*.tmp"))
        self.assertEqual(written[0]["word"], "alpha")
        self.assertEqual(written[0][f"{PREDICTOR_PREFIX}1"], "1.25")
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
