"""CONTEXT-LIMITED: Verify complete multi-subtoken target aggregation."""

from pathlib import Path
import sys
import types
import unittest

import torch


# CONTEXT-LIMITED: Import the scorer directly from the repository source tree.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from h01_data.get_context_limited_surprisals import (  # noqa: E402
    build_example,
    score_examples,
)


class MultiPieceTokenizer:
    """CONTEXT-LIMITED: Map one target word to a BOW plus continuation piece."""

    bos_token_id = 0
    eos_token_id = 0
    pad_token_id = 0

    @staticmethod
    def encode(text, add_special_tokens=False):
        mapping = {
            "alpha": [1],
            " complex": [4, 2],
            " alpha complex": [3, 4, 2],
        }
        return mapping[text]


class PositionSensitiveModel(torch.nn.Module):
    """CONTEXT-LIMITED: Give each sequence position a different distribution."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = types.SimpleNamespace(max_position_embeddings=16)

    def forward(self, input_ids, attention_mask, use_cache=False):
        base = torch.arange(5, dtype=torch.float32, device=input_ids.device)
        scales = 1 + torch.arange(
            input_ids.shape[1], dtype=torch.float32,
            device=input_ids.device)
        logits = scales.view(1, -1, 1) * base.view(1, 1, -1)
        return types.SimpleNamespace(
            logits=logits.expand(input_ids.shape[0], -1, -1).clone())


class MultiPieceWrapper:
    """CONTEXT-LIMITED: Supply word-boundary masks for the toy vocabulary."""

    def __init__(self):
        self.tokenizer = MultiPieceTokenizer()
        self.model = PositionSensitiveModel()
        self.vocab_masks = {
            "bow": torch.tensor([0, 0, 0, 1, 1], dtype=torch.float32),
            "mid": torch.tensor([0, 1, 1, 0, 0], dtype=torch.float32),
            "punct": torch.zeros(5),
            "eos": torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32),
        }


class MultiSubtokenScoringTest(unittest.TestCase):
    """CONTEXT-LIMITED: Sum all target pieces and use the final-piece EOW fix."""

    def test_multi_piece_target_span_and_score(self):
        wrapper = MultiPieceWrapper()
        example = build_example(
            ["alpha", "complex"],
            text_id=0,
            word_id=1,
            context_length=1,
            tokenizer=wrapper.tokenizer,
            bos_token_id=0,
        )
        self.assertEqual(example.input_ids, (3, 4, 2))
        self.assertEqual((example.target_start, example.target_end), (1, 2))

        value = score_examples(
            [example], wrapper, batch_size=1)[(0, 1, 1)]

        # CONTEXT-LIMITED: Reconstruct the expected two-piece NLL, BOW start
        # correction at position 0, and EOW correction after final position 2.
        logits = wrapper.model(
            input_ids=torch.tensor([[3, 4, 2]]),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            use_cache=False,
        ).logits[0]
        raw = torch.nn.functional.cross_entropy(
            logits[0:2], torch.tensor([4, 2]), reduction="sum")
        boundary_ids = torch.tensor([0, 3, 4])
        start_log_probs = torch.log_softmax(logits[0], dim=-1)
        end_log_probs = torch.log_softmax(logits[2], dim=-1)
        start_boundary = -torch.logsumexp(
            start_log_probs[boundary_ids], dim=0)
        end_boundary = -torch.logsumexp(
            end_log_probs[boundary_ids], dim=0)
        expected = float(raw - start_boundary + end_boundary)

        self.assertAlmostEqual(value, expected, places=6)


if __name__ == "__main__":
    unittest.main()
