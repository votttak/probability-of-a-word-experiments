import unittest

import numpy as np
import torch

from scripts.wordsprobability_cuda_compat import (
    _get_stable_surprisal,
    _move_vocab_masks_to_model_device,
    _stable_weighted_boundary_surprisal,
    cuda_safe_concatenate,
)


class WordsprobabilityCudaCompatTests(unittest.TestCase):
    def test_concatenates_numpy_arrays_and_torch_tensors(self):
        observed = cuda_safe_concatenate(
            [np.asarray([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        )

        np.testing.assert_array_equal(observed, np.asarray([1.0, 2.0, 3.0, 4.0]))

    def test_detaches_tensors_before_numpy_conversion(self):
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)

        observed = cuda_safe_concatenate([tensor, np.asarray([3.0])])

        np.testing.assert_array_equal(observed, np.asarray([1.0, 2.0, 3.0]))

    def test_moves_all_tensor_vocab_masks_to_model_device(self):
        class FakeModel:
            device = torch.device("cpu")
            vocab_masks = {
                "bow": torch.tensor([1.0]),
                "metadata": "unchanged",
            }

        model = FakeModel()
        _move_vocab_masks_to_model_device(model)

        self.assertEqual(model.vocab_masks["bow"].device, model.device)
        self.assertEqual(model.vocab_masks["metadata"], "unchanged")

    def test_surprisal_ignores_mixed_dtype_diagnostic_loss(self):
        logits = torch.tensor(
            [[[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]]],
            dtype=torch.float16,
        )
        labels = torch.tensor([[2, 0]])
        output = {"loss": torch.tensor(0.0, dtype=torch.float32)}

        observed = _get_stable_surprisal(logits, labels, output, labels)
        float_logits = logits.float()
        expected = (
            torch.logsumexp(float_logits, dim=-1)
            - float_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        ).reshape(-1).numpy()

        np.testing.assert_array_equal(observed, expected)

    def test_boundary_mass_is_stable_when_fp16_softmax_underflows(self):
        logits = torch.tensor([[[0.0, -30.0]]], dtype=torch.float16)
        weights = torch.tensor([0.0, 1.0])

        observed = _stable_weighted_boundary_surprisal(logits, weights)

        self.assertTrue(np.isfinite(observed[0]))
        self.assertAlmostEqual(float(observed[0]), 30.0, places=5)


if __name__ == "__main__":
    unittest.main()
