import unittest

import numpy as np
import torch

from scripts.wordsprobability_cuda_compat import cuda_safe_concatenate


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


if __name__ == "__main__":
    unittest.main()
