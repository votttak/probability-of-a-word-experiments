#!/usr/bin/env python3
"""Run wordsprobability 0.17 safely when model outputs live on CUDA.

The package leaves its vocabulary masks on CPU and concatenates NumPy arrays
with Torch tensors. Both assumptions fail when the model runs on CUDA. Move
the masks to the model device and convert Torch inputs explicitly at the
NumPy boundary while leaving model loading and inference on the selected
device.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


_NUMPY_CONCATENATE = np.concatenate


def _numpy_compatible(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def cuda_safe_concatenate(
    arrays: Iterable[Any], *args: Any, **kwargs: Any
) -> np.ndarray:
    """Delegate to NumPy after moving any Torch tensors to host memory."""

    return _NUMPY_CONCATENATE(
        tuple(_numpy_compatible(value) for value in arrays), *args, **kwargs
    )


def _move_vocab_masks_to_model_device(model: Any) -> None:
    model.vocab_masks = {
        name: value.to(model.device) if torch.is_tensor(value) else value
        for name, value in model.vocab_masks.items()
    }


def _get_surprisal_without_dtype_assert(
    logits: torch.Tensor,
    labels: torch.Tensor,
    _output: Any,
    _tensor_input: torch.Tensor,
) -> np.ndarray:
    surprisals = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="none",
    )
    return surprisals.detach().cpu().numpy()


def main() -> int:
    from wordsprobability.models.bow_lm import BaseBOWModel

    original = np.concatenate
    original_mask_initializer = BaseBOWModel._initialise_vocab_masks
    original_surprisal = BaseBOWModel._get_surprisal

    def initialise_device_safe_vocab_masks(model: Any) -> None:
        original_mask_initializer(model)
        _move_vocab_masks_to_model_device(model)

    np.concatenate = cuda_safe_concatenate
    BaseBOWModel._initialise_vocab_masks = initialise_device_safe_vocab_masks
    BaseBOWModel._get_surprisal = staticmethod(_get_surprisal_without_dtype_assert)
    try:
        from wordsprobability.main import main as wordsprobability_main

        result = wordsprobability_main()
        return 0 if result is None else int(result)
    finally:
        BaseBOWModel._get_surprisal = staticmethod(original_surprisal)
        BaseBOWModel._initialise_vocab_masks = original_mask_initializer
        np.concatenate = original


if __name__ == "__main__":
    raise SystemExit(main())
