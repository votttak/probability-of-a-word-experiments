#!/usr/bin/env python3
"""Run wordsprobability 0.17 safely when model outputs live on CUDA.

The package leaves its vocabulary masks on CPU, concatenates NumPy arrays
with Torch tensors, and reduces FP16 probabilities in native precision. Those
assumptions fail or underflow when Pythia runs on CUDA. Move the masks to the
model device, use float32 log-space reductions, and convert Torch inputs at
the NumPy boundary while leaving model loading and inference in the selected
runtime dtype.
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


def _get_stable_surprisal(
    logits: torch.Tensor,
    labels: torch.Tensor,
    _output: Any,
    _tensor_input: torch.Tensor,
) -> np.ndarray:
    float_logits = logits.float()
    log_normalizer = torch.logsumexp(float_logits, dim=-1)
    target_logits = float_logits.gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    surprisals = log_normalizer - target_logits
    return surprisals.reshape(-1).detach().cpu().numpy()


def _stable_weighted_boundary_surprisal(
    logits: torch.Tensor,
    weights: torch.Tensor,
) -> np.ndarray:
    float_logits = logits.float()
    log_normalizer = torch.logsumexp(float_logits, dim=-1)
    positive = weights > 0
    positive_weights = weights[positive].to(
        device=float_logits.device,
        dtype=float_logits.dtype,
    )
    weighted_logits = (
        float_logits[..., positive] + torch.log(positive_weights)
    )
    result = log_normalizer - torch.logsumexp(weighted_logits, dim=-1)
    return result.reshape(-1).detach().cpu().numpy()


def _get_stable_bow_fix(
    model: Any,
    logits: torch.Tensor,
    _labels: torch.Tensor,
    _output: Any,
    _tensor_input: torch.Tensor,
) -> np.ndarray:
    weights = model.vocab_masks["bow"] + model.vocab_masks["eos"]
    return _stable_weighted_boundary_surprisal(logits, weights)


def _get_stable_bos_fix(
    model: Any,
    logits: torch.Tensor,
    _labels: torch.Tensor,
    _output: Any,
    _tensor_input: torch.Tensor,
) -> np.ndarray:
    weights = (
        model.vocab_masks["mid"]
        + model.vocab_masks["punct"]
        + model.vocab_masks["eos"]
    )
    return _stable_weighted_boundary_surprisal(logits, weights)


def main() -> int:
    from wordsprobability.models.bow_lm import BaseBOWModel

    original = np.concatenate
    original_mask_initializer = BaseBOWModel._initialise_vocab_masks
    original_surprisal = BaseBOWModel._get_surprisal
    original_bow_fix = BaseBOWModel._get_bow_fix
    original_bos_fix = BaseBOWModel._get_bos_fix

    def initialise_device_safe_vocab_masks(model: Any) -> None:
        original_mask_initializer(model)
        _move_vocab_masks_to_model_device(model)

    np.concatenate = cuda_safe_concatenate
    BaseBOWModel._initialise_vocab_masks = initialise_device_safe_vocab_masks
    BaseBOWModel._get_surprisal = staticmethod(_get_stable_surprisal)
    BaseBOWModel._get_bow_fix = _get_stable_bow_fix
    BaseBOWModel._get_bos_fix = _get_stable_bos_fix
    try:
        from wordsprobability.main import main as wordsprobability_main

        result = wordsprobability_main()
        return 0 if result is None else int(result)
    finally:
        BaseBOWModel._get_bos_fix = original_bos_fix
        BaseBOWModel._get_bow_fix = original_bow_fix
        BaseBOWModel._get_surprisal = staticmethod(original_surprisal)
        BaseBOWModel._initialise_vocab_masks = original_mask_initializer
        np.concatenate = original


if __name__ == "__main__":
    raise SystemExit(main())
