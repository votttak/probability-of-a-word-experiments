#!/usr/bin/env python3
"""Run wordsprobability 0.17 safely when model outputs live on CUDA.

The package concatenates NumPy arrays with Torch tensors. CPU tensors are
implicitly converted by NumPy, but CUDA tensors raise before the package can
finish a passage. Convert Torch inputs explicitly at that narrow boundary
while leaving model loading and inference on the selected device.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch


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


def main() -> int:
    original = np.concatenate
    np.concatenate = cuda_safe_concatenate
    try:
        from wordsprobability.main import main as wordsprobability_main

        result = wordsprobability_main()
        return 0 if result is None else int(result)
    finally:
        np.concatenate = original


if __name__ == "__main__":
    raise SystemExit(main())
