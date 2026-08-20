#!/usr/bin/env python3

"""Canonical model registry for the multi-model internal-layer experiment.

The aliases deliberately match the ten models with completed context-limited
(``C``) results.  Keeping Hugging Face names and layer counts here gives the
Python scorer, Make orchestration, and cluster runner one shared source of
truth without importing PyTorch or Transformers.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from types import MappingProxyType
from typing import Iterable


GPT2_ANCHOR_TOLERANCE = 5e-4
PYTHIA_ANCHOR_TOLERANCE = 5e-4


@dataclass(frozen=True)
class InternalLayerModel:
    """Static metadata needed to score and validate one causal LM."""

    alias: str
    hf_name: str
    family: str
    final_layer: int
    reference_policy: str
    default_anchor_tolerance: float

    @property
    def layer_ids(self) -> tuple[int, ...]:
        """One-based transformer-block outputs, excluding embeddings."""

        return tuple(range(1, self.final_layer + 1))


# Reference policy is a provenance constraint, not only a file-selection hint.
# The validated GPT-2-small run retains its tracked reference.  Every new model
# must generate an ordinary-surprisal reference from the same runtime and model
# snapshot immediately before L scoring.
#
# Fresh Pythia references retain FP16 model inference but use the same stable
# float32 log-space token and boundary reductions as L. This avoids native-FP16
# boundary underflow and supports the same strict anchor gate as GPT-2. It does
# not permit using old tracked Pythia references, whose provenance is unknown.
# Order is intentional: GPT-2 by scale, followed by Pythia by scale.
MODEL_SPECS = (
    InternalLayerModel(
        "gpt2-small", "gpt2", "gpt2", 12,
        "tracked", GPT2_ANCHOR_TOLERANCE,
    ),
    InternalLayerModel(
        "gpt2-medium", "gpt2-medium", "gpt2", 24,
        "fresh", GPT2_ANCHOR_TOLERANCE,
    ),
    InternalLayerModel(
        "gpt2-large", "gpt2-large", "gpt2", 36,
        "fresh", GPT2_ANCHOR_TOLERANCE,
    ),
    InternalLayerModel(
        "gpt2-xl", "gpt2-xl", "gpt2", 48,
        "fresh", GPT2_ANCHOR_TOLERANCE,
    ),
    InternalLayerModel(
        "pythia-70m", "EleutherAI/pythia-70m", "pythia", 6,
        "fresh", PYTHIA_ANCHOR_TOLERANCE,
    ),
    InternalLayerModel(
        "pythia-160m", "EleutherAI/pythia-160m", "pythia", 12,
        "fresh", PYTHIA_ANCHOR_TOLERANCE,
    ),
    InternalLayerModel(
        "pythia-410m", "EleutherAI/pythia-410m", "pythia", 24,
        "fresh", PYTHIA_ANCHOR_TOLERANCE,
    ),
    InternalLayerModel(
        "pythia-14b", "EleutherAI/pythia-1.4b", "pythia", 24,
        "fresh", PYTHIA_ANCHOR_TOLERANCE,
    ),
    InternalLayerModel(
        "pythia-28b", "EleutherAI/pythia-2.8b", "pythia", 32,
        "fresh", PYTHIA_ANCHOR_TOLERANCE,
    ),
    InternalLayerModel(
        "pythia-69b", "EleutherAI/pythia-6.9b", "pythia", 32,
        "fresh", PYTHIA_ANCHOR_TOLERANCE,
    ),
)


def _validate_specs(specs: Iterable[InternalLayerModel]) -> None:
    """Fail at import time if the supposedly canonical registry is invalid."""

    specs = tuple(specs)
    aliases = [spec.alias for spec in specs]
    hf_names = [spec.hf_name for spec in specs]
    if len(specs) != 10:
        raise RuntimeError(
            f"internal-layer registry has {len(specs)} models; expected 10"
        )
    if len(set(aliases)) != len(aliases):
        raise RuntimeError("internal-layer model aliases must be unique")
    if len(set(hf_names)) != len(hf_names):
        raise RuntimeError("internal-layer Hugging Face names must be unique")
    for spec in specs:
        if spec.family not in {"gpt2", "pythia"}:
            raise RuntimeError(
                f"unsupported family {spec.family!r} for {spec.alias}"
            )
        expected_policy = (
            "tracked" if spec.alias == "gpt2-small" else "fresh"
        )
        if spec.reference_policy != expected_policy:
            raise RuntimeError(
                f"reference policy for {spec.alias} must be {expected_policy}"
            )
        if (
            isinstance(spec.final_layer, bool)
            or not isinstance(spec.final_layer, int)
            or spec.final_layer < 1
        ):
            raise RuntimeError(
                f"invalid final layer {spec.final_layer!r} for {spec.alias}"
            )
        if (
            not math.isfinite(spec.default_anchor_tolerance)
            or spec.default_anchor_tolerance <= 0
        ):
            raise RuntimeError(
                "default anchor tolerance must be finite and positive for "
                f"{spec.alias}"
            )


_validate_specs(MODEL_SPECS)
MODEL_SPECS_BY_ALIAS = MappingProxyType(
    {spec.alias: spec for spec in MODEL_SPECS}
)


def model_aliases() -> tuple[str, ...]:
    """Return the canonical aliases in stable scale order."""

    return tuple(spec.alias for spec in MODEL_SPECS)


def get_model_spec(alias: str) -> InternalLayerModel:
    """Resolve one canonical alias, raising a useful error if it is unknown."""

    try:
        return MODEL_SPECS_BY_ALIAS[alias]
    except KeyError as error:
        raise KeyError(
            f"unknown internal-layer model {alias!r}; expected one of "
            + ", ".join(model_aliases())
        ) from error


CLI_FIELDS = (
    "alias",
    "hf_name",
    "family",
    "final_layer",
    "layer_ids",
    "reference_policy",
    "default_anchor_tolerance",
)


def _field_text(spec: InternalLayerModel, field: str) -> str:
    if field == "layer_ids":
        return " ".join(str(layer) for layer in spec.layer_ids)
    value = getattr(spec, field)
    if field == "default_anchor_tolerance":
        return format(value, ".10g")
    return str(value)


def _spec_payload(spec: InternalLayerModel) -> dict:
    payload = asdict(spec)
    payload["layer_ids"] = list(spec.layer_ids)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the canonical internal-layer model registry"
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--list", action="store_true", help="print all model aliases"
    )
    selection.add_argument("--model", choices=model_aliases())
    parser.add_argument(
        "--field",
        choices=CLI_FIELDS,
        help="print one shell-friendly field instead of the model JSON",
    )
    args = parser.parse_args(argv)
    if args.list and args.field is not None:
        parser.error("--field requires --model")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        print("\n".join(model_aliases()))
        return 0

    spec = get_model_spec(args.model)
    if args.field is not None:
        print(_field_text(spec, args.field))
    else:
        print(json.dumps(_spec_payload(spec), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
