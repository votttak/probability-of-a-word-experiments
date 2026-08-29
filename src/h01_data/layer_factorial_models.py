#!/usr/bin/env python3

"""Pinned model and tuned-lens identities for the layer factorial."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from types import MappingProxyType


TUNED_LENS_REPOSITORY = "AlignmentResearch/tuned-lens"
TUNED_LENS_REPOSITORY_TYPE = "space"
TUNED_LENS_REVISION = "1ac7285852a22309f571c2555efc37375d0c4cda"
TUNED_LENS_PACKAGE_VERSION = "0.2.0"


@dataclass(frozen=True)
class LayerFactorialModel:
    """Immutable identities required for one faithful tuned-lens run."""

    alias: str
    hf_name: str
    final_layer: int
    base_model_revision: str
    base_weight_file: str
    lens_artifact: str
    lens_config_sha256: str
    lens_config_size: int
    lens_params_sha256: str
    lens_params_size: int


# GPT-2 Medium is deliberately absent: the official tuned-lens repository has
# no corresponding artifact. Pythia is deliberately absent because
# Kuribayashi's tuned-lens commands use the deduplicated Pythia checkpoints,
# while the existing project aliases identify different, non-deduplicated
# checkpoints. A silent substitution would not be a replication.
MODEL_SPECS = (
    LayerFactorialModel(
        alias="gpt2-small",
        hf_name="gpt2",
        final_layer=12,
        base_model_revision="e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
        base_weight_file="model.safetensors",
        lens_artifact="gpt2",
        lens_config_sha256=(
            "84764e9fb4aef06fe3007d08531ce7ea9213f8436291089f3e8fa4af36126549"
        ),
        lens_config_size=191,
        lens_params_sha256=(
            "1e0494dcf4a56a77b73b421820941ea948ffae0c6a0391d88c9cb10b48bc19c8"
        ),
        lens_params_size=28_353_795,
    ),
    LayerFactorialModel(
        alias="gpt2-large",
        hf_name="gpt2-large",
        final_layer=36,
        base_model_revision="212095d5832abbf9926672e1c1e8d14312a3be20",
        base_weight_file="model.safetensors",
        lens_artifact="gpt2-large",
        lens_config_sha256=(
            "36c946f2dddabb150648d2d2c954cfa8422dd4e227f1d3216f8b790022eb7d55"
        ),
        lens_config_size=198,
        lens_params_sha256=(
            "291a85a7f524378221e2af0814c2a98c68f740c38993a0b62863f50adb3231db"
        ),
        lens_params_size=236_130_371,
    ),
    LayerFactorialModel(
        alias="gpt2-xl",
        hf_name="gpt2-xl",
        final_layer=48,
        base_model_revision="33cdb5c0db5423c1879b1b9f16c352988e8754a8",
        base_weight_file="pytorch_model.bin",
        lens_artifact="gpt2-xl",
        lens_config_sha256=(
            "1ac180e5f01e9492a9c92f2275882dab3879df3a6856bbba4e40c58f3bb2414a"
        ),
        lens_config_size=195,
        lens_params_sha256=(
            "7c9b8eaf307a87965188d2311ee4d90d3c7868611e20aab263bc8e30b51320b6"
        ),
        lens_params_size=491_849_251,
    ),
)
MODEL_SPECS_BY_ALIAS = MappingProxyType(
    {spec.alias: spec for spec in MODEL_SPECS}
)


def model_aliases() -> tuple[str, ...]:
    return tuple(spec.alias for spec in MODEL_SPECS)


def get_model_spec(alias: str) -> LayerFactorialModel:
    try:
        return MODEL_SPECS_BY_ALIAS[alias]
    except KeyError as error:
        raise KeyError(
            f"unsupported layer-factorial model {alias!r}; expected one of "
            + ", ".join(model_aliases())
        ) from error


def _validate_specs() -> None:
    if len(MODEL_SPECS_BY_ALIAS) != len(MODEL_SPECS):
        raise RuntimeError("layer-factorial aliases must be unique")
    for spec in MODEL_SPECS:
        if spec.final_layer < 1:
            raise RuntimeError(f"invalid final layer for {spec.alias}")
        if spec.base_weight_file not in {
            "model.safetensors",
            "pytorch_model.bin",
        }:
            raise RuntimeError(
                f"unsupported base weight format for {spec.alias}"
            )
        for label, digest in (
            ("config", spec.lens_config_sha256),
            ("params", spec.lens_params_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise RuntimeError(
                    f"invalid {label} SHA-256 for {spec.alias}"
                )


_validate_specs()


FIELDS = tuple(asdict(MODEL_SPECS[0])) + (
    "tuned_lens_repository",
    "tuned_lens_revision",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the pinned layer-factorial model registry"
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list", action="store_true")
    selection.add_argument("--model", choices=model_aliases())
    parser.add_argument("--field", choices=FIELDS)
    args = parser.parse_args(argv)
    if args.list and args.field:
        parser.error("--field requires --model")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        print("\n".join(model_aliases()))
        return 0
    spec = get_model_spec(args.model)
    if args.field == "tuned_lens_repository":
        print(TUNED_LENS_REPOSITORY)
    elif args.field == "tuned_lens_revision":
        print(TUNED_LENS_REVISION)
    elif args.field:
        print(getattr(spec, args.field))
    else:
        payload = asdict(spec)
        payload.update({
            "tuned_lens_repository": TUNED_LENS_REPOSITORY,
            "tuned_lens_repository_type": TUNED_LENS_REPOSITORY_TYPE,
            "tuned_lens_revision": TUNED_LENS_REVISION,
            "tuned_lens_package_version": TUNED_LENS_PACKAGE_VERSION,
        })
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
