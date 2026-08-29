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
    base_weight_files: tuple[str, ...]
    base_tokenizer_files: tuple[str, ...]
    lens_artifact: str
    lens_base_model_revision: str | None
    lens_config_sha256: str
    lens_config_size: int
    lens_params_sha256: str
    lens_params_size: int


# GPT-2 Medium is deliberately absent because the official tuned-lens
# repository has no corresponding artifact. The Pythia aliases deliberately
# retain the project's short names while pinning Kuribayashi's distinct
# deduplicated checkpoints at the final training step.
MODEL_SPECS = (
    LayerFactorialModel(
        alias="gpt2-small",
        hf_name="gpt2",
        final_layer=12,
        base_model_revision="e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
        base_weight_files=("model.safetensors",),
        base_tokenizer_files=("vocab.json", "merges.txt", "tokenizer.json"),
        lens_artifact="gpt2",
        lens_base_model_revision="e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
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
        base_weight_files=("model.safetensors",),
        base_tokenizer_files=("vocab.json", "merges.txt", "tokenizer.json"),
        lens_artifact="gpt2-large",
        lens_base_model_revision="212095d5832abbf9926672e1c1e8d14312a3be20",
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
        base_weight_files=("pytorch_model.bin",),
        base_tokenizer_files=("vocab.json", "merges.txt", "tokenizer.json"),
        lens_artifact="gpt2-xl",
        lens_base_model_revision="33cdb5c0db5423c1879b1b9f16c352988e8754a8",
        lens_config_sha256=(
            "1ac180e5f01e9492a9c92f2275882dab3879df3a6856bbba4e40c58f3bb2414a"
        ),
        lens_config_size=195,
        lens_params_sha256=(
            "7c9b8eaf307a87965188d2311ee4d90d3c7868611e20aab263bc8e30b51320b6"
        ),
        lens_params_size=491_849_251,
    ),
    LayerFactorialModel(
        alias="pythia-70m",
        hf_name="EleutherAI/pythia-70m-deduped",
        final_layer=6,
        base_model_revision="9a7c847e93250c8f24d4b7e7134dbf369e8fc9cb",
        base_weight_files=("model.safetensors",),
        base_tokenizer_files=(
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ),
        lens_artifact="EleutherAI/pythia-70m-deduped",
        lens_base_model_revision=None,
        lens_config_sha256=(
            "7d8fbce626f53f8eef6470e47c3352f956fc8240d3f964f8af06ce6ee2cb0389"
        ),
        lens_config_size=177,
        lens_params_sha256=(
            "c363b641564af68e9f9b73af56015b6b5c5b08caea5880787744eaabfab01343"
        ),
        lens_params_size=6_306_803,
    ),
    LayerFactorialModel(
        alias="pythia-160m",
        hf_name="EleutherAI/pythia-160m-deduped",
        final_layer=12,
        base_model_revision="c54a0e0b28cc667b6f278803024438d57f847b5d",
        base_weight_files=("model.safetensors",),
        base_tokenizer_files=(
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ),
        lens_artifact="EleutherAI/pythia-160m-deduped",
        lens_base_model_revision=None,
        lens_config_sha256=(
            "cde1c23fcd004678209150528f2f6e6938357af0f4d7ca59c58161271b704500"
        ),
        lens_config_size=179,
        lens_params_sha256=(
            "1d16b68baa1eb903bbf74eaf859cee5c34361f3e8dff331b17f55b8ce52a2dbe"
        ),
        lens_params_size=28_354_051,
    ),
    LayerFactorialModel(
        alias="pythia-410m",
        hf_name="EleutherAI/pythia-410m-deduped",
        final_layer=24,
        base_model_revision="c66f7467608ffee8fca0d28cf1f46a7574b53cec",
        base_weight_files=("model.safetensors",),
        base_tokenizer_files=(
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ),
        lens_artifact="EleutherAI/pythia-410m-deduped",
        lens_base_model_revision=None,
        lens_config_sha256=(
            "c66232b38ed6e1682dd3fc6f4d8ec89d5e978aff62baf33a2052c0894bce124f"
        ),
        lens_config_size=180,
        lens_params_sha256=(
            "166ea259b35481e1eb2feba50b5ac4d9a8faed47b0937ede0d7bd6d9830dbc95"
        ),
        lens_params_size=100_773_155,
    ),
    LayerFactorialModel(
        alias="pythia-14b",
        hf_name="EleutherAI/pythia-1.4b-deduped",
        final_layer=24,
        base_model_revision="6d1288ca1da05b700367a229ed2000de0eab8c4d",
        base_weight_files=("model.safetensors",),
        base_tokenizer_files=(
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ),
        lens_artifact="EleutherAI/pythia-1.4b-deduped",
        lens_base_model_revision=None,
        lens_config_sha256=(
            "a9c41eafcb120512f4569327b17831d78e789bfbd8ab4d2d7bed2a8f5a1ce033"
        ),
        lens_config_size=180,
        lens_params_sha256=(
            "b56db530d2c0df1bc5916bae58b241cfd4389dd4b1aa29e7210395df97164824"
        ),
        lens_params_size=402_861_347,
    ),
    LayerFactorialModel(
        alias="pythia-28b",
        hf_name="EleutherAI/pythia-2.8b-deduped",
        final_layer=32,
        base_model_revision="346f515745789fe4b4acbc74b105707cc9d5a36d",
        base_weight_files=("pytorch_model.bin",),
        base_tokenizer_files=(
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ),
        lens_artifact="EleutherAI/pythia-2.8b-deduped",
        lens_base_model_revision=None,
        lens_config_sha256=(
            "ca5f7e2395d78885fc1ed654714677cb601efb3e5a55c32fd7efe96eeba61376"
        ),
        lens_config_size=180,
        lens_params_sha256=(
            "314d403f2e1b1bf575ab2e851419d3a687c54138bfae5feace3ff00f1a96fd60"
        ),
        lens_params_size=839_204_003,
    ),
    LayerFactorialModel(
        alias="pythia-69b",
        hf_name="EleutherAI/pythia-6.9b-deduped",
        final_layer=32,
        base_model_revision="f9a3856d3d568fc12eb0c68d5b0cce1be9013642",
        base_weight_files=(
            "pytorch_model.bin.index.json",
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
        ),
        base_tokenizer_files=(
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ),
        lens_artifact="EleutherAI/pythia-6.9b-deduped",
        lens_base_model_revision=None,
        lens_config_sha256=(
            "82bb2ea0da41dd20de51479728974642c3e14605b6f5ab4991437dae164ccce8"
        ),
        lens_config_size=180,
        lens_params_sha256=(
            "e2d00af6f64631b932b089fac5ca031061e207d51132a0f7433bd7a34fb06da0"
        ),
        lens_params_size=2_148_023_459,
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
        for label, filenames in (
            ("weight", spec.base_weight_files),
            ("tokenizer", spec.base_tokenizer_files),
        ):
            if (
                not filenames
                or len(set(filenames)) != len(filenames)
                or any(
                    not filename
                    or filename.startswith(("/", "."))
                    or "/" in filename
                    for filename in filenames
                )
            ):
                raise RuntimeError(
                    f"invalid base {label} files for {spec.alias}"
                )
        for label, revision in (
            ("base-model", spec.base_model_revision),
            ("lens base-model", spec.lens_base_model_revision),
        ):
            if revision is not None and (
                len(revision) != 40
                or any(
                    character not in "0123456789abcdef"
                    for character in revision
                )
            ):
                raise RuntimeError(
                    f"invalid {label} revision for {spec.alias}"
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
