#!/usr/bin/env python3

"""Compute internal-layer word surprisal with configurable context and lens.

The legacy defaults deliberately mirror wordsprobability 0.17: one passage per
line, BOS/EOS framing, 1,022-token overlapping chunks with stride 200, and its
weighted word-boundary correction. The factorial experiment additionally
supports sentence-reset context, raw (``surprisal_buggy``) subtoken aggregation,
and a tuned lens. Hidden-state indices 1..N are transformer block outputs.
The legacy default excludes hidden state 0; --include-embedding-layer enables
Kuribayashi et al.'s exact 0..N enumeration.
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import sys
import tempfile

try:
    from .build_natural_stories_sentence_manifest import read_sentence_manifest
    from .get_context_limited_surprisals import (
        load_wordsprobability_model,
        read_texts,
    )
    from .internal_layer_models import get_model_spec, model_aliases
    from .tuned_lens_decoder import (
        inspect_local_tuned_lens_artifact,
        load_local_tuned_lens_decoder,
    )
except ImportError:  # Support direct execution from src/h01_data.
    from build_natural_stories_sentence_manifest import read_sentence_manifest
    from get_context_limited_surprisals import (
        load_wordsprobability_model,
        read_texts,
    )
    from internal_layer_models import get_model_spec, model_aliases
    from tuned_lens_decoder import (
        inspect_local_tuned_lens_artifact,
        load_local_tuned_lens_decoder,
    )


PREDICTOR_PREFIX = "internal_layer_surprisal_layer_"
BUGGY_PREDICTOR_PREFIX = "internal_layer_surprisal_buggy_layer_"
MAX_ENCODED_TOKENS = 1022
CHUNK_STRIDE = 200
# Boundary correction subtracts separately rounded float32 log-probabilities.
# At early logit-lens layers their cancellation can leave a one-ULP negative
# residue (observed as 2**-15 for Pythia); this is still far below the final
# reference-anchor tolerance and is mathematically a zero surprisal.
NEGATIVE_ROUNDOFF_TOLERANCE = 1e-4
PASSAGE_CHECKPOINT_SCHEMA_VERSION = 2
CONTEXT_UNITS = ("passage", "sentence")
FIRST_WORD_POLICIES = ("bos", "bow")
LENS_METHODS = ("logit-lens", "tuned-lens")


@dataclass(frozen=True)
class PassageChunk:
    """One exact wordsprobability-style model window."""

    input_ids: tuple
    retained_token_ids: tuple
    retained_offset: int
    is_final: bool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute corrected full-prefix surprisal with a logit lens"
    )
    parser.add_argument("--input-fname", required=True)
    parser.add_argument("--output-fname", required=True)
    parser.add_argument("--model", required=True, choices=model_aliases())
    parser.add_argument(
        "--hf-model-name",
        help=(
            "explicit Hugging Face repository loaded through the selected "
            "wordsprobability wrapper; used for faithful deduplicated Pythia runs"
        ),
    )
    parser.add_argument(
        "--model-revision",
        help=(
            "pinned Hugging Face base-model revision; a tuned-lens artifact "
            "revision is used automatically when present and must agree when "
            "both are set"
        ),
    )
    parser.add_argument(
        "--context-unit",
        choices=CONTEXT_UNITS,
        default="passage",
        help="reset model context at passages (legacy) or authoritative sentences",
    )
    parser.add_argument(
        "--sentence-map-fname",
        help="validated sentence manifest; required with --context-unit sentence",
    )
    parser.add_argument(
        "--sentence-first-token-policy",
        choices=FIRST_WORD_POLICIES,
        default="bos",
        help=(
            "tokenize a sentence-initial word without a leading space (bos) or "
            "with Kuribayashi's leading-space/BOW framing (bow)"
        ),
    )
    parser.add_argument(
        "--return-buggy-surprisals",
        action="store_true",
        help="also emit raw subtoken-NLL sums for every layer",
    )
    parser.add_argument(
        "--lens-method",
        choices=LENS_METHODS,
        default="logit-lens",
    )
    parser.add_argument(
        "--tuned-lens-path",
        help="explicit local tuned-lens artifact directory; required for tuned-lens",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="transformer block indices to emit; omit for every block",
    )
    parser.add_argument(
        "--include-embedding-layer",
        action="store_true",
        help=(
            "allow and, when --layers is omitted, emit hidden state 0 to "
            "match Kuribayashi et al.'s exact layer enumeration"
        ),
    )
    parser.add_argument(
        "--passage-checkpoint-dir",
        help=(
            "optional resumable per-passage cache; cache identity includes "
            "the scorer, model, layers, runtime, and precision"
        ),
    )
    parser.add_argument(
        "--reference-surprisal-fname",
        help="optional ordinary full-context TSV used to validate the final layer",
    )
    parser.add_argument("--anchor-tolerance", type=float, default=5e-4)
    return parser.parse_args()


def validate_factorial_options(context_unit, sentence_map_fname,
                               first_word_policy, lens_method,
                               tuned_lens_path):
    """Reject ambiguous factor combinations before loading a model."""

    if context_unit == "sentence" and not sentence_map_fname:
        raise ValueError(
            "--sentence-map-fname is required with --context-unit sentence"
        )
    if context_unit == "passage" and sentence_map_fname:
        raise ValueError(
            "--sentence-map-fname is only valid with --context-unit sentence"
        )
    if context_unit == "passage" and first_word_policy != "bos":
        raise ValueError(
            "--sentence-first-token-policy bow requires sentence context"
        )
    if lens_method == "tuned-lens" and not tuned_lens_path:
        raise ValueError("--tuned-lens-path is required for tuned-lens")
    if lens_method == "logit-lens" and tuned_lens_path:
        raise ValueError(
            "--tuned-lens-path cannot be supplied with logit-lens"
        )


def model_layer_count(model):
    """Return the number of transformer blocks advertised by the model."""

    config = getattr(model, "config", None)
    for attribute in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise RuntimeError("Unable to determine the model's transformer-layer count")


def validate_registered_model_layer_count(model_name, model):
    """Require the loaded config to match the canonical model registry."""

    expected = get_model_spec(model_name).final_layer
    observed = model_layer_count(model)
    if observed != expected:
        raise RuntimeError(
            f"Loaded model layer-count mismatch for {model_name}: "
            f"registry expects {expected}, config advertises {observed}"
        )
    return observed


def validate_layers(model, layers, include_embedding_layer=False):
    """Normalize requested hidden-state indices."""

    final_layer = model_layer_count(model)
    minimum_layer = 0 if include_embedding_layer else 1
    if layers is None:
        return list(range(minimum_layer, final_layer + 1))
    layers = sorted(set(layers))
    if not layers or layers[0] < minimum_layer or layers[-1] > final_layer:
        lower_bound = (
            "0 (embedding) with --include-embedding-layer"
            if include_embedding_layer else "1"
        )
        raise ValueError(
            f"layers must be between {lower_bound} and {final_layer}, inclusive; "
            "hidden state 0 is available only with --include-embedding-layer"
        )
    return layers


def logit_lens_modules(model):
    """Resolve the final normalization and vocabulary head for GPT-2/Pythia."""

    transformer = getattr(model, "transformer", None)
    final_norm = getattr(transformer, "ln_f", None)
    output_head = getattr(model, "lm_head", None)
    if final_norm is not None and output_head is not None:
        return final_norm, output_head

    gpt_neox = getattr(model, "gpt_neox", None)
    final_norm = getattr(gpt_neox, "final_layer_norm", None)
    output_head = getattr(model, "embed_out", None)
    if final_norm is not None and output_head is not None:
        return final_norm, output_head

    raise RuntimeError(
        "Unsupported causal-LM architecture: expected GPT-2 or GPT-NeoX "
        "final normalization/output-head attributes"
    )


def log_internal_model_runtime(model_name, wrapper):
    """Record precision, device, and size for an auditable checkpoint."""

    model = wrapper.model
    try:
        first_parameter = next(model.parameters())
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        print(
            "INTERNAL-LAYER model "
            f"name={model_name} parameters={parameter_count} "
            f"dtype={first_parameter.dtype} device={first_parameter.device}",
            file=sys.stderr,
            flush=True,
        )
    except (AttributeError, StopIteration):
        print(
            f"INTERNAL-LAYER model name={model_name} "
            "runtime metadata unavailable",
            file=sys.stderr,
            flush=True,
        )


def _package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unavailable"


def normalized_texts_sha256(texts):
    """Hash the exact normalized passages consumed by the scorer."""

    digest = hashlib.sha256()
    for words in texts:
        digest.update(" ".join(words).encode("utf8"))
        digest.update(b"\n")
    return digest.hexdigest()


def passage_checkpoint_identity(model_name, wrapper, layers, torch,
                                texts=None, context_unit="passage",
                                segmentation_sha256=None,
                                first_word_policy="bos",
                                return_buggy_surprisals=False,
                                lens_method="logit-lens",
                                lens_identity=None):
    """Describe every setting that can change a resumable passage score."""

    model = wrapper.model
    try:
        first_parameter = next(model.parameters())
        dtype = str(first_parameter.dtype)
        device_type = str(first_parameter.device.type)
    except (AttributeError, StopIteration):
        dtype = "unavailable"
        device_type = "unavailable"
    scorer_hash = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    config = getattr(model, "config", None)
    identity = {
        "schema_version": PASSAGE_CHECKPOINT_SCHEMA_VERSION,
        "scorer_sha256": scorer_hash,
        "model_alias": model_name,
        "model_name_or_path": getattr(config, "_name_or_path", None),
        "model_commit": getattr(config, "_commit_hash", None),
        "tokenizer_name_or_path": getattr(
            wrapper.tokenizer, "name_or_path", None
        ),
        "normalized_texts_sha256": (
            normalized_texts_sha256(texts) if texts is not None else None
        ),
        "layers": list(layers),
        "context_unit": context_unit,
        "segmentation_sha256": segmentation_sha256,
        "first_word_policy": first_word_policy,
        "score_kinds": (
            ["corrected", "buggy"]
            if return_buggy_surprisals else ["corrected"]
        ),
        "lens_method": lens_method,
        "lens_identity": lens_identity,
        "max_encoded_tokens": MAX_ENCODED_TOKENS,
        "chunk_stride": CHUNK_STRIDE,
        "negative_roundoff_tolerance": NEGATIVE_ROUNDOFF_TOLERANCE,
        "parameter_dtype": dtype,
        "device_type": device_type,
        "torch_version": getattr(torch, "__version__", "unavailable"),
        "transformers_version": _package_version("transformers"),
        "wordsprobability_version": _package_version("wordsprobability"),
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf8")
    identity["identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return identity


def write_json_atomic(payload, output_fname):
    """Atomically write a deterministic JSON object."""

    output_path = Path(output_fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_fname = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf8") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        os.replace(temporary_fname, output_path)
    except Exception:
        if os.path.exists(temporary_fname):
            os.unlink(temporary_fname)
        raise


def prepare_passage_checkpoint_dir(root_dir, identity):
    """Select an identity-specific cache directory without mixing runtimes."""

    if root_dir is None:
        return None
    root_path = Path(root_dir)
    run_path = root_path / identity["identity_sha256"][:16]
    run_path.mkdir(parents=True, exist_ok=True)
    manifest_path = run_path / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf8") as input_file:
            observed = json.load(input_file)
        if observed != identity:
            raise ValueError(
                f"Passage checkpoint manifest mismatch: {manifest_path}"
            )
    else:
        existing_passages = list(run_path.glob("text-*.tsv"))
        if existing_passages:
            raise ValueError(
                f"Passage checkpoints exist without a manifest in {run_path}"
            )
        write_json_atomic(identity, manifest_path)
    print(
        f"INTERNAL-LAYER passage_checkpoint_dir={run_path}",
        file=sys.stderr,
        flush=True,
    )
    return run_path


def build_passage_chunks(passage, tokenizer, bos_token_id, eos_token_id,
                         max_encoded_tokens=MAX_ENCODED_TOKENS,
                         stride=CHUNK_STRIDE):
    """Reproduce wordsprobability's character-aligned overlapping windows."""

    if not passage:
        return []
    if bos_token_id is None or eos_token_id is None:
        raise ValueError("The selected tokenizer has no BOS/EOS token")
    if max_encoded_tokens < 1 or stride < 2 or stride > max_encoded_tokens:
        raise ValueError("invalid full-context chunk configuration")

    chunks = []
    start_index = 0
    while True:
        encodings = tokenizer(
            passage[start_index:],
            max_length=max_encoded_tokens,
            truncation=True,
            return_offsets_mapping=True,
        )
        encoded_ids = list(encodings["input_ids"])
        offsets = list(encodings["offset_mapping"])
        if not encoded_ids or len(encoded_ids) != len(offsets):
            raise ValueError(
                f"Tokenizer returned invalid chunk at character {start_index}"
            )

        retained_offset = 0 if start_index == 0 else stride - 1
        if retained_offset >= len(encoded_ids):
            raise ValueError("A continuation chunk is shorter than its overlap")
        is_final = offsets[-1][1] + start_index == len(passage)
        chunks.append(PassageChunk(
            input_ids=tuple([bos_token_id] + encoded_ids + [eos_token_id]),
            retained_token_ids=tuple(encoded_ids[retained_offset:]),
            retained_offset=retained_offset,
            is_final=is_final,
        ))
        if is_final:
            break
        if len(offsets) < stride:
            raise ValueError("Tokenizer truncated before the configured overlap")
        advance = offsets[-stride][1]
        if advance <= 0:
            raise ValueError("Tokenizer offsets did not advance the passage")
        start_index += advance
    return chunks


def weighted_boundary_masks(wrapper, device, vocabulary_size):
    """Return the exact weighted BOW/BOS masks used by wordsprobability 0.17."""

    try:
        masks = wrapper.vocab_masks
        bow_weights = masks["bow"] + masks["eos"]
        bos_weights = masks["mid"] + masks["punct"] + masks["eos"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "The installed wordsprobability package lacks required boundary masks"
        ) from error
    if len(bow_weights) != vocabulary_size or len(bos_weights) != vocabulary_size:
        raise RuntimeError(
            "wordsprobability boundary masks do not match the model vocabulary"
        )
    if not bow_weights.any() or not bos_weights.any():
        raise RuntimeError("wordsprobability returned an empty boundary mask")
    return bow_weights.to(device=device), bos_weights.to(device=device)


def weighted_boundary_surprisal(logits, weights, torch,
                                log_normalizer=None):
    """Negative log weighted probability mass for one or more positions."""

    float_logits = logits if logits.dtype == torch.float32 else logits.float()
    if log_normalizer is None:
        log_normalizer = torch.logsumexp(float_logits, dim=-1)
    positive = weights > 0
    positive_weights = weights[positive].to(
        device=float_logits.device, dtype=float_logits.dtype
    )
    weighted_logits = (
        float_logits[..., positive]
        + torch.log(positive_weights)
    )
    log_mass_numerator = torch.logsumexp(weighted_logits, dim=-1)
    result = log_normalizer - log_mass_numerator
    if torch.any(~torch.isfinite(result)):
        raise ValueError("Invalid probability mass for a boundary class")
    return result


def layer_logits(layer_id, final_layer, hidden_states, ordinary_logits,
                 final_norm, output_head, position_offset=0,
                 lens_method="logit-lens", tuned_lens=None):
    """Decode one block output with the selected lens."""

    if layer_id == final_layer:
        # The final hidden state is already normalized; use ordinary logits to
        # prevent an accidental second application of the final norm.
        return ordinary_logits[:, position_offset:, :]
    if lens_method == "tuned-lens":
        if tuned_lens is None:
            raise ValueError("tuned-lens decoding requires a loaded lens")
        return tuned_lens.layer_logits(
            layer_id,
            hidden_states,
            ordinary_logits,
            position_offset=position_offset,
        )
    if lens_method != "logit-lens":
        raise ValueError(f"Unsupported lens method: {lens_method}")
    return output_head(
        final_norm(hidden_states[layer_id][:, position_offset:, :])
    )


def token_word_ids(token_ids, tokenizer, bow_symbol, expected_words,
                   first_word_policy="bos"):
    """Map retained GPT-style tokens to whitespace-word IDs as the package does."""

    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    if len(tokens) != len(token_ids):
        raise RuntimeError("Tokenizer did not return one token string per ID")
    is_bow = [
        isinstance(token, str) and token.startswith(bow_symbol)
        for token in tokens
    ]
    if not is_bow:
        raise ValueError("The score unit has no retained subtokens")
    if first_word_policy == "bos" and is_bow[0]:
        raise ValueError("The first subtoken is not a BOS-class token")
    if first_word_policy == "bow" and not is_bow[0]:
        raise ValueError("The first subtoken is not a leading-space/BOW token")
    if first_word_policy not in FIRST_WORD_POLICIES:
        raise ValueError(f"Unsupported first-word policy: {first_word_policy}")

    word_ids = []
    word_id = 0
    for token_index, begins_word in enumerate(is_bow):
        if token_index > 0 and begins_word:
            word_id += 1
        word_ids.append(word_id)
    if word_id + 1 != expected_words:
        raise ValueError(
            f"Model tokenization produced {word_id + 1} words; "
            f"the project input has {expected_words}"
        )
    is_eow = is_bow[1:] + [True]
    return word_ids, is_bow, is_eow


def aggregate_layer_scores(raw, bow_fix, bos_fix, final_bow_fix, word_ids,
                           is_bow, is_eow, word_count):
    """Apply per-token boundary corrections and sum into project words."""

    token_count = len(word_ids)
    if not (len(raw) == len(bow_fix) == len(bos_fix) == token_count):
        raise RuntimeError("Layer metric lengths do not match retained tokens")
    eow_fix = bow_fix[1:] + [final_bow_fix]
    word_scores = [0.0] * word_count
    for token_index in range(token_count):
        corrected = raw[token_index]
        if is_bow[token_index]:
            corrected -= bow_fix[token_index]
        # A score unit begins in exactly one known start class. Ordinary
        # wordsprobability framing has a non-BOW first token and conditions on
        # BOS. Kuribayashi's leading-space framing has a BOW first token and
        # conditions on BOW. Never subtract both corrections.
        if token_index == 0 and not is_bow[token_index]:
            corrected -= bos_fix[token_index]
        if is_eow[token_index]:
            corrected += eow_fix[token_index]
        word_scores[word_ids[token_index]] += corrected

    for word_id, value in enumerate(word_scores):
        if (
            not math.isfinite(value)
            or value < -NEGATIVE_ROUNDOFF_TOLERANCE
        ):
            raise ValueError(
                f"Invalid corrected surprisal {value} for word {word_id}"
            )
        # Boundary correction combines several float32 log probabilities.
        # Clamp only cancellation-scale negatives; substantive negatives fail.
        word_scores[word_id] = max(0.0, value)
    return word_scores


def aggregate_buggy_layer_scores(raw, word_ids, word_count):
    """Sum uncorrected subtoken NLLs into project words."""

    if len(raw) != len(word_ids):
        raise RuntimeError("Raw metric length does not match retained tokens")
    word_scores = [0.0] * word_count
    for token_index, word_id in enumerate(word_ids):
        value = raw[token_index]
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"Invalid raw surprisal {value} at subtoken {token_index}"
            )
        word_scores[word_id] += value
    if any(not math.isfinite(value) or value < 0 for value in word_scores):
        raise ValueError("Invalid aggregated buggy surprisal")
    return word_scores


def score_passage(words, text_id, wrapper, layers, torch, device,
                  final_norm, output_head, boundary_masks,
                  return_buggy_surprisals=False,
                  first_word_policy="bos", lens_method="logit-lens",
                  tuned_lens=None, allow_multiple_chunks=True):
    """Score one independently framed word sequence at selected blocks."""

    passage = " ".join(words)
    if first_word_policy == "bow":
        passage = " " + passage
    elif first_word_policy != "bos":
        raise ValueError(f"Unsupported first-word policy: {first_word_policy}")
    chunks = build_passage_chunks(
        passage,
        wrapper.tokenizer,
        wrapper.tokenizer.bos_token_id,
        wrapper.tokenizer.eos_token_id,
    )
    if not allow_multiple_chunks and len(chunks) != 1:
        raise ValueError(
            "Sentence exceeds the model's single-window limit; refusing an "
            "unreported within-sentence context reset"
        )
    retained_token_ids = []
    metrics = {
        layer_id: {"raw": [], "bow": [], "bos": [], "final_bow": None}
        for layer_id in layers
    }
    final_layer = model_layer_count(wrapper.model)

    for chunk in chunks:
        tensor_input = torch.tensor(
            [chunk.input_ids], dtype=torch.long, device=device
        )
        output = wrapper.model(
            input_ids=tensor_input,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        ordinary_logits = output.logits
        hidden_states = output.hidden_states
        if hidden_states is None or len(hidden_states) != final_layer + 1:
            raise RuntimeError(
                "Causal LM returned an unexpected number of hidden states"
            )
        if ordinary_logits.ndim != 3 or ordinary_logits.shape[:2] != tensor_input.shape:
            raise RuntimeError("Causal LM returned logits with an unexpected shape")
        if ordinary_logits.shape[-1] != len(boundary_masks[0]):
            raise RuntimeError("Causal LM vocabulary changed during scoring")

        labels = tensor_input[0, 1:]
        for layer_id in layers:
            position_offset = chunk.retained_offset
            logits = layer_logits(
                layer_id,
                final_layer,
                hidden_states,
                ordinary_logits,
                final_norm,
                output_head,
                position_offset=position_offset,
                lens_method=lens_method,
                tuned_lens=tuned_lens,
            )
            shifted = logits[0, :-1, :]
            retained_labels = labels[position_offset:]
            float_logits = (
                shifted
                if shifted.dtype == torch.float32
                else shifted.float()
            )
            log_normalizer = torch.logsumexp(float_logits, dim=-1)
            target_logits = float_logits.gather(
                -1, retained_labels.unsqueeze(-1)
            ).squeeze(-1)
            raw = log_normalizer - target_logits
            bow_fix = weighted_boundary_surprisal(
                float_logits,
                boundary_masks[0],
                torch,
                log_normalizer=log_normalizer,
            )
            bos_fix = weighted_boundary_surprisal(
                float_logits,
                boundary_masks[1],
                torch,
                log_normalizer=log_normalizer,
            )
            retained = slice(None, -1)
            metrics[layer_id]["raw"].extend(
                raw[retained].detach().cpu().tolist()
            )
            metrics[layer_id]["bow"].extend(
                bow_fix[retained].detach().cpu().tolist()
            )
            metrics[layer_id]["bos"].extend(
                bos_fix[retained].detach().cpu().tolist()
            )
            metrics[layer_id]["final_bow"] = float(
                bow_fix[-1].detach().cpu()
            )
            if layer_id != final_layer:
                del logits
            del float_logits
        retained_token_ids.extend(chunk.retained_token_ids)

    word_ids, is_bow, is_eow = token_word_ids(
        retained_token_ids,
        wrapper.tokenizer,
        wrapper.bow_symbol,
        len(words),
        first_word_policy=first_word_policy,
    )
    scores = {}
    buggy_scores = {}
    for layer_id in layers:
        layer_metrics = metrics[layer_id]
        word_scores = aggregate_layer_scores(
            layer_metrics["raw"],
            layer_metrics["bow"],
            layer_metrics["bos"],
            layer_metrics["final_bow"],
            word_ids,
            is_bow,
            is_eow,
            len(words),
        )
        for word_id, value in enumerate(word_scores):
            scores[(text_id, word_id, layer_id)] = value
        if return_buggy_surprisals:
            buggy_word_scores = aggregate_buggy_layer_scores(
                layer_metrics["raw"], word_ids, len(words)
            )
            for word_id, value in enumerate(buggy_word_scores):
                buggy_scores[(text_id, word_id, layer_id)] = value
    if return_buggy_surprisals:
        return scores, buggy_scores
    return scores


def layer_output_fields(layers, return_buggy_surprisals=False):
    """Return the deterministic predictor field order for one artifact."""

    fields = [f"{PREDICTOR_PREFIX}{layer_id}" for layer_id in layers]
    if return_buggy_surprisals:
        fields.extend(
            f"{BUGGY_PREDICTOR_PREFIX}{layer_id}" for layer_id in layers
        )
    return fields


def passage_rows(words, text_id, layers, scores, buggy_scores=None):
    """Convert one passage's keyed scores into stable TSV rows."""

    rows = []
    for word_id, word in enumerate(words):
        row = {"text_id": text_id, "word_id": word_id, "word": word}
        for layer_id in layers:
            key = (text_id, word_id, layer_id)
            if key not in scores:
                raise ValueError(f"Missing internal-layer score key: {key}")
            row[f"{PREDICTOR_PREFIX}{layer_id}"] = scores[key]
            if buggy_scores is not None:
                if key not in buggy_scores:
                    raise ValueError(
                        f"Missing buggy internal-layer score key: {key}"
                    )
                row[f"{BUGGY_PREDICTOR_PREFIX}{layer_id}"] = buggy_scores[key]
        rows.append(row)
    return rows


def read_passage_checkpoint(fname, words, text_id, layers,
                            return_buggy_surprisals=False):
    """Validate and load one completed passage checkpoint."""

    expected_fields = ["text_id", "word_id", "word"] + layer_output_fields(
        layers, return_buggy_surprisals=return_buggy_surprisals
    )
    with open(fname, "r", encoding="utf8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        if reader.fieldnames != expected_fields:
            raise ValueError(
                f"Passage checkpoint has unexpected columns: {fname}"
            )
        rows = list(reader)
    if len(rows) != len(words):
        raise ValueError(
            f"Passage checkpoint {fname} has {len(rows)} rows; "
            f"expected {len(words)}"
        )

    scores = {}
    buggy_scores = {}
    for word_id, (row, word) in enumerate(zip(rows, words)):
        try:
            observed_text_id = int(row["text_id"])
            observed_word_id = int(row["word_id"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Passage checkpoint has invalid keys: {fname}"
            ) from error
        if (
            observed_text_id != text_id
            or observed_word_id != word_id
            or row["word"] != word
        ):
            raise ValueError(
                f"Passage checkpoint key/word mismatch at text {text_id}, "
                f"word {word_id}: {fname}"
            )
        for layer_id in layers:
            column = f"{PREDICTOR_PREFIX}{layer_id}"
            try:
                value = float(row[column])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Passage checkpoint has invalid value in {column}: {fname}"
                ) from error
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"Passage checkpoint has non-finite/negative value: {fname}"
                )
            scores[(text_id, word_id, layer_id)] = value
            if return_buggy_surprisals:
                buggy_column = f"{BUGGY_PREDICTOR_PREFIX}{layer_id}"
                try:
                    buggy_value = float(row[buggy_column])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Passage checkpoint has invalid value in "
                        f"{buggy_column}: {fname}"
                    ) from error
                if not math.isfinite(buggy_value) or buggy_value < 0:
                    raise ValueError(
                        "Passage checkpoint has non-finite/negative buggy "
                        f"value: {fname}"
                    )
                buggy_scores[(text_id, word_id, layer_id)] = buggy_value
    if return_buggy_surprisals:
        return scores, buggy_scores
    return scores


def score_sentence_bounded_text(words, text_id, sentence_units, wrapper,
                                layers, torch, device, final_norm,
                                output_head, boundary_masks,
                                return_buggy_surprisals=False,
                                first_word_policy="bos",
                                lens_method="logit-lens",
                                tuned_lens=None):
    """Score one story as independently BOS-framed authoritative sentences."""

    scores = {}
    buggy_scores = {}
    observed_word_ids = []
    for sentence in sentence_units:
        sentence_words = list(sentence.words)
        sentence_word_ids = list(sentence.word_ids)
        if len(sentence_words) != len(sentence_word_ids) or not sentence_words:
            raise ValueError(
                f"Invalid sentence {sentence.sentence_id} in text {text_id}"
            )
        expected_words = [words[word_id] for word_id in sentence_word_ids]
        if sentence_words != expected_words:
            raise ValueError(
                f"Sentence-map word mismatch in text {text_id}, "
                f"sentence {sentence.sentence_id}"
            )
        loaded_scores = score_passage(
            sentence_words,
            text_id,
            wrapper,
            layers,
            torch,
            device,
            final_norm,
            output_head,
            boundary_masks,
            return_buggy_surprisals=return_buggy_surprisals,
            first_word_policy=first_word_policy,
            lens_method=lens_method,
            tuned_lens=tuned_lens,
            allow_multiple_chunks=False,
        )
        if return_buggy_surprisals:
            local_scores, local_buggy_scores = loaded_scores
        else:
            local_scores = loaded_scores
            local_buggy_scores = None
        for local_word_id, global_word_id in enumerate(sentence_word_ids):
            observed_word_ids.append(global_word_id)
            for layer_id in layers:
                local_key = (text_id, local_word_id, layer_id)
                global_key = (text_id, global_word_id, layer_id)
                if global_key in scores:
                    raise ValueError(
                        f"Duplicate sentence-map word ID: {global_key}"
                    )
                scores[global_key] = local_scores[local_key]
                if local_buggy_scores is not None:
                    buggy_scores[global_key] = local_buggy_scores[local_key]

    if observed_word_ids != list(range(len(words))):
        raise ValueError(
            f"Sentence map does not cover text {text_id} contiguously"
        )
    if return_buggy_surprisals:
        return scores, buggy_scores
    return scores


def score_passages(texts, wrapper, layers, passage_checkpoint_dir=None,
                   model_name="unspecified",
                   return_buggy_surprisals=False,
                   first_word_policy="bos", lens_method="logit-lens",
                   tuned_lens=None, lens_identity=None,
                   context_unit="passage", sentence_map=None,
                   segmentation_sha256=None):
    """Score every nonempty passage and return stable keyed values."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for internal-layer scoring") from error

    model = wrapper.model
    model.eval()
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = getattr(wrapper, "device", torch.device("cpu"))
    final_norm, output_head = logit_lens_modules(model)
    vocabulary_size = getattr(output_head, "out_features", None)
    if not isinstance(vocabulary_size, int):
        vocabulary_size = len(wrapper.tokenizer)
    boundary_masks = weighted_boundary_masks(wrapper, device, vocabulary_size)
    identity = passage_checkpoint_identity(
        model_name,
        wrapper,
        layers,
        torch,
        texts=texts,
        context_unit=context_unit,
        segmentation_sha256=segmentation_sha256,
        first_word_policy=first_word_policy,
        return_buggy_surprisals=return_buggy_surprisals,
        lens_method=lens_method,
        lens_identity=lens_identity,
    )
    checkpoint_run_dir = prepare_passage_checkpoint_dir(
        passage_checkpoint_dir, identity
    )

    scores = {}
    buggy_scores = {}
    with torch.inference_mode():
        for text_id, words in enumerate(texts):
            if not words:
                continue
            checkpoint_fname = (
                checkpoint_run_dir / f"text-{text_id:05d}.tsv"
                if checkpoint_run_dir is not None else None
            )
            if checkpoint_fname is not None and checkpoint_fname.exists():
                loaded_scores = read_passage_checkpoint(
                    checkpoint_fname,
                    words,
                    text_id,
                    layers,
                    return_buggy_surprisals=return_buggy_surprisals,
                )
                action = "reused"
            else:
                if context_unit == "passage":
                    loaded_scores = score_passage(
                        words,
                        text_id,
                        wrapper,
                        layers,
                        torch,
                        device,
                        final_norm,
                        output_head,
                        boundary_masks,
                        return_buggy_surprisals=return_buggy_surprisals,
                        first_word_policy=first_word_policy,
                        lens_method=lens_method,
                        tuned_lens=tuned_lens,
                    )
                elif context_unit == "sentence":
                    if sentence_map is None:
                        raise ValueError(
                            "sentence context requires a validated sentence map"
                        )
                    loaded_scores = score_sentence_bounded_text(
                        words,
                        text_id,
                        sentence_map[text_id],
                        wrapper,
                        layers,
                        torch,
                        device,
                        final_norm,
                        output_head,
                        boundary_masks,
                        return_buggy_surprisals=return_buggy_surprisals,
                        first_word_policy=first_word_policy,
                        lens_method=lens_method,
                        tuned_lens=tuned_lens,
                    )
                else:
                    raise ValueError(f"Unsupported context unit: {context_unit}")
                action = "scored"
            if return_buggy_surprisals:
                passage_scores, passage_buggy_scores = loaded_scores
            else:
                passage_scores = loaded_scores
                passage_buggy_scores = None
            if checkpoint_fname is not None and not checkpoint_fname.exists():
                write_rows_atomic(
                    passage_rows(
                        words,
                        text_id,
                        layers,
                        passage_scores,
                        buggy_scores=passage_buggy_scores,
                    ),
                    checkpoint_fname,
                    layers,
                    return_buggy_surprisals=return_buggy_surprisals,
                )
            overlap = set(scores).intersection(passage_scores)
            if overlap:
                raise ValueError(
                    f"Duplicate internal-layer score key: {next(iter(overlap))}"
                )
            scores.update(passage_scores)
            if passage_buggy_scores is not None:
                buggy_overlap = set(buggy_scores).intersection(
                    passage_buggy_scores
                )
                if buggy_overlap:
                    raise ValueError(
                        "Duplicate buggy internal-layer score key: "
                        f"{next(iter(buggy_overlap))}"
                    )
                buggy_scores.update(passage_buggy_scores)
            print(
                f"INTERNAL-LAYER {action} text={text_id} words={len(words)}",
                file=sys.stderr,
                flush=True,
            )
    if return_buggy_surprisals:
        return scores, buggy_scores
    return scores


def build_rows(texts, wrapper, layers, passage_checkpoint_dir=None,
               model_name="unspecified", return_buggy_surprisals=False,
               first_word_policy="bos", lens_method="logit-lens",
               tuned_lens=None, lens_identity=None,
               context_unit="passage", sentence_map=None,
               segmentation_sha256=None):
    """Create a merge-compatible table from internal-layer scores."""

    loaded_scores = score_passages(
        texts,
        wrapper,
        layers,
        passage_checkpoint_dir=passage_checkpoint_dir,
        model_name=model_name,
        return_buggy_surprisals=return_buggy_surprisals,
        first_word_policy=first_word_policy,
        lens_method=lens_method,
        tuned_lens=tuned_lens,
        lens_identity=lens_identity,
        context_unit=context_unit,
        sentence_map=sentence_map,
        segmentation_sha256=segmentation_sha256,
    )
    if return_buggy_surprisals:
        scores, buggy_scores = loaded_scores
    else:
        scores = loaded_scores
        buggy_scores = None
    rows = []
    for text_id, words in enumerate(texts):
        rows.extend(
            passage_rows(
                words,
                text_id,
                layers,
                scores,
                buggy_scores=buggy_scores,
            )
        )
    return rows


def _sha256_file(fname):
    digest = hashlib.sha256()
    with open(fname, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_final_layer_reference(rows, layers, model, reference_fname,
                                   anchor_tolerance,
                                   return_buggy_surprisals=False):
    """Validate the ordinary final layer against an established checkpoint."""

    if reference_fname is None:
        return {
            "validated": False,
            "reason": "no reference surprisal file supplied",
        }
    if not math.isfinite(anchor_tolerance) or anchor_tolerance < 0:
        raise ValueError("anchor tolerance must be finite and nonnegative")

    final_layer = model_layer_count(model)
    if final_layer not in layers:
        raise ValueError(
            f"Reference validation requires final layer {final_layer}"
        )
    final_column = f"{PREDICTOR_PREFIX}{final_layer}"
    with open(reference_fname, "r", encoding="utf8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        required = {"text_id", "word_id", "word", "surprisal"}
        if return_buggy_surprisals:
            required.add("surprisal_buggy")
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Reference surprisal table lacks required keyed columns"
            )
        reference_rows = list(reader)
    if len(reference_rows) != len(rows):
        raise ValueError(
            f"Reference surprisal has {len(reference_rows)} rows; "
            f"internal layers have {len(rows)}"
        )

    differences = []
    buggy_differences = []
    for index, (row, reference) in enumerate(zip(rows, reference_rows)):
        try:
            reference_key = (
                int(reference["text_id"]),
                int(reference["word_id"]),
            )
            reference_value = float(reference["surprisal"])
            layer_value = float(row[final_column])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid reference value at row {index}"
            ) from error
        row_key = (int(row["text_id"]), int(row["word_id"]))
        if row_key != reference_key or row["word"] != reference["word"]:
            raise ValueError(
                f"Reference key/word mismatch at row {index}: "
                f"{row_key} versus {reference_key}"
            )
        difference = abs(layer_value - reference_value)
        if not math.isfinite(difference):
            raise ValueError(
                f"Non-finite final-layer anchor difference at row {index}"
            )
        differences.append(difference)
        if return_buggy_surprisals:
            buggy_column = f"{BUGGY_PREDICTOR_PREFIX}{final_layer}"
            try:
                buggy_reference_value = float(reference["surprisal_buggy"])
                buggy_layer_value = float(row[buggy_column])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid buggy reference value at row {index}"
                ) from error
            buggy_difference = abs(
                buggy_layer_value - buggy_reference_value
            )
            if not math.isfinite(buggy_difference):
                raise ValueError(
                    f"Non-finite buggy anchor difference at row {index}"
                )
            buggy_differences.append(buggy_difference)

    maximum = max(differences, default=0.0)
    mean = sum(differences) / len(differences) if differences else 0.0
    sorted_differences = sorted(differences)
    p99_index = max(0, math.ceil(0.99 * len(sorted_differences)) - 1)
    p99 = (
        sorted_differences[p99_index] if sorted_differences else 0.0
    )
    if maximum > anchor_tolerance:
        raise ValueError(
            f"{final_column} differs from ordinary surprisal by "
            f"{maximum:.6g}, above tolerance {anchor_tolerance}"
        )
    buggy_maximum = max(buggy_differences, default=0.0)
    buggy_mean = (
        sum(buggy_differences) / len(buggy_differences)
        if buggy_differences else None
    )
    if return_buggy_surprisals and buggy_maximum > anchor_tolerance:
        raise ValueError(
            f"{BUGGY_PREDICTOR_PREFIX}{final_layer} differs from ordinary "
            f"surprisal_buggy by {buggy_maximum:.6g}, above tolerance "
            f"{anchor_tolerance}"
        )
    report = {
        "validated": True,
        "reference_fname": str(Path(reference_fname).resolve()),
        "reference_sha256": _sha256_file(reference_fname),
        "final_layer": final_layer,
        "rows": len(rows),
        "max_abs_difference": maximum,
        "mean_abs_difference": mean,
        "p99_abs_difference": p99,
        "buggy_max_abs_difference": (
            buggy_maximum if return_buggy_surprisals else None
        ),
        "buggy_mean_abs_difference": buggy_mean,
        "tolerance": anchor_tolerance,
    }
    print(
        "INTERNAL-LAYER final_anchor "
        f"layer={final_layer} rows={len(rows)} max_abs={maximum:.9g} "
        f"mean_abs={mean:.9g} p99_abs={p99:.9g} "
        f"tolerance={anchor_tolerance}",
        file=sys.stderr,
        flush=True,
    )
    return report


def write_rows_atomic(rows, output_fname, layers,
                      return_buggy_surprisals=False):
    """Atomically publish only a complete layer-predictor TSV."""

    output_path = Path(output_fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["text_id", "word_id", "word"] + layer_output_fields(
        layers, return_buggy_surprisals=return_buggy_surprisals
    )
    descriptor, temporary_fname = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file, fieldnames=fieldnames, delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_fname, output_path)
    except Exception:
        if os.path.exists(temporary_fname):
            os.unlink(temporary_fname)
        raise


def main():
    args = parse_args()
    validate_factorial_options(
        args.context_unit,
        args.sentence_map_fname,
        args.sentence_first_token_policy,
        args.lens_method,
        args.tuned_lens_path,
    )
    texts = read_texts(args.input_fname)
    sentence_map = None
    segmentation_sha256 = None
    if args.context_unit == "sentence":
        sentence_map, segmentation_sha256 = read_sentence_manifest(
            Path(args.sentence_map_fname), texts
        )
    registered_model = get_model_spec(args.model)
    effective_hf_model_name = args.hf_model_name or registered_model.hf_name
    effective_model_revision = args.model_revision
    tuned_artifact = None
    if args.lens_method == "tuned-lens":
        tuned_artifact = inspect_local_tuned_lens_artifact(
            args.tuned_lens_path
        )
        artifact_model_name = tuned_artifact.config.get(
            "base_model_name_or_path"
        )
        if artifact_model_name != effective_hf_model_name:
            raise ValueError(
                "loaded Hugging Face model name disagrees with the tuned-lens "
                f"artifact: {effective_hf_model_name!r} versus "
                f"{artifact_model_name!r}"
            )
        artifact_revision = tuned_artifact.config.get("base_model_revision")
        if (
            effective_model_revision is not None
            and artifact_revision is not None
            and effective_model_revision != artifact_revision
        ):
            raise ValueError(
                "--model-revision disagrees with the tuned-lens artifact: "
                f"{effective_model_revision!r} versus {artifact_revision!r}"
            )
        if effective_model_revision is None:
            effective_model_revision = artifact_revision
    wrapper = load_wordsprobability_model(
        args.model,
        revision=effective_model_revision,
        hf_model_name=args.hf_model_name,
    )
    loaded_hf_model_name = wrapper.hf_model_name
    loaded_model_revision = wrapper.hf_model_revision
    if loaded_hf_model_name != effective_hf_model_name:
        raise RuntimeError(
            "Loaded Hugging Face model identity mismatch: "
            f"wrapper reports {loaded_hf_model_name!r}, expected "
            f"{effective_hf_model_name!r}"
        )
    if loaded_model_revision is not None:
        effective_model_revision = loaded_model_revision
    validate_registered_model_layer_count(args.model, wrapper.model)
    layers = validate_layers(
        wrapper.model,
        args.layers,
        include_embedding_layer=args.include_embedding_layer,
    )
    tuned_lens = None
    lens_identity = None
    if args.lens_method == "tuned-lens":
        tuned_lens = load_local_tuned_lens_decoder(
            wrapper.model,
            args.tuned_lens_path,
            expected_base_model_name=loaded_hf_model_name,
        )
        lens_identity = tuned_lens.provenance()
    log_internal_model_runtime(args.model, wrapper)
    print(
        f"INTERNAL-LAYER method={args.lens_method} "
        f"hf_model_name={loaded_hf_model_name} "
        f"model_revision={loaded_model_revision or effective_model_revision} "
        f"context_unit={args.context_unit} "
        f"first_word_policy={args.sentence_first_token_policy} "
        f"score_kinds={'corrected,buggy' if args.return_buggy_surprisals else 'corrected'} "
        f"layers={','.join(str(layer) for layer in layers)} "
        f"chunk_tokens={MAX_ENCODED_TOKENS} stride={CHUNK_STRIDE}",
        file=sys.stderr,
        flush=True,
    )
    rows = build_rows(
        texts,
        wrapper,
        layers,
        passage_checkpoint_dir=args.passage_checkpoint_dir,
        model_name=args.model,
        return_buggy_surprisals=args.return_buggy_surprisals,
        first_word_policy=args.sentence_first_token_policy,
        lens_method=args.lens_method,
        tuned_lens=tuned_lens,
        lens_identity=lens_identity,
        context_unit=args.context_unit,
        sentence_map=sentence_map,
        segmentation_sha256=segmentation_sha256,
    )
    anchor_report = validate_final_layer_reference(
        rows,
        layers,
        wrapper.model,
        args.reference_surprisal_fname,
        args.anchor_tolerance,
        return_buggy_surprisals=args.return_buggy_surprisals,
    )
    anchor_report["experiment"] = {
        "model": args.model,
        "hf_model_name_requested": args.hf_model_name,
        "hf_model_name_effective": loaded_hf_model_name,
        "model_revision_requested": args.model_revision,
        "model_revision_loaded": loaded_model_revision,
        "model_revision_effective": effective_model_revision,
        "context_unit": args.context_unit,
        "sentence_first_token_policy": args.sentence_first_token_policy,
        "sentence_manifest_sha256": segmentation_sha256,
        "lens_method": args.lens_method,
        "lens_identity": lens_identity,
        "score_kinds": (
            ["corrected", "buggy"]
            if args.return_buggy_surprisals else ["corrected"]
        ),
        "include_embedding_layer": args.include_embedding_layer,
        "layers": layers,
    }
    write_json_atomic(
        anchor_report, f"{args.output_fname}.anchor.json"
    )
    write_rows_atomic(
        rows,
        args.output_fname,
        layers,
        return_buggy_surprisals=args.return_buggy_surprisals,
    )


if __name__ == "__main__":
    main()
