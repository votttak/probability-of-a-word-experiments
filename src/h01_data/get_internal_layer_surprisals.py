#!/usr/bin/env python3

"""Compute boundary-corrected full-context surprisal with a logit lens.

The scorer deliberately mirrors wordsprobability 0.17: one passage per line,
BOS/EOS framing, 1,022-token overlapping chunks with stride 200, and its
weighted word-boundary correction. Hidden-state indices 1..N are transformer
block outputs; index 0 (the uncontextualized embedding stream) is excluded.
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
    from .get_context_limited_surprisals import (
        SUPPORTED_MODELS,
        load_wordsprobability_model,
        read_texts,
    )
except ImportError:  # Support direct execution from src/h01_data.
    from get_context_limited_surprisals import (
        SUPPORTED_MODELS,
        load_wordsprobability_model,
        read_texts,
    )


PREDICTOR_PREFIX = "internal_layer_surprisal_layer_"
MAX_ENCODED_TOKENS = 1022
CHUNK_STRIDE = 200
NEGATIVE_ROUNDOFF_TOLERANCE = 1e-5
PASSAGE_CHECKPOINT_SCHEMA_VERSION = 1


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
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="transformer block indices to emit; omit for every block",
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


def model_layer_count(model):
    """Return the number of transformer blocks advertised by the model."""

    config = getattr(model, "config", None)
    for attribute in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise RuntimeError("Unable to determine the model's transformer-layer count")


def validate_layers(model, layers):
    """Normalize requested block-output indices."""

    final_layer = model_layer_count(model)
    if layers is None:
        return list(range(1, final_layer + 1))
    layers = sorted(set(layers))
    if not layers or layers[0] < 1 or layers[-1] > final_layer:
        raise ValueError(
            f"layers must be between 1 and {final_layer}, inclusive; "
            "hidden state 0 is the embedding stream"
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
                                texts=None):
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
                 final_norm, output_head, position_offset=0):
    """Decode one block output according to the logit-lens definition."""

    if layer_id == final_layer:
        # The final hidden state is already normalized; use ordinary logits to
        # prevent an accidental second application of the final norm.
        return ordinary_logits[:, position_offset:, :]
    return output_head(
        final_norm(hidden_states[layer_id][:, position_offset:, :])
    )


def token_word_ids(token_ids, tokenizer, bow_symbol, expected_words):
    """Map retained GPT-style tokens to whitespace-word IDs as the package does."""

    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    if len(tokens) != len(token_ids):
        raise RuntimeError("Tokenizer did not return one token string per ID")
    is_bow = [
        isinstance(token, str) and token.startswith(bow_symbol)
        for token in tokens
    ]
    if not is_bow or is_bow[0]:
        raise ValueError("The passage's first subtoken is not a BOS-class token")

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
        if token_index == 0:
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


def score_passage(words, text_id, wrapper, layers, torch, device,
                  final_norm, output_head, boundary_masks):
    """Score one passage at all selected transformer blocks."""

    passage = " ".join(words)
    chunks = build_passage_chunks(
        passage,
        wrapper.tokenizer,
        wrapper.tokenizer.bos_token_id,
        wrapper.tokenizer.eos_token_id,
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
    )
    scores = {}
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
    return scores


def passage_rows(words, text_id, layers, scores):
    """Convert one passage's keyed scores into stable TSV rows."""

    rows = []
    for word_id, word in enumerate(words):
        row = {"text_id": text_id, "word_id": word_id, "word": word}
        for layer_id in layers:
            key = (text_id, word_id, layer_id)
            if key not in scores:
                raise ValueError(f"Missing internal-layer score key: {key}")
            row[f"{PREDICTOR_PREFIX}{layer_id}"] = scores[key]
        rows.append(row)
    return rows


def read_passage_checkpoint(fname, words, text_id, layers):
    """Validate and load one completed passage checkpoint."""

    expected_fields = ["text_id", "word_id", "word"] + [
        f"{PREDICTOR_PREFIX}{layer_id}" for layer_id in layers
    ]
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
    return scores


def score_passages(texts, wrapper, layers, passage_checkpoint_dir=None,
                   model_name="unspecified"):
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
        model_name, wrapper, layers, torch, texts=texts
    )
    checkpoint_run_dir = prepare_passage_checkpoint_dir(
        passage_checkpoint_dir, identity
    )

    scores = {}
    with torch.inference_mode():
        for text_id, words in enumerate(texts):
            if not words:
                continue
            checkpoint_fname = (
                checkpoint_run_dir / f"text-{text_id:05d}.tsv"
                if checkpoint_run_dir is not None else None
            )
            if checkpoint_fname is not None and checkpoint_fname.exists():
                passage_scores = read_passage_checkpoint(
                    checkpoint_fname, words, text_id, layers
                )
                action = "reused"
            else:
                passage_scores = score_passage(
                    words,
                    text_id,
                    wrapper,
                    layers,
                    torch,
                    device,
                    final_norm,
                    output_head,
                    boundary_masks,
                )
                if checkpoint_fname is not None:
                    write_rows_atomic(
                        passage_rows(
                            words, text_id, layers, passage_scores
                        ),
                        checkpoint_fname,
                        layers,
                    )
                action = "scored"
            overlap = set(scores).intersection(passage_scores)
            if overlap:
                raise ValueError(
                    f"Duplicate internal-layer score key: {next(iter(overlap))}"
                )
            scores.update(passage_scores)
            print(
                f"INTERNAL-LAYER {action} text={text_id} words={len(words)}",
                file=sys.stderr,
                flush=True,
            )
    return scores


def build_rows(texts, wrapper, layers, passage_checkpoint_dir=None,
               model_name="unspecified"):
    """Create a merge-compatible table from internal-layer scores."""

    scores = score_passages(
        texts,
        wrapper,
        layers,
        passage_checkpoint_dir=passage_checkpoint_dir,
        model_name=model_name,
    )
    rows = []
    for text_id, words in enumerate(texts):
        rows.extend(passage_rows(words, text_id, layers, scores))
    return rows


def _sha256_file(fname):
    digest = hashlib.sha256()
    with open(fname, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_final_layer_reference(rows, layers, model, reference_fname,
                                   anchor_tolerance):
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
    report = {
        "validated": True,
        "reference_fname": str(Path(reference_fname).resolve()),
        "reference_sha256": _sha256_file(reference_fname),
        "final_layer": final_layer,
        "rows": len(rows),
        "max_abs_difference": maximum,
        "mean_abs_difference": mean,
        "p99_abs_difference": p99,
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


def write_rows_atomic(rows, output_fname, layers):
    """Atomically publish only a complete layer-predictor TSV."""

    output_path = Path(output_fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["text_id", "word_id", "word"] + [
        f"{PREDICTOR_PREFIX}{layer_id}" for layer_id in layers
    ]
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
    texts = read_texts(args.input_fname)
    wrapper = load_wordsprobability_model(args.model)
    layers = validate_layers(wrapper.model, args.layers)
    log_internal_model_runtime(args.model, wrapper)
    print(
        "INTERNAL-LAYER method=logit-lens "
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
    )
    anchor_report = validate_final_layer_reference(
        rows,
        layers,
        wrapper.model,
        args.reference_surprisal_fname,
        args.anchor_tolerance,
    )
    write_json_atomic(
        anchor_report, f"{args.output_fname}.anchor.json"
    )
    write_rows_atomic(rows, args.output_fname, layers)


if __name__ == "__main__":
    main()
