#!/usr/bin/env python3

"""CONTEXT-LIMITED: Compute word surprisal from fixed preceding-word windows.

CONTEXT-LIMITED: Every input line is one passage.  For each target word and
requested context length, the scorer keeps at most that many preceding
whitespace-delimited words, resets only at the passage boundary, and emits a
stable ``(text_id, word_id)`` keyed TSV.  Word probabilities use the same
boundary corrections as the project's installed ``wordsprobability`` package.
"""

import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile


# CONTEXT-LIMITED: These names exactly match the models accepted by the
# existing ``wordsprobability`` command in the main Makefile.
SUPPORTED_MODELS = (
    "gpt2-small",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-14b",
    "pythia-28b",
    "pythia-69b",
    "pythia-120b",
)
DEFAULT_CONTEXT_LENGTHS = (1, 2, 4)
PREDICTOR_PREFIX = "context_limited_surprisal_context_"


@dataclass(frozen=True)
class ContextExample:
    """CONTEXT-LIMITED: One independently scored context/target window."""

    text_id: int
    word_id: int
    context_length: int
    context_word_count: int
    input_ids: tuple
    target_start: int
    target_end: int
    uses_bos: bool


def parse_args():
    """CONTEXT-LIMITED: Parse the opt-in scorer's command-line interface."""

    parser = argparse.ArgumentParser(
        description="Compute corrected LM surprisal with fixed word contexts"
    )
    parser.add_argument("--input-fname", required=True)
    parser.add_argument("--output-fname", required=True)
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTEXT_LENGTHS),
        help="maximum numbers of preceding whitespace-delimited words",
    )
    # CONTEXT-LIMITED: Batching amortizes per-target forward passes without
    # changing any score, so it is a runtime option rather than a file tag.
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def validate_options(context_lengths, batch_size):
    """CONTEXT-LIMITED: Normalize contexts and reject unsafe configurations."""

    # CONTEXT-LIMITED: Sorting and deduplication make output columns
    # deterministic regardless of how the lengths were written on the CLI.
    context_lengths = sorted(set(context_lengths))
    if not context_lengths or context_lengths[0] < 1:
        raise ValueError("context lengths must be positive integers")
    if batch_size < 1:
        raise ValueError("batch size must be at least 1")
    return context_lengths


def read_texts(fname):
    """CONTEXT-LIMITED: Read one passage per line using project word units."""

    # CONTEXT-LIMITED: Blank lines retain their zero-based text_id even though
    # they contain no targets, matching the n-gram checkpoint convention.
    with open(fname, "r", encoding="utf8") as input_file:
        return [line.strip().split() for line in input_file]


def _encode(tokenizer, text):
    """CONTEXT-LIMITED: Tokenize without silently adding model special tokens."""

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(token_ids, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in token_ids):
        raise ValueError(f"Tokenizer returned invalid IDs for {text!r}: {token_ids!r}")
    return token_ids


def build_example(words, text_id, word_id, context_length, tokenizer,
                  bos_token_id):
    """CONTEXT-LIMITED: Construct one verified context/target token sequence."""

    target_word = words[word_id]
    context_start = max(0, word_id - context_length)
    context_words = words[context_start:word_id]

    if context_words:
        # CONTEXT-LIMITED: A nonempty truncated window starts directly with its
        # oldest retained word.  Deliberately do not prepend BOS mid-passage.
        # Its leading space preserves ordinary in-passage BOW tokenization.
        context_text = " " + " ".join(context_words)
        target_text = " " + target_word
        input_ids = _encode(tokenizer, context_text + target_text)
        target_ids = _encode(tokenizer, target_text)

        # CONTEXT-LIMITED: Suffix equality catches tokenizer boundary behavior
        # that would otherwise assign the wrong subtoken losses to the target.
        if not target_ids or input_ids[-len(target_ids):] != target_ids:
            raise ValueError(
                "Target tokenization is not a suffix of its context window "
                f"for text {text_id}, word {word_id}, context {context_length}"
            )
        target_start = len(input_ids) - len(target_ids)
        if target_start < 1:
            raise ValueError(
                "A nonempty word context produced no preceding model token "
                f"for text {text_id}, word {word_id}"
            )
        uses_bos = False
    else:
        # CONTEXT-LIMITED: BOS is used only when the retained context is empty,
        # following the reference implementation's explicit methodological note.
        target_ids = _encode(tokenizer, target_word)
        if not target_ids:
            raise ValueError(
                f"Tokenizer returned no target IDs for text {text_id}, word {word_id}"
            )
        if bos_token_id is None:
            raise ValueError("The selected tokenizer has no BOS/EOS token for empty context")
        input_ids = [bos_token_id] + target_ids
        target_start = 1
        uses_bos = True

    target_end = len(input_ids) - 1
    return ContextExample(
        text_id=text_id,
        word_id=word_id,
        context_length=context_length,
        context_word_count=len(context_words),
        input_ids=tuple(input_ids),
        target_start=target_start,
        target_end=target_end,
        uses_bos=uses_bos,
    )


def build_examples(words, text_id, context_lengths, tokenizer, bos_token_id):
    """CONTEXT-LIMITED: Build every target/window pair for one passage."""

    return [
        build_example(
            words,
            text_id,
            word_id,
            context_length,
            tokenizer,
            bos_token_id,
        )
        for word_id in range(len(words))
        for context_length in context_lengths
    ]


def corrected_word_surprisal(raw_surprisal, start_boundary_surprisal,
                              end_boundary_surprisal):
    """CONTEXT-LIMITED: Apply the published word-boundary correction."""

    # CONTEXT-LIMITED: This is the exact aggregation used by
    # wordsprobability: raw - BOW/BOS correction + end-of-word correction.
    return raw_surprisal - start_boundary_surprisal + end_boundary_surprisal


def load_wordsprobability_model(model_name):
    """CONTEXT-LIMITED: Load one compatible model wrapper for the whole run."""

    # CONTEXT-LIMITED: Import lazily so pure unit tests never initialize or
    # download Transformer weights.
    try:
        from wordsprobability.models import get_model
    except ImportError as error:
        raise RuntimeError(
            "wordsprobability is required; install the same package used by "
            "the main surprisal pipeline"
        ) from error

    wrapper = get_model(model_name)
    required_attributes = ("model", "tokenizer", "vocab_masks")
    missing = [name for name in required_attributes if not hasattr(wrapper, name)]
    if missing:
        raise RuntimeError(
            "The installed wordsprobability model wrapper is incompatible; "
            f"missing: {', '.join(missing)}"
        )
    return wrapper


def _model_max_positions(model):
    """CONTEXT-LIMITED: Find the causal model's hard positional limit."""

    config = getattr(model, "config", None)
    for attribute in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = getattr(config, attribute, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _boundary_masks(wrapper, device, vocabulary_size):
    """CONTEXT-LIMITED: Reuse wordsprobability's exact vocabulary partitions."""

    try:
        masks = wrapper.vocab_masks
        bow_mask = (masks["bow"] + masks["eos"]) > 0
        bos_mask = (masks["mid"] + masks["punct"] + masks["eos"]) > 0
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "The installed wordsprobability package lacks required boundary masks"
        ) from error

    if len(bow_mask) != vocabulary_size or len(bos_mask) != vocabulary_size:
        raise RuntimeError(
            "wordsprobability boundary masks do not match the model vocabulary"
        )
    if not bow_mask.any() or not bos_mask.any():
        raise RuntimeError("wordsprobability returned an empty boundary mask")
    return bow_mask.to(device=device), bos_mask.to(device=device)


def _boundary_surprisal(logits, mask, torch):
    """CONTEXT-LIMITED: Return negative log mass assigned to a token class."""

    # CONTEXT-LIMITED: Compute in float32 even when model weights use a lower
    # precision; boundary mass can otherwise underflow for large vocabularies.
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    return -torch.logsumexp(log_probs[mask], dim=0)


def score_examples(examples, wrapper, batch_size):
    """CONTEXT-LIMITED: Batch-score examples and restore their stable keys."""

    if not examples:
        return {}

    # CONTEXT-LIMITED: Torch is imported only for actual scoring so schema and
    # windowing tests remain lightweight and network-free.
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for context-limited scoring") from error

    model = wrapper.model
    model.eval()
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = getattr(wrapper, "device", torch.device("cpu"))

    tokenizer = wrapper.tokenizer
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.bos_token_id
    if pad_token_id is None:
        raise ValueError("The selected tokenizer has no usable padding token ID")

    max_positions = _model_max_positions(model)
    too_long = [example for example in examples
                if max_positions is not None and len(example.input_ids) > max_positions]
    if too_long:
        example = too_long[0]
        raise ValueError(
            f"Context window for text {example.text_id}, word {example.word_id}, "
            f"context {example.context_length} needs {len(example.input_ids)} "
            f"tokens but the model limit is {max_positions}"
        )

    # CONTEXT-LIMITED: Length sorting reduces padding while the dictionary keys
    # below restore the original text/word/context association.
    ordered_examples = sorted(examples, key=lambda example: len(example.input_ids))
    scores = {}
    bow_mask = None
    bos_mask = None

    with torch.inference_mode():
        for batch_start in range(0, len(ordered_examples), batch_size):
            batch = ordered_examples[batch_start:batch_start + batch_size]
            maximum_length = max(len(example.input_ids) for example in batch)
            input_ids = torch.full(
                (len(batch), maximum_length),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros(
                (len(batch), maximum_length),
                dtype=torch.long,
                device=device,
            )
            for row_index, example in enumerate(batch):
                sequence = torch.tensor(
                    example.input_ids, dtype=torch.long, device=device)
                input_ids[row_index, :len(sequence)] = sequence
                attention_mask[row_index, :len(sequence)] = 1

            # CONTEXT-LIMITED: KV caching is unused for full-window scoring and
            # would consume substantial extra memory, especially for large LMs.
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = output.logits if hasattr(output, "logits") else output[0]
            if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
                raise RuntimeError(
                    "Causal LM returned logits with an unexpected batch/sequence shape"
                )

            if bow_mask is None:
                bow_mask, bos_mask = _boundary_masks(
                    wrapper, device, logits.shape[-1])

            for row_index, example in enumerate(batch):
                # CONTEXT-LIMITED: Enforce the same one-start-class invariant as
                # wordsprobability before applying a BOW or BOS correction.
                first_target_id = int(input_ids[
                    row_index, example.target_start].item())
                belongs_to_bow = bool(bow_mask[first_target_id].item())
                belongs_to_bos = bool(bos_mask[first_target_id].item())
                if example.uses_bos:
                    valid_start = belongs_to_bos and not belongs_to_bow
                else:
                    valid_start = belongs_to_bow and not belongs_to_bos
                if not valid_start:
                    raise ValueError(
                        "Target begins in the wrong boundary class for text "
                        f"{example.text_id}, word {example.word_id}, context "
                        f"{example.context_length}"
                    )

                # CONTEXT-LIMITED: Each target token is predicted by the logit
                # immediately before it; all retained context tokens are ignored.
                prediction_logits = logits[
                    row_index,
                    example.target_start - 1:example.target_end,
                    :,
                ]
                target_ids = input_ids[
                    row_index,
                    example.target_start:example.target_end + 1,
                ]
                raw_surprisal = torch.nn.functional.cross_entropy(
                    prediction_logits.float(), target_ids, reduction="sum")

                # CONTEXT-LIMITED: Empty contexts use the BOS vocabulary class;
                # truncated nonempty contexts use the ordinary BOW class.
                start_mask = bos_mask if example.uses_bos else bow_mask
                start_boundary = _boundary_surprisal(
                    logits[row_index, example.target_start - 1, :],
                    start_mask,
                    torch,
                )
                end_boundary = _boundary_surprisal(
                    logits[row_index, example.target_end, :],
                    bow_mask,
                    torch,
                )
                corrected = corrected_word_surprisal(
                    raw_surprisal, start_boundary, end_boundary)
                value = float(corrected.detach().cpu())

                if not math.isfinite(value) or value < 0:
                    raise ValueError(
                        f"Invalid corrected surprisal {value} for text "
                        f"{example.text_id}, word {example.word_id}, context "
                        f"{example.context_length}"
                    )
                key = (example.text_id, example.word_id, example.context_length)
                if key in scores:
                    raise ValueError(f"Duplicate context-limited score key: {key}")
                scores[key] = value

    return scores


def build_rows(texts, context_lengths, wrapper, batch_size):
    """CONTEXT-LIMITED: Score passages and build merge-compatible TSV rows."""

    tokenizer = wrapper.tokenizer
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        # CONTEXT-LIMITED: GPT-style models commonly use EOS as BOS; this
        # mirrors wordsprobability's own wrapper attributes.
        bos_token_id = tokenizer.eos_token_id

    rows = []
    for text_id, words in enumerate(texts):
        examples = build_examples(
            words, text_id, context_lengths, tokenizer, bos_token_id)
        scores = score_examples(examples, wrapper, batch_size)

        for word_id, word in enumerate(words):
            row = {"text_id": text_id, "word_id": word_id, "word": word}
            for context_length in context_lengths:
                column = f"{PREDICTOR_PREFIX}{context_length}"
                key = (text_id, word_id, context_length)
                if key not in scores:
                    raise ValueError(f"Missing context-limited score key: {key}")
                row[column] = scores[key]
            rows.append(row)
    return rows


def write_rows_atomic(rows, output_fname, context_lengths):
    """CONTEXT-LIMITED: Atomically publish only a complete predictor TSV."""

    output_path = Path(output_fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["text_id", "word_id", "word"] + [
        f"{PREDICTOR_PREFIX}{context_length}"
        for context_length in context_lengths
    ]

    # CONTEXT-LIMITED: The temporary file is created beside the destination so
    # ``os.replace`` cannot cross filesystems and remains atomic.
    descriptor, temporary_fname = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_fname, output_path)
    except Exception:
        if os.path.exists(temporary_fname):
            os.unlink(temporary_fname)
        raise


def main():
    """CONTEXT-LIMITED: Run one model over all requested context conditions."""

    args = parse_args()
    context_lengths = validate_options(args.context_lengths, args.batch_size)
    texts = read_texts(args.input_fname)
    wrapper = load_wordsprobability_model(args.model)
    rows = build_rows(texts, context_lengths, wrapper, args.batch_size)
    write_rows_atomic(rows, args.output_fname, context_lengths)


if __name__ == "__main__":
    main()
