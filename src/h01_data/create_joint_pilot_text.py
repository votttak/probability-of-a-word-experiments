#!/usr/bin/env python3

"""Create a passage-balanced prefix sample for the joint surprisal pilot."""

import argparse
import os
from pathlib import Path
import tempfile


def read_texts(fname):
    """Read the project's one-passage-per-line text format."""

    with open(fname, "r", encoding="utf8") as input_file:
        return [line.strip().split() for line in input_file]


def select_prefixes(texts, words_per_text):
    """Keep exactly the first ``words_per_text`` words of every passage."""

    if words_per_text < 1:
        raise ValueError("words_per_text must be at least 1")

    prefixes = []
    for text_id, words in enumerate(texts):
        if len(words) < words_per_text:
            raise ValueError(
                f"text {text_id} has only {len(words)} words; "
                f"cannot select {words_per_text}"
            )
        prefixes.append(words[:words_per_text])
    if not prefixes:
        raise ValueError("input contains no texts")
    return prefixes


def write_texts_atomic(texts, output_fname):
    """Write the sampled passages atomically."""

    output_path = Path(output_fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_fname = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf8", newline="\n") as output_file:
            for words in texts:
                output_file.write(" ".join(words) + "\n")
        os.replace(temporary_fname, output_path)
    except Exception:
        if os.path.exists(temporary_fname):
            os.unlink(temporary_fname)
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select a fixed word-prefix from every RT passage"
    )
    parser.add_argument("--input-fname", required=True)
    parser.add_argument("--output-fname", required=True)
    parser.add_argument("--words-per-text", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    texts = read_texts(args.input_fname)
    prefixes = select_prefixes(texts, args.words_per_text)
    write_texts_atomic(prefixes, args.output_fname)


if __name__ == "__main__":
    main()
