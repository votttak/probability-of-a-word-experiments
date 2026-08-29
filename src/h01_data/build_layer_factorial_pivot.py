#!/usr/bin/env python3

"""Build a small, sentence-complete Natural Stories factorial pivot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

try:
    from .build_natural_stories_sentence_manifest import (
        FIELDNAMES,
        read_sentence_manifest,
        write_manifest,
    )
    from .get_context_limited_surprisals import read_texts
except ImportError:
    from build_natural_stories_sentence_manifest import (
        FIELDNAMES,
        read_sentence_manifest,
        write_manifest,
    )
    from get_context_limited_surprisals import read_texts


KEY_COLUMNS = ("text_id", "word_id")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_atomic(lines, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf8") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")
        os.replace(temporary, output_path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def write_json_atomic(payload, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output_path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def select_sentence_prefixes(texts, sentence_map, sentences_per_text):
    if (
        isinstance(sentences_per_text, bool)
        or not isinstance(sentences_per_text, int)
        or sentences_per_text < 1
    ):
        raise ValueError("sentences_per_text must be a positive integer")

    selected_texts = []
    manifest_rows = []
    selected_keys = set()
    selected_sentence_counts = {}
    for text_id, words in enumerate(texts):
        units = sentence_map.get(text_id)
        if not units:
            raise ValueError(f"sentence manifest has no units for text {text_id}")
        selected_units = units[:sentences_per_text]
        selected_words = [
            word for unit in selected_units for word in unit.words
        ]
        selected_word_ids = [
            word_id for unit in selected_units for word_id in unit.word_ids
        ]
        if selected_word_ids != list(range(len(selected_word_ids))):
            raise ValueError(
                f"selected sentences are not a prefix of text {text_id}"
            )
        if selected_words != words[:len(selected_words)]:
            raise ValueError(
                f"selected sentence words do not match text {text_id}"
            )
        selected_texts.append(selected_words)
        selected_sentence_counts[text_id] = len(selected_units)
        for unit in selected_units:
            for sentence_word_id, (word_id, word) in enumerate(
                zip(unit.word_ids, unit.words)
            ):
                manifest_rows.append({
                    "text_id": text_id,
                    "sentence_id": unit.sentence_id,
                    "sentence_word_id": sentence_word_id,
                    "word_id": word_id,
                    "word": word,
                })
                selected_keys.add((text_id + 1, word_id))

    return (
        selected_texts,
        manifest_rows,
        selected_keys,
        selected_sentence_counts,
    )


def filter_joint_rows(joint_path, selected_keys, selected_texts):
    with Path(joint_path).open("r", encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError("joint table has no header")
        required = set(KEY_COLUMNS)
        if not required.issubset(reader.fieldnames):
            raise ValueError("joint table lacks text_id/word_id keys")
        word_column = (
            "ref_token" if "ref_token" in reader.fieldnames else "word"
        )
        if word_column not in reader.fieldnames:
            raise ValueError("joint table has neither ref_token nor word")
        rows = []
        seen = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                key = (int(row["text_id"]), int(row["word_id"]))
            except ValueError as error:
                raise ValueError(
                    f"invalid joint key at line {line_number}"
                ) from error
            if key not in selected_keys:
                continue
            if key in seen:
                raise ValueError(f"duplicate joint key: {key}")
            seen.add(key)
            expected_word = selected_texts[key[0] - 1][key[1]]
            if row[word_column] != expected_word:
                raise ValueError(
                    f"joint word mismatch at {key}: "
                    f"{row[word_column]!r} versus {expected_word!r}"
                )
            rows.append(row)
    missing = selected_keys - seen
    if missing:
        raise ValueError(
            f"joint table is missing {len(missing)} selected keys"
        )
    rows.sort(key=lambda row: (int(row["text_id"]), int(row["word_id"])))
    return reader.fieldnames, rows


def write_joint_atomic(fieldnames, rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output_path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def build_pivot(
    text_path,
    sentence_manifest_path,
    joint_path,
    sentences_per_text,
    output_text_path,
    output_manifest_path,
    output_joint_path,
    output_metadata_path,
):
    texts = read_texts(text_path)
    sentence_map, sentence_manifest_sha256 = read_sentence_manifest(
        Path(sentence_manifest_path), texts
    )
    (
        selected_texts,
        manifest_rows,
        selected_keys,
        selected_sentence_counts,
    ) = select_sentence_prefixes(
        texts, sentence_map, sentences_per_text
    )
    joint_fields, joint_rows = filter_joint_rows(
        joint_path, selected_keys, selected_texts
    )

    write_text_atomic(
        (" ".join(words) for words in selected_texts), output_text_path
    )
    write_manifest(manifest_rows, Path(output_manifest_path))
    write_joint_atomic(joint_fields, joint_rows, output_joint_path)
    # Re-read the emitted pair so publication itself is covered by validation.
    emitted_texts = read_texts(output_text_path)
    _, emitted_manifest_sha256 = read_sentence_manifest(
        Path(output_manifest_path), emitted_texts
    )
    if emitted_texts != selected_texts:
        raise RuntimeError("published pivot text changed during writing")

    payload = {
        "schema_version": 1,
        "selection": "first-complete-sentences-per-text",
        "requested_sentences_per_text": sentences_per_text,
        "selected_sentence_counts": selected_sentence_counts,
        "texts": len(selected_texts),
        "words": len(selected_keys),
        "source": {
            "text_path": str(Path(text_path).resolve()),
            "text_sha256": sha256_file(text_path),
            "sentence_manifest_path": str(
                Path(sentence_manifest_path).resolve()
            ),
            "sentence_manifest_sha256": sentence_manifest_sha256,
            "joint_path": str(Path(joint_path).resolve()),
            "joint_sha256": sha256_file(joint_path),
        },
        "outputs": {
            "text_sha256": sha256_file(output_text_path),
            "sentence_manifest_sha256": emitted_manifest_sha256,
            "joint_sha256": sha256_file(output_joint_path),
        },
    }
    write_json_atomic(payload, output_metadata_path)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select complete sentence prefixes for a local layer pivot"
    )
    parser.add_argument("--text-fname", required=True)
    parser.add_argument("--sentence-manifest-fname", required=True)
    parser.add_argument("--joint-data-fname", required=True)
    parser.add_argument("--sentences-per-text", type=int, default=2)
    parser.add_argument("--output-text-fname", required=True)
    parser.add_argument("--output-sentence-manifest-fname", required=True)
    parser.add_argument("--output-joint-data-fname", required=True)
    parser.add_argument("--output-metadata-fname", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_pivot(
        args.text_fname,
        args.sentence_manifest_fname,
        args.joint_data_fname,
        args.sentences_per_text,
        args.output_text_fname,
        args.output_sentence_manifest_fname,
        args.output_joint_data_fname,
        args.output_metadata_fname,
    )
    print(
        f"Wrote pivot with {payload['texts']} texts and "
        f"{payload['words']} complete-sentence words"
    )


if __name__ == "__main__":
    main()
