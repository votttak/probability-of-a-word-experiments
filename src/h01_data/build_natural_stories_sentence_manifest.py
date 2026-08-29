#!/usr/bin/env python3
"""Build a word-to-sentence manifest from aligned Natural Stories CoNLL-X.

The aligned parse is authoritative for sentence boundaries and the canonical
passage text is authoritative for word spelling. A processed RT variant spells
one canonical "peeked" token as "peaked"; that known mismatch is checked
explicitly rather than generalized into normalization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


FIELDNAMES = (
    "text_id",
    "sentence_id",
    "sentence_word_id",
    "word_id",
    "word",
)
TOKEN_ID_RE = re.compile(
    r"^(?P<story>[1-9][0-9]*)\.(?P<zone>[1-9][0-9]*)"
    r"(?:\.(?P<suffix>[1-9][0-9]*|word))?$"
)
PEEKED_COMPATIBILITY_KEY = (2, 749)
PTB_FORM_NORMALIZATION = {
    chr(96) * 2: "'",
    "''": "'",
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
}


@dataclass(frozen=True)
class AlignedToken:
    form: str
    token_id: str
    story_id: int
    zone_id: int
    alignment_suffix: str | None
    line_number: int


@dataclass(frozen=True)
class AlignedSentence:
    tokens: tuple[AlignedToken, ...]
    first_line_number: int


@dataclass(frozen=True)
class SentenceUnit:
    sentence_id: int
    word_ids: tuple[int, ...]
    words: tuple[str, ...]


def read_canonical_passages(path: Path) -> list[list[str]]:
    """Read one whitespace-tokenized canonical passage per line."""
    passages: list[list[str]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                raise ValueError(
                    "canonical passage file contains a blank passage "
                    f"at line {line_number}"
                )
            passages.append(stripped.split())

    if not passages:
        raise ValueError("canonical passage file is empty")
    return passages


def _parse_token_id(
    misc: str, line_number: int
) -> tuple[str, int, int, str | None]:
    values = [
        value
        for item in re.split(r"[;|]", misc)
        for key, separator, value in [item.strip().partition("=")]
        if separator and key == "TokenId"
    ]
    if len(values) != 1:
        raise ValueError(
            f"aligned CoNLL-X line {line_number} must contain exactly one "
            f"TokenId in MISC; found {len(values)}"
        )

    token_id = values[0]
    match = TOKEN_ID_RE.fullmatch(token_id)
    if match is None:
        raise ValueError(
            f"aligned CoNLL-X line {line_number} has malformed "
            f"TokenId={token_id!r}; expected STORY.ZONE or "
            "STORY.ZONE.SUFFIX, where SUFFIX is a positive integer or 'word'"
        )

    return (
        token_id,
        int(match.group("story")),
        int(match.group("zone")),
        match.group("suffix"),
    )


def read_aligned_sentences(path: Path) -> list[AlignedSentence]:
    """Read blank-line-delimited sentences from aligned CoNLL-X."""
    sentences: list[AlignedSentence] = []
    sentence_tokens: list[AlignedToken] = []
    seen_token_ids: dict[str, int] = {}

    def finish_sentence() -> None:
        if not sentence_tokens:
            return
        story_ids = {token.story_id for token in sentence_tokens}
        if len(story_ids) != 1:
            locations = ", ".join(str(story_id) for story_id in sorted(story_ids))
            raise ValueError(
                "sentence beginning at aligned CoNLL-X line "
                f"{sentence_tokens[0].line_number} mixes story IDs: {locations}"
            )
        sentences.append(
            AlignedSentence(tuple(sentence_tokens), sentence_tokens[0].line_number)
        )
        sentence_tokens.clear()

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                finish_sentence()
                continue
            if line.lstrip().startswith("#"):
                continue

            fields = line.split("\t")
            if len(fields) != 10:
                raise ValueError(
                    f"aligned CoNLL-X line {line_number} has {len(fields)} "
                    "fields; expected 10"
                )
            try:
                conll_row_id = int(fields[0])
            except ValueError as error:
                raise ValueError(
                    f"aligned CoNLL-X line {line_number} has non-integer "
                    f"row ID {fields[0]!r}"
                ) from error
            expected_row_id = len(sentence_tokens) + 1
            if conll_row_id != expected_row_id:
                raise ValueError(
                    f"aligned CoNLL-X line {line_number} has row ID "
                    f"{conll_row_id}; expected {expected_row_id} within sentence"
                )
            if not fields[1] or fields[1] == "_":
                raise ValueError(
                    f"aligned CoNLL-X line {line_number} has an empty FORM"
                )
            token_id, story_id, zone_id, alignment_suffix = _parse_token_id(
                fields[9], line_number
            )
            if token_id in seen_token_ids:
                raise ValueError(
                    f"duplicate TokenId={token_id!r} at aligned CoNLL-X "
                    f"line {line_number}; first seen at line "
                    f"{seen_token_ids[token_id]}"
                )
            seen_token_ids[token_id] = line_number
            sentence_tokens.append(
                AlignedToken(
                    form=fields[1],
                    token_id=token_id,
                    story_id=story_id,
                    zone_id=zone_id,
                    alignment_suffix=alignment_suffix,
                    line_number=line_number,
                )
            )
    finish_sentence()

    if not sentences:
        raise ValueError("aligned CoNLL-X file contains no token sentences")
    return sentences


def _validate_aligned_word(
    canonical_word: str,
    parse_forms: Sequence[str],
    story_id: int,
    zone_id: int,
) -> None:
    """Require FORM concatenation to match the canonical word exactly."""
    aligned_word = "".join(
        PTB_FORM_NORMALIZATION.get(form, form) for form in parse_forms
    )
    if aligned_word == canonical_word:
        return

    # Kuribayashi's processed RT table spells this one stimulus "peaked";
    # the aligned CoNLL-X and canonical passage both spell it "peeked".
    # Keep the compatibility exception key-specific and always emit canonical.
    if (
        (story_id, zone_id) == PEEKED_COMPATIBILITY_KEY
        and canonical_word == "peeked"
        and aligned_word == "peaked"
    ):
        return

    raise ValueError(
        f"aligned word mismatch at story {story_id}, zone {zone_id}: "
        f"canonical={canonical_word!r}, aligned forms={list(parse_forms)!r}, "
        f"normalized concatenation={aligned_word!r}"
    )


def _collapse_sentence(
    sentence: AlignedSentence,
) -> list[tuple[int, tuple[str, ...]]]:
    """Collapse consecutive parse subtokens sharing a story.zone."""
    collapsed: list[tuple[int, list[str], list[str | None]]] = []
    closed_zones: set[int] = set()
    for token in sentence.tokens:
        if collapsed and collapsed[-1][0] == token.zone_id:
            collapsed[-1][1].append(token.form)
            collapsed[-1][2].append(token.alignment_suffix)
            continue
        if token.zone_id in closed_zones:
            raise ValueError(
                f"zone {token.story_id}.{token.zone_id} repeats "
                "non-contiguously in sentence beginning at aligned CoNLL-X "
                f"line {sentence.first_line_number}"
            )
        if collapsed:
            closed_zones.add(collapsed[-1][0])
        collapsed.append(
            (token.zone_id, [token.form], [token.alignment_suffix])
        )

    result: list[tuple[int, tuple[str, ...]]] = []
    for zone_id, forms, suffixes in collapsed:
        if len(suffixes) > 1 and any(suffix is None for suffix in suffixes):
            raise ValueError(
                f"zone {sentence.tokens[0].story_id}.{zone_id} mixes a "
                "suffixless TokenId with suffixed TokenIds"
            )
        result.append((zone_id, tuple(forms)))
    return result


def build_manifest_rows(
    sentences: Sequence[AlignedSentence],
    passages: Sequence[Sequence[str]],
) -> list[dict[str, int | str]]:
    """Combine aligned sentence boundaries with canonical passage words."""
    rows: list[dict[str, int | str]] = []
    expected_zone_by_story = {
        story_id: 1 for story_id in range(1, len(passages) + 1)
    }
    next_sentence_id_by_story = {
        story_id: 0 for story_id in range(1, len(passages) + 1)
    }
    seen_stories: set[int] = set()
    completed_stories: set[int] = set()
    previous_story_id: int | None = None

    for sentence in sentences:
        story_id = sentence.tokens[0].story_id
        if story_id < 1 or story_id > len(passages):
            raise ValueError(
                f"aligned sentence at line {sentence.first_line_number} "
                f"references story {story_id}, but canonical text has "
                f"stories 1..{len(passages)}"
            )
        if previous_story_id is None and story_id != 1:
            raise ValueError(
                f"aligned stories must begin with story 1; got story "
                f"{story_id} at line {sentence.first_line_number}"
            )
        if previous_story_id is not None and story_id != previous_story_id:
            completed_stories.add(previous_story_id)
            if story_id in completed_stories:
                raise ValueError(
                    f"story {story_id} reappears at aligned CoNLL-X line "
                    f"{sentence.first_line_number}; each story must form one "
                    "contiguous block"
                )
            if story_id != previous_story_id + 1:
                raise ValueError(
                    f"aligned stories must occur in canonical order; after "
                    f"story {previous_story_id}, got story {story_id} at line "
                    f"{sentence.first_line_number}"
                )
        previous_story_id = story_id
        seen_stories.add(story_id)

        canonical_words = passages[story_id - 1]
        sentence_id = next_sentence_id_by_story[story_id]
        collapsed = _collapse_sentence(sentence)
        for sentence_word_id, (zone_id, parse_forms) in enumerate(collapsed):
            expected_zone = expected_zone_by_story[story_id]
            if zone_id != expected_zone:
                relation = (
                    "duplicate or out-of-order"
                    if zone_id < expected_zone
                    else "missing"
                )
                raise ValueError(
                    f"{relation} zone in story {story_id} at aligned "
                    f"CoNLL-X line {sentence.first_line_number}: expected "
                    f"zone {expected_zone}, got {zone_id}"
                )
            if zone_id > len(canonical_words):
                raise ValueError(
                    f"aligned story {story_id} has zone {zone_id}, but "
                    f"canonical passage has only {len(canonical_words)} words"
                )

            canonical_word = canonical_words[zone_id - 1]
            _validate_aligned_word(
                canonical_word, parse_forms, story_id, zone_id
            )
            rows.append(
                {
                    "text_id": story_id - 1,
                    "sentence_id": sentence_id,
                    "sentence_word_id": sentence_word_id,
                    "word_id": zone_id - 1,
                    "word": canonical_word,
                }
            )
            expected_zone_by_story[story_id] += 1
        next_sentence_id_by_story[story_id] += 1

    expected_stories = set(range(1, len(passages) + 1))
    if seen_stories != expected_stories:
        missing = sorted(expected_stories - seen_stories)
        extra = sorted(seen_stories - expected_stories)
        raise ValueError(
            "aligned story IDs do not match canonical passages; "
            f"missing={missing}, extra={extra}"
        )
    for story_id, canonical_words in enumerate(passages, start=1):
        observed_count = expected_zone_by_story[story_id] - 1
        if observed_count != len(canonical_words):
            raise ValueError(
                f"aligned story {story_id} contains {observed_count} word "
                f"zones, but canonical passage contains "
                f"{len(canonical_words)} words"
            )

    validate_manifest_rows(rows, passages)
    return rows


def validate_manifest_rows(
    rows: Sequence[dict[str, int | str]],
    passages: Sequence[Sequence[str]],
) -> None:
    """Validate IDs and exact flattened equality to canonical passages."""
    expected = [
        (text_id, word_id, word)
        for text_id, passage in enumerate(passages)
        for word_id, word in enumerate(passage)
    ]
    keys = [(int(row["text_id"]), int(row["word_id"])) for row in rows]
    key_counts = Counter(keys)
    if any(count > 1 for count in key_counts.values()):
        duplicates = sorted(
            key for key, count in key_counts.items() if count > 1
        )
        raise ValueError(
            "manifest contains duplicate (text_id, word_id) keys: "
            f"{duplicates}"
        )

    observed = [
        (int(row["text_id"]), int(row["word_id"]), str(row["word"]))
        for row in rows
    ]
    if observed != expected:
        first_difference = next(
            (
                index
                for index, (observed_item, expected_item) in enumerate(
                    zip(observed, expected)
                )
                if observed_item != expected_item
            ),
            min(len(observed), len(expected)),
        )
        observed_item = (
            observed[first_difference]
            if first_difference < len(observed)
            else None
        )
        expected_item = (
            expected[first_difference]
            if first_difference < len(expected)
            else None
        )
        raise ValueError(
            "manifest does not flatten exactly to canonical passage text; "
            f"first difference at row {first_difference}: "
            f"observed={observed_item!r}, expected={expected_item!r}; row "
            f"counts observed={len(observed)}, expected={len(expected)}"
        )

    grouped: dict[tuple[int, int], list[dict[str, int | str]]] = {}
    sentence_ids_by_text: dict[int, list[int]] = {}
    for row in rows:
        text_id = int(row["text_id"])
        sentence_id = int(row["sentence_id"])
        sentence_word_id = int(row["sentence_word_id"])
        if min(text_id, sentence_id, sentence_word_id, int(row["word_id"])) < 0:
            raise ValueError(f"manifest IDs must be nonnegative: {row!r}")
        grouped.setdefault((text_id, sentence_id), []).append(row)
        sentence_ids_by_text.setdefault(text_id, [])
        if sentence_id not in sentence_ids_by_text[text_id]:
            sentence_ids_by_text[text_id].append(sentence_id)

    for text_id, sentence_ids in sentence_ids_by_text.items():
        expected_sentence_ids = list(range(len(sentence_ids)))
        if sentence_ids != expected_sentence_ids:
            raise ValueError(
                f"sentence_id values for text {text_id} must start at 0 "
                f"and be contiguous; observed={sentence_ids}, "
                f"expected={expected_sentence_ids}"
            )
    for (text_id, sentence_id), sentence_rows in grouped.items():
        observed_word_ids = [
            int(row["sentence_word_id"]) for row in sentence_rows
        ]
        expected_word_ids = list(range(len(sentence_rows)))
        if observed_word_ids != expected_word_ids:
            raise ValueError(
                f"sentence_word_id values for text {text_id}, sentence "
                f"{sentence_id} must start at 0 and be contiguous; "
                f"observed={observed_word_ids}, expected={expected_word_ids}"
            )


def write_manifest(
    rows: Iterable[dict[str, int | str]], output_path: Path
) -> None:
    """Write atomically so interruption cannot leave a partial manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=FIELDNAMES,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _coerce_canonical_passages(
    canonical_texts: Sequence[str | Sequence[str]],
) -> list[list[str]]:
    passages: list[list[str]] = []
    for text_id, passage in enumerate(canonical_texts):
        words = passage.split() if isinstance(passage, str) else list(passage)
        if not words:
            raise ValueError(f"canonical text {text_id} contains no words")
        passages.append(words)
    if not passages:
        raise ValueError("canonical texts contain no passages")
    return passages


def read_sentence_manifest(
    path: str | Path,
    canonical_texts: Sequence[str | Sequence[str]],
) -> tuple[dict[int, list[SentenceUnit]], str]:
    """Read and validate a manifest, returning sentence units and file SHA-256."""
    path = Path(path)
    passages = _coerce_canonical_passages(canonical_texts)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows: list[dict[str, int | str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != list(FIELDNAMES):
            raise ValueError(
                f"manifest header must be exactly {list(FIELDNAMES)!r}; "
                f"observed={reader.fieldnames!r}"
            )
        for line_number, raw_row in enumerate(reader, start=2):
            if None in raw_row or any(raw_row[field] is None for field in FIELDNAMES):
                raise ValueError(
                    f"manifest line {line_number} does not have exactly "
                    f"{len(FIELDNAMES)} fields"
                )
            try:
                rows.append(
                    {
                        "text_id": int(raw_row["text_id"]),
                        "sentence_id": int(raw_row["sentence_id"]),
                        "sentence_word_id": int(raw_row["sentence_word_id"]),
                        "word_id": int(raw_row["word_id"]),
                        "word": raw_row["word"],
                    }
                )
            except ValueError as error:
                raise ValueError(
                    f"manifest line {line_number} contains a non-integer ID"
                ) from error

    validate_manifest_rows(rows, passages)
    grouped: dict[tuple[int, int], list[dict[str, int | str]]] = {}
    for row in rows:
        grouped.setdefault(
            (int(row["text_id"]), int(row["sentence_id"])), []
        ).append(row)

    mapping: dict[int, list[SentenceUnit]] = {
        text_id: [] for text_id in range(len(passages))
    }
    for (text_id, sentence_id), sentence_rows in grouped.items():
        mapping[text_id].append(
            SentenceUnit(
                sentence_id=sentence_id,
                word_ids=tuple(int(row["word_id"]) for row in sentence_rows),
                words=tuple(str(row["word"]) for row in sentence_rows),
            )
        )
    return mapping, digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Natural Stories sentence manifest using aligned CoNLL-X "
            "sentence boundaries and canonical passage words."
        )
    )
    parser.add_argument(
        "--aligned-conllx-fname",
        required=True,
        type=Path,
        help="Kuribayashi-style aligned Natural Stories CoNLL-X file",
    )
    parser.add_argument(
        "--canonical-passage-fname",
        required=True,
        type=Path,
        help="canonical passage text, one whitespace-tokenized story per line",
    )
    parser.add_argument(
        "--output-fname",
        required=True,
        type=Path,
        help="output TSV manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    passages = read_canonical_passages(args.canonical_passage_fname)
    sentences = read_aligned_sentences(args.aligned_conllx_fname)
    rows = build_manifest_rows(sentences, passages)
    write_manifest(rows, args.output_fname)
    print(
        f"Wrote {len(rows)} words from {len(passages)} passages "
        f"to {args.output_fname}"
    )


if __name__ == "__main__":
    main()
