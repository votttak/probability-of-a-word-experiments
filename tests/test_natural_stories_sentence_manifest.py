import csv
import hashlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from h01_data import build_natural_stories_sentence_manifest as manifest


def conll_row(row_id, form, token_id, *, misc=None):
    misc = misc if misc is not None else f"TokenId={token_id}"
    return "\t".join(
        [str(row_id), form, "_", "X", "X", "_", "0", "dep", "_", misc]
    )


class NaturalStoriesSentenceManifestTests(unittest.TestCase):
    def write_inputs(self, directory, canonical_lines, conll_text):
        directory = Path(directory)
        canonical_path = directory / "canonical.txt"
        conll_path = directory / "stories-aligned.conllx"
        canonical_path.write_text(
            "\n".join(canonical_lines) + "\n", encoding="utf-8"
        )
        conll_path.write_text(conll_text, encoding="utf-8")
        return canonical_path, conll_path

    def parse_and_build(self, canonical_lines, conll_text):
        with tempfile.TemporaryDirectory() as directory:
            canonical_path, conll_path = self.write_inputs(
                directory, canonical_lines, conll_text
            )
            passages = manifest.read_canonical_passages(canonical_path)
            sentences = manifest.read_aligned_sentences(conll_path)
            return manifest.build_manifest_rows(sentences, passages)

    def test_cli_builds_exact_zero_based_manifest_and_reader_mapping(self):
        canonical = ["If England, 'hi' (again)", "Second story"]
        conll = "\n".join(
            [
                "# comments do not create sentence blocks",
                conll_row(1, "If", "1.1"),
                conll_row(2, "England", "1.2.word"),
                conll_row(3, ",", "1.2.4"),
                conll_row(4, chr(96) * 2, "1.3.word"),
                conll_row(5, "hi", "1.3.4"),
                conll_row(6, "''", "1.3.5"),
                "",
                "",
                conll_row(1, "-LRB-", "1.4.word"),
                conll_row(2, "again", "1.4.4"),
                conll_row(3, "-RRB-", "1.4.5"),
                "",
                conll_row(
                    1,
                    "Second",
                    "2.1",
                    misc="Source=fixture;TokenId=2.1",
                ),
                conll_row(2, "story", "2.2"),
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            canonical_path, conll_path = self.write_inputs(
                directory, canonical, conll
            )
            output_path = Path(directory) / "manifest.tsv"
            argv = [
                "build_natural_stories_sentence_manifest.py",
                "--aligned-conllx-fname",
                str(conll_path),
                "--canonical-passage-fname",
                str(canonical_path),
                "--output-fname",
                str(output_path),
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                manifest.main()

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(list(rows[0]), list(manifest.FIELDNAMES))
            self.assertEqual(
                rows,
                [
                    {
                        "text_id": "0",
                        "sentence_id": "0",
                        "sentence_word_id": "0",
                        "word_id": "0",
                        "word": "If",
                    },
                    {
                        "text_id": "0",
                        "sentence_id": "0",
                        "sentence_word_id": "1",
                        "word_id": "1",
                        "word": "England,",
                    },
                    {
                        "text_id": "0",
                        "sentence_id": "0",
                        "sentence_word_id": "2",
                        "word_id": "2",
                        "word": "'hi'",
                    },
                    {
                        "text_id": "0",
                        "sentence_id": "1",
                        "sentence_word_id": "0",
                        "word_id": "3",
                        "word": "(again)",
                    },
                    {
                        "text_id": "1",
                        "sentence_id": "0",
                        "sentence_word_id": "0",
                        "word_id": "0",
                        "word": "Second",
                    },
                    {
                        "text_id": "1",
                        "sentence_id": "0",
                        "sentence_word_id": "1",
                        "word_id": "1",
                        "word": "story",
                    },
                ],
            )

            mapping, digest = manifest.read_sentence_manifest(
                output_path, canonical
            )
            self.assertEqual(
                mapping[0],
                [
                    manifest.SentenceUnit(0, (0, 1, 2), ("If", "England,", "'hi'")),
                    manifest.SentenceUnit(1, (3,), ("(again)",)),
                ],
            )
            self.assertEqual(
                mapping[1],
                [manifest.SentenceUnit(0, (0, 1), ("Second", "story"))],
            )
            self.assertEqual(
                digest, hashlib.sha256(output_path.read_bytes()).hexdigest()
            )

    def test_rejects_duplicate_full_token_id(self):
        conll = "\n".join(
            [
                conll_row(1, "one", "1.1"),
                conll_row(2, "one", "1.1"),
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            _, conll_path = self.write_inputs(directory, ["one"], conll)
            with self.assertRaisesRegex(ValueError, "duplicate TokenId"):
                manifest.read_aligned_sentences(conll_path)

    def test_rejects_missing_duplicate_and_noncontiguous_zones(self):
        cases = {
            "missing": (
                "\n".join(
                    [
                        conll_row(1, "one", "1.1"),
                        conll_row(2, "two", "1.3"),
                        "",
                    ]
                ),
                "missing zone",
            ),
            "duplicate-across-sentences": (
                "\n".join(
                    [
                        conll_row(1, "one", "1.1"),
                        "",
                        conll_row(1, "one", "1.1.word"),
                        "",
                    ]
                ),
                "duplicate or out-of-order zone",
            ),
            "noncontiguous-in-sentence": (
                "\n".join(
                    [
                        conll_row(1, "one", "1.1.word"),
                        conll_row(2, "two", "1.2"),
                        conll_row(3, "one", "1.1.4"),
                        "",
                    ]
                ),
                "repeats non-contiguously",
            ),
        }
        for name, (conll, message) in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    self.parse_and_build(["one two"], conll)

    def test_rejects_suffixless_and_suffixed_rows_in_one_zone(self):
        conll = "\n".join(
            [
                conll_row(1, "ca", "1.1"),
                conll_row(2, "n't", "1.1.4"),
                "",
            ]
        )
        with self.assertRaisesRegex(ValueError, "mixes a suffixless TokenId"):
            self.parse_and_build(["can't"], conll)

    def test_rejects_malformed_conll_rows_misc_ids_and_forms(self):
        valid_fields = conll_row(1, "one", "1.1").split("\t")
        cases = {
            "nine-fields": ("\t".join(valid_fields[:-1]) + "\n", "expected 10"),
            "row-id-gap": (
                conll_row(2, "one", "1.1") + "\n",
                "expected 1 within sentence",
            ),
            "empty-form": (
                conll_row(1, "_", "1.1") + "\n",
                "empty FORM",
            ),
            "invalid-suffix": (
                conll_row(1, "one", "1.1.foo") + "\n",
                "malformed TokenId",
            ),
            "duplicate-token-id-attribute": (
                conll_row(
                    1,
                    "one",
                    "1.1",
                    misc="TokenId=1.1;TokenId=1.2",
                )
                + "\n",
                "exactly one TokenId",
            ),
        }
        for name, (conll, message) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                _, conll_path = self.write_inputs(directory, ["one"], conll)
                with self.assertRaisesRegex(ValueError, message):
                    manifest.read_aligned_sentences(conll_path)

    def test_rejects_mixed_or_out_of_order_story_blocks(self):
        mixed = "\n".join(
            [
                conll_row(1, "one", "1.1"),
                conll_row(2, "two", "2.1"),
                "",
            ]
        )
        with self.assertRaisesRegex(ValueError, "mixes story IDs"):
            self.parse_and_build(["one", "two"], mixed)

        starts_at_two = conll_row(1, "two", "2.1") + "\n"
        with self.assertRaisesRegex(ValueError, "begin with story 1"):
            self.parse_and_build(["one", "two"], starts_at_two)

    def test_aligned_forms_must_reconstruct_canonical_word(self):
        bad = conll_row(1, "won", "1.1") + "\n"
        with self.assertRaisesRegex(ValueError, "aligned word mismatch"):
            self.parse_and_build(["one"], bad)

    def test_peeked_compatibility_is_explicit_and_key_scoped(self):
        manifest._validate_aligned_word("peeked", ("peeked",), 2, 749)
        manifest._validate_aligned_word("peeked", ("peaked",), 2, 749)
        with self.assertRaisesRegex(ValueError, "aligned word mismatch"):
            manifest._validate_aligned_word("peeked", ("peaked",), 1, 749)
        with self.assertRaisesRegex(ValueError, "aligned word mismatch"):
            manifest._validate_aligned_word("peaked", ("peeked",), 2, 749)

    def test_manifest_validator_rejects_duplicates_flattening_and_bad_ids(self):
        passages = [["one", "two"]]
        valid = [
            {
                "text_id": 0,
                "sentence_id": 0,
                "sentence_word_id": 0,
                "word_id": 0,
                "word": "one",
            },
            {
                "text_id": 0,
                "sentence_id": 0,
                "sentence_word_id": 1,
                "word_id": 1,
                "word": "two",
            },
        ]
        manifest.validate_manifest_rows(valid, passages)

        duplicated = valid + [deepcopy(valid[0])]
        with self.assertRaisesRegex(ValueError, "duplicate .* keys"):
            manifest.validate_manifest_rows(duplicated, passages)

        wrong_word = deepcopy(valid)
        wrong_word[1]["word"] = "too"
        with self.assertRaisesRegex(ValueError, "does not flatten exactly"):
            manifest.validate_manifest_rows(wrong_word, passages)

        bad_sentence = deepcopy(valid)
        for row in bad_sentence:
            row["sentence_id"] = 1
        with self.assertRaisesRegex(ValueError, "sentence_id values"):
            manifest.validate_manifest_rows(bad_sentence, passages)

        bad_sentence_word = deepcopy(valid)
        bad_sentence_word[1]["sentence_word_id"] = 2
        with self.assertRaisesRegex(ValueError, "sentence_word_id values"):
            manifest.validate_manifest_rows(bad_sentence_word, passages)

    def test_manifest_reader_requires_exact_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.tsv"
            path.write_text(
                "text_id\tword_id\tword\n0\t0\tone\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "header must be exactly"):
                manifest.read_sentence_manifest(path, ["one"])


if __name__ == "__main__":
    unittest.main()
