#!/usr/bin/env python3

"""Build and verify the small, portable Natural Stories factorial inputs."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import tempfile

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = 10_256
WORDFREQ_VERSION = "3.1.1"
WORDFREQ_EPSILON = 1e-7
PEAKED_PAPER_KEY = (2, 748)
NATURAL_STORIES_REPOSITORY = "https://github.com/languageMIT/naturalstories.git"
NATURAL_STORIES_REVISION = "4700daad696e942f5aba23c957a7423d0de66612"
HASHES = {
    "aligned_conllx": (
        "21e8bfa0fec0484f7fb66aa8220ef1817225cbfebe9ac11334ce916db1d12f41"
    ),
    "paper_rt_source": (
        "208ee6c451f3827734d5b2de32b91e2cf3bd76aa177c412cb426373fa7d50b22"
    ),
    "canonical_joint": (
        "a66f7ae10a5f1b342a2d55c7acc1360d9ba119c37db2d29a4be1573f1b84ad84"
    ),
    "full_text": (
        "04578a7187ec7edb779362f912df97befc74f7945c4d554902e2049041579da4"
    ),
    "sentence_manifest": (
        "f9e14f6ac9d1d7624dba51c9d658721a11c1a596560ade6f33fba6767e4f8263"
    ),
    "paper_rt": (
        "cef406dfb4eaef3fdd12b4f94f7f20418fc394dcceb42981a7171370cfc6c145"
    ),
    "frequency": (
        "208f9749acec1894e9e6b46f56ca889621fdbc8e4a5bb9e879743920f448c47a"
    ),
}


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: observed {observed}, expected {expected}"
        )


def write_dataframe_atomic(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    try:
        dataframe.to_csv(
            temporary, sep="\t", index=False, lineterminator="\n"
        )
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def write_json_atomic(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def build_paper_rt(source_path: Path) -> pd.DataFrame:
    require_hash(
        source_path, HASHES["paper_rt_source"], "Natural Stories paper RT source"
    )
    dataframe = pd.read_csv(
        source_path,
        sep="\t",
        usecols=["item", "zone", "word", "meanItemRT"],
        low_memory=False,
    )
    dataframe = (
        dataframe.drop_duplicates()
        .sort_values(["item", "zone"], kind="stable")
        .reset_index(drop=True)
    )
    if len(dataframe) != EXPECTED_ROWS:
        raise ValueError(
            f"paper RT has {len(dataframe)} unique word rows; "
            f"expected {EXPECTED_ROWS}"
        )
    if dataframe.duplicated(["item", "zone"], keep=False).any():
        raise ValueError("paper RT has conflicting values for an item/zone key")
    return dataframe


def build_frequency(joint_path: Path) -> pd.DataFrame:
    require_hash(joint_path, HASHES["canonical_joint"], "canonical joint")
    try:
        installed_version = version("wordfreq")
    except PackageNotFoundError as error:
        raise RuntimeError("wordfreq==3.1.1 is required to build controls") from error
    if installed_version != WORDFREQ_VERSION:
        raise RuntimeError(
            f"wordfreq=={WORDFREQ_VERSION} is required; found {installed_version}"
        )
    from wordfreq import word_frequency

    joint = pd.read_csv(
        joint_path,
        sep="\t",
        usecols=["text_id", "word_id", "ref_token"],
        keep_default_na=False,
        low_memory=False,
    )
    if len(joint) != EXPECTED_ROWS:
        raise ValueError(
            f"canonical joint has {len(joint)} rows; expected {EXPECTED_ROWS}"
        )
    if joint.duplicated(["text_id", "word_id"], keep=False).any():
        raise ValueError("canonical joint has duplicate word keys")

    cache: dict[str, float] = {}
    output_words = []
    values = []
    for row in joint.itertuples(index=False):
        word = str(row.ref_token)
        key = (int(row.text_id), int(row.word_id))
        scoring_word = word
        if key == PEAKED_PAPER_KEY:
            if word != "peeked":
                raise ValueError(
                    "paper compatibility key (2, 748) no longer contains 'peeked'"
                )
            scoring_word = "peaked"
        if scoring_word not in cache:
            raw = float(word_frequency(scoring_word, "en"))
            if not math.isfinite(raw) or raw < 0:
                raise ValueError(
                    f"invalid wordfreq value for {scoring_word!r}: {raw!r}"
                )
            cache[scoring_word] = math.log(raw + WORDFREQ_EPSILON)
        output_words.append(scoring_word)
        values.append(cache[scoring_word])

    return pd.DataFrame({
        "text_id": joint["text_id"],
        "word_id": joint["word_id"],
        "word": output_words,
        "paper_log_gmean_freq": values,
    })


def verify_portable_inputs(paper_path: Path, frequency_path: Path) -> None:
    require_hash(paper_path, HASHES["paper_rt"], "portable paper RT")
    require_hash(frequency_path, HASHES["frequency"], "portable frequency")
    paper = pd.read_csv(paper_path, sep="\t", low_memory=False)
    frequency = pd.read_csv(
        frequency_path, sep="\t", keep_default_na=False, low_memory=False
    )
    if len(paper) != EXPECTED_ROWS or len(frequency) != EXPECTED_ROWS:
        raise ValueError("portable inputs must each contain 10,256 rows")
    if list(paper.columns) != ["item", "zone", "word", "meanItemRT"]:
        raise ValueError("portable paper RT has an unexpected schema")
    if list(frequency.columns) != [
        "text_id",
        "word_id",
        "word",
        "paper_log_gmean_freq",
    ]:
        raise ValueError("portable frequency has an unexpected schema")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify committed layer-factorial input tables"
    )
    parser.add_argument(
        "--paper-rt-source",
        default=(
            ".cache/naturalstories-source/"
            "naturalstories_RTS/processed_RTs.tsv"
        ),
    )
    parser.add_argument(
        "--joint-data-fname",
        default="checkpoints/rt/merged_data/natural_stories-gpt2-small.tsv",
    )
    parser.add_argument(
        "--paper-output-fname",
        default=(
            "checkpoints/rt/layer_factorial/inputs/"
            "natural-stories-paper-time.tsv"
        ),
    )
    parser.add_argument(
        "--frequency-output-fname",
        default=(
            "checkpoints/rt/layer_factorial/inputs/"
            "natural-stories-paper-frequency.tsv"
        ),
    )
    parser.add_argument(
        "--provenance-output-fname",
        default=(
            "checkpoints/rt/layer_factorial/inputs/"
            "PROVENANCE.json"
        ),
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paper_output = repo_path(args.paper_output_fname)
    frequency_output = repo_path(args.frequency_output_fname)
    if not args.verify_only:
        paper = build_paper_rt(repo_path(args.paper_rt_source))
        frequency = build_frequency(repo_path(args.joint_data_fname))
        write_dataframe_atomic(paper, paper_output)
        write_dataframe_atomic(frequency, frequency_output)
    verify_portable_inputs(paper_output, frequency_output)

    if not args.verify_only:
        provenance = {
            "schema_version": 1,
            "natural_stories": {
                "repository": NATURAL_STORIES_REPOSITORY,
                "revision": NATURAL_STORIES_REVISION,
                "aligned_conllx_sha256": HASHES["aligned_conllx"],
                "processed_rt_sha256": HASHES["paper_rt_source"],
            },
            "canonical_joint_sha256": HASHES["canonical_joint"],
            "canonical_text_sha256": HASHES["full_text"],
            "sentence_manifest_sha256": HASHES["sentence_manifest"],
            "portable_paper_rt": {
                "rows": EXPECTED_ROWS,
                "sha256": HASHES["paper_rt"],
            },
            "portable_frequency": {
                "rows": EXPECTED_ROWS,
                "sha256": HASHES["frequency"],
                "wordfreq_version": WORDFREQ_VERSION,
                "epsilon": WORDFREQ_EPSILON,
                "peeked_compatibility_key": [2, 748],
            },
        }
        write_json_atomic(
            provenance, repo_path(args.provenance_output_fname)
        )
    print("Portable layer-factorial inputs verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
