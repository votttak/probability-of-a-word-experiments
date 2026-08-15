#!/usr/bin/env python3

"""Compute fixed-context n-gram surprisal predictors with Infini-gram counts.

N-GRAM: This module adapts the count-ratio and Stupid Backoff method from the
``ngram-reading-time`` project to this repository's one-line-per-text format.
It emits one keyed row per whitespace-delimited word so the values can be
validated and merged with the existing ``wordsprobability`` output.
"""

import argparse
import csv
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import requests
from tqdm import tqdm


# N-GRAM: The public endpoint and index match the current Infini-gram API.
DEFAULT_API_URL = "https://api.infini-gram.io/"
DEFAULT_INDEX = "v4_piletrain_llama"
# N-GRAM: Pin the public Llama-2 tokenizer revision whose IDs were verified
# against the configured Infini-gram index; model weights are never loaded.
DEFAULT_TOKENIZER = "NousResearch/Llama-2-7b-hf"
DEFAULT_TOKENIZER_REVISION = "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
DEFAULT_TOKENIZER_CACHE_DIR = (
    Path(__file__).resolve().parents[2] / ".cache" / "huggingface"
)
DEFAULT_CONTEXT_LENGTHS = (0, 1, 2, 3, 4)
DEFAULT_BACKOFF_ALPHA = 0.4
DEFAULT_UNSEEN_UNIGRAM_COUNT = 1


class InfiniGramAPI:
    """Fetch exact n-gram counts from the public Infini-gram API."""

    # N-GRAM: Retries are restricted to transient network/server failures;
    # malformed requests and invalid API results fail after bounded retries.
    def __init__(self, api_url, index, timeout, max_retries, tokenizer_name=None,
                 tokenizer_revision=None, tokenizer_cache_dir=None):
        self.api_url = api_url
        self.index = index
        self.timeout = timeout
        self.max_retries = max_retries

        # N-GRAM: Initialize and use the shared fast tokenizer under one lock;
        # short local encodes are negligible beside remote count requests.
        self.tokenizer_name = tokenizer_name or DEFAULT_TOKENIZER
        self.tokenizer_revision = (
            tokenizer_revision or DEFAULT_TOKENIZER_REVISION
        )
        self.tokenizer_cache_dir = (
            tokenizer_cache_dir or str(DEFAULT_TOKENIZER_CACHE_DIR)
        )
        self.tokenizer = None
        self.tokenizer_lock = threading.Lock()

    def _post(self, payload):
        return requests.post(
            self.api_url,
            json=payload,
            timeout=self.timeout,
        )

    @staticmethod
    def _validate_token_ids(token_ids, query):
        if (
                not isinstance(token_ids, list)
                or any(isinstance(token_id, bool) or not isinstance(token_id, int)
                       for token_id in token_ids)):
            raise ValueError(
                f"Invalid tokenizer IDs for {query!r}: {token_ids!r}"
            )
        return token_ids

    def _get_query_token_ids(self, query):
        with self.tokenizer_lock:
            if self.tokenizer is None:
                # N-GRAM: Import lazily so importing pure scoring functions in
                # tests does not require or initialize Transformers.
                try:
                    from transformers import AutoTokenizer

                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.tokenizer_name,
                        revision=self.tokenizer_revision,
                        add_bos_token=False,
                        add_eos_token=False,
                        cache_dir=self.tokenizer_cache_dir,
                        use_fast=True,
                    )
                except (ImportError, OSError) as error:
                    raise RuntimeError(
                        "Unable to load the pinned Llama-2 tokenizer needed "
                        "for Infini-gram query IDs. Install the n-gram "
                        "requirements and rerun the same command."
                    ) from error

            token_ids = self.tokenizer.encode(query, add_special_tokens=False)

        token_ids = self._validate_token_ids(token_ids, query)
        # N-GRAM: Empty IDs are valid only for the empty corpus-size query.
        if query and not token_ids:
            raise ValueError(f"Tokenizer returned no token IDs for {query!r}")
        return token_ids

    def count(self, query):
        # N-GRAM: Flat query_ids force a simple contiguous n-gram. Raw strings
        # can trigger external filters or be parsed as CNF when they contain
        # standalone uppercase AND/OR, changing the requested count.
        query_ids = self._get_query_token_ids(query)
        payload = {
            "index": self.index,
            "query_type": "count",
            "query_ids": query_ids,
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._post(payload)

                # N-GRAM: Infini-gram/WAF 403s observed during long runs were
                # temporary. Retry them like throttling and server failures,
                # with the same finite exponential backoff below.
                if (
                        response.status_code in {403, 408, 429}
                        or response.status_code >= 500):
                    raise requests.ConnectionError(
                        f"transient Infini-gram HTTP status {response.status_code}"
                    )
                response.raise_for_status()

                result = response.json()
                if "error" in result:
                    raise ValueError(f"Infini-gram API error: {result['error']}")

                returned_ids = self._validate_token_ids(
                    result.get("token_ids"), query
                )
                if returned_ids != query_ids:
                    raise ValueError(
                        f"Infini-gram changed token IDs for {query!r}: "
                        f"sent {query_ids!r}, received {returned_ids!r}"
                    )

                count = result.get("count")
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(
                        f"Invalid Infini-gram count for {query!r}: {count!r}"
                    )
                return count
            except (requests.Timeout, requests.ConnectionError, ValueError) as error:
                last_error = error
                if attempt + 1 == self.max_retries:
                    break
                time.sleep(min(2 ** attempt, 30))
            except requests.RequestException as error:
                raise RuntimeError(
                    f"Infini-gram request failed for {query!r}: {error}"
                ) from error

        raise RuntimeError(
            f"Infini-gram request failed after {self.max_retries} attempts "
            f"for {query!r}: {last_error}"
        ) from last_error


class SQLiteCountCache:
    """Persist counts so long API runs can be resumed without repeated queries."""

    # N-GRAM: Cache keys include the index because counts differ across corpora.
    def __init__(self, fname, index):
        self.fname = Path(fname)
        self.index = index
        self.fname.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.fname)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ngram_counts (
                index_name TEXT NOT NULL,
                query TEXT NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (index_name, query)
            )
            """
        )
        self.connection.commit()

    def close(self):
        self.connection.close()

    def get_many(self, queries):
        results = {}
        queries = list(queries)

        # N-GRAM: Chunk lookups to stay below SQLite's bound-parameter limit.
        for start in range(0, len(queries), 400):
            chunk = queries[start:start + 400]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"""
                SELECT query, count
                FROM ngram_counts
                WHERE index_name = ? AND query IN ({placeholders})
                """,
                [self.index, *chunk],
            ).fetchall()
            results.update(rows)
        return results

    def put_many(self, counts):
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO ngram_counts (index_name, query, count)
            VALUES (?, ?, ?)
            """,
            [(self.index, query, count) for query, count in counts.items()],
        )
        self.connection.commit()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute word-level fixed-context n-gram surprisals"
    )
    parser.add_argument("--input-fname", required=True)
    parser.add_argument("--output-fname", required=True)

    # N-GRAM: A context length of k corresponds to a (k + 1)-gram predictor.
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTEXT_LENGTHS),
        help="maximum numbers of preceding whitespace-delimited words",
    )
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    # N-GRAM: These settings pin the tokenizer used for all query_ids.
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION
    )
    parser.add_argument(
        "--tokenizer-cache-dir", default=str(DEFAULT_TOKENIZER_CACHE_DIR)
    )
    parser.add_argument("--cache-fname")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--backoff-alpha", type=float, default=DEFAULT_BACKOFF_ALPHA)
    parser.add_argument(
        "--unseen-unigram-count",
        type=int,
        default=DEFAULT_UNSEEN_UNIGRAM_COUNT,
    )
    return parser.parse_args()


def validate_options(context_lengths, workers, max_retries, backoff_alpha,
                     unseen_unigram_count):
    """Normalize CLI options and reject configurations that cannot be scored."""

    # N-GRAM: Sorting gives deterministic columns regardless of CLI order.
    context_lengths = sorted(set(context_lengths))
    if not context_lengths or context_lengths[0] < 0:
        raise ValueError("context lengths must be non-negative integers")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if max_retries < 1:
        raise ValueError("max retries must be at least 1")
    if not 0 < backoff_alpha <= 1:
        raise ValueError("backoff alpha must be in (0, 1]")
    if unseen_unigram_count < 1:
        raise ValueError("unseen unigram count must be at least 1")
    return context_lengths


def validate_index(index):
    """Reject indexes that do not use the configured Llama-2 tokenizer."""

    # N-GRAM: Query IDs are meaningful only for an index built with the same
    # tokenizer. Official Infini-gram Llama-2 index names end in ``_llama``.
    if not index.endswith("_llama"):
        raise ValueError(
            f"Unsupported n-gram index {index!r}: this generator requires an "
            "Infini-gram Llama-2 index whose name ends in '_llama'"
        )


def read_texts(fname):
    """Read one text per line and split it exactly as the RT pipeline does."""

    # N-GRAM: Blank lines retain their text_id even though they emit no words.
    with open(fname, "r", encoding="utf8") as input_file:
        return [line.strip().split() for line in input_file]


def suffix_query(words, word_index, order):
    """Return the whitespace-word suffix ending at ``word_index``."""

    start = word_index - order + 1
    return " ".join(words[start:word_index + 1])


def required_queries(texts, max_context_length):
    """Collect every distinct count needed for all requested predictors."""

    # N-GRAM: Context denominators are suffixes ending at the previous word,
    # so collecting every suffix once also collects all required denominators.
    queries = {""}
    max_order = max_context_length + 1
    for words in texts:
        for word_index in range(len(words)):
            available_order = min(max_order, word_index + 1)
            for order in range(1, available_order + 1):
                queries.add(suffix_query(words, word_index, order))
    return sorted(queries)


def fetch_counts(queries, api, cache=None, workers=4):
    """Fetch missing queries concurrently while saving resumable cache batches."""

    queries = list(dict.fromkeys(queries))
    counts = cache.get_many(queries) if cache else {}
    missing_queries = [query for query in queries if query not in counts]
    if not missing_queries:
        return counts

    pending_cache = {}
    query_iterator = iter(missing_queries)
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {}

    # N-GRAM: Keep at most one active request per worker. The earlier version
    # queued every query at once, so one exhausted retry could freeze progress
    # while the executor silently drained tens of thousands of queued requests.
    def submit_next():
        try:
            query = next(query_iterator)
        except StopIteration:
            return False
        futures[executor.submit(api.count, query)] = query
        return True

    for _ in range(workers):
        if not submit_next():
            break

    try:
        with tqdm(
                total=len(queries),
                initial=len(counts),
                desc="Infini-gram counts",
                unit="query") as progress:
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    query = futures.pop(future)
                    count = future.result()
                    counts[query] = count
                    pending_cache[query] = count
                    progress.update(1)

                    if cache and len(pending_cache) >= 25:
                        cache.put_many(pending_cache)
                        pending_cache.clear()
                    submit_next()
    except BaseException:
        # N-GRAM: Cancel requests that have not started and surface the original
        # error immediately; at most ``workers`` active requests can remain.
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    finally:
        # N-GRAM: Preserve even a short final batch on API failure or Ctrl+C.
        if cache and pending_cache:
            cache.put_many(pending_cache)
    return counts


def score_word(words, word_index, context_length, counts, total_tokens,
               backoff_alpha=DEFAULT_BACKOFF_ALPHA,
               unseen_unigram_count=DEFAULT_UNSEEN_UNIGRAM_COUNT):
    """Return Stupid-Backoff surprisal for one word and maximum context length."""

    # N-GRAM: Missing text-start context shortens the requested order without a
    # penalty; zero corpus counts back off with one factor of alpha per order.
    available_order = min(context_length + 1, word_index + 1)
    failed_orders = 0

    for order in range(available_order, 1, -1):
        full_query = suffix_query(words, word_index, order)
        full_count = counts[full_query]
        if full_count > 0:
            context_query = suffix_query(words, word_index - 1, order - 1)
            context_count = counts[context_query]
            if context_count <= 0:
                raise ValueError(
                    f"Positive count for {full_query!r} has non-positive "
                    f"context count for {context_query!r}"
                )
            log_score = (
                failed_orders * math.log(backoff_alpha)
                + math.log(full_count)
                - math.log(context_count)
            )
            return -log_score
        failed_orders += 1

    unigram_query = suffix_query(words, word_index, 1)
    unigram_count = counts[unigram_query] or unseen_unigram_count
    log_score = (
        failed_orders * math.log(backoff_alpha)
        + math.log(unigram_count)
        - math.log(total_tokens)
    )
    return -log_score


def build_rows(texts, context_lengths, counts, backoff_alpha,
               unseen_unigram_count):
    """Build the keyed TSV rows consumed by the project merge step."""

    total_tokens = counts[""]
    if total_tokens <= 0:
        raise ValueError(f"Infini-gram returned invalid corpus size: {total_tokens}")

    rows = []
    for text_id, words in enumerate(texts):
        for word_id, word in enumerate(words):
            row = {"text_id": text_id, "word_id": word_id, "word": word}
            for context_length in context_lengths:
                column = f"ngram_surprisal_context_{context_length}"
                value = score_word(
                    words,
                    word_id,
                    context_length,
                    counts,
                    total_tokens,
                    backoff_alpha=backoff_alpha,
                    unseen_unigram_count=unseen_unigram_count,
                )
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite {column} for text {text_id}, word {word_id}"
                    )
                row[column] = value
            rows.append(row)
    return rows


def write_rows_atomic(rows, output_fname, context_lengths):
    """Write a complete TSV atomically so Make never accepts partial output."""

    output_path = Path(output_fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["text_id", "word_id", "word"] + [
        f"ngram_surprisal_context_{length}" for length in context_lengths
    ]

    # N-GRAM: The temporary file lives beside the destination so os.replace is
    # atomic on the target filesystem.
    descriptor, temporary_fname = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_fname, output_path)
    except Exception:
        if os.path.exists(temporary_fname):
            os.unlink(temporary_fname)
        raise


def main():
    args = parse_args()
    context_lengths = validate_options(
        args.context_lengths,
        args.workers,
        args.max_retries,
        args.backoff_alpha,
        args.unseen_unigram_count,
    )
    validate_index(args.index)
    texts = read_texts(args.input_fname)
    queries = required_queries(texts, max(context_lengths))

    # N-GRAM: The API client is deliberately isolated from pure scoring logic so
    # unit tests can verify the mathematics without network access.
    api = InfiniGramAPI(
        api_url=args.api_url,
        index=args.index,
        timeout=args.timeout,
        max_retries=args.max_retries,
        tokenizer_name=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
    )
    cache = SQLiteCountCache(args.cache_fname, args.index) if args.cache_fname else None
    try:
        counts = fetch_counts(queries, api, cache=cache, workers=args.workers)
    finally:
        if cache:
            cache.close()

    rows = build_rows(
        texts,
        context_lengths,
        counts,
        backoff_alpha=args.backoff_alpha,
        unseen_unigram_count=args.unseen_unigram_count,
    )
    write_rows_atomic(rows, args.output_fname, context_lengths)


if __name__ == "__main__":
    main()
