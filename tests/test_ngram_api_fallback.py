"""N-GRAM: Regression tests for token-ID transport and API retries."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import requests


# N-GRAM: Import project code without installing it as a package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from h01_data.get_ngram_surprisals import (  # noqa: E402
    InfiniGramAPI,
    validate_index,
)


class FakeResponse:
    """N-GRAM: Provide only the response behavior used by InfiniGramAPI."""

    def __init__(self, status_code, result):
        self.status_code = status_code
        self.result = result

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.result


class FakeTokenizer:
    """N-GRAM: Return known whole-query tokenizations without network I/O."""

    def __init__(self):
        self.calls = []

    def encode(self, query, add_special_tokens):
        self.calls.append((query, add_special_tokens))
        if query == "word1 OR word2":
            return [10, 20, 21]
        if query == "":
            return []
        raise AssertionError("unexpected tokenizer input")


def make_api():
    """N-GRAM: Build a client with a fake preloaded matching tokenizer."""

    api = InfiniGramAPI(
        "https://example.invalid/",
        "test_llama",
        timeout=1,
        max_retries=2,
    )
    api.tokenizer = FakeTokenizer()
    return api


class TokenIDTransportTest(unittest.TestCase):
    """N-GRAM: Cover deterministic IDs and bounded transient retries."""

    @patch("h01_data.get_ngram_surprisals.requests.post")
    def test_count_uses_flat_ids_not_raw_or_syntax(self, mock_post):
        # N-GRAM: Even reserved uppercase OR remains part of one contiguous
        # sequence because the request contains a flat token-ID list.
        mock_post.return_value = FakeResponse(
            200, {"count": 7, "token_ids": [10, 20, 21]}
        )
        api = make_api()

        self.assertEqual(api.count("word1 OR word2"), 7)
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["query_ids"], [10, 20, 21])
        self.assertNotIn("query", payload)
        self.assertEqual(api.tokenizer.calls, [("word1 OR word2", False)])

    @patch("h01_data.get_ngram_surprisals.time.sleep")
    @patch("h01_data.get_ngram_surprisals.requests.post")
    def test_transient_403_retries_token_ids(self, mock_post, mock_sleep):
        # N-GRAM: A temporary WAF 403 is retried with finite backoff instead of
        # aborting a resumable long run after one response.
        mock_post.side_effect = [
            FakeResponse(403, {}),
            FakeResponse(200, {"count": 7, "token_ids": [10, 20, 21]}),
        ]
        api = make_api()

        self.assertEqual(api.count("word1 OR word2"), 7)
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(1)

    @patch("h01_data.get_ngram_surprisals.requests.post")
    def test_empty_corpus_query_uses_empty_ids(self, mock_post):
        # N-GRAM: Infini-gram defines an empty count query as corpus token total.
        mock_post.return_value = FakeResponse(
            200, {"count": 1234, "token_ids": []}
        )
        api = make_api()

        self.assertEqual(api.count(""), 1234)
        self.assertEqual(mock_post.call_args.kwargs["json"]["query_ids"], [])

    def test_non_llama_index_is_rejected(self):
        # N-GRAM: Token IDs cannot be silently reused across tokenizer families.
        with self.assertRaisesRegex(ValueError, "requires.*Llama-2"):
            validate_index("v4_pileval_gpt2")


if __name__ == "__main__":
    unittest.main()
