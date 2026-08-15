"""N-GRAM: Regression tests for bounded API scheduling and visible failures."""

from pathlib import Path
import sys
import threading
import time
import unittest


# N-GRAM: Import project code without installing it as a package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from h01_data.get_ngram_surprisals import fetch_counts  # noqa: E402


class FailingAPI:
    """Fail the first query while making other active requests observable."""

    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    def count(self, query):
        with self.lock:
            self.calls += 1
        if query == "bad":
            raise RuntimeError("expected API failure")
        time.sleep(0.05)
        return 1


class BoundedFetchTest(unittest.TestCase):
    """Ensure one failure cannot leave thousands of silent queued requests."""

    def test_failure_cancels_unsubmitted_queries(self):
        api = FailingAPI()
        queries = ["bad"] + [f"query-{index}" for index in range(100)]

        with self.assertRaisesRegex(RuntimeError, "expected API failure"):
            fetch_counts(queries, api, workers=3)

        # N-GRAM: Only the bounded initial worker set may have started. The old
        # implementation submitted and silently drained every remaining query.
        self.assertLessEqual(api.calls, 3)


if __name__ == "__main__":
    unittest.main()
