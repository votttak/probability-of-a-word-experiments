"""Focused tests for factorial result aggregation and reporting."""

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from h03_paper.analyze_layer_factorial_results import (  # noqa: E402
    analyze,
    validate_and_select_best,
)


def make_valid_rows():
    rows = []
    for context in ("passage", "sentence"):
        for lens in ("logit-lens", "tuned-lens"):
            for score in ("corrected", "buggy"):
                for layer in range(3):
                    best_layer = (
                        0 if (context, lens, score) ==
                        ("sentence", "tuned-lens", "buggy") else 1
                    )
                    rows.append({
                        "response_column": "time",
                        "context_unit": context,
                        "lens_method": lens,
                        "score_kind": score,
                        "model": "gpt2-small",
                        "analysis_mode": "paper-exact",
                        "layer": layer,
                        "min_layer": 0,
                        "max_layer": 2,
                        "delta_ll": 2.0 if layer == best_layer else 1.0,
                        "ppp_x1000": 4.0 if layer == best_layer else 2.0,
                        "input_rows": 500,
                        "analysis_rows": 480,
                    })
    return rows


class LayerFactorialResultsTest(unittest.TestCase):
    def test_eight_cells_are_selected_and_reported(self):
        rows = make_valid_rows()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "layers.tsv"
            pd.DataFrame(rows).to_csv(source, sep="\t", index=False)
            _, best, payload = analyze(
                [source],
                directory / "combined.tsv",
                directory / "best.tsv",
                directory / "REPORT.md",
                directory / "summary.json",
                title="Pivot",
                note="Small diagnostic.",
            )
            self.assertEqual(len(best), 8)
            self.assertEqual(
                payload["by_response"]["time"]["best_in_first_20pct"], 1
            )
            report = (directory / "REPORT.md").read_text(encoding="utf8")
            self.assertIn("1 of 8 factorial cells", report)
            self.assertIn("Small diagnostic.", report)

    def test_missing_cell_is_rejected(self):
        rows = [
            row for row in make_valid_rows()
            if (
                row["context_unit"],
                row["lens_method"],
                row["score_kind"],
            ) != ("passage", "logit-lens", "corrected")
        ]
        with self.assertRaisesRegex(ValueError, "all eight cells"):
            validate_and_select_best(pd.DataFrame(rows))

    def test_layer_ids_must_be_nonnegative_integers(self):
        for value, message in (
            (0.5, "layer must contain integers"),
            (-1, "layer must contain non-negative integers"),
        ):
            with self.subTest(value=value):
                rows = make_valid_rows()
                rows[0]["layer"] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_and_select_best(pd.DataFrame(rows))

    def test_declared_bounds_must_match_complete_observed_range(self):
        rows = make_valid_rows()
        rows[0]["max_layer"] = 3
        with self.assertRaisesRegex(ValueError, "inconsistent max_layer"):
            validate_and_select_best(pd.DataFrame(rows))

        rows = make_valid_rows()
        rows = [
            row for row in rows
            if not (
                row["context_unit"] == "passage"
                and row["lens_method"] == "logit-lens"
                and row["score_kind"] == "corrected"
                and row["layer"] == 1
            )
        ]
        with self.assertRaisesRegex(ValueError, "incomplete layer range"):
            validate_and_select_best(pd.DataFrame(rows))

    def test_all_factorial_cells_must_use_the_same_layer_range(self):
        rows = make_valid_rows()
        target = ("sentence", "tuned-lens", "buggy")
        rows = [
            row for row in rows
            if not (
                (
                    row["context_unit"], row["lens_method"],
                    row["score_kind"],
                ) == target and row["layer"] == 0
            )
        ]
        for row in rows:
            if (
                row["context_unit"], row["lens_method"], row["score_kind"]
            ) == target:
                row["min_layer"] = 1
        with self.assertRaisesRegex(ValueError, "inconsistent layer ranges"):
            validate_and_select_best(pd.DataFrame(rows))

    def test_experiment_metadata_and_response_samples_are_consistent(self):
        for column, value, message in (
            ("model", "gpt2-large", "inconsistent model"),
            ("analysis_mode", "project-bridge", "inconsistent analysis_mode"),
        ):
            with self.subTest(column=column):
                rows = make_valid_rows()
                rows[0][column] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_and_select_best(pd.DataFrame(rows))

        rows = make_valid_rows()
        for row in rows:
            if (
                row["context_unit"], row["lens_method"], row["score_kind"]
            ) == ("passage", "logit-lens", "corrected"):
                row["analysis_rows"] = 479
        with self.assertRaisesRegex(ValueError, "inconsistent analysis_rows"):
            validate_and_select_best(pd.DataFrame(rows))

    def test_best_layer_uses_exact_first_maximum_semantics(self):
        target = ("passage", "logit-lens", "corrected")
        rows = make_valid_rows()
        target_rows = []
        other_rows = []
        for row in rows:
            if (
                row["context_unit"], row["lens_method"], row["score_kind"]
            ) == target:
                row["delta_ll"] = 5.0 if row["layer"] in (0, 1) else 4.0
                target_rows.append(row)
            else:
                other_rows.append(row)
        target_rows.sort(key=lambda row: {1: 0, 0: 1, 2: 2}[row["layer"]])
        _, best = validate_and_select_best(
            pd.DataFrame(target_rows + other_rows)
        )
        selected = best.loc[
            (best["context_unit"] == target[0])
            & (best["lens_method"] == target[1])
            & (best["score_kind"] == target[2])
        ]
        self.assertEqual(selected["layer"].iloc[0], 1)

        rows = make_valid_rows()
        for row in rows:
            if (
                row["context_unit"], row["lens_method"], row["score_kind"]
            ) == target:
                row["delta_ll"] = {
                    0: 5.0,
                    1: 5.0 + 5e-13,
                    2: 4.0,
                }[row["layer"]]
        _, best = validate_and_select_best(pd.DataFrame(rows))
        selected = best.loc[
            (best["context_unit"] == target[0])
            & (best["lens_method"] == target[1])
            & (best["score_kind"] == target[2])
        ]
        self.assertEqual(selected["layer"].iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
