from __future__ import annotations

import unittest

from analysis.common import (
    duplicate_signatures,
    infer_query_tags,
    outcome_label,
    rank_of_answer,
    sample_case_rows,
)


class Phase4AnalysisTests(unittest.TestCase):
    def test_rank_of_answer_returns_none_when_missing(self) -> None:
        self.assertIsNone(rank_of_answer([1, 2, 3], 9))

    def test_rank_of_answer_returns_1_based_rank(self) -> None:
        self.assertEqual(rank_of_answer([5, 7, 9], 7), 2)

    def test_infer_query_tags_detects_identifier_and_failure_modes(self) -> None:
        tags = infer_query_tags(
            "Resolve user_profile.getEmailAddress",
            "Resolve user profile contact",
        )
        self.assertIn("identifier_heavy", tags)

    def test_outcome_label(self) -> None:
        self.assertEqual(outcome_label(0.1), "improved")
        self.assertEqual(outcome_label(-0.1), "worse")
        self.assertEqual(outcome_label(0.0), "same")

    def test_duplicate_signatures_groups_runs(self) -> None:
        duplicates = duplicate_signatures(
            [
                {
                    "run_name": "raw_valid_clean",
                    "split": "valid",
                    "condition": "clean",
                    "query_source": "original",
                    "prompt_version": None,
                    "prompt_path": None,
                },
                {
                    "run_name": "raw_valid_clean_copy",
                    "split": "valid",
                    "condition": "clean",
                    "query_source": "original",
                    "prompt_version": None,
                    "prompt_path": None,
                },
            ]
        )
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]["count"], 2)

    def test_sample_case_rows_balances_buckets(self) -> None:
        rows = [
            {
                "example_id": "1",
                "fixed_outcome": "improved",
                "evolved_outcome": "same",
                "evolved_vs_fixed_outcome": "worse",
                "fixed_delta_rr": 0.5,
                "evolved_delta_rr": 0.0,
                "evolved_vs_fixed_delta_rr": -0.5,
                "tags": ["identifier_heavy"],
            },
            {
                "example_id": "2",
                "fixed_outcome": "worse",
                "evolved_outcome": "improved",
                "evolved_vs_fixed_outcome": "improved",
                "fixed_delta_rr": -0.25,
                "evolved_delta_rr": 0.25,
                "evolved_vs_fixed_delta_rr": 0.5,
                "tags": ["ambiguous_natural_language"],
            },
        ]
        sampled = sample_case_rows(rows, 2)
        self.assertEqual(len(sampled), 2)
        self.assertEqual({row["example_id"] for row in sampled}, {"1", "2"})


if __name__ == "__main__":
    unittest.main()
