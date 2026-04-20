from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.compare_runs import _method_row
from src.rewriting.cache_manager import RewriteCacheManager, load_rewrite_records
from src.rewriting.validators import validate_rewritten_query


class RewriteValidationTests(unittest.TestCase):
    def test_validator_accepts_reasonable_rewrite(self) -> None:
        result = validate_rewritten_query(
            "Helper which expand_dims `is_accepted` then applies tf.where.",
            "Helper which expand dims is accepted then applies tf where.",
        )
        self.assertEqual(result.validation_status, "passed")
        self.assertIn("expand", result.rewritten_query_tokens)

    def test_validator_falls_back_on_prompt_leakage(self) -> None:
        result = validate_rewritten_query(
            "Original query text.",
            "Original query: rewrite the input query to be better.",
        )
        self.assertEqual(result.validation_status, "fallback")
        self.assertEqual(result.reason, "prompt_leakage")

    def test_validator_falls_back_on_generated_code(self) -> None:
        result = validate_rewritten_query("Return user id", "def user_id(): return 1")
        self.assertEqual(result.validation_status, "fallback")
        self.assertEqual(result.reason, "generated_code")


class RewriteCacheTests(unittest.TestCase):
    def test_cache_manager_resumes_and_finalizes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rewrite_root = Path(temp_dir) / "rewrites"
            log_root = Path(temp_dir) / "logs"
            manager = RewriteCacheManager(
                rewrite_root=rewrite_root,
                log_root=log_root,
                condition="adv",
                run_name="sample",
            )
            manager.append({"example_id": "2", "rewritten_query": "two"})
            manager.append({"example_id": "1", "rewritten_query": "one"})
            manager.finalize(
                ordered_records=[
                    {"example_id": "1", "rewritten_query": "one"},
                    {"example_id": "2", "rewritten_query": "two"},
                ],
                metadata={"run_name": "sample"},
            )

            cache_path = rewrite_root / "adv" / "sample" / "rewrites.jsonl"
            records = load_rewrite_records(cache_path)
            self.assertEqual(set(records), {"1", "2"})
            first_line = cache_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(json.loads(first_line)["example_id"], "1")


class ComparisonTests(unittest.TestCase):
    def test_method_row_computes_drop(self) -> None:
        row = _method_row(
            "Raw Query BM25",
            {"run_name": "raw_clean", "metrics": {"mrr": 0.2}},
            {"run_name": "raw_adv", "metrics": {"mrr": 0.1}},
        )
        self.assertAlmostEqual(row["robustness_drop"], 0.1)
        self.assertAlmostEqual(row["relative_drop"], 0.5)


if __name__ == "__main__":
    unittest.main()
