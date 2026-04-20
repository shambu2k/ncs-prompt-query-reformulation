from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.schema import CanonicalRecord
from src.evaluation.official_wrapper import score_prediction_file
from src.retrieval.bm25 import build_rankings


def _canonical(idx: int, query: list[str], code: list[str]) -> CanonicalRecord:
    return CanonicalRecord(
        example_id=str(idx),
        query_text=" ".join(query),
        code_text=" ".join(code),
        query_tokens=query,
        code_tokens=code,
        language="python",
        split="valid",
        condition="adv",
        source_task="unit_test",
        metadata={"idx": idx, "url": f"url-{idx}"},
    )


class Bm25EvaluationTests(unittest.TestCase):
    def test_build_rankings_is_deterministic(self) -> None:
        docs = [["alpha", "beta"], ["gamma", "delta"], ["alpha", "gamma"]]
        queries = [["alpha"], ["gamma"]]
        first = build_rankings(docs, queries, top_k=3)
        second = build_rankings(docs, queries, top_k=3)
        self.assertEqual(first, second)

    def test_official_wrapper_scores_perfect_predictions(self) -> None:
        records = [
            _canonical(10, ["alpha"], ["alpha", "beta"]),
            _canonical(11, ["gamma"], ["gamma", "delta"]),
        ]
        predictions = [
            {"url": "url-10", "answers": [10, 11]},
            {"url": "url-11", "answers": [11, 10]},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            prediction_path = Path(temp_dir) / "predictions.jsonl"
            with prediction_path.open("w", encoding="utf-8") as handle:
                for row in predictions:
                    handle.write(json.dumps(row))
                    handle.write("\n")

            scores = score_prediction_file(records, prediction_path)
        self.assertEqual(scores["MRR"], 1.0)


if __name__ == "__main__":
    unittest.main()
