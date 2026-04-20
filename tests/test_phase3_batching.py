from __future__ import annotations

import unittest

from src.evolution.scorer import _parse_batch_response as parse_score_batch_response
from src.rewriting.rewrite_queries import _parse_batch_response as parse_rewrite_batch_response


class Phase3BatchingTests(unittest.TestCase):
    def test_score_batch_parser_accepts_json(self) -> None:
        parsed = parse_score_batch_response('{"a":"rewrite one","b":"rewrite two"}', ["a", "b"])
        self.assertEqual(parsed["a"], "rewrite one")
        self.assertEqual(parsed["b"], "rewrite two")

    def test_score_batch_parser_accepts_code_fenced_json(self) -> None:
        parsed = parse_score_batch_response(
            '```json\n{"a":"rewrite one","b":"rewrite two"}\n```',
            ["a", "b"],
        )
        self.assertEqual(set(parsed), {"a", "b"})

    def test_rewrite_batch_parser_rejects_missing_id(self) -> None:
        with self.assertRaises(ValueError):
            parse_rewrite_batch_response('{"a":"rewrite one"}', ["a", "b"])


if __name__ == "__main__":
    unittest.main()
