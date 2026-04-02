#!/usr/bin/env python3
"""Run a lightweight BM25 baseline for CodeXGLUE NL-code-search-Adv."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def document_tokens(record: dict) -> list[str]:
    if "function_tokens" in record:
        return record["function_tokens"]
    if "code_tokens" in record:
        return record["code_tokens"]
    raise KeyError("Expected either function_tokens or code_tokens in record.")


def query_tokens(record: dict) -> list[str]:
    if "docstring_tokens" in record:
        return record["docstring_tokens"]
    raise KeyError("Expected docstring_tokens in record.")


class BM25Index:
    def __init__(self, corpus: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avg_doc_length = sum(self.doc_lengths) / max(1, self.doc_count)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self._build()

    def _build(self) -> None:
        doc_freq = Counter()
        for doc_id, doc in enumerate(self.corpus):
            term_counts = Counter(doc)
            for term, count in term_counts.items():
                self.postings[term].append((doc_id, count))
            doc_freq.update(term_counts.keys())

        for term, freq in doc_freq.items():
            # Standard Okapi BM25 IDF with a +1 inside the log to keep values positive.
            self.idf[term] = math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))

    def score(self, query: list[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        term_counts = Counter(query)
        for term, query_tf in term_counts.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, doc_tf in self.postings[term]:
                doc_len = self.doc_lengths[doc_id]
                denom = doc_tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_length)
                scores[doc_id] += query_tf * idf * (doc_tf * (self.k1 + 1.0) / denom)
        return scores


def ranked_doc_ids(index: BM25Index, query: list[str], top_k: int) -> list[int]:
    scores = index.score(query)
    ranked = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]]
    if len(ranked) == top_k:
        return ranked

    seen = set(ranked)
    for doc_id in range(index.doc_count):
        if doc_id not in seen:
            ranked.append(doc_id)
            if len(ranked) == top_k:
                break
    return ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="baseline/data",
        help="Directory containing valid.jsonl/test.jsonl prepared by prepare_codexglue_adv.py.",
    )
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        required=True,
        help="Dataset split to score.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Prediction output path. Defaults to baseline/out/<split>.predictions.jsonl.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of candidate indices to emit per query.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    split_path = data_dir / f"{args.split}.jsonl"
    output_path = Path(args.output or f"baseline/out/{args.split}.predictions.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(split_path)
    docs = [document_tokens(row) for row in rows]
    queries = [query_tokens(row) for row in rows]
    idx_by_doc_id = [row["idx"] for row in rows]
    urls_by_query_id = [row["url"] for row in rows]

    index = BM25Index(docs)
    with output_path.open("w", encoding="utf-8") as handle:
        for query_id, query in enumerate(queries):
            ranked = ranked_doc_ids(index, query, args.top_k)
            payload = {
                "url": urls_by_query_id[query_id],
                "answers": [idx_by_doc_id[doc_id] for doc_id in ranked],
            }
            handle.write(json.dumps(payload))
            handle.write("\n")

    print(
        json.dumps(
            {
                "split": args.split,
                "examples": len(rows),
                "output": str(output_path),
                "top_k": args.top_k,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
