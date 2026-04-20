"""Deterministic BM25 retrieval helpers."""

from __future__ import annotations

import math
from collections import Counter, defaultdict


class BM25Index:
    """Simple Okapi BM25 index with stable tie-breaking."""

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
    ranked = [
        doc_id
        for doc_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    ]
    if len(ranked) == top_k:
        return ranked

    seen = set(ranked)
    for doc_id in range(index.doc_count):
        if doc_id not in seen:
            ranked.append(doc_id)
            if len(ranked) == top_k:
                break
    return ranked


def build_rankings(
    document_tokens: list[list[str]],
    query_tokens: list[list[str]],
    *,
    top_k: int,
) -> list[list[int]]:
    index = BM25Index(document_tokens)
    return [ranked_doc_ids(index, query, top_k) for query in query_tokens]
