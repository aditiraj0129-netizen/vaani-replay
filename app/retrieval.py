"""Step 3: BM25 keyword retrieval, written by hand so every line can be explained.

BM25 score of a chunk = sum over query terms t of
    idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * chunk_len / avg_len))
idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))   (never negative, unlike the textbook form)
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from app.chunking import Chunk, build_corpus
from app.config import RetrievalConfig, Settings

# Question words and glue words. "not" is deliberately kept: negation matters in questions.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "which with we our their these those can also but if than then into such what how does do why "
    "when where who whom whose about between".split()
)


def tokenize(text: str) -> List[str]:
    """Lower-case, split on anything that is not a letter or digit, drop stop words."""
    text = unicodedata.normalize("NFKC", text).lower()
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int  # 1 = best


class BM25Index:
    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks: List[Chunk] = list(chunks)
        self.k1, self.b = k1, b
        self._postings: Dict[str, List[Tuple[int, int]]] = {}  # term -> [(chunk_index, term_count)]
        self._doc_len: List[int] = []
        for i, chunk in enumerate(self.chunks):
            tokens = tokenize(chunk.text)
            self._doc_len.append(len(tokens))
            for term, count in Counter(tokens).items():
                self._postings.setdefault(term, []).append((i, count))
        n = len(self.chunks)
        self._avg_len = sum(self._doc_len) / n if n else 0.0
        self._idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self._postings.items()}

    def search(self, query: str, k: int, papers: Optional[Sequence[str]] = None) -> List[Hit]:
        """Top-k chunks with a positive score. Ties break by chunk order, so results are deterministic.

        `papers` (paper keys) restricts the search to those papers; scores still use corpus-wide statistics.
        """
        allowed = set(papers) if papers else None
        scores: Dict[int, float] = {}
        for term in set(tokenize(query)):  # a repeated query word does not count twice
            for i, tf in self._postings.get(term, []):
                if allowed is not None and self.chunks[i].paper_key not in allowed:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self._doc_len[i] / self._avg_len)
                scores[i] = scores.get(i, 0.0) + self._idf[term] * tf * (self.k1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [Hit(chunk=self.chunks[i], score=s, rank=r) for r, (i, s) in enumerate(ranked, 1)]


class Retriever:
    """One retrieval configuration = one chunk size + one top_k (+ optional re-rank, added next stage)."""

    def __init__(self, chunks: Sequence[Chunk], config: RetrievalConfig) -> None:
        self.config = config
        self.index = BM25Index(chunks)

    @classmethod
    def from_settings(cls, settings: Settings, config: RetrievalConfig) -> "Retriever":
        return cls(build_corpus(settings, config), config)

    def retrieve(self, query: str, papers: Optional[Sequence[str]] = None) -> List[Hit]:
        if not self.config.rerank:
            return self.index.search(query, self.config.top_k, papers)
        return rerank(query, self.index.search(query, self.config.candidates, papers), self.config.top_k)


# Re-ranking weights are NOT tuned on data; they are a plain, explainable blend (see rerank()).
RERANK_WEIGHTS = {"bm25": 0.5, "coverage": 0.25, "phrase": 0.15, "section": 0.10}


def _bigrams(tokens: Sequence[str]) -> set:
    return set(zip(tokens, tokens[1:]))


def rerank(query: str, candidates: Sequence[Hit], top_k: int) -> List[Hit]:
    """Re-order BM25 candidates with three extra signals, then keep the best top_k.

    score = 0.50 * bm25 / best_bm25            (the original evidence, scaled to 0..1)
          + 0.25 * share of distinct query terms that appear in the chunk       (coverage)
          + 0.15 * share of query word-pairs that appear next to each other     (phrase match)
          + 0.10 * share of query terms that appear in the chunk's section title
    """
    if not candidates:
        return []
    q_tokens = tokenize(query)
    q_terms = set(q_tokens)
    q_pairs = _bigrams(q_tokens)
    best = max(h.score for h in candidates) or 1.0
    rescored = []
    for h in candidates:
        c_tokens = tokenize(h.chunk.text)
        coverage = len(q_terms & set(c_tokens)) / len(q_terms) if q_terms else 0.0
        phrase = len(q_pairs & _bigrams(c_tokens)) / len(q_pairs) if q_pairs else 0.0
        section = len(q_terms & set(tokenize(h.chunk.section))) / len(q_terms) if q_terms else 0.0
        w = RERANK_WEIGHTS
        score = w["bm25"] * h.score / best + w["coverage"] * coverage + w["phrase"] * phrase + w["section"] * section
        rescored.append((score, h))
    rescored.sort(key=lambda item: (-item[0], item[1].rank))
    return [Hit(chunk=h.chunk, score=sc, rank=r) for r, (sc, h) in enumerate(rescored[:top_k], 1)]
