"""Okapi BM25 over one company's chunks, in pure Python + numpy (no extra dependency).

Tokens are the same as the hashing embedder's: lowercase words and numbers, with
thousands separators and decimals kept inside one token, so a question's "64,377" or
"2025" matches a table row's `$64,377` or a heading's `2025`. A short stopword list
drops question words ("what", "how", "did") that would otherwise favor chunks that
happen to repeat them.

Scoring is the standard Okapi BM25 with Lucene's non-negative IDF,
`log(1 + (N - df + 0.5) / (df + 0.5))`, `k1 = 1.5`, `b = 0.75`. A filing has a few
thousand chunks, so scoring every chunk for every query takes about a millisecond and
no inverted-index tricks are needed.
"""

import math
from collections import Counter

import numpy as np

from groundcheck.retrieval.embedder import TOKEN_RE

_STOPWORDS = """a an and are as at be been by can could did do does for from had has have how
in is it its of on or our than that the their there these this those to was were what when
where which who will with would"""
STOPWORDS = frozenset(_STOPWORDS.split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(documents)
        self._postings: dict[str, list[tuple[int, int]]] = {}
        lengths = []
        for i, text in enumerate(documents):
            counts = Counter(tokenize(text))
            lengths.append(sum(counts.values()))
            for token, tf in counts.items():
                self._postings.setdefault(token, []).append((i, tf))
        self._lengths = np.asarray(lengths, dtype=np.float64)
        self._avg_length = float(self._lengths.mean()) if self.n else 0.0

    def idf(self, token: str) -> float:
        df = len(self._postings.get(token, ()))
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float64)
        if not self.n:
            return out
        norm = self.k1 * (1 - self.b + self.b * self._lengths / (self._avg_length or 1.0))
        for token in set(tokenize(query)):
            postings = self._postings.get(token)
            if not postings:
                continue
            ids = np.fromiter((i for i, _ in postings), dtype=np.int64, count=len(postings))
            tf = np.fromiter((f for _, f in postings), dtype=np.float64, count=len(postings))
            out[ids] += self.idf(token) * tf * (self.k1 + 1) / (tf + norm[ids])
        return out

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        """(document position, score) for the best `k` documents with a score above 0,
        best first; ties keep document order."""
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
