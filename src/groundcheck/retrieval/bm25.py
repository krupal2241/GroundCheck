"""Okapi BM25 over one company's chunks, in pure Python + numpy (no extra dependency).

Tokens are lowercase words and numbers (`tokenize`):

- a number is one token in the canonical form of `finnum.number_token`: thousands
  separators and trailing decimal zeros removed, so "12,345", "$12,345" and "12345" are
  the same token, "391.0" is "391", and "89.3" stays "89.3";
- a comma only joins digits when exactly three digits follow it, so "2025,2024" is two
  years, not one number;
- letters and digits are split, so "FY2025" gives "fy" and "2025" (and matches a
  heading's "2025"), "Q4" gives "q" and "4", "10-K" gives "10" and "k".

A short stopword list drops question words ("what", "how", "did") that would otherwise
favor chunks that happen to repeat them. (The hashing test embedder keeps its own,
older token pattern, `embedder.TOKEN_RE`; BM25 no longer shares it.)

Scoring is the standard Okapi BM25 with Lucene's non-negative IDF,
`log(1 + (N - df + 0.5) / (df + 0.5))`, `k1 = 1.5`, `b = 0.75`. A filing has a few
thousand chunks, so scoring every chunk for every query takes about a millisecond and
no inverted-index tricks are needed.
"""

import math
import re
from collections import Counter

import numpy as np

from groundcheck.finnum import number_token

# A number with proper thousands separators, a plain number, or a run of letters.
TOKEN_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?!\d)(?:\.\d+)?|\d+(?:\.\d+)?|[a-z]+")

_STOPWORDS = """a an and are as at be been by can could did do does for from had has have how
in is it its of on or our than that the their there these this those to was were what when
where which who will with would"""
STOPWORDS = frozenset(_STOPWORDS.split())


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in TOKEN_RE.findall(text.lower()):
        if token[0].isdigit():
            token = number_token(token)  # "12,345" -> "12345", "391.0" -> "391"
        if token not in STOPWORDS:
            tokens.append(token)
    return tokens


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
