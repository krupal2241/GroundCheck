"""`FakeVerifier`: a deterministic stand-in for tests, `--dry-run` and the offline demo.

By default it is a crude lexical rule (no model):

- find the unit (table row or sentence) that best covers the claim's words and numbers;
- supported if it covers at least `support_at` of them and contains every number in
  the claim (years aside: in a table the year is in the header row, not the value row);
- contradicted if the claim has numbers, the unit covers its words well
  (`words_at`) but a number differs;
- otherwise unsupported.

Tests can pass `labels` (claim text -> label) or a `responder(claim, chunks) -> label`
instead. It never touches the network or a model.
"""

import re
from collections.abc import Callable

from groundcheck.verify.base import BaseVerifier, EvidenceChunk, Label, Verification
from groundcheck.verify.spans import (
    best_unit_span,
    content_words,
    numbers,
    overlap_score,
)

YEAR_RE = re.compile(r"(?:19|20)\d\d")


def values(text: str) -> set[str]:
    """The text's numbers without years."""
    return {n for n in numbers(text) if not YEAR_RE.fullmatch(n)}


class FakeVerifier(BaseVerifier):
    name = "fake"
    model = "lexical-rule"
    local = True

    def __init__(
        self,
        labels: dict[str, Label] | None = None,
        responder: Callable[[str, list[EvidenceChunk]], Label] | None = None,
        support_at: float = 0.8,
        words_at: float = 0.6,
    ):
        self.labels = labels
        self.responder = responder
        self.support_at = support_at
        self.words_at = words_at
        self.calls: list[tuple[str, list[str]]] = []

    def _verify(self, claim: str, chunks: list[EvidenceChunk]) -> Verification:
        self.calls.append((claim, [c.chunk_id for c in chunks]))
        spans = [s for c in chunks if (s := best_unit_span(claim, c.chunk_id, c.text))]
        best = max(spans, key=lambda s: overlap_score(claim, s.text), default=None)
        coverage = overlap_score(claim, best.text) if best else 0.0
        if self.labels is not None or self.responder is not None:
            label = (
                self.labels.get(claim, "unsupported")
                if self.labels is not None
                else self.responder(claim, chunks)
            )
        else:
            label = self._rule(claim, best.text if best else "", coverage)
        score = {"supported": max(coverage, 0.5), "contradicted": 0.1, "unsupported": 0.3}[label]
        evidence = best if label != "unsupported" else None
        return Verification(
            label=label, score=round(min(score, 1.0), 4), evidence=evidence,
            details={"coverage": round(coverage, 4)},
        )  # fmt: skip

    def _rule(self, claim: str, unit: str, coverage: float) -> Label:
        claim_values, unit_values = values(claim), values(unit)
        words = content_words(claim)
        word_cover = len(words & content_words(unit)) / len(words) if words else 0.0
        if claim_values <= unit_values and (
            coverage >= self.support_at or (claim_values and word_cover >= self.words_at)
        ):
            return "supported"
        if claim_values and word_cover >= self.words_at:
            return "contradicted"
        return "unsupported"
