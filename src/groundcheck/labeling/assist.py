"""Labeling assistance: an evidence-span suggestion and an optional verifier label.

1. **Evidence span** (deterministic, always available): every retrieved chunk is cut
   into units (table rows; sentences in prose, `verify.spans.units`) and the unit
   containing the most of the claim's numbers (weighted 3x) and content words is
   offered. It is checked to be an exact substring of its chunk after whitespace
   normalization (the gold-span rule) before it is shown. It is offered only *after*
   the annotator has chosen supported or contradicted, so it can't steer the label,
   only save copy-pasting; one key (`a`) accepts it.

2. **Label suggestion** (`groundcheck label --suggest VERIFIER`, OFF by default): a
   verifier's proposed label is shown before the annotator decides. That can anchor
   the annotator, which is why it is off by default and why every label made with it
   records what was shown and whether the annotator agreed (`schema.Assistance`), so
   `label-stats` can compare labels made with and without suggestions.
"""

from collections.abc import Callable
from dataclasses import dataclass

from groundcheck.generation.pipeline import RetrievedChunk
from groundcheck.labeling.schema import EvidenceSpan, make_span
from groundcheck.retrieval.evaluate import normalize_for_match
from groundcheck.verify.base import EvidenceChunk, Verifier, VerifierError
from groundcheck.verify.spans import overlap_score, units


@dataclass
class SpanSuggestion:
    span: EvidenceSpan
    rank: int
    score: float  # overlap in [0, 1]


def suggest_span(claim: str, chunks: list[RetrievedChunk]) -> SpanSuggestion | None:
    """The unit with the highest overlap over all chunks (ties: better-ranked chunk,
    then shorter unit, then earlier). None if nothing overlaps at all."""
    best: tuple[float, int, int, int] | None = None
    found: SpanSuggestion | None = None
    for chunk in sorted(chunks, key=lambda c: c.rank):
        for position, unit in enumerate(units(chunk.text)):
            score = overlap_score(claim, unit)
            if score <= 0:
                continue
            key = (-score, chunk.rank, len(unit), position)
            if best is not None and key >= best:
                continue
            span = make_span(chunk.chunk_id, chunk.text, unit)
            # Exact-substring check, exactly as the gold-span validator does it.
            if span is None or normalize_for_match(chunk.text)[span.start : span.end] != unit:
                continue
            best, found = key, SpanSuggestion(span, chunk.rank, round(score, 3))
    return found


@dataclass
class LabelSuggestion:
    label: str
    score: float
    verifier: str


LabelSuggester = Callable[[str, list[RetrievedChunk]], LabelSuggestion | None]


def verifier_suggester(verifier: Verifier) -> LabelSuggester:
    """Wrap a verifier: claim + the candidate's retrieved chunks -> its proposed label.
    A verifier error gives no suggestion (and the label records none was shown)."""

    def suggest(claim: str, chunks: list[RetrievedChunk]) -> LabelSuggestion | None:
        evidence = [EvidenceChunk(c.chunk_id, c.text) for c in sorted(chunks, key=lambda c: c.rank)]
        try:
            result = verifier.verify(claim, evidence)
        except VerifierError:
            return None
        return LabelSuggestion(result.label, round(result.score, 3), verifier.name)

    return suggest
