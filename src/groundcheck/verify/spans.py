"""Pick the evidence span inside a chunk, and locate a quoted span.

Cross-encoder verifiers (MiniCheck, HHEM, the fine-tuned model) score a claim against
whole chunks; they say *which chunk* but not which sentence. To give every verdict a
citable span, the chunk is cut into units (table rows, and sentences inside prose
lines) and the unit that best covers the claim's words is returned. Numbers count
three times as much as words, because in 10-K claims the number is what the verdict
hinges on. This is a cheap lexical choice, made only *after* the model has picked the
chunk; it never changes the label.

Spans use the same convention as gold labels: whitespace-normalized text with offsets
into the normalized chunk (`labeling.schema.EvidenceSpan`), so a predicted span can be
compared with a gold span directly.
"""

import re

from groundcheck.labeling.schema import EvidenceSpan, make_span
from groundcheck.retrieval.evaluate import normalize_for_match

NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
WORD_RE = re.compile(r"[a-z][a-z'’-]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\"“$])")
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
        "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was",
        "were", "which", "with", "during", "company", "company's", "year", "fiscal",
        "total",
    }
)  # fmt: skip
TABLE_MARKERS = ("[TABLE]", "[/TABLE]")


def numbers(text: str) -> set[str]:
    """Numbers as written, without thousands separators: '$12,345' -> '12345'."""
    return {n.replace(",", "") for n in NUMBER_RE.findall(text)}


def content_words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS}


def units(chunk_text: str) -> list[str]:
    """Table rows and prose sentences, in order, each whitespace-normalized."""
    out = []
    for line in chunk_text.splitlines():
        line = line.strip()
        if not line or line in TABLE_MARKERS:
            continue
        if " | " in line:
            out.append(normalize_for_match(line))
            continue
        out.extend(normalize_for_match(s) for s in SENTENCE_SPLIT_RE.split(line) if s.strip())
    return [u for u in out if u]


def overlap_score(claim: str, unit: str) -> float:
    """Share of the claim's numbers (x3) and content words found in `unit`, in [0, 1]."""
    claim_numbers, claim_words = numbers(claim), content_words(claim)
    total = 3 * len(claim_numbers) + len(claim_words)
    if total == 0:
        return 0.0
    unit_numbers, unit_words = numbers(unit), content_words(unit)
    hit = 3 * len(claim_numbers & unit_numbers) + len(claim_words & unit_words)
    return hit / total


def best_unit_span(claim: str, chunk_id: str, chunk_text: str) -> EvidenceSpan | None:
    """The unit of the chunk that best covers the claim (ties: the earlier, shorter one)."""
    candidates = units(chunk_text)
    if not candidates:
        return None
    best = max(candidates, key=lambda u: (overlap_score(claim, u), -len(u)))
    return make_span(chunk_id, chunk_text, best)


def find_quote(quote: str, chunks: list[tuple[str, str]]) -> EvidenceSpan | None:
    """Locate a quoted span (e.g. from an LLM) in the chunks; first chunk that has it."""
    for chunk_id, text in chunks:
        span = make_span(chunk_id, text, quote)
        if span is not None:
            return span
    return None
