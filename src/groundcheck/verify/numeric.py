"""`NumericVerifier`: a rule-based baseline that only checks numbers. No model.

It reads every number in the claim and looks for it in the passages under unit and
scale normalization (`groundcheck.finnum`): "$89.3 billion" matches a table cell
"89,300" (in millions), and correct rounding to the claim's own precision counts
("$12.3 billion" for "12,345"). A number only counts where the surrounding words match
the claim ("context"), and where the period matches when both sides name one.

How evidence is read:

- **Table rows.** The row label is the context; each cell is a fact; the cell's period
  is the year in its column header, when the header rows line up with the row. The
  table's scale note ("(in millions)") is used when the chunk has one; without it a
  bare cell may be in units, thousands, millions or billions.
- **Prose sentences.** The sentence's words are the context; each number's period is
  the nearest year in the sentence (a year right after the number wins).

The verdict:

- **supported** if every number in the claim (years aside) is found in a matching
  context, for the same period;
- **contradicted** if some number is not found, but a matching context has a different
  number of the same kind (percent vs amount) for the same period; or the number is
  found in a table cell whose sign disagrees with the claim's direction word
  ("increased" against "(4)%");
- **unsupported** otherwise, including **every claim without a number**: this verifier
  has no opinion on those and says so in `details["no_numbers"]`.

Honest limits (docs/DESIGN.md, "NumericVerifier"):

- it does not read: a claim whose number, period and row label all match is supported
  even if the sentence says something else about that number (wrong entity in the same
  row label, "excluding", "pro forma", a forecast instead of a result);
- negation and direction are only checked against signed table cells, never in prose;
- claims without numbers are always unsupported, so its recall on supported
  qualitative claims is zero by construction;
- derived numbers (a difference, a sum, a growth rate the passage does not print) are
  not computed: a correct derived claim comes out contradicted or unsupported;
- context matching is word overlap, so a different metric with similar words ("net
  sales" vs "cost of sales") can supply a "different number" and a false contradicted;
- without a scale note, a scale error ("billion" for "million") cannot be seen against
  a bare table cell.

It costs nothing to run (microseconds per claim) and is reported next to the model
verifiers as the floor a model has to beat on numeric claims.
"""

import re
from dataclasses import dataclass

from groundcheck.finnum import Quantity, parse_quantities, same_number, scale_hint
from groundcheck.labeling.schema import make_span
from groundcheck.retrieval.evaluate import normalize_for_match
from groundcheck.verify.base import BaseVerifier, EvidenceChunk, Verification
from groundcheck.verify.spans import SENTENCE_SPLIT_RE, TABLE_MARKERS, WORD_RE

# Words that carry no context: function words, unit words and reporting verbs.
FILLER = frozenset(
    """
    a an and are as at be by for from has have had in is it its of on or that the this to
    was were which with during company company's year years fiscal total approximately
    about million billion thousand trillion percent dollars reported recorded amounted
    compared ended quarter annual respectively also than
    """.split()  # noqa: SIM905
)
UP = frozenset("increased increase rose grew grow growth higher up gain gained".split())  # noqa: SIM905
DOWN = frozenset(
    "decreased decrease declined decline fell dropped lower down loss lost".split()  # noqa: SIM905
)
PERIOD_WINDOW = 60  # characters: how far a year may be from the number it dates


def words(text: str) -> set[str]:
    return {w.removesuffix("'s").removesuffix("’s") for w in WORD_RE.findall(text.lower())} - FILLER


@dataclass
class Fact:
    """One number in the evidence, with what it is about."""

    chunk_id: str
    chunk_text: str
    unit: str  # the table row or sentence, whitespace-normalized
    quantity: Quantity
    context: set[str]  # row label words (table) or sentence words (prose)
    period: int | None  # the year this number belongs to, if known
    table: bool
    hint: float | None  # the table's stated scale, if any


def nearest_year(quantity: Quantity, years: list[Quantity]) -> int | None:
    """The year a number in running text belongs to: the closest one, with a year that
    follows the number ("$5 million in 2025") preferred over one that precedes it."""
    best, best_distance = None, None
    for year in years:
        after = year.start >= quantity.end
        distance = (year.start - quantity.end) if after else (quantity.start - year.end) + 15
        if distance > PERIOD_WINDOW:
            continue
        if best_distance is None or distance < best_distance:
            best, best_distance = int(year.raw), distance
    if best is None and len({int(y.raw) for y in years}) == 1:
        return int(years[0].raw)  # the only year mentioned
    return best


def split_quantities(text: str) -> tuple[list[Quantity], list[Quantity]]:
    """(values, years) in `text`."""
    found = parse_quantities(text)
    return [q for q in found if q.kind != "year"], [q for q in found if q.kind == "year"]


def table_facts(chunk: EvidenceChunk, rows: list[str], hint: float | None) -> list[Fact]:
    cells = [[c.strip() for c in row.split("|")] for row in rows]
    # Header rows have an empty label cell; they run until the first labeled row.
    n_header = 0
    for row in cells:
        if row[0]:
            break
        n_header += 1
    columns: dict[int, int] = {}
    for header in cells[:n_header]:
        for j, cell in enumerate(header):
            years = [q for q in parse_quantities(cell) if q.kind == "year"]
            if j and years:
                columns[j] = int(years[-1].raw)
    facts = []
    for row, line in zip(cells[n_header:], rows[n_header:], strict=True):
        context = words(row[0])
        if not context:
            continue
        aligned = all(j < len(row) for j in columns)
        for j, cell in enumerate(row[1:], start=1):
            for quantity in parse_quantities(cell):
                if quantity.kind == "year":
                    continue
                period = columns.get(j) if aligned else None
                facts.append(
                    Fact(
                        chunk.chunk_id,
                        chunk.text,
                        normalize_for_match(line),
                        quantity,
                        context,
                        period,
                        True,
                        hint,
                    )  # fmt: skip
                )
    return facts


def prose_facts(chunk: EvidenceChunk, line: str) -> list[Fact]:
    facts = []
    for sentence in SENTENCE_SPLIT_RE.split(line):
        sentence = normalize_for_match(sentence)
        values, years = split_quantities(sentence)
        context = words(sentence)
        for quantity in values:
            facts.append(
                Fact(
                    chunk.chunk_id,
                    chunk.text,
                    sentence,
                    quantity,
                    context,
                    nearest_year(quantity, years),
                    False,
                    None,
                )  # fmt: skip
            )
    return facts


def chunk_facts(chunk: EvidenceChunk) -> list[Fact]:
    hint = scale_hint(chunk.text)
    facts: list[Fact] = []
    table: list[str] = []
    for line in [*chunk.text.splitlines(), ""]:
        line = line.strip()
        is_row = " | " in line or line.startswith("|") or line.endswith("|")
        if is_row and line not in TABLE_MARKERS:
            table.append(line)
            continue
        if table:
            facts += table_facts(chunk, table, hint)
            table = []
        if line and line not in TABLE_MARKERS:
            facts += prose_facts(chunk, line)
    return facts


def context_score(claim_words: set[str], fact: Fact) -> float:
    """How well a fact's context matches the claim, in [0, 1]. A table row label must be
    (mostly) contained in the claim; a sentence must contain (much of) the claim."""
    if not claim_words or not fact.context:
        return 0.0
    shared = len(claim_words & fact.context)
    return shared / len(fact.context) if fact.table else shared / len(claim_words)


def direction(claim: str) -> int:
    """+1 if the claim says something rose, -1 if it fell, 0 if neither or both."""
    found = set(re.findall(r"[a-z]+", claim.lower()))
    up, down = bool(found & UP), bool(found & DOWN)
    return (up > down) - (down > up)


class NumericVerifier(BaseVerifier):
    name = "numeric"
    model = "rules: number matching (no model)"
    local = True

    def __init__(self, table_context: float = 0.6, prose_context: float = 0.5):
        self.table_context = table_context
        self.prose_context = prose_context

    def identity(self) -> dict:
        return {
            "verifier": self.name,
            "model": self.model,
            "table_context": self.table_context,
            "prose_context": self.prose_context,
        }

    def _matches_context(self, claim_words: set[str], fact: Fact) -> bool:
        needed = self.table_context if fact.table else self.prose_context
        return context_score(claim_words, fact) >= needed

    def _verify(self, claim: str, chunks: list[EvidenceChunk]) -> Verification:
        values, years = split_quantities(claim)
        if not values:
            return Verification(
                label="unsupported", score=0.3, details={"no_numbers": True, "rule": "no opinion"}
            )
        claim_words = words(claim)
        trend = direction(claim)
        facts = [f for c in chunks for f in chunk_facts(c)]
        relevant = [f for f in facts if self._matches_context(claim_words, f)]
        found: list[Fact] = []
        conflicts: list[Fact] = []
        missing = 0
        for value in values:
            period = nearest_year(value, years)
            same_period = [
                f for f in relevant if period is None or f.period is None or f.period == period
            ]
            hits = [
                f
                for f in same_period
                if same_number(value, f.quantity, rounding=True, implied=True, hint=f.hint)
            ]
            signed = [f for f in hits if self._sign_conflict(trend, value, f)]
            if hits and len(signed) < len(hits):
                found.append(max((f for f in hits if f not in signed), key=self._rank(claim_words)))
                continue
            missing += 1
            # A different number of the same kind, in the same context and period. A
            # prose fact with no period of its own does not contradict a dated claim.
            rivals = signed + [
                f
                for f in same_period
                if f not in hits
                and (f.quantity.kind == "percent") == (value.kind == "percent")
                and (period is None or f.period == period)
            ]
            if rivals:
                conflicts.append(max(rivals, key=self._rank(claim_words)))
        details = {
            "claim_numbers": len(values),
            "found": len(found),
            "conflicting": len(conflicts),
            "facts_in_context": len(relevant),
        }
        if missing == 0:
            best = max(found, key=self._rank(claim_words))
            score = 0.5 + 0.5 * context_score(claim_words, best)
            return self._verdict("supported", score, best, details)
        if conflicts:
            best = max(conflicts, key=self._rank(claim_words))
            return self._verdict("contradicted", 0.05, best, details)
        return Verification(label="unsupported", score=0.3, details=details)

    @staticmethod
    def _sign_conflict(trend: int, value: Quantity, fact: Fact) -> bool:
        """A table cell whose sign disagrees with the claim's direction word."""
        if not trend or not fact.table or value.kind != "percent":
            return False
        return (trend > 0) == fact.quantity.negative

    @staticmethod
    def _rank(claim_words: set[str]):
        return lambda fact: (context_score(claim_words, fact), fact.table, -len(fact.unit))

    @staticmethod
    def _verdict(label: str, score: float, fact: Fact, details: dict) -> Verification:
        return Verification(
            label=label,
            score=round(min(max(score, 0.0), 1.0), 4),
            evidence=make_span(fact.chunk_id, fact.chunk_text, fact.unit),
            details={**details, "evidence_chunk": fact.chunk_id, "matched": fact.quantity.text},
        )
