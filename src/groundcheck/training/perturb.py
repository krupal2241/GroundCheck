"""Rule-based claim rewrites for synthetic training data.

Every function takes a supported claim, the evidence text it is supported by and a
`random.Random`, and returns the rewritten claim, or None when the rule does not apply
(or would accidentally produce a claim the evidence still supports). The label each
rewrite is meant to produce follows docs/labeling_guide.md:

Still **supported** (the meaning is unchanged):
- `synonym`: increased -> rose, decreased -> declined, approximately -> about, ...
- `rescale`: "$12,345 million" -> "$12.3 billion" (correct rounding to the claim's own
  precision is supported, PLAN.md's rounding rule).

**Contradicted** (the evidence shows the new claim is false):
- `numeric`: one number (not a year) changed by 10-50%, formatted like the original,
  and only if the new number does not appear anywhere in the evidence;
- `scale`: million <-> billion, thousand -> million (a scale error is contradicted);
- `negation`: increased <-> decreased, higher <-> lower, ... or "was" -> "was not";
- `year_swap`: a year replaced by *another year that appears in the evidence*, so the
  value is attached to the wrong period. (A year the evidence never mentions would make
  the claim unsupported, not contradicted, so such swaps are not made.);
- `entity_swap`: a table row label replaced by another row label of the same table
  whose values differ, or a capitalized name replaced by another name from the
  evidence.

The rules are heuristics, so some examples will be mislabeled; docs/DESIGN.md lists
the known failure modes. They never see the hand-labeled gold set.
"""

import random
import re

from groundcheck.verify.spans import NUMBER_RE, numbers

YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
SCALE_RE = re.compile(r"\b(thousand|million|billion)\b", re.IGNORECASE)
RESCALE_RE = re.compile(r"\$(\d{1,3}(?:,\d{3})+|\d{4,})(?:\.\d+)?\s+million\b")
NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")

SYNONYMS = {
    "increased": "rose",
    "decreased": "declined",
    "approximately": "about",
    "primarily": "mainly",
    "purchased": "bought",
    "reported": "recorded",
    "compared with": "versus",
    "compared to": "versus",
}
ANTONYMS = {
    "increased": "decreased",
    "decreased": "increased",
    "increase": "decrease",
    "decrease": "increase",
    "higher": "lower",
    "lower": "higher",
    "rose": "fell",
    "fell": "rose",
    "grew": "declined",
    "gain": "loss",
    "loss": "gain",
    "above": "below",
    "below": "above",
}
NEGATABLE_RE = re.compile(r"\b(was|were|is|are|did|has|have)\b(?! not)")
SCALE_SWAP = {"million": "billion", "billion": "million", "thousand": "million"}
COMMON_CAPITALS = frozenset(
    {
        "The", "This", "These", "That", "In", "On", "For", "During", "As", "Our", "We",
        "Total", "Net", "Item", "Part", "Company", "Table", "Fiscal", "Year", "Note",
        "Notes", "See", "Such", "Any", "All", "Other",
    }
)  # fmt: skip


def _replace_word(text: str, mapping: dict[str, str], rng: random.Random) -> str | None:
    hits = [
        (m, key)
        for key in mapping
        for m in re.finditer(rf"\b{re.escape(key)}\b", text, flags=re.IGNORECASE)
    ]
    if not hits:
        return None
    match, key = rng.choice(hits)
    word = mapping[key]
    if match.group(0)[0].isupper():
        word = word[0].upper() + word[1:]
    return text[: match.start()] + word + text[match.end() :]


def synonym(claim: str, evidence: str, rng: random.Random) -> str | None:
    return _replace_word(claim, SYNONYMS, rng)


def rescale(claim: str, evidence: str, rng: random.Random) -> str | None:
    """ "$12,345 million" -> "$12.3 billion" (rounded to one decimal)."""
    match = RESCALE_RE.search(claim)
    if not match:
        return None
    millions = float(match.group(0)[1:].split()[0].replace(",", ""))
    billions = f"{millions / 1000:.1f}".rstrip("0").rstrip(".")
    return claim[: match.start()] + f"${billions} billion" + claim[match.end() :]


def format_like(value: float, original: str) -> str:
    """Format `value` with the original's thousands separators and decimal places."""
    decimals = len(original.split(".")[1]) if "." in original else 0
    text = f"{value:,.{decimals}f}" if "," in original else f"{value:.{decimals}f}"
    return text


def numeric(claim: str, evidence: str, rng: random.Random) -> str | None:
    candidates = [
        m for m in NUMBER_RE.finditer(claim) if not YEAR_RE.fullmatch(m.group(0).replace(",", ""))
    ]
    candidates = [m for m in candidates if float(m.group(0).replace(",", "")) > 0]
    if not candidates:
        return None
    match = rng.choice(candidates)
    original = match.group(0)
    value = float(original.replace(",", ""))
    present = numbers(evidence)
    for factor in rng.sample([0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5], 8):
        new = format_like(value * factor, original)
        if new != original and new.replace(",", "") not in present:
            return claim[: match.start()] + new + claim[match.end() :]
    return None


def scale(claim: str, evidence: str, rng: random.Random) -> str | None:
    match = SCALE_RE.search(claim)
    if not match:
        return None
    word = SCALE_SWAP[match.group(1).lower()]
    return claim[: match.start()] + word + claim[match.end() :]


def negation(claim: str, evidence: str, rng: random.Random) -> str | None:
    flipped = _replace_word(claim, ANTONYMS, rng)
    if flipped is not None:
        return flipped
    if re.search(r"\bnot\b|n't\b", claim):
        return None
    match = NEGATABLE_RE.search(claim)
    if not match:
        return None
    return claim[: match.end()] + " not" + claim[match.end() :]


def year_swap(claim: str, evidence: str, rng: random.Random) -> str | None:
    claim_years = YEAR_RE.findall(claim)
    if not claim_years:
        return None
    others = sorted(set(YEAR_RE.findall(evidence)) - set(claim_years))
    if not others:
        return None
    old, new = rng.choice(claim_years), rng.choice(others)
    return re.sub(rf"\b{old}\b", new, claim, count=1)


def table_rows(evidence: str) -> list[list[str]]:
    rows = []
    for line in evidence.splitlines():
        if " | " in line:
            rows.append([cell.strip() for cell in line.split("|")])
    return rows


def entity_swap(claim: str, evidence: str, rng: random.Random) -> str | None:
    rows = [r for r in table_rows(evidence) if r and r[0] and re.search(r"[A-Za-z]", r[0])]
    labels = {r[0]: numbers(" ".join(r[1:])) for r in rows}
    in_claim = [label for label in labels if label in claim]
    if in_claim:
        old = max(in_claim, key=len)
        claim_numbers = numbers(claim)
        others = [
            label
            for label, values in labels.items()
            if label != old and label not in claim and not (claim_numbers & values)
        ]
        if others:
            return claim.replace(old, rng.choice(sorted(others)), 1)
        return None
    names = [n for n in NAME_RE.findall(claim) if n.split()[0] not in COMMON_CAPITALS]
    pool = sorted(
        {n for n in NAME_RE.findall(evidence) if n.split()[0] not in COMMON_CAPITALS} - set(names)
    )
    if not names or not pool:
        return None
    return claim.replace(rng.choice(names), rng.choice(pool), 1)


SUPPORTED_REWRITES = {"synonym": synonym, "rescale": rescale}
CONTRADICTING_REWRITES = {
    "numeric": numeric,
    "scale": scale,
    "negation": negation,
    "year_swap": year_swap,
    "entity_swap": entity_swap,
}
