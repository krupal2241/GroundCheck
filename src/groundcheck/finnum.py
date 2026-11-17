"""Financial number normalization: one parser for every place numbers are compared.

10-K text and LLM answers write the same amount in many ways: "$89.3 billion",
"89,300 million", a table cell "89,300" under "(in millions)", "$89.30B". Comparing
them as strings ("89.3" vs "89300") misses the match; this module parses each number
into a `Quantity` and compares values.

`parse_quantities(text)` returns, for every number in the text:

- `raw`: the number as written, sign applied ("(1,234)" and "-1,234" are -1234);
- `scale` and `scale_stated`: 1e3 / 1e6 / 1e9 / 1e12 from "thousand" / "million" /
  "billion" / "trillion" (or "$1.2B", "$5M", "$3bn" directly after a dollar amount);
  1 and False when no scale is written, as in a table cell;
- `value` = raw * scale;
- `kind`: `currency` ("$", "dollars"), `percent` ("%", "percent", "percentage points"),
  `year` (a bare four-digit 1900-2099 integer, or "fiscal 2025", "FY2025", "FY25"),
  else `count`;
- `decimals`: decimal places written, which sets the rounding tolerance;
- `range_role`: "low" / "high" for the two ends of "$25 billion to $27 billion",
  "between $25 and $27 billion" (the first number inherits the scale) or "4-5%".

`same_number(a, b, ...)` compares two quantities:

- **exact** (default): equal values. "$89.3 billion" = "$89,300 million" = "$89.30B".
- `rounding=True`: equal after rounding the more precise one to the coarser one's last
  written digit, the project's rounding rule: "$12.3 billion" matches "12,345 million",
  "$12.4 billion" does not.
- `implied=True`: when exactly one side states a scale, the other (a table cell) may be
  in units, thousands, millions or billions; `hint` (from `scale_hint`, e.g. "(in
  millions)") narrows that to one scale.
- `raw_ok=True`: equal numbers as written also match, whatever the scale words say.

Signs are ignored unless `signed=True`: a table's "(4)%" and a sentence's "decreased
4%" are the same number.

Known limits (see docs/DESIGN.md): a bare four-digit integer between 1900 and 2099 is
read as a year even when it is a count ("2025 employees"); a small integer in
parentheses is read as a footnote marker, not a negative, unless it is a whole table
cell; spelled-out numbers ("twenty") and non-dollar currencies are not parsed; "basis
points" are plain counts.
"""

import re
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Literal

Kind = Literal["currency", "percent", "count", "year"]

SCALE_WORDS = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
# Only directly after a dollar amount: "$1.2B", "$5M", "$3bn". ("10K" is a form name.)
SCALE_ABBREVIATIONS = {
    "k": 1e3,
    "m": 1e6,
    "mm": 1e6,
    "mn": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "t": 1e12,
    "tn": 1e12,
}
IMPLIED_SCALES = (1.0, 1e3, 1e6, 1e9)
SCALE_NAMES = {1e3: "thousand", 1e6: "million", 1e9: "billion", 1e12: "trillion"}

# Digits with proper thousands separators (exactly three digits after each comma), or
# plain digits; optional decimals. ".5" (no leading zero) is read as 0.5.
NUMBER_RE = re.compile(r"(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?|(?<![\d.])\.\d+")
_PERCENT_RE = re.compile(r"\)?\s?%|\)?\s(?:percent(?:age\s+points?)?|per\s+cent)\b", re.I)
_SCALE_RE = re.compile(r"\)?[\s-]?(thousand|million|billion|trillion)s?\b", re.I)
_ABBREVIATION_RE = re.compile(r"(bn|mm|mn|tn|k|m|b|t)\b", re.I)
_DOLLARS_RE = re.compile(
    r"\)?\s(?:(?:thousand|million|billion|trillion)s?\s)?(?:dollars|USD)\b", re.I
)
_FISCAL_RE = re.compile(r"\b(?:fy|fiscal(?:\s+years?)?)\s?['’]?$", re.I)
_RANGE_RE = re.compile(r"^\s*(?:to|and|through|-|–|—)\s*$", re.I)
_SCALE_HINT_RE = re.compile(
    r"\b(?:in|\$\s?in|dollars\s+in|amounts\s+in)\s+(thousand|million|billion)s\b"
    r"|\b(thousand|million|billion)s\s+of\s+(?:U\.S\.\s+)?dollars\b",
    re.I,
)
_SIGNS = "-−"
_PREFIX_CHARS = "($" + _SIGNS


@dataclass(frozen=True)
class Quantity:
    text: str  # the number as it appears, with its sign, "$", "%" and scale word
    start: int
    end: int
    raw: float  # as written, sign applied
    scale: float = 1.0
    scale_stated: bool = False
    kind: Kind = "count"
    decimals: int = 0
    fiscal: bool = False  # a year written as "fiscal 2025" / "FY2025"
    range_role: Literal["low", "high"] | None = None

    @property
    def value(self) -> float:
        return self.raw * self.scale

    @property
    def unit(self) -> float:
        """The size of the last written digit, in base units: 1e8 for "$12.3 billion"."""
        return 10.0**-self.decimals * self.scale

    @property
    def negative(self) -> bool:
        return self.raw < 0

    @property
    def token(self) -> str:
        """The number as written, canonical: no sign, separators or trailing zeros."""
        return format_number(abs(self.raw))

    @property
    def value_token(self) -> str:
        """The value in base units, canonical: "$89.3 billion" -> "89300000000"."""
        return format_number(abs(self.value))


def format_number(x: float) -> str:
    """12345.0 -> "12345", 89.30 -> "89.3", 0.5 -> "0.5": plain decimal notation with
    15 significant digits, which hides floating-point noise (89.3 * 1e9 is
    89300000000.00002) without inventing digits."""
    text = format(Decimal(f"{x:.15g}"), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def number_token(text: str) -> str:
    """Canonical form of one written number: "12,345" -> "12345", "391.0" -> "391",
    ".50" -> "0.5". Text that is not one well-formed number only loses its commas."""
    text = text.replace(",", "")
    if text.count(".") != 1:
        return text or "0"
    whole, fraction = text.split(".")
    fraction = fraction.rstrip("0")
    whole = whole or "0"
    return f"{whole}.{fraction}" if fraction else whole


def _is_cell(text: str, open_at: int, close_at: int) -> bool:
    """Whether "(...)" between the two positions is a whole table cell."""
    before = text[:open_at].rstrip(" \t$")
    after = text[close_at + 1 :].lstrip(" \t%")
    return (not before or before[-1] in "|\n") and (not after or after[0] in "|\n")


def _parse_one(text: str, match: re.Match) -> Quantity:
    start, end = match.span()
    number = match.group(0)
    decimals = len(number.split(".")[1]) if "." in number else 0
    raw = float(number.replace(",", "") if not number.startswith(".") else "0" + number)

    # Left: a run of "(", "$" and minus signs touching the number ("($1,234", "-$5").
    left = start
    while left > 0 and text[left - 1] in _PREFIX_CHARS:
        left -= 1
    prefix = text[left:start]
    before = text[left - 1] if left > 0 else ""
    currency = "$" in prefix
    # "2024-2025" and "10-K": a hyphen after a letter, digit or ")" is not a minus sign.
    minus = any(s in prefix for s in _SIGNS) and not (before.isalnum() or before == ")")
    if any(s in prefix for s in _SIGNS) and not minus:
        left = start - len(prefix.lstrip(_SIGNS)) if prefix.lstrip(_SIGNS) != prefix else start
        prefix = text[left:start]
    opened = "(" in prefix

    # Right: ")", "%" / "percent", a scale word or abbreviation, "dollars".
    rest = text[end:]
    right = end
    closed = opened and rest.startswith(")")
    percent = _PERCENT_RE.match(rest)
    scale_word = None if percent else _SCALE_RE.match(rest)
    abbreviation = None
    if currency and not percent and not scale_word:
        abbreviation = _ABBREVIATION_RE.match(rest)
    scale, stated = 1.0, False
    if percent:
        right = end + percent.end()
    elif scale_word:
        scale, stated = SCALE_WORDS[scale_word.group(1).lower()], True
        right = end + scale_word.end()
    elif abbreviation:
        scale, stated = SCALE_ABBREVIATIONS[abbreviation.group(1).lower()], True
        right = end + abbreviation.end()
    elif closed:
        right = end + 1
    if not currency and not percent and _DOLLARS_RE.match(text[right:] if stated else rest):
        currency = True

    # Accounting negatives: "(1,234)", "($5.2)", "(4)%". A bare "(4)" after a word is a
    # footnote marker unless it fills a table cell.
    accounting = closed and (
        currency
        or bool(percent)
        or "," in number
        or decimals > 0
        or len(number) >= 3
        or _is_cell(text, left + prefix.index("("), end)
    )
    if minus or accounting:
        raw = -raw
    if not (opened and closed):
        # An unmatched "(" is ordinary punctuation, not part of the number.
        left = start - len(prefix.lstrip("("))

    plain = not (currency or percent or stated or opened or minus)
    fiscal = plain and decimals == 0 and "," not in number and bool(_FISCAL_RE.search(text[:left]))
    kind: Kind
    if fiscal and len(number) in (2, 4):
        kind = "year"
        if len(number) == 2:
            raw += 2000
    elif plain and decimals == 0 and len(number) == 4 and 1900 <= raw <= 2099:
        kind, fiscal = "year", False
    elif percent:
        kind, fiscal = "percent", False
    elif currency:
        kind, fiscal = "currency", False
    else:
        kind, fiscal = "count", False
    return Quantity(
        text=text[left:right],
        start=left,
        end=right,
        raw=raw,
        scale=scale,
        scale_stated=stated,
        kind=kind,
        decimals=decimals,
        fiscal=fiscal,
    )


def parse_quantities(text: str) -> list[Quantity]:
    """Every number in `text`, in order. Never raises, whatever the text."""
    found = [_parse_one(text, m) for m in NUMBER_RE.finditer(text)]
    # Ranges: "$25 billion to $27 billion", "between $25 and $27 billion", "4-5%".
    for i in range(len(found) - 1):
        low, high = found[i], found[i + 1]
        if low.kind == "year" or high.kind == "year":
            continue
        if not _RANGE_RE.match(text[low.end : high.start]):
            continue
        if high.kind == "percent" and low.kind == "count" and not low.scale_stated:
            low = replace(low, kind="percent")
        elif high.scale_stated and not low.scale_stated and low.kind in ("count", "currency"):
            kind = "currency" if "currency" in (low.kind, high.kind) else low.kind
            low = replace(low, scale=high.scale, scale_stated=True, kind=kind)
            high = replace(high, kind=kind)
        if low.kind != high.kind:
            continue
        found[i] = replace(low, range_role="low")
        found[i + 1] = replace(high, range_role="high")
    return found


def scale_hint(text: str) -> float | None:
    """The scale a table states for its cells: "(in millions)", "$ in thousands",
    "millions of dollars". None if the text has no such note."""
    match = _SCALE_HINT_RE.search(text)
    if not match:
        return None
    return SCALE_WORDS[(match.group(1) or match.group(2)).lower()]


def _close(x: float, y: float, tolerance: float) -> bool:
    return abs(x - y) <= tolerance * (1 + 1e-9) + 1e-9 * max(abs(x), abs(y), 1e-12)


def same_number(
    a: Quantity,
    b: Quantity,
    *,
    rounding: bool = False,
    implied: bool = False,
    hint: float | None = None,
    raw_ok: bool = False,
    signed: bool = False,
) -> bool:
    """Whether two quantities are the same number; see the module docs for the modes."""

    def mag(x: float) -> float:
        return x if signed else abs(x)

    if a.kind == "year" or b.kind == "year":
        if a.kind == b.kind:
            return a.raw == b.raw
        other = b if a.kind == "year" else a
        # A year against a plain unscaled integer ("2025" in a table header cell).
        plain = other.kind == "count" and not other.scale_stated and other.decimals == 0
        return plain and abs(other.raw) == abs(a.raw if a.kind == "year" else b.raw)
    if (a.kind == "percent") != (b.kind == "percent"):
        # A percent column often has no "%" in its cells; never a scaled or dollar amount.
        other = b if a.kind == "percent" else a
        if other.kind != "count" or other.scale_stated:
            return False

    def equal(x: float, unit_x: float, y: float, unit_y: float) -> bool:
        tolerance = 0.5 * max(unit_x, unit_y) if rounding else 0.0
        return _close(mag(x), mag(y), tolerance)

    if equal(a.value, a.unit, b.value, b.unit):
        return True
    if raw_ok and equal(a.raw, 10.0**-a.decimals, b.raw, 10.0**-b.decimals):
        return True
    if implied and a.scale_stated != b.scale_stated and "percent" not in (a.kind, b.kind):
        stated, bare = (a, b) if a.scale_stated else (b, a)
        for scale in (hint,) if hint else IMPLIED_SCALES:
            if equal(stated.value, stated.unit, bare.raw * scale, 10.0**-bare.decimals * scale):
                return True
    return False


def find_match(quantity: Quantity, candidates: list[Quantity], **modes) -> Quantity | None:
    """The first of `candidates` that is the same number as `quantity`, or None."""
    return next((c for c in candidates if same_number(quantity, c, **modes)), None)


def _clean(value: float, places: int = 3) -> str | None:
    """`value` as a token if it has at most `places` decimals, else None."""
    rounded = round(value, places)
    if rounded == 0 or not _close(value, rounded, 0.0):
        return None
    return format_number(rounded)


def aliases(quantity: Quantity) -> list[str]:
    """Other ways the same amount is written, as number tokens (no separators).

    With a stated scale, the amount in units, thousands, millions and billions:
    "$89.3 billion" -> 89300 (millions), 89300000 (thousands), 89300000000. Without one,
    a number of four or more digits (a table-style amount) also in the next scale up,
    exactly and rounded to one decimal: "12,345" -> 12.345, 12.3. Years and percents
    have no aliases.
    """
    if quantity.kind in ("year", "percent"):
        return []
    out: list[str] = []
    value = abs(quantity.value)
    if quantity.scale_stated:
        for scale in (1.0, 1e3, 1e6, 1e9):
            token = _clean(value / scale)
            if token and scale != quantity.scale:
                out.append(token)
    elif abs(quantity.raw) >= 1000:
        up = abs(quantity.raw) / 1000
        out += [t for t in (_clean(up), format_number(round(up, 1))) if t]
    seen = {quantity.token}
    return [t for t in out if not (t in seen or seen.add(t))]


def expand_query(query: str) -> str:
    """The query plus the numeric aliases of its numbers (for a lexical retriever)."""
    extra: list[str] = []
    for quantity in parse_quantities(query):
        extra += [t for t in aliases(quantity) if t not in extra]
    return query if not extra else query + " " + " ".join(extra)
