"""Split a filing's plain text (from `html_to_text`) into 10-K Item sections.

The input is a list of blocks separated by blank lines; `[TABLE]` blocks contain
single newlines only, so splitting on blank lines never cuts a table apart.

A block is a *heading candidate* if it starts with "Item 7.", "ITEM 1A", "Items 1 and
2." and so on, and is short. A table that is a single row (some filers lay a heading
out as `Item 7. | Management's Discussion ...`) also counts. A multi-row table never
does, so a table of contents laid out as a table is ignored entirely.

Every Item heading usually appears at least twice: once in the table of contents and
once where the section really starts (and sometimes again as "(continued)" on later
pages). Items are picked in canonical order (1, 1A, 1B, ..., 16), each one after the
previous pick:

1. the first candidate after the previous pick that is followed by real content
   (at least `MIN_SECTION_CHARS` before the next candidate), else
2. the last candidate after the previous pick (an item whose whole body is "None.").

Table-of-contents lines are followed by almost nothing, so rule 1 skips them, and the
ordering means a TOC entry can't be picked once the real sections have started.

Each section runs from its heading to the next picked heading. Missing expected items
(1, 1A, 7, 8) are logged as WARNINGs and returned, never silently skipped.

`check_section_sizes` adds a second tripwire: a key Item that is suspiciously short (a
cross-reference stub) or any other Item that is suspiciously long (it swallowed text
that belongs elsewhere). The thresholds are the `SECTION SIZE THRESHOLDS` block below.
"""

import logging
import re
from dataclasses import dataclass, field

from groundcheck.parsing.html_text import CELL_SEPARATOR, TABLE_START

logger = logging.getLogger(__name__)

EXPECTED_ITEMS = ("1", "1A", "7", "8")

# --- SECTION SIZE THRESHOLDS ------------------------------------------------------------
# Items 1, 1A, 7 and 8 are the long, substantive parts of every 10-K. One of them under
# MIN_KEY_SECTION_CHARS is almost always a cross-reference stub ("see pages 50-160"),
# e.g. JPM Item 7 (396 chars) and XOM Item 8 (710 chars). Any *other* Item over
# MAX_OTHER_SECTION_CHARS has almost always swallowed text that belongs to another
# Item, e.g. JPM Item 15 (1,050,977 chars) holding the whole annual report. Real
# sections that must stay quiet: JNJ Item 8 (227,361), BRK-B Item 1 (126,647) and
# XOM Item 2 (40,860). Both checks are WARNINGs, and together they trigger
# cross-reference layout recovery (parsing/recovery.py).
KEY_ITEMS = ("1", "1A", "7", "8")
MIN_KEY_SECTION_CHARS = 1_500
MAX_OTHER_SECTION_CHARS = 60_000
# -----------------------------------------------------------------------------------------

MIN_SECTION_CHARS = 200
MAX_HEADING_CHARS = 200
BLOCK_SEPARATOR = "\n\n"

# "Item 1A.", "ITEM 7 -", "Items 1 and 2.", "PART II, Item 5.", "Item 9A(T)".
ITEM_RE = re.compile(
    r"^(?:PART\s+[IV]+\s*[,.:\-–—]?\s*)?"
    r"ITEMS?\s*(?P<id>\d{1,2}[A-C]?)\b"
    r"(?P<more>(?:\s*(?:,|and|&)\s*\d{1,2}[A-C]?\b)*)"
    r"(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)
ITEM_ID_RE = re.compile(r"\d{1,2}[A-C]?", re.IGNORECASE)
TRAILING_PAGE_RE = re.compile(r"\s+\d{1,3}(?:\s*[-–—]\s*\d{1,3})?$")  # "45", "50–160"
PART_HEADING_RE = re.compile(r"^PART\s+[IV]+\b.{0,80}$", re.IGNORECASE)
HEADING_PUNCTUATION = ".:-–—"


@dataclass
class Section:
    item: str  # canonical id: "1", "1A", "7", ...
    title: str
    text: str  # heading block included, blocks joined by blank lines
    covers: list[str] = field(default_factory=list)  # e.g. ["1", "2"] for "Items 1 and 2"
    recovered: bool = False  # True if relabeled by cross-reference layout recovery

    @property
    def items(self) -> list[str]:
        return self.covers or [self.item]

    @property
    def label(self) -> str:
        return "Items " + ", ".join(self.covers) if self.covers else f"Item {self.item}"


@dataclass
class SplitResult:
    sections: list[Section]
    warnings: list[str]
    front_matter_chars: int  # text before the first picked heading (cover page, TOC)


@dataclass
class _Candidate:
    block: int  # index into the block list
    item: str
    covers: list[str]
    title: str
    span: int = 0  # characters of content until the next candidate


def item_sort_key(item: str) -> tuple[int, str]:
    """Canonical 10-K order: 1 < 1A < 1B < 1C < 2 < ... < 16."""
    match = re.match(r"(\d+)([A-Z]?)", item)
    return (int(match.group(1)), match.group(2)) if match else (999, item)


def parse_heading(block: str) -> tuple[str, list[str], str] | None:
    """Return (item, covers, title) if `block` looks like an Item heading, else None."""
    if block.startswith(TABLE_START):
        lines = block.split("\n")
        if len(lines) != 3:  # [TABLE], one row, [/TABLE]
            return None
        block = " ".join(c.strip() for c in lines[1].split(CELL_SEPARATOR.strip()) if c.strip())
    if len(block) > MAX_HEADING_CHARS or "\n" in block:
        return None
    match = ITEM_RE.match(block.strip())
    if not match:
        return None
    rest = match.group("rest").strip()
    # "Item 7 of this report discusses ..." is a sentence, not a heading.
    if rest and rest[0] not in HEADING_PUNCTUATION and not rest[0].isupper() and rest[0] != "(":
        return None
    item = match.group("id").upper()
    covers = [item, *(m.upper() for m in ITEM_ID_RE.findall(match.group("more")))]
    title = TRAILING_PAGE_RE.sub("", rest.lstrip(HEADING_PUNCTUATION + " ").strip())
    return item, covers if len(covers) > 1 else [], title


def split_sections(
    text: str, expected: tuple[str, ...] = EXPECTED_ITEMS, label: str = "filing"
) -> SplitResult:
    """Split `text` into Item sections. Warnings are logged (prefixed with `label`)."""
    blocks = [b for b in text.split(BLOCK_SEPARATOR) if b.strip()]
    candidates = []
    for i, block in enumerate(blocks):
        parsed = parse_heading(block)
        if parsed:
            candidates.append(_Candidate(i, *parsed))
    for this, nxt in _with_next(candidates):
        end = nxt.block if nxt else len(blocks)
        this.span = sum(len(b) for b in blocks[this.block + 1 : end])

    warnings: list[str] = []
    picked = _pick(candidates, warnings)

    sections = []
    for this, nxt in _with_next(picked):
        end = nxt.block if nxt else len(blocks)
        body = blocks[this.block : end]
        while len(body) > 1 and PART_HEADING_RE.match(body[-1]):
            body.pop()  # "PART II" belongs to the next section's context, not this one
        title = this.title
        if not title and len(body) > 1 and _looks_like_title(body[1]):
            title = body[1]  # "Item 1A." alone, title in the next block
        sections.append(
            Section(
                item=this.item, title=title, text=BLOCK_SEPARATOR.join(body), covers=this.covers
            )
        )

    found = {item for s in sections for item in (s.covers or [s.item])}
    for item in expected:
        if item not in found:
            warnings.append(f"missing Item {item}")
    for message in warnings:
        logger.warning("%s: %s", label, message)

    front = sum(len(b) for b in blocks[: picked[0].block]) if picked else len(text)
    return SplitResult(sections=sections, warnings=warnings, front_matter_chars=front)


def is_key_section(section: Section) -> bool:
    return any(item in KEY_ITEMS for item in section.items)


def is_stub(section: Section) -> bool:
    """A key Item too short to be the real section (usually a cross-reference)."""
    return is_key_section(section) and len(section.text) < MIN_KEY_SECTION_CHARS


def is_oversized(section: Section) -> bool:
    """A non-key Item long enough that it has probably swallowed other Items' text."""
    return not is_key_section(section) and len(section.text) > MAX_OTHER_SECTION_CHARS


def section_size_warning(section: Section) -> str | None:
    """The warning for a stub-sized key Item or an oversized other Item, else None."""
    n = len(section.text)
    if is_stub(section):
        return (
            f"{section.label} is only {n:,} characters (expected at least "
            f"{MIN_KEY_SECTION_CHARS:,}); probably a cross-reference stub"
        )
    if is_oversized(section):
        return (
            f"{section.label} has {n:,} characters (expected at most "
            f"{MAX_OTHER_SECTION_CHARS:,} outside Items {', '.join(KEY_ITEMS)}); "
            "it probably contains other Items' text"
        )
    return None


def check_section_sizes(sections: list[Section]) -> list[str]:
    """Warnings for stub-sized key Items and oversized other Items (not logged here)."""
    return [w for s in sections if (w := section_size_warning(s))]


def _pick(candidates: list[_Candidate], warnings: list[str]) -> list[_Candidate]:
    picked: list[_Candidate] = []
    items = sorted({c.item for c in candidates}, key=item_sort_key)
    for item in items:
        after = picked[-1].block if picked else -1
        mine = [c for c in candidates if c.item == item and c.block > after]
        if not mine:
            warnings.append(f"Item {item} heading only appears out of order; skipped")
            continue
        substantial = [c for c in mine if c.span >= MIN_SECTION_CHARS]
        picked.append(substantial[0] if substantial else mine[-1])
    return picked


def _with_next(items: list[_Candidate]) -> list[tuple[_Candidate, _Candidate | None]]:
    return list(zip(items, [*items[1:], None], strict=False))


def _looks_like_title(block: str) -> bool:
    return len(block) <= 120 and not block.startswith(TABLE_START) and not block.endswith(".")
