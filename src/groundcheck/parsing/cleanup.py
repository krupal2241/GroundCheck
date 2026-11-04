"""Remove page furniture (page numbers, running footers) from `html_to_text` output.

Only blocks that are clearly furniture are dropped, using a few conservative patterns:

- a bare page number: `23`, `Page 23`, `- 23 -`, `ii`;
- a "Table of Contents" back-link, which Workiva filings repeat at the top of each page;
- a short running footer that mentions "10-K" and starts or ends with a page number,
  e.g. `Apple Inc. | 2025 Form 10-K | 23`.

Anything that looks like an Item heading is never dropped ("Item 16. Form 10-K
Summary 95" in a table of contents). Tables are never touched.

`remove_repeated_lines` then drops running page headers that match no pattern: any
short line repeated many times in the filing (JPM prints "Management's discussion and
analysis" at the top of every MD&A page; XOM prints "MANAGEMENT'S DISCUSSION AND
ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS" every 20 to 40 lines). The
first occurrence that is a genuine heading, i.e. followed by prose, is kept.
"""

import logging
import re
from collections import Counter

from groundcheck.parsing.html_text import TABLE_START
from groundcheck.parsing.sections import BLOCK_SEPARATOR, parse_heading

PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s+)?(?:\d{1,3}|[ivxl]{1,6})$|^[-–—]\s*\d{1,3}\s*[-–—]$", re.I
)
TOC_LINK_RE = re.compile(r"^(?:back to\s+)?table of contents$", re.IGNORECASE)
FOOTER_RE = re.compile(r"(?:^\d{1,3}\s*\|?\s+.*10-K)|(?:10-K.*\s\|?\s*\d{1,3}$)", re.IGNORECASE)
MAX_FOOTER_CHARS = 120

# A line this short that repeats this often is page furniture (a running header).
MAX_REPEATED_LINE_CHARS = 120
MIN_REPEATS = 8
# "Followed by prose": a prose block within this many blocks, skipping subheadings
# and data tables (XOM's MD&A heading is followed by a subheading and a table).
PROSE_LOOKAHEAD_BLOCKS = 4
MIN_PROSE_CHARS = 100
MIN_PROSE_WORDS = 15
TOC_ROW_RE = re.compile(r"\|\s*(?:page\s+)?(\d{1,3})(?:\s*[-–—]\s*\d{1,3})?\s*$", re.IGNORECASE)

logger = logging.getLogger(__name__)


def is_page_furniture(block: str) -> bool:
    block = block.strip()
    if not block or block.startswith(TABLE_START) or parse_heading(block):
        return False
    if PAGE_NUMBER_RE.match(block) or TOC_LINK_RE.match(block):
        return True
    return len(block) <= MAX_FOOTER_CHARS and bool(FOOTER_RE.search(block))


def remove_page_furniture(text: str) -> tuple[str, int]:
    """Return the text without furniture blocks, and how many blocks were removed."""
    blocks = text.split(BLOCK_SEPARATOR)
    kept = [b for b in blocks if not is_page_furniture(b)]
    return BLOCK_SEPARATOR.join(kept), len(blocks) - len(kept)


def is_prose(block: str) -> bool:
    """A paragraph of running text, not a heading, a table or a one-line label."""
    block = block.strip()
    return (
        not block.startswith(TABLE_START)
        and len(block) >= MIN_PROSE_CHARS
        and len(block.split()) >= MIN_PROSE_WORDS
        and parse_heading(block) is None
    )


def is_toc_table(block: str) -> bool:
    """A table of contents: at least two rows, most of them ending in a page number,
    and those page numbers never going down (a data table's last column jumps around).
    """
    if not block.startswith(TABLE_START):
        return False
    rows = [r for r in block.split("\n")[1:-1] if r.strip()]
    pages = [int(m.group(1)) for r in rows if (m := TOC_ROW_RE.search(r))]
    return len(pages) >= 2 and len(pages) * 2 >= len(rows) and pages == sorted(pages)


def followed_by_prose(blocks: list[str], i: int, skip=lambda block: False) -> bool:
    """True if block `i` is followed by prose, as a genuine heading is.

    Looks at the next `PROSE_LOOKAHEAD_BLOCKS` blocks, ignoring blocks for which
    `skip(block)` is true (e.g. other running headers). Subheadings and data tables
    are passed over; a table-of-contents table means "no".
    """
    seen = 0
    for block in blocks[i + 1 :]:
        if not block.strip() or skip(block):
            continue
        if is_prose(block):
            return True
        if is_toc_table(block):
            return False
        seen += 1
        if seen >= PROSE_LOOKAHEAD_BLOCKS:
            return False
    return False


def _line_key(block: str) -> str:
    return block.strip().casefold()


def _repeat_eligible(block: str) -> bool:
    block = block.strip()
    return (
        bool(block)
        and not block.startswith(TABLE_START)
        and "\n" not in block
        and len(block) <= MAX_REPEATED_LINE_CHARS
        and parse_heading(block) is None
    )


def remove_repeated_lines(text: str, label: str = "filing") -> tuple[str, int]:
    """Drop short lines repeated `MIN_REPEATS`+ times (running page headers).

    Lines are compared case-insensitively, ignoring surrounding whitespace. For each
    repeated line, the first occurrence that is followed by prose (a genuine heading)
    is kept; every other occurrence is removed. Item headings and tables are never
    touched. Returns the text and the number of lines removed; each removed line and
    its count are logged at INFO level (`groundcheck -v parse`).
    """
    blocks = text.split(BLOCK_SEPARATOR)
    counts = Counter(_line_key(b) for b in blocks if _repeat_eligible(b))
    repeated = {key for key, n in counts.items() if n >= MIN_REPEATS}
    if not repeated:
        return text, 0

    def is_repeat(block: str) -> bool:
        return _repeat_eligible(block) and _line_key(block) in repeated

    kept_at: dict[str, int] = {}  # repeated line -> index of the occurrence kept
    for i, block in enumerate(blocks):
        key = _line_key(block)
        if is_repeat(block) and key not in kept_at and followed_by_prose(blocks, i, is_repeat):
            kept_at[key] = i
    keep = set(kept_at.values())
    kept = [b for i, b in enumerate(blocks) if i in keep or not is_repeat(b)]
    removed = len(blocks) - len(kept)
    for key in sorted(repeated):
        heading = "kept once as a heading" if key in kept_at else "no heading kept"
        logger.info("%s: removed repeated line %r (%d times, %s)", label, key, counts[key], heading)
    logger.info("%s: removed %d repeated header line(s)", label, removed)
    return BLOCK_SEPARATOR.join(kept), removed
