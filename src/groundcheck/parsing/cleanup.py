"""Remove page furniture (page numbers, running footers) from `html_to_text` output.

Only blocks that are clearly furniture are dropped, using a few conservative patterns:

- a bare page number: `23`, `Page 23`, `- 23 -`, `ii`;
- a "Table of Contents" back-link, which Workiva filings repeat at the top of each page;
- a short running footer that mentions "10-K" and starts or ends with a page number,
  e.g. `Apple Inc. | 2025 Form 10-K | 23`.

Anything that looks like an Item heading is never dropped ("Item 16. Form 10-K
Summary 95" in a table of contents). Tables are never touched.
"""

import re

from groundcheck.parsing.html_text import TABLE_START
from groundcheck.parsing.sections import BLOCK_SEPARATOR, parse_heading

PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s+)?(?:\d{1,3}|[ivxl]{1,6})$|^[-–—]\s*\d{1,3}\s*[-–—]$", re.I
)
TOC_LINK_RE = re.compile(r"^(?:back to\s+)?table of contents$", re.IGNORECASE)
FOOTER_RE = re.compile(r"(?:^\d{1,3}\s*\|?\s+.*10-K)|(?:10-K.*\s\|?\s*\d{1,3}$)", re.IGNORECASE)
MAX_FOOTER_CHARS = 120


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
