"""Recover Items 7 and 8 from filings that use a cross-reference layout.

Some 10-Ks (JPM, XOM) don't print MD&A and the financial statements under their
Item 7 and Item 8 headings. Those Items are one-line pointers ("see pages 50-160";
JPM Item 7 is 396 characters), and the real text sits later in the document, inside
whatever Item heading comes before it (JPM's Item 15 holds the whole annual report,
1,050,977 characters; XOM's Item 16 holds the "Financial Section", 342,034).

Trigger: Item 7 or Item 8 is a stub (under `MIN_KEY_SECTION_CHARS`) *and* a later,
non-key section is oversized (over `MAX_OTHER_SECTION_CHARS`). Inside the largest such
section we look for:

1. the MD&A start: the first line without a `|` (so never a table-of-contents row)
   that contains "Management's discussion and analysis" (JPM) or "MANAGEMENT'S
   DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS" (XOM),
   with either apostrophe (' or ’), and is followed by prose. The line is either the
   heading itself or a sentence that opens with it (JPM's "The following is
   Management's discussion and analysis of ..."); a sentence that points at a page
   ("see Management's discussion ... on page 50") never counts;
2. the financial statements start: the first line without a `|`, after the MD&A start,
   equal to "Report of Independent Registered Public Accounting Firm".

MD&A start → financial statements start becomes Item 7, the rest of the oversized
section becomes Item 8 (which is "Financial Statements and Supplementary Data", so the
notes and supplementary schedules belong there), and the oversized section keeps what
came before the MD&A start. The stubs' own text is kept at the top of the recovered
sections, so nothing is dropped. Recovered sections carry `recovered=True`.

If either start can't be found, nothing changes and a WARNING naming the filing is
returned (and logged).
"""

import logging
import re
from dataclasses import dataclass, field, replace

from groundcheck.parsing.cleanup import followed_by_prose, is_prose
from groundcheck.parsing.html_text import TABLE_START
from groundcheck.parsing.sections import (
    BLOCK_SEPARATOR,
    Section,
    is_oversized,
    is_stub,
    item_sort_key,
)

logger = logging.getLogger(__name__)

MDA_PHRASES = (
    "management's discussion and analysis",  # JPM
    "management's discussion and analysis of financial condition and results of operations",  # XOM
)
FS_START_LINE = "report of independent registered public accounting firm"
# In a prose line, the phrase must come this early ("The following is Management's
# discussion and analysis ...") so a passing mention deep in a paragraph doesn't count.
MAX_PHRASE_OFFSET = 40
PAGE_REFERENCE_RE = re.compile(r"\bpages?\s+\d", re.IGNORECASE)
DEFAULT_TITLES = {
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "8": "Financial Statements and Supplementary Data",
}
RECOVERED_ITEMS = ("7", "8")


@dataclass
class RecoveryResult:
    sections: list[Section]
    notes: list[str] = field(default_factory=list)  # what was recovered (informational)
    warnings: list[str] = field(default_factory=list)


def normalize_line(line: str) -> str:
    """Lowercase, one apostrophe style, single spaces, no trailing period or colon."""
    line = line.replace("’", "'").replace("‘", "'").casefold()
    return " ".join(line.split()).rstrip(".:")


def _is_candidate_line(block: str) -> bool:
    return bool(block.strip()) and not block.startswith(TABLE_START) and "|" not in block


def is_mda_start_line(block: str) -> bool:
    """The MD&A heading itself, or a sentence that opens MD&A by naming it."""
    if not _is_candidate_line(block):
        return False
    line = normalize_line(block)
    if line in MDA_PHRASES:
        return True
    if not is_prose(block) or PAGE_REFERENCE_RE.search(block):
        return False
    return any(0 <= line.find(p) <= MAX_PHRASE_OFFSET for p in MDA_PHRASES)


def find_mda_start(blocks: list[str], start: int = 0) -> int | None:
    """Index of the block where MD&A really starts, or None."""
    for i in range(start, len(blocks)):
        if is_mda_start_line(blocks[i]) and followed_by_prose(blocks, i):
            return i
    return None


def find_fs_start(blocks: list[str], start: int) -> int | None:
    """Index of the first non-TOC "Report of Independent Registered ..." line."""
    for i in range(start, len(blocks)):
        if _is_candidate_line(blocks[i]) and normalize_line(blocks[i]) == FS_START_LINE:
            return i
    return None


def recover_cross_reference_layout(
    sections: list[Section], label: str = "filing", full_text: str | None = None
) -> RecoveryResult:
    """Relabel MD&A and the financial statements as Items 7 and 8 (see module docs).

    If `full_text` (the text the sections were split from) is given, each note says on
    which line of it the recovered section starts, to check against `<TICKER>.txt`.
    """
    by_item = {s.item: s for s in sections}
    stubs = [s for s in sections if s.item in RECOVERED_ITEMS and is_stub(s)]
    if not stubs:
        return RecoveryResult(sections)
    first_stub = sections.index(stubs[0])
    oversized = [s for s in sections[first_stub + 1 :] if is_oversized(s)]
    if not oversized:
        return RecoveryResult(sections)
    host = max(oversized, key=lambda s: len(s.text))
    context = (
        ", ".join(f"{s.label} is {len(s.text):,} chars" for s in stubs)
        + f"; {host.label} is {len(host.text):,} chars"
    )

    def give_up(reason: str) -> RecoveryResult:
        message = (
            f"cross-reference layout suspected ({context}) but {reason}; sections left unchanged"
        )
        logger.warning("%s: %s", label, message)
        return RecoveryResult(sections, warnings=[message])

    full = [by_item[i] for i in RECOVERED_ITEMS if i in by_item and not is_stub(by_item[i])]
    if full:
        return give_up(f"{full[0].label} already has {len(full[0].text):,} chars")

    blocks = host.text.split(BLOCK_SEPARATOR)
    mda = find_mda_start(blocks, start=1)  # block 0 is the host's own heading
    if mda is None:
        return give_up(f"no start of Management's discussion and analysis in {host.label}")
    fs = find_fs_start(blocks, mda + 1)
    if fs is None:
        return give_up(
            f"no 'Report of Independent Registered Public Accounting Firm' line after the "
            f"MD&A start in {host.label}"
        )

    spans = {"7": blocks[mda:fs], "8": blocks[fs:]}
    recovered = []
    for item in RECOVERED_ITEMS:
        stub = by_item.get(item)
        body = ([stub.text] if stub else []) + spans[item]
        title = (stub.title if stub else "") or DEFAULT_TITLES[item]
        recovered.append(
            Section(item=item, title=title, text=BLOCK_SEPARATOR.join(body), recovered=True)
        )
    remainder = replace(host, text=BLOCK_SEPARATOR.join(blocks[:mda]))

    out = [remainder if s is host else s for s in sections if s.item not in RECOVERED_ITEMS]
    out = sorted([*out, *recovered], key=lambda s: item_sort_key(s.item))
    notes = []
    for section, start in zip(recovered, (mda, fs), strict=True):
        preview = " ".join(blocks[start].split())[:80]
        where = ""
        if full_text is not None:
            offset = full_text.find(BLOCK_SEPARATOR.join(blocks[start : start + 20]))
            if offset >= 0:
                line_number = full_text.count("\n", 0, offset) + 1
                where = f" at line {line_number}"
        note = (
            f"recovered {section.label} ({len(section.text):,} chars) from {host.label}"
            f"{where}, starting: {preview!r}"
        )
        notes.append(note)
        logger.info("%s: %s", label, note)
    notes.append(f"{host.label} shrunk to {len(remainder.text):,} chars")
    return RecoveryResult(out, notes=notes)
