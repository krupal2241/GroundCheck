"""Convert a 10-K HTML document into clean plain text.

Output format:

- Each block of prose (a paragraph, heading, list item, `<div>`, ...) is one line, and
  blocks are separated by a blank line. Whitespace inside a block is collapsed.
- Each data table becomes a block like

      [TABLE]
      | 2025 | 2024
      Net sales | $416,161 | $391,035
      Net income (loss) | (1,234) | 5.6%
      [/TABLE]

  One line per row, cells joined by " | ". Cells that SEC filers split apart for
  alignment are merged back together: "$" joins the number after it, and ")" or "%"
  joins the number before it. Negative numbers like "(1,234)" are kept verbatim.
  Columns that only hold padding are dropped, and a heading cell that spans several
  columns (colspan) is placed in the first column that actually holds data.
- A table with a single column left after that is a layout table (a bordered note,
  a footnote list), and a two-column table whose first column holds only bullets is a
  bulleted list. Their rows are emitted as ordinary prose blocks instead.

Hidden content (`display:none`, which is where inline-XBRL filings keep their
machine-readable header) is dropped. Inline-XBRL tags such as `<ix:nonFraction>` are
unwrapped: their visible text is kept.
"""

import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, Comment, NavigableString, Tag, XMLParsedAsHTMLWarning

TABLE_START = "[TABLE]"
TABLE_END = "[/TABLE]"
CELL_SEPARATOR = " | "

# Tags that start a new block of text. Anything else (span, font, b, a, ix:*) is inline.
BLOCK_TAGS = frozenset(
    {
        *("address", "article", "aside", "blockquote", "body", "br", "center", "dd", "div"),
        *("dl", "dt", "figcaption", "figure", "footer", "form", "header", "hr", "li"),
        *("h1", "h2", "h3", "h4", "h5", "h6", "main", "nav", "ol", "p", "pre", "section"),
        *("table", "td", "th", "tr", "ul"),
    }
)
# Tags whose content is never visible text.
SKIP_TAGS = frozenset(["head", "script", "style", "noscript", "template", "title", "ix:header"])

HIDDEN_STYLE_RE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
# Characters that render as nothing: zero-width space/joiners, BOM, soft hyphen.
INVISIBLE_RE = re.compile("[​‌‍﻿­]")
# A cell holding only an opening "$" or "(" belongs to the value on its right; one
# holding only ")" or "%" belongs to the value on its left.
PREFIX_CELL_RE = re.compile(r"^[$(]+$")
SUFFIX_CELL_RE = re.compile(r"^[)%]+$")
# Filers often lay out a bulleted list as a two-column table: bullet | text.
BULLET_RE = re.compile(r"^[\u2022\u25cf\u25e6\u25aa\u00b7\u2013\u2014-]$")
MAX_COLSPAN = 100  # guards against a malformed colspan="100000"


def html_to_text(html: str | bytes) -> str:
    """Return the visible text of an HTML document, with tables as pipe rows.

    `html` may be bytes, in which case BeautifulSoup detects the encoding (from the
    `<meta charset>` or by sniffing).
    """
    # Inline-XBRL filings are XHTML and start with an <?xml ...?> declaration, which
    # makes bs4 suggest its XML parser. We parse them as HTML on purpose: the HTML
    # parser tolerates the malformed markup older filings contain.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    root = soup.body or soup
    return "\n\n".join(_blocks(root))


def table_rows(table: Tag) -> list[list[str]]:
    """Return a table's rows as lists of cell strings, after merging and column cleanup.

    Every row has the same number of cells. Rows with no text at all are dropped.
    """
    rows = [
        _row_cells(tr)
        for tr in table.find_all("tr")
        if tr.find_parent("table") is table and not _is_hidden(tr)
    ]
    for row in rows:
        _merge_split_cells(row)

    # A column is kept if some ordinary (colspan=1) cell has text in it. Cells that span
    # several columns don't count: their text sits over padding columns like the one a
    # "$" lived in before merging.
    used = {c.start for row in rows for c in row if c.text and c.span == 1}
    for row in rows:
        for cell in row:
            if cell.text and cell.span > 1 and not used.intersection(cell.columns):
                used.add(cell.start)  # nothing below it; keep its own column

    columns = sorted(used)
    position = {col: i for i, col in enumerate(columns)}
    result = []
    for row in rows:
        out = [""] * len(columns)
        for cell in row:
            if cell.text:
                target = min(used.intersection(cell.columns))
                out[position[target]] = cell.text
        if any(out):
            result.append(out)
    return result


@dataclass
class _Cell:
    text: str
    start: int  # first grid column this cell covers
    span: int

    @property
    def columns(self) -> range:
        return range(self.start, self.start + self.span)


def _row_cells(tr: Tag) -> list[_Cell]:
    cells, column = [], 0
    for td in tr.find_all(["td", "th"], recursive=False):
        span = _colspan(td)
        text = "" if _is_hidden(td) else " ".join(_blocks(td, flatten_tables=True))
        cells.append(_Cell(text=text, start=column, span=span))
        column += span
    return cells


def _colspan(td: Tag) -> int:
    match = re.match(r"\s*(\d+)", str(td.get("colspan", "1")))
    return min(max(int(match.group(1)), 1), MAX_COLSPAN) if match else 1


def _merge_split_cells(row: list[_Cell]) -> None:
    """Join "$", "(", ")" and "%" cells onto the neighbouring value, in place."""
    previous = None
    for cell in row:
        if not cell.text:
            continue
        if SUFFIX_CELL_RE.match(cell.text) and previous is not None:
            previous.text += cell.text
            cell.text = ""
        else:
            previous = cell

    prefixes: list[_Cell] = []
    for cell in row:
        if not cell.text:
            continue
        if PREFIX_CELL_RE.match(cell.text):
            prefixes.append(cell)
        elif prefixes:
            cell.text = "".join(p.text for p in prefixes) + cell.text
            for p in prefixes:
                p.text = ""
            prefixes = []
    # A trailing "$" with nothing after it is left where it was.


class _Preformatted(str):
    """Text that must not be whitespace-collapsed (a rendered table)."""


_BREAK = object()  # marks a block boundary in the stream of pieces


def _blocks(root: Tag, flatten_tables: bool = False) -> list[str]:
    """Return the non-empty text blocks under `root`, in document order."""
    pieces: list = []
    _walk(root, pieces, flatten_tables)

    blocks, current = [], []

    def flush() -> None:
        text = WHITESPACE_RE.sub(" ", INVISIBLE_RE.sub("", "".join(current))).strip()
        if text:
            blocks.append(text)
        current.clear()

    for piece in pieces:
        if piece is _BREAK:
            flush()
        elif isinstance(piece, _Preformatted):
            flush()
            blocks.append(str(piece))
        else:
            current.append(piece)
    flush()
    return blocks


def _walk(node: Tag, pieces: list, flatten_tables: bool) -> None:
    for child in node.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            pieces.append(str(child))
            continue
        if not isinstance(child, Tag) or child.name in SKIP_TAGS or _is_hidden(child):
            continue
        if child.name == "table" and not flatten_tables:
            pieces.append(_BREAK)
            pieces.extend(_render_table(child))
            pieces.append(_BREAK)
            continue
        is_block = child.name in BLOCK_TAGS
        if is_block:
            pieces.append(_BREAK)
        _walk(child, pieces, flatten_tables)
        if is_block:
            pieces.append(_BREAK)


def _render_table(table: Tag) -> list:
    rows = table_rows(table)
    if not rows:
        return []
    if len(rows[0]) == 1:  # layout table: treat each row as a paragraph
        return [piece for row in rows for piece in (row[0], _BREAK)]
    if len(rows[0]) == 2 and all(BULLET_RE.match(row[0]) for row in rows):  # bullet list
        return [piece for row in rows for piece in (f"{row[0]} {row[1]}", _BREAK)]
    lines = [TABLE_START, *(CELL_SEPARATOR.join(row).strip() for row in rows), TABLE_END]
    return [_Preformatted("\n".join(lines))]


def _is_hidden(tag: Tag) -> bool:
    return bool(HIDDEN_STYLE_RE.search(str(tag.get("style", "")))) or tag.has_attr("hidden")
