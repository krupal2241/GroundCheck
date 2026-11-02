"""The baseline chunker: fixed-size word windows for prose, row-bounded pieces for tables.

Within each Item section:

- **Prose** (every block that isn't a `[TABLE]`) is treated as one stream of words and
  cut into windows of `prose_words` words, each starting `prose_words - overlap` words
  after the previous one. The overlap means a sentence cut at one window's edge is
  whole in the next one.
- **Tables** are chunked separately and only between rows. Each piece repeats the
  table's header rows and stays within `table_words` words (a single row longer than
  that becomes its own piece). Table pieces are smaller than prose windows because
  numbers and pipes cost more tokens per word.

Both limits keep chunks under bge-small's 512-token input limit. Word counts only
approximate tokens; check them with the real tokenizer (`groundcheck index` reports
any chunk the model would truncate).
"""

from dataclasses import dataclass

from groundcheck.chunking.base import Chunk, chunk_id
from groundcheck.parsing.filing import ParsedFiling, ParsedSection
from groundcheck.parsing.html_text import CELL_SEPARATOR, TABLE_END, TABLE_START
from groundcheck.parsing.sections import BLOCK_SEPARATOR

MAX_HEADER_ROWS = 3


@dataclass(frozen=True)
class FixedSizeChunker:
    prose_words: int = 250
    overlap: int = 50
    table_words: int = 120

    def __post_init__(self) -> None:
        if not 0 <= self.overlap < self.prose_words:
            raise ValueError("overlap must be at least 0 and smaller than prose_words")
        if self.table_words < 1:
            raise ValueError("table_words must be positive")

    @property
    def name(self) -> str:
        return f"fixed-p{self.prose_words}-o{self.overlap}-t{self.table_words}"

    def chunk(self, filing: ParsedFiling) -> list[Chunk]:
        chunks = []
        for section in filing.sections:
            chunks.extend(self.chunk_section(filing, section))
        return chunks

    def chunk_section(self, filing: ParsedFiling, section: ParsedSection) -> list[Chunk]:
        blocks = section.text.split(BLOCK_SEPARATOR)
        tables = [b for b in blocks if b.startswith(TABLE_START)]
        prose = " ".join(b for b in blocks if not b.startswith(TABLE_START))

        def make(kind: str, index: int, text: str) -> Chunk:
            return Chunk(
                chunk_id=chunk_id(filing.ticker, section.item, kind, index),
                ticker=filing.ticker,
                accession=filing.accession,
                item=section.item,
                section_title=section.title,
                kind=kind,
                index=index,
                text=text,
                word_count=len(text.split()),
                chunker=self.name,
            )

        out = [make("prose", i, text) for i, text in enumerate(self.prose_windows(prose))]
        pieces = [piece for table in tables for piece in self.table_pieces(table)]
        out.extend(make("table", i, text) for i, text in enumerate(pieces))
        return out

    def prose_windows(self, text: str) -> list[str]:
        words = text.split()
        step = self.prose_words - self.overlap
        windows = []
        for start in range(0, len(words), step):
            windows.append(" ".join(words[start : start + self.prose_words]))
            if start + self.prose_words >= len(words):
                break
        return windows

    def table_pieces(self, table: str) -> list[str]:
        rows = table.split("\n")[1:-1]  # drop the [TABLE] / [/TABLE] markers
        if not rows:
            return []
        n_header = header_row_count(rows)
        header, body = rows[:n_header], rows[n_header:]
        if not body:
            return [_render(header)]
        budget = self.table_words - _words(header)
        pieces, current, used = [], [], 0
        for row in body:
            size = _words([row])
            if current and used + size > budget:
                pieces.append(_render(header + current))
                current, used = [], 0
            current.append(row)
            used += size
        pieces.append(_render(header + current))
        return pieces


def header_row_count(rows: list[str]) -> int:
    """How many leading rows are the table's header.

    Header rows in 10-K tables have an empty first (label) cell: `| 2025 | 2024`, or
    `| Years ended December 31,`. They run until the first row that has a label. If
    every row lacks a label, only the first row counts as the header.
    """
    n = 0
    for row in rows[:MAX_HEADER_ROWS]:
        if row.split(CELL_SEPARATOR.strip(), 1)[0].strip():
            break
        n += 1
    return n if 0 < n < len(rows) else 1


def _words(rows: list[str]) -> int:
    return sum(len(r.split()) for r in rows)


def _render(rows: list[str]) -> str:
    return "\n".join([TABLE_START, *rows, TABLE_END])
