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

Word limits only approximate tokens: number-dense prose runs well over 1.3 tokens per
word (JNJ-1-p0001 was 632 tokens at 250 words). So when a token counter is given
(`groundcheck chunk` uses bge-small's tokenizer), any chunk over `max_tokens` is split:

- prose at the sentence boundary nearest its middle (else the middle word), the second
  half starting `overlap` words before the cut, recursively until every part fits;
- tables between rows (each part repeats the header), and a single row that is too
  long between words.

Only windows that need it are split. A split window keeps its ID with a letter
appended (`JNJ-1-p0001a`, `JNJ-1-p0001b`), so every other chunk ID stays exactly as it
was. `max_tokens` (480) leaves headroom under bge-small's 512-token input limit.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from groundcheck.chunking.base import Chunk, chunk_id, part_suffix
from groundcheck.parsing.filing import ParsedFiling, ParsedSection
from groundcheck.parsing.html_text import CELL_SEPARATOR, TABLE_END, TABLE_START
from groundcheck.parsing.sections import BLOCK_SEPARATOR

MAX_HEADER_ROWS = 3
DEFAULT_MAX_TOKENS = 480
# Words ending in "." that rarely end a sentence in a 10-K.
ABBREVIATIONS = {"inc.", "co.", "corp.", "ltd.", "llc.", "no.", "nos.", "mr.", "ms.", "dr.", "vs."}
SENTENCE_END = ".!?"
CLOSERS = "\"'”’)]"


@dataclass(frozen=True)
class FixedSizeChunker:
    prose_words: int = 250
    overlap: int = 50
    table_words: int = 120
    max_tokens: int = DEFAULT_MAX_TOKENS
    # Counts a text's tokens (special tokens included). None: no token check (tests, CI).
    count_tokens: Callable[[str], int] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.overlap < self.prose_words:
            raise ValueError("overlap must be at least 0 and smaller than prose_words")
        if self.table_words < 1:
            raise ValueError("table_words must be positive")
        if self.max_tokens < 16:
            raise ValueError("max_tokens must be at least 16")

    @property
    def name(self) -> str:
        name = f"fixed-p{self.prose_words}-o{self.overlap}-t{self.table_words}"
        return name + (f"-k{self.max_tokens}" if self.count_tokens else "")

    def fits(self, text: str) -> bool:
        return self.count_tokens is None or self.count_tokens(text) <= self.max_tokens

    def chunk(self, filing: ParsedFiling) -> list[Chunk]:
        chunks = []
        for section in filing.sections:
            chunks.extend(self.chunk_section(filing, section))
        return chunks

    def chunk_section(self, filing: ParsedFiling, section: ParsedSection) -> list[Chunk]:
        blocks = section.text.split(BLOCK_SEPARATOR)
        tables = [b for b in blocks if b.startswith(TABLE_START)]
        prose = " ".join(b for b in blocks if not b.startswith(TABLE_START))

        def make(kind: str, index: int, text: str, part: str = "") -> Chunk:
            return Chunk(
                chunk_id=chunk_id(filing.ticker, section.item, kind, index, part),
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

        def make_all(kind: str, index: int, parts: list[str]) -> list[Chunk]:
            if len(parts) == 1:
                return [make(kind, index, parts[0])]
            return [make(kind, index, text, part_suffix(j)) for j, text in enumerate(parts)]

        out = []
        for i, window in enumerate(self.prose_windows(prose)):
            out.extend(make_all("prose", i, self.fit_prose(window)))
        pieces = [piece for table in tables for piece in self.row_groups(table)]
        for i, (header, rows) in enumerate(pieces):
            out.extend(make_all("table", i, self.fit_table(header, rows)))
        return out

    def fit_prose(self, text: str) -> list[str]:
        """Split `text` until every part is within `max_tokens` (see module docs)."""
        if self.fits(text):
            return [text]
        words = text.split()
        if len(words) < 2:
            return [text]  # one enormous "word": nothing to split at
        cut = sentence_cut(words)
        overlap = min(self.overlap, cut // 2)
        first = " ".join(words[:cut])
        second = " ".join(words[cut - overlap :])
        return self.fit_prose(first) + self.fit_prose(second)

    def fit_table(self, header: list[str], rows: list[str]) -> list[str]:
        """Render a table piece, split between rows (then words) to fit `max_tokens`."""
        text = _render(header + rows)
        if self.fits(text) or not rows:
            return [text]
        if len(rows) > 1:
            mid = len(rows) // 2
            return self.fit_table(header, rows[:mid]) + self.fit_table(header, rows[mid:])
        words = rows[0].split()
        if len(words) < 2:
            return [text]
        mid = len(words) // 2
        halves = [" ".join(words[:mid]), " ".join(words[mid:])]
        return [part for half in halves for part in self.fit_table(header, [half])]

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
        """A table's chunk texts: row groups within `table_words`, each within the tokens."""
        return [p for header, rows in self.row_groups(table) for p in self.fit_table(header, rows)]

    def row_groups(self, table: str) -> list[tuple[list[str], list[str]]]:
        """Split a table between rows into (header rows, body rows) within `table_words`."""
        rows = table.split("\n")[1:-1]  # drop the [TABLE] / [/TABLE] markers
        if not rows:
            return []
        n_header = header_row_count(rows)
        header, body = rows[:n_header], rows[n_header:]
        if not body:
            return [(header, [])]
        budget = self.table_words - _words(header)
        pieces, current, used = [], [], 0
        for row in body:
            size = _words([row])
            if current and used + size > budget:
                pieces.append((header, current))
                current, used = [], 0
            current.append(row)
            used += size
        pieces.append((header, current))
        return pieces


def is_sentence_end(word: str) -> bool:
    core = word.rstrip(CLOSERS)
    if not core or core[-1] not in SENTENCE_END:
        return False
    # "U.S.", "e.g.", "Inc.", "No.", "J." are abbreviations, not sentence ends.
    abbreviation = "." in core[:-1] or len(core) <= 2 or core.casefold() in ABBREVIATIONS
    return core[-1] != "." or not abbreviation


def sentence_cut(words: list[str]) -> int:
    """Where to split a word list: the sentence boundary nearest the middle (within the
    middle third), else the middle word. Always 1 <= cut < len(words)."""
    n = len(words)
    mid = n // 2
    lo, hi = max(1, n // 3), min(n - 1, (2 * n) // 3)
    boundaries = [
        i
        for i in range(lo, hi + 1)
        if is_sentence_end(words[i - 1]) and (words[i][0].isupper() or words[i][0].isdigit())
    ]
    return min(boundaries, key=lambda i: abs(i - mid)) if boundaries else max(1, mid)


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
