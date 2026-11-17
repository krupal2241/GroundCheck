"""The chunk record, the `Chunker` interface, and JSONL input/output.

Every chunker turns a `ParsedFiling` into a list of `Chunk`s. Chunks never cross an
Item section, and their IDs are deterministic (`<TICKER>-<item>-<p|t><nnnn>`, plus a
letter suffix when a window had to be split to fit the token limit), so the
same parsed filing and chunker settings always produce byte-identical JSONL. That's
what lets retrieval results and labels refer to chunks by ID.
"""

import json
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from groundcheck.config import REPO_ROOT, ConfigError
from groundcheck.parsing.filing import ParsedFiling

DEFAULT_CHUNKS_DIR = REPO_ROOT / "data" / "chunks"


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    ticker: str
    accession: str
    item: str  # the Item section this chunk comes from: "1", "1A", "7", ...
    section_title: str
    kind: Literal["prose", "table"]
    index: int  # position among this section's windows of the same kind (split parts share it)
    text: str
    word_count: int
    chunker: str  # e.g. "fixed-p250-o50-t120": the chunker and its settings
    # Table chunks only, and only when the chunker was asked to record it: the table's
    # title or caption, i.e. the nearest preceding heading or line of the filing. It is
    # metadata for retrieval (BM25 `table_context`); it is NOT part of `text`, so gold
    # strings and evidence spans are unaffected. Empty: none recorded.
    caption: str = ""


def chunk_id(ticker: str, item: str, kind: str, index: int, part: str = "") -> str:
    """`AAPL-7-p0003`; a window split to fit the token limit gets `a`, `b`, ... appended."""
    return f"{ticker}-{item}-{kind[0]}{index:04d}{part}"


def part_suffix(i: int) -> str:
    """0 → "a", ..., 24 → "y", 25 → "za", 26 → "zb", ...: sorts in order, never ambiguous."""
    return chr(ord("a") + i) if i < 25 else "z" + part_suffix(i - 25)


def is_split_part(chunk_id: str) -> bool:
    return chunk_id[-1].isalpha()


class Chunker(Protocol):
    """Anything with a `name` and a `chunk(filing)` method is a chunker."""

    @property
    def name(self) -> str: ...

    def chunk(self, filing: ParsedFiling) -> list[Chunk]: ...


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    """Write one chunk per line, keys in a fixed order, so diffs stay readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # An empty caption is left out, so files written without `--table-captions` are
    # byte-identical to those written before the field existed (same sha, same indexes).
    rows = [c.model_dump() for c in chunks]
    lines = [
        json.dumps({k: v for k, v in row.items() if k != "caption" or v}, ensure_ascii=False)
        for row in rows
    ]
    tmp = path.with_name(path.name + ".part")
    tmp.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    tmp.replace(path)


def read_chunks(path: Path) -> list[Chunk]:
    if not path.is_file():
        raise ConfigError(f"{path} not found; run `groundcheck chunk`")
    with path.open(encoding="utf-8") as f:
        return [Chunk.model_validate_json(line) for line in f if line.strip()]


def chunks_path(ticker: str, chunks_dir: Path = DEFAULT_CHUNKS_DIR) -> Path:
    return chunks_dir / f"{ticker}.jsonl"
