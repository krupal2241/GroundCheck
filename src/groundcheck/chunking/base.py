"""The chunk record, the `Chunker` interface, and JSONL input/output.

Every chunker turns a `ParsedFiling` into a list of `Chunk`s. Chunks never cross an
Item section, and their IDs are deterministic (`<TICKER>-<item>-<p|t><nnnn>`), so the
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
    index: int  # position among this section's chunks of the same kind
    text: str
    word_count: int
    chunker: str  # e.g. "fixed-p250-o50-t120": the chunker and its settings


def chunk_id(ticker: str, item: str, kind: str, index: int) -> str:
    return f"{ticker}-{item}-{kind[0]}{index:04d}"


class Chunker(Protocol):
    """Anything with a `name` and a `chunk(filing)` method is a chunker."""

    @property
    def name(self) -> str: ...

    def chunk(self, filing: ParsedFiling) -> list[Chunk]: ...


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    """Write one chunk per line, keys in a fixed order, so diffs stay readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(c.model_dump(), ensure_ascii=False) for c in chunks]
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
