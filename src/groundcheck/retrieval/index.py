"""One exact-cosine FAISS index per company.

Layout (gitignored, rebuilt from the committed chunks). Retrieval variants keep their
indexes in one folder per embedding (see `retrieval/retriever.py`), e.g.
`data/index/BAAI_bge-small-en-v1.5+headers-h1/`, and inside it:

    <TICKER>/index.faiss   the vectors (IndexFlatIP over unit vectors)
    <TICKER>/meta.json     embedder name and dim, the variant that built the index, its
                           embedding key, the sha256 of the exact texts that were
                           embedded, the chunk IDs in vector order, and the sha256 of
                           the chunks file

Why one index per company: every question is about one filing, so searching only that
filing's index gives it the full `top_k` from the right 10-K, and a similar sentence
in another company's filing can never crowd it out.

`faiss` is imported only when an index is built or loaded, so commands that never touch
an index (e.g. `eval-verifiers`) don't load it. On macOS, faiss and torch each bring an
OpenMP runtime, and loading both into one process can abort it (docs/DESIGN.md, Part 0).

The meta file makes stale indexes loud: loading fails if the chunks file has changed
since the index was built, if the texts a variant would embed (e.g. with or without
contextual headers) differ from the ones that were embedded, or if the query embedder
differs from the one used to build.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from groundcheck.chunking.base import Chunk, chunks_path, read_chunks
from groundcheck.config import REPO_ROOT, ConfigError
from groundcheck.retrieval.embedder import Embedder
from groundcheck.retrieval.variants import texts_sha256

if TYPE_CHECKING:
    import faiss

DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "index"


def _faiss() -> Any:
    import faiss

    return faiss


@dataclass
class Hit:
    chunk: Chunk
    score: float  # cosine similarity
    rank: int  # 1-based


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_index(vectors: np.ndarray) -> "faiss.IndexFlatIP":
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    index = _faiss().IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def build_company_index(
    ticker: str,
    embedder: Embedder,
    chunks_dir: Path,
    index_dir: Path = DEFAULT_INDEX_DIR,
    *,
    texts: list[str] | None = None,
    variant: str | None = None,
    embedding: str | None = None,
) -> int:
    """Embed a company's chunks and write its index. Returns the number of vectors.

    `texts` (one per chunk, in chunk order) is what gets embedded: by default the chunk
    text itself; a variant with contextual headers passes header + text. Only the
    vectors depend on it; the chunks file is never rewritten."""
    source = chunks_path(ticker, chunks_dir)
    chunks = read_chunks(source)
    if not chunks:
        raise ConfigError(f"{source} has no chunks")
    texts = [c.text for c in chunks] if texts is None else texts
    if len(texts) != len(chunks):
        raise ConfigError(f"{ticker}: {len(texts)} texts for {len(chunks)} chunks")
    vectors = embedder.embed_documents(texts)
    index = build_index(vectors)
    out = index_dir / ticker
    out.mkdir(parents=True, exist_ok=True)
    _faiss().write_index(index, str(out / "index.faiss"))
    meta = {
        "ticker": ticker,
        "embedder": embedder.name,
        "dim": embedder.dim,
        "variant": variant,
        "embedding": embedding or embedder.name,
        "embedded_texts_sha256": texts_sha256(texts),
        "chunker": chunks[0].chunker,
        "chunks_file": source.name,
        "chunks_sha256": file_sha256(source),
        "chunk_ids": [c.chunk_id for c in chunks],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return index.ntotal


class CompanyIndex:
    """A loaded per-company index plus the chunks its vectors point to."""

    def __init__(
        self,
        ticker: str,
        index: "faiss.Index",
        chunks: list[Chunk],
        embedder_name: str,
        meta: dict | None = None,
    ):
        if index.ntotal != len(chunks):
            raise ConfigError(f"{ticker}: index has {index.ntotal} vectors, {len(chunks)} chunks")
        self.ticker = ticker
        self.index = index
        self.chunks = chunks
        self.embedder_name = embedder_name
        self.meta = meta or {}

    @classmethod
    def load(
        cls,
        ticker: str,
        chunks_dir: Path,
        index_dir: Path = DEFAULT_INDEX_DIR,
        *,
        variant: str | None = None,
        embedding: str | None = None,
        expected_texts: list[str] | None = None,
    ) -> "CompanyIndex":
        """Load and check an index. A variant passes its name, its embedding key and the
        texts it would embed; an index built for anything else is refused."""
        folder = index_dir / ticker
        hint = f"`groundcheck index --variant {variant}`" if variant else "`groundcheck index`"
        if not (folder / "meta.json").is_file():
            raise ConfigError(f"no index for {ticker} in {index_dir}; run {hint}")
        meta = json.loads((folder / "meta.json").read_text())
        source = chunks_path(ticker, chunks_dir)
        if not source.is_file() or file_sha256(source) != meta["chunks_sha256"]:
            raise ConfigError(
                f"{ticker}: {source} changed since the index was built; re-run {hint}"
            )
        built_by = f"variant {meta.get('variant')!r} ({meta.get('embedding', meta['embedder'])})"
        if embedding is not None and meta.get("embedding", meta["embedder"]) != embedding:
            raise ConfigError(
                f"{ticker}: the index in {folder} was built by {built_by}, but variant "
                f"{variant!r} needs {embedding}; run {hint}"
            )
        if expected_texts is not None and meta.get("embedded_texts_sha256") != texts_sha256(
            expected_texts
        ):
            raise ConfigError(
                f"{ticker}: the index in {folder} was built by {built_by} from different "
                f"texts than variant {variant!r} embeds; run {hint}"
            )
        by_id = {c.chunk_id: c for c in read_chunks(source)}
        chunks = [by_id[i] for i in meta["chunk_ids"]]
        index = _faiss().read_index(str(folder / "index.faiss"))
        return cls(ticker, index, chunks, meta["embedder"], meta)

    def search(self, embedder: Embedder, query: str, k: int = 10) -> list[Hit]:
        if embedder.name != self.embedder_name:
            raise ConfigError(
                f"{self.ticker}: index was built with {self.embedder_name!r}, "
                f"but the query embedder is {embedder.name!r}"
            )
        vector = np.ascontiguousarray(embedder.embed_queries([query]), dtype=np.float32)
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(vector, k)
        return [
            Hit(chunk=self.chunks[i], score=float(s), rank=rank)
            for rank, (s, i) in enumerate(zip(scores[0], ids[0], strict=True), start=1)
            if i >= 0
        ]
