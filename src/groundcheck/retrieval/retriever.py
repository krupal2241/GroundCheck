"""The `Retriever` interface and the configurable retriever behind every variant.

A `Retriever` answers `search(ticker, query, k)` with ranked `Hit`s from that company's
filing only. `VariantRetriever` runs the stages a `VariantSpec` turns on:

1. first stage: dense (FAISS, cosine), BM25, or both merged by reciprocal rank fusion:
   `score(chunk) = sum over rankings of 1 / (rrf_k + rank)`, over each ranking's top
   `pool`. RRF only uses ranks, so the two incomparable score scales never need
   calibrating;
2. optional rerank: the first stage's top `pool` candidates are rescored by a
   cross-encoder and the best `k` returned.

Contextual headers (`variants.model_text`) are built on the fly for the embedder and
the reranker. The `Chunk` objects in every `Hit` are the stored chunks, unchanged.

`Hit.score` is the last stage's score: cosine similarity (dense), BM25 score (BM25
only), RRF score (hybrid) or reranker score (rerank).

Indexes live in `<index_dir>/<embedding key>/<TICKER>/`, one folder per embedding (model
plus header format), so variants that embed the same texts (e.g. `dense` and `hybrid`)
share one index instead of embedding the filings twice. The index's `meta.json` records
the variant that built it, and loading checks that it holds exactly the texts the
requesting variant embeds.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from groundcheck.chunking.base import Chunk, chunks_path, read_chunks
from groundcheck.config import ConfigError
from groundcheck.retrieval.bm25 import BM25
from groundcheck.retrieval.embedder import Embedder, make_embedder
from groundcheck.retrieval.index import CompanyIndex, Hit, build_company_index
from groundcheck.retrieval.rerank import Reranker, make_reranker
from groundcheck.retrieval.variants import (
    HEADER_VERSION,
    RetrievalConfig,
    VariantSpec,
    embedding_key,
    index_folder_name,
    model_text,
)


class Retriever(Protocol):
    @property
    def variant(self) -> str: ...

    @property
    def description(self) -> str: ...

    def search(self, ticker: str, query: str, k: int) -> list[Hit]: ...


def rrf_merge(rankings: list[list[int]], rrf_k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion of rankings of chunk positions (best first). Ties go to
    the better single rank, then to the earlier chunk."""
    scores: dict[int, float] = {}
    best: dict[int, int] = {}
    for ranking in rankings:
        for rank, pos in enumerate(ranking, start=1):
            scores[pos] = scores.get(pos, 0.0) + 1.0 / (rrf_k + rank)
            best[pos] = min(best.get(pos, rank), rank)
    return sorted(scores.items(), key=lambda item: (-item[1], best[item[0]], item[0]))


def describe(
    spec: VariantSpec,
    embedder: Embedder | None,
    reranker: Reranker | None,
    config: RetrievalConfig,
) -> str:
    """One line for logs and candidate metadata, with the real model names."""
    parts = []
    if embedder is not None:
        parts.append(f"dense={embedder.name}")
    if spec.headers:
        parts.append(f"headers={HEADER_VERSION}")
    if spec.bm25:
        parts.append("bm25")
    if spec.bm25 and embedder is not None:
        parts.append(f"rrf_k={config.rrf_k}")
    if reranker is not None:
        parts.append(f"rerank={reranker.name}")
    if (spec.bm25 and embedder is not None) or reranker is not None:
        parts.append(f"pool={config.pool}")
    return " ".join(parts)


@dataclass
class _Company:
    chunks: list[Chunk]
    texts: list[str]  # what the embedder and reranker read (with headers if enabled)
    index: CompanyIndex | None
    bm25: BM25 | None


class VariantRetriever:
    def __init__(
        self,
        name: str,
        spec: VariantSpec,
        config: RetrievalConfig,
        companies: dict[str, _Company],
        embedder: Embedder | None,
        reranker: Reranker | None,
    ):
        self.name, self.spec, self.config = name, spec, config
        self.companies = companies
        self.embedder = embedder
        self.reranker = reranker

    @property
    def variant(self) -> str:
        return self.name

    @property
    def description(self) -> str:
        return describe(self.spec, self.embedder, self.reranker, self.config)

    def chunker(self, ticker: str) -> str:
        return self.companies[ticker].chunks[0].chunker

    def search(self, ticker: str, query: str, k: int) -> list[Hit]:
        if ticker not in self.companies:
            raise ConfigError(f"variant {self.name!r} has no index loaded for {ticker}")
        company = self.companies[ticker]
        hybrid = company.index is not None and company.bm25 is not None
        pool = max(k, self.config.pool) if hybrid or self.reranker else k
        positions = {c.chunk_id: i for i, c in enumerate(company.chunks)}

        ranked: list[tuple[int, float]]
        dense = lexical = None
        if company.index is not None:
            hits = company.index.search(self.embedder, query, pool)
            dense = [(positions[h.chunk.chunk_id], h.score) for h in hits]
        if company.bm25 is not None:
            lexical = company.bm25.top(query, pool)
        if dense is not None and lexical is not None:
            ranked = rrf_merge([[p for p, _ in dense], [p for p, _ in lexical]], self.config.rrf_k)
        else:
            ranked = dense if dense is not None else lexical
        ranked = ranked[:pool]

        if self.reranker is not None and ranked:
            scores = self.reranker.score(query, [company.texts[p] for p, _ in ranked])
            order = sorted(range(len(ranked)), key=lambda i: (-scores[i], i))
            ranked = [(ranked[i][0], scores[i]) for i in order]
        return [
            Hit(chunk=company.chunks[pos], score=float(score), rank=rank)
            for rank, (pos, score) in enumerate(ranked[:k], start=1)
        ]


def _texts(chunks: list[Chunk], spec: VariantSpec, fiscal_year: str | None) -> list[str]:
    return [model_text(c, spec.headers, fiscal_year) for c in chunks]


def variant_index_dir(index_dir: Path, embedder: Embedder, spec: VariantSpec) -> Path:
    return index_dir / index_folder_name(embedding_key(embedder.name, spec.headers))


def build_variant_index(
    name: str,
    spec: VariantSpec,
    ticker: str,
    embedder: Embedder,
    chunks_dir: Path,
    index_dir: Path,
    fiscal_year: str | None = None,
) -> tuple[int, Path]:
    """Embed one company's chunks the way variant `name` does and write the index.
    Returns (number of vectors, folder)."""
    if spec.embedder is None:
        raise ConfigError(f"variant {name!r} has no embedder, so it needs no index")
    chunks = read_chunks(chunks_path(ticker, chunks_dir))
    folder = variant_index_dir(index_dir, embedder, spec)
    n = build_company_index(
        ticker,
        embedder,
        chunks_dir,
        folder,
        texts=_texts(chunks, spec, fiscal_year),
        variant=name,
        embedding=embedding_key(embedder.name, spec.headers),
    )
    return n, folder / ticker


def open_retriever(
    name: str,
    spec: VariantSpec,
    config: RetrievalConfig,
    tickers: list[str],
    chunks_dir: Path,
    index_dir: Path,
    fiscal_years: dict[str, str] | None = None,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    device: str | None = None,
) -> VariantRetriever:
    """Load everything variant `name` needs for `tickers`, so a missing or mismatching
    index fails here, before any query. Models are created from the spec unless passed
    in (tests pass fakes; `--compare` reuses one model across variants)."""
    fiscal_years = fiscal_years or {}
    if spec.embedder is not None and embedder is None:
        embedder = make_embedder(spec.embedder, device)
    if spec.embedder is None:
        embedder = None
    if spec.rerank is not None and reranker is None:
        reranker = make_reranker(spec.rerank, device)
    if spec.rerank is None:
        reranker = None
    companies = {}
    for ticker in tickers:
        chunks = read_chunks(chunks_path(ticker, chunks_dir))
        texts = _texts(chunks, spec, fiscal_years.get(ticker))
        index = None
        if embedder is not None:
            index = CompanyIndex.load(
                ticker,
                chunks_dir,
                variant_index_dir(index_dir, embedder, spec),
                variant=name,
                embedding=embedding_key(embedder.name, spec.headers),
                expected_texts=texts,
            )
            by_id = {c.chunk_id: (c, t) for c, t in zip(chunks, texts, strict=True)}
            chunks = [by_id[c.chunk_id][0] for c in index.chunks]
            texts = [by_id[c.chunk_id][1] for c in index.chunks]
        bm25 = BM25([c.text for c in chunks]) if spec.bm25 else None
        companies[ticker] = _Company(chunks, texts, index, bm25)
    return VariantRetriever(name, spec, config, companies, embedder, reranker)
