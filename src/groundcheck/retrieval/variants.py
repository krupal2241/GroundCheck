"""Retrieval variants: which stages a retriever runs, as named in config/retrieval.yaml.

A variant combines up to four independent choices:

- `embedder`: the dense model (`bge-small`, `bge-base`, `hashing` for tests), or none
  for a lexical-only retriever;
- `headers`: prepend a short contextual header to each chunk *only when it is
  embedded* (and when the reranker reads it). The stored chunk text, and so every gold
  string and evidence span, is never changed;
- `bm25`: add a BM25 ranking and merge it with the dense ranking by reciprocal rank
  fusion (RRF), or use BM25 alone when there is no embedder;
- `rerank`: rerank the first stage's top `pool` candidates with a cross-encoder.

`pool` (candidates fused and reranked, default 30) and `rrf_k` (default 60) are
shared settings at the top of the file.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from groundcheck.chunking.base import Chunk
from groundcheck.config import REPO_ROOT, CompaniesConfig, ConfigError

DEFAULT_RETRIEVAL_CONFIG_PATH = REPO_ROOT / "config" / "retrieval.yaml"

# Bump when the header format changes: indexes built with another version are refused.
HEADER_VERSION = "h1"

EmbedderKind = Literal["bge-small", "bge-base", "hashing"]
RerankerKind = Literal["bge-reranker-base", "overlap"]


class VariantSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = ""
    # Compared by --compare but never recommended as the default (e.g. BM25 alone).
    diagnostic: bool = False
    embedder: EmbedderKind | None = "bge-small"
    headers: bool = False
    bm25: bool = False
    rerank: RerankerKind | None = None

    @model_validator(mode="after")
    def _has_a_first_stage(self) -> "VariantSpec":
        if self.embedder is None and not self.bm25:
            raise ValueError("a variant needs an embedder, bm25, or both")
        if self.headers and self.embedder is None and self.rerank is None:
            raise ValueError("headers only apply to embedding and reranking")
        return self

    def with_overrides(
        self, embedder: str | None = None, reranker: str | None = None
    ) -> "VariantSpec":
        """Swap the models (e.g. the hashing fake for a smoke run) but keep the stages."""
        update: dict = {}
        if embedder is not None and self.embedder is not None:
            update["embedder"] = embedder
        if reranker is not None and self.rerank is not None:
            update["rerank"] = reranker
        return self.model_validate({**self.model_dump(), **update})


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: str
    pool: int = Field(default=30, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1)
    variants: dict[str, VariantSpec]

    @model_validator(mode="after")
    def _default_exists(self) -> "RetrievalConfig":
        if self.default not in self.variants:
            raise ValueError(f"default {self.default!r} is not one of the variants")
        return self

    def get(self, name: str | None = None) -> tuple[str, VariantSpec]:
        name = name or self.default
        if name not in self.variants:
            raise ConfigError(
                f"unknown retrieval variant {name!r}; configured: {', '.join(self.variants)}"
            )
        return name, self.variants[name]


def load_retrieval_config(path: Path | None = None) -> RetrievalConfig:
    target = Path(path) if path is not None else DEFAULT_RETRIEVAL_CONFIG_PATH
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"retrieval config not found: {target}") from exc
    try:
        return RetrievalConfig.model_validate(raw or {})
    except ValidationError as exc:
        raise ConfigError(f"invalid retrieval config {target}:\n{exc}") from exc


# --- contextual headers ------------------------------------------------------------------


def fiscal_years(companies: CompaniesConfig) -> dict[str, str]:
    """`FY2025` per ticker, from the period of report (the fiscal year-end date). The
    year of the end date is how each company names its own fiscal year (Walmart's year
    ending 2026-01-31 is fiscal 2026)."""
    return {c.ticker: f"FY{c.period[:4]}" for c in companies.companies if c.period}


def contextual_header(chunk: Chunk, fiscal_year: str | None) -> str:
    """`AAPL FY2025 10-K | Item 7 Management's Discussion ... | table`."""
    filing = " ".join(p for p in (chunk.ticker, fiscal_year, "10-K") if p)
    section = f"Item {chunk.item} {chunk.section_title}".strip()
    return f"{filing} | {section} | {chunk.kind}"


def model_text(chunk: Chunk, headers: bool, fiscal_year: str | None) -> str:
    """The text a model reads for this chunk. A new string: `chunk.text` is untouched."""
    if not headers:
        return chunk.text
    return contextual_header(chunk, fiscal_year) + "\n" + chunk.text


def embedding_key(embedder_name: str, headers: bool) -> str:
    """What a dense index depends on: the model, and the header format if any."""
    return embedder_name + (f"+headers-{HEADER_VERSION}" if headers else "")


def index_folder_name(key: str) -> str:
    return key.replace("/", "_")


def texts_sha256(texts: list[str]) -> str:
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()


# --- choosing a default ------------------------------------------------------------------


def recommend_default(
    rows: list[dict], tolerance: float = 0.05, k: int = 5
) -> tuple[str, str] | None:
    """The rule from PLAN/DESIGN: the best recall@k, but among the variants within
    `tolerance` of the best, the one with the lowest average query latency. Returns
    (variant, reason), or None with no rows. Each row needs `variant`, `recall`
    (keyed by k) and `latency_ms`; rows with `diagnostic: True` are left out."""
    rows = [r for r in rows if not r.get("diagnostic")]
    if not rows:
        return None
    best = max(r["recall"][k] for r in rows)
    close = [r for r in rows if r["recall"][k] >= best - tolerance - 1e-9]
    pick = min(close, key=lambda r: (r["latency_ms"], -r["recall"][k]))
    reason = (
        f"best recall@{k} is {best:.2f}; {len(close)} variant(s) within {tolerance:.2f} of it; "
        f"{pick['variant']} is the fastest of those ({pick['latency_ms']:.1f} ms/query, "
        f"recall@{k} {pick['recall'][k]:.2f})"
    )
    return pick["variant"], reason
