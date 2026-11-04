"""Second-stage rerankers behind one `Reranker` interface.

- `BgeReranker`: the cross-encoder `BAAI/bge-reranker-base` through
  sentence-transformers' `CrossEncoder`. It reads the question and each candidate
  passage together, which is slower than comparing two precomputed vectors but much
  better at "does this passage answer this question". Needs the `[rerank]` extra and
  downloads the model (about 1.1 GB) from Hugging Face on first use, so it only runs on
  your machine, never in CI.
- `OverlapReranker`: a deterministic fake for tests and offline smoke runs: the score is
  the IDF-free fraction of the question's tokens that occur in the passage.
"""

from typing import Protocol

from groundcheck.config import ConfigError
from groundcheck.retrieval.bm25 import tokenize

BGE_RERANKER = "BAAI/bge-reranker-base"
RERANKER_MAX_TOKENS = 512


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class OverlapReranker:
    name = "overlap"

    def score(self, query: str, passages: list[str]) -> list[float]:
        wanted = set(tokenize(query))
        if not wanted:
            return [0.0] * len(passages)
        return [len(wanted & set(tokenize(p))) / len(wanted) for p in passages]


class BgeReranker:
    def __init__(self, model_name: str = BGE_RERANKER, device: str | None = None, batch_size=8):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ConfigError(
                'the reranker needs the [rerank] extra: pip install -e ".[rerank]"'
            ) from exc
        from groundcheck.retrieval.embedder import pick_device

        self.model_name = model_name
        self.device = device or pick_device()
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, max_length=RERANKER_MAX_TOKENS, device=self.device)

    @property
    def name(self) -> str:
        return self.model_name

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict(
            [(query, p) for p in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]


RERANKERS = ("bge-reranker-base", "overlap")


def make_reranker(kind: str, device: str | None = None) -> Reranker:
    if kind == "bge-reranker-base":
        return BgeReranker(device=device)
    if kind == "overlap":
        return OverlapReranker()
    raise ConfigError(f"unknown reranker {kind!r}; use one of {', '.join(RERANKERS)}")
