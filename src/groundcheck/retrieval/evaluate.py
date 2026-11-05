"""Retrieval evaluation: recall@k and section-hit@k against human-written gold evidence.

The question file (`data/eval/retrieval_questions.yaml`) lists questions, each with
one or more **gold evidence strings**: exact text copied from the parsed filing
(`data/parsed/<TICKER>.txt`) that answers it. A retrieved chunk "contains" a gold
string if the string appears in the chunk's text after whitespace normalization on
both sides (`normalize_for_match`).

Per question, for each k:

- **recall@k** = the fraction of the question's gold strings that appear in at least
  one of the top-k chunks. A two-part multi_step question that finds one of its two
  values scores 0.5.
- **section-hit@k** = 1 if any of the top-k chunks comes from a gold Item section,
  else 0. Gold sections are `gold_items` if you give them; otherwise they're the
  sections whose chunks contain a gold string.
- **reciprocal rank** = 1 / the rank of the first retrieved chunk containing any gold
  string, 0 if none is in the top max(k). MRR is its mean.
- **latency** = wall time of the retrieve call for that question, in milliseconds
  (query embedding, search, fusion and reranking; model loading is not included).

Reported numbers are means over the evaluated questions. Questions are skipped (and
counted) when they're still unfilled template entries, `unanswerable` (nothing to
retrieve), or when a gold string doesn't occur anywhere in that company's chunks. That
last case is a typo or a normalization mismatch, not a retrieval miss, so it's
reported as a warning rather than scored as a zero.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from groundcheck.chunking.base import Chunk
from groundcheck.config import REPO_ROOT, ConfigError

DEFAULT_QUESTIONS_PATH = REPO_ROOT / "data" / "eval" / "retrieval_questions.yaml"
DEFAULT_KS = (1, 3, 5, 10)

QType = Literal["numeric_lookup", "table", "multi_step", "yes_no", "temporal", "unanswerable"]


class RetrievalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    ticker: str
    qtype: QType
    question: str = ""
    gold_evidence: list[str] = Field(default_factory=list)
    gold_items: list[str] = Field(default_factory=list)
    notes: str = ""

    @property
    def is_filled(self) -> bool:
        if not self.question.strip():
            return False
        return self.qtype == "unanswerable" or any(g.strip() for g in self.gold_evidence)


class QuestionFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    questions: list[RetrievalQuestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "QuestionFile":
        ids = [q.id for q in self.questions]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate question ids: {duplicates}")
        return self


def load_questions(path: Path = DEFAULT_QUESTIONS_PATH) -> QuestionFile:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"question file not found: {path}") from exc
    try:
        return QuestionFile.model_validate(raw or {})
    except ValidationError as exc:
        raise ConfigError(f"invalid question file {path}:\n{exc}") from exc


def normalize_for_match(text: str) -> str:
    """Collapse all whitespace (spaces, newlines, tabs) to single spaces and strip."""
    return " ".join(text.split())


def contains(chunk: Chunk, gold: str) -> bool:
    return normalize_for_match(gold) in normalize_for_match(chunk.text)


class QuestionResult(BaseModel):
    id: str
    ticker: str
    qtype: str
    status: Literal["evaluated", "unfilled", "unanswerable", "gold_not_in_corpus"]
    gold_items: list[str] = Field(default_factory=list)
    missing_gold: list[str] = Field(default_factory=list)
    retrieved: list[str] = Field(default_factory=list)  # chunk IDs, best first
    first_gold_rank: int | None = None  # rank of the first chunk containing any gold
    reciprocal_rank: float | None = None
    latency_ms: float | None = None
    recall: dict[int, float] = Field(default_factory=dict)
    section_hit: dict[int, float] = Field(default_factory=dict)


class Summary(BaseModel):
    n: int
    recall: dict[int, float]
    section_hit: dict[int, float]
    mrr: float = 0.0
    latency_ms: float = 0.0  # mean per query


class RetrievalReport(BaseModel):
    variant: str = ""
    embedder: str  # the retriever's description (models and stages)
    chunker: str
    ks: list[int]
    overall: Summary
    by_ticker: dict[str, Summary]
    by_qtype: dict[str, Summary]
    skipped: dict[str, int]
    questions: list[QuestionResult]


Retriever = Callable[[str, str, int], list[Chunk]]  # (ticker, question, k) -> chunks


def gold_strings(question: RetrievalQuestion) -> list[str]:
    return [g for g in question.gold_evidence if g.strip()]


def check_gold(golds: list[str], chunks: list[Chunk]) -> tuple[list[str], list[str]]:
    """Return (gold strings not found in any chunk, items whose chunks contain gold)."""
    missing, items = [], []
    for gold in golds:
        holders = [c for c in chunks if contains(c, gold)]
        if not holders:
            missing.append(gold)
        for c in holders:
            if c.item not in items:
                items.append(c.item)
    return missing, items


def evaluate_question(
    question: RetrievalQuestion,
    corpus: list[Chunk],
    retrieve: Retriever,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> QuestionResult:
    base = {"id": question.id, "ticker": question.ticker, "qtype": question.qtype}
    if not question.is_filled:
        return QuestionResult(**base, status="unfilled")
    if question.qtype == "unanswerable":
        return QuestionResult(**base, status="unanswerable")
    golds = gold_strings(question)
    missing, found_items = check_gold(golds, corpus)
    if missing:
        return QuestionResult(**base, status="gold_not_in_corpus", missing_gold=missing)
    gold_items = question.gold_items or found_items

    start = time.perf_counter()
    hits = retrieve(question.ticker, question.question, max(ks))
    latency_ms = (time.perf_counter() - start) * 1000
    first = next((r for r, c in enumerate(hits, 1) if any(contains(c, g) for g in golds)), None)
    recall, section_hit = {}, {}
    for k in ks:
        top = hits[:k]
        found = sum(any(contains(c, g) for c in top) for g in golds)
        recall[k] = found / len(golds)
        section_hit[k] = float(any(c.item in gold_items for c in top))
    return QuestionResult(
        **base,
        status="evaluated",
        gold_items=gold_items,
        retrieved=[c.chunk_id for c in hits],
        first_gold_rank=first,
        reciprocal_rank=1.0 / first if first else 0.0,
        latency_ms=latency_ms,
        recall=recall,
        section_hit=section_hit,
    )


def check_questions(
    questions: list[RetrievalQuestion], corpora: dict[str, list[Chunk]]
) -> list[str]:
    """Offline sanity check (no embedding): every filled question's gold must be findable."""
    problems = []
    for q in questions:
        if not q.is_filled or q.qtype == "unanswerable":
            continue
        if q.ticker not in corpora:
            problems.append(f"{q.id}: no chunks for ticker {q.ticker}")
            continue
        missing, items = check_gold(gold_strings(q), corpora[q.ticker])
        problems += [
            f"{q.id}: gold evidence not found in {q.ticker} chunks: {g!r}" for g in missing
        ]
        stray = [i for i in q.gold_items if i not in {c.item for c in corpora[q.ticker]}]
        if stray:
            problems.append(f"{q.id}: gold_items {stray} are not sections of {q.ticker}")
        elif not missing and q.gold_items and not set(items) & set(q.gold_items):
            problems.append(
                f"{q.id}: gold evidence found only in Items {items}, not {q.gold_items}"
            )
    return problems


def summarize(results: list[QuestionResult], ks: tuple[int, ...]) -> Summary:
    scored = [r for r in results if r.status == "evaluated"]
    n = len(scored)

    def mean(values: list[float]) -> float:
        return sum(values) / n if n else 0.0

    return Summary(
        n=n,
        recall={k: mean([r.recall[k] for r in scored]) for k in ks},
        section_hit={k: mean([r.section_hit[k] for r in scored]) for k in ks},
        mrr=mean([r.reciprocal_rank or 0.0 for r in scored]),
        latency_ms=mean([r.latency_ms or 0.0 for r in scored]),
    )


def evaluate(
    questions: list[RetrievalQuestion],
    corpora: dict[str, list[Chunk]],
    retrieve: Retriever,
    embedder: str,
    ks: tuple[int, ...] = DEFAULT_KS,
    variant: str = "",
) -> RetrievalReport:
    unknown = sorted({q.ticker for q in questions if q.is_filled} - set(corpora))
    if unknown:
        raise ConfigError(f"questions refer to tickers with no chunks: {unknown}")
    results = [evaluate_question(q, corpora.get(q.ticker, []), retrieve, ks) for q in questions]
    scored = [r for r in results if r.status == "evaluated"]
    chunkers = sorted({c.chunker for t in {r.ticker for r in scored} for c in corpora[t]})
    skipped: dict[str, int] = {}
    for r in results:
        if r.status != "evaluated":
            skipped[r.status] = skipped.get(r.status, 0) + 1
    return RetrievalReport(
        variant=variant,
        embedder=embedder,
        chunker=", ".join(chunkers),
        ks=list(ks),
        overall=summarize(results, ks),
        by_ticker={
            t: summarize([r for r in results if r.ticker == t], ks)
            for t in sorted({r.ticker for r in scored})
        },
        by_qtype={
            q: summarize([r for r in results if r.qtype == q], ks)
            for q in sorted({r.qtype for r in scored})
        },
        skipped=skipped,
        questions=results,
    )


def format_report(report: RetrievalReport) -> str:
    ks = report.ks
    header = f"{'':<18} {'n':>4}  " + "  ".join(f"R@{k:<3}" for k in ks) + "  "
    header += "  ".join(f"S@{k:<3}" for k in ks) + "  MRR    ms/q"

    def row(label: str, s: Summary) -> str:
        cells = [f"{s.recall[k]:.2f} " for k in ks] + [f"{s.section_hit[k]:.2f} " for k in ks]
        cells += [f"{s.mrr:.2f} ", f"{s.latency_ms:6.1f}"]
        return f"{label:<18} {s.n:>4}  " + "  ".join(cells)

    variant = f"Variant: {report.variant}   " if report.variant else ""
    lines = [
        f"{variant}Retriever: {report.embedder}   Chunker: {report.chunker or '-'}",
        "R@k = recall@k, S@k = section-hit@k, MRR = mean reciprocal rank, ms/q = mean latency",
        header,
        row("overall", report.overall),
    ]
    lines += [row(t, s) for t, s in report.by_ticker.items()]
    lines += [row(q, s) for q, s in report.by_qtype.items()]
    if report.skipped:
        lines.append("Skipped: " + ", ".join(f"{n} {k}" for k, n in sorted(report.skipped.items())))
    for r in report.questions:
        for gold in r.missing_gold:
            lines.append(f"WARNING: {r.id}: gold evidence not found in {r.ticker} chunks: {gold!r}")
    return "\n".join(lines)
