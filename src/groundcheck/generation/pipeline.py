"""The candidate pipeline and its JSONL records.

Per question: retrieve the top-k chunks from the company's FAISS index, ask the LLM to
answer from only those chunks, then ask it to split the answer into atomic claims
(structured JSON, validated, with bounded retries).

Output: `data/candidates/<TICKER>.jsonl`, one `CandidateRecord` per line. Files are
append-only: a record is written (and flushed to disk) as soon as it is complete, and
a re-run skips every `candidate_id` already present, so an interrupted run resumes
where it stopped. The candidate ID is `<question id>@<provider>:<model>`, so running a
second model appends its own records instead of being skipped.
"""

import datetime as dt
import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from groundcheck import __version__
from groundcheck.config import REPO_ROOT, LLMConfig
from groundcheck.generation.prompts import (
    ANSWER_SYSTEM,
    CLAIMS_SYSTEM,
    PROMPT_VERSION,
    ClaimSplit,
    answer_prompt,
    claims_prompt,
    make_request,
)
from groundcheck.generation.questions import GenerationQuestion
from groundcheck.llm.base import LLMError, LLMMetadata, LLMProvider
from groundcheck.llm.structured import generate_structured
from groundcheck.retrieval.index import Hit

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATES_DIR = REPO_ROOT / "data" / "candidates"
DRY_RUN_CANDIDATES_DIR = REPO_ROOT / "data" / "cache" / "dry-run" / "candidates"


class RetrievalInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedder: str
    chunker: str
    top_k: int
    chunks_sha256: str  # of data/chunks/<TICKER>.jsonl when the index was searched


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    chunk_id: str
    score: float
    item: str
    kind: Literal["prose", "table"]
    text: str


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str  # <candidate_id>#c01
    text: str


class CandidateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    candidate_id: str
    question_id: str
    ticker: str
    qtype: str
    question: str
    question_source: str
    retrieval: RetrievalInfo
    retrieved: list[RetrievedChunk]
    answer: str
    claims: list[Claim]
    answer_model: LLMMetadata
    claims_model: LLMMetadata
    prompt_version: str
    groundcheck_version: str
    generated_at: str  # UTC, ISO 8601


def model_tag(provider: LLMProvider) -> str:
    return f"{provider.name}:{provider.model}"


def candidate_id(question_id: str, provider: LLMProvider) -> str:
    return f"{question_id}@{model_tag(provider)}"


def candidates_path(ticker: str, candidates_dir: Path = DEFAULT_CANDIDATES_DIR) -> Path:
    return candidates_dir / f"{ticker}.jsonl"


def read_candidates(path: Path) -> list[CandidateRecord]:
    """Every valid record in `path`. A broken line (e.g. a crash mid-write) is skipped
    with a warning, never fatal: the rest of the file is still usable."""
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(CandidateRecord.model_validate_json(line))
            except ValidationError:
                logger.warning("%s:%d: not a valid candidate record; skipped", path, number)
    return records


def append_record(record: CandidateRecord, path: Path) -> None:
    """Append one line and flush it to disk. If the file ends in a partial line (an
    earlier crash), a newline is added first so the new record stays parseable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.model_dump(), ensure_ascii=False) + "\n"
    if path.is_file() and path.stat().st_size > 0:
        with path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                line = "\n" + line
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


Retriever = Callable[[str, str, int], list[Hit]]


@dataclass
class Generator:
    provider: LLMProvider
    retrieve: Retriever  # (ticker, question, k) -> hits, best first
    retrieval_info: Callable[[str], RetrievalInfo]  # per ticker
    config: LLMConfig

    def _request(self, system: str, prompt: str):
        c = self.config
        return make_request(system, prompt, c.temperature, c.seed, c.max_tokens, c.context_window)

    def generate(self, question: GenerationQuestion) -> CandidateRecord:
        hits = self.retrieve(question.ticker, question.question, self.config.top_k)
        retrieved = [
            RetrievedChunk(
                rank=h.rank,
                chunk_id=h.chunk.chunk_id,
                score=round(h.score, 6),
                item=h.chunk.item,
                kind=h.chunk.kind,
                text=h.chunk.text,
            )
            for h in hits
        ]
        passages = [(c.chunk_id, c.text) for c in retrieved]
        prompt = answer_prompt(question.ticker, question.question, passages)
        answer = self.provider.generate(self._request(ANSWER_SYSTEM, prompt))
        split, claims_response = generate_structured(
            self.provider,
            self._request(CLAIMS_SYSTEM, claims_prompt(question.question, answer.text)),
            ClaimSplit,
            max_attempts=self.config.json_max_attempts,
        )
        cid = candidate_id(question.id, self.provider)
        return CandidateRecord(
            candidate_id=cid,
            question_id=question.id,
            ticker=question.ticker,
            qtype=question.qtype,
            question=question.question,
            question_source=question.source,
            retrieval=self.retrieval_info(question.ticker),
            retrieved=retrieved,
            answer=answer.text.strip(),
            claims=[
                Claim(claim_id=f"{cid}#c{i:02d}", text=text)
                for i, text in enumerate(split.claims, start=1)
            ],
            answer_model=answer.metadata,
            claims_model=claims_response.metadata,
            prompt_version=PROMPT_VERSION,
            groundcheck_version=__version__,
            generated_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        )


@dataclass
class RunSummary:
    written: int = 0
    already_done: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)  # (question id, error)
    claims: int = 0


def batches(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run(
    generator: Generator,
    questions: list[GenerationQuestion],
    candidates_dir: Path,
    batch_size: int = 10,
    limit: int | None = None,
    sleep_between_batches: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    echo: Callable[[str], None] = print,
) -> RunSummary:
    """Generate every question that has no record yet, in batches; see module docs."""
    summary = RunSummary()
    done: set[str] = set()
    for ticker in sorted({q.ticker for q in questions}):
        done |= {r.candidate_id for r in read_candidates(candidates_path(ticker, candidates_dir))}
    pending = [q for q in questions if candidate_id(q.id, generator.provider) not in done]
    summary.already_done = len(questions) - len(pending)
    if limit is not None:
        pending = pending[:limit]
    echo(
        f"{len(questions)} question(s): {summary.already_done} already generated, "
        f"{len(pending)} to do now with {model_tag(generator.provider)}"
    )
    position = 0
    for number, batch in enumerate(batches(pending, batch_size)):
        if number > 0 and sleep_between_batches > 0:
            echo(f"  sleeping {sleep_between_batches:g}s between batches")
            sleep(sleep_between_batches)
        for question in batch:
            position += 1
            prefix = f"[{position}/{len(pending)}] {question.id} ({question.qtype})"
            try:
                record = generator.generate(question)
            except LLMError as exc:
                summary.failed.append((question.id, str(exc)))
                echo(f"  {prefix}: FAILED: {exc}")
                continue
            append_record(record, candidates_path(question.ticker, candidates_dir))
            summary.written += 1
            summary.claims += len(record.claims)
            cached = " (cached)" if record.answer_model.cached else ""
            echo(f"  {prefix}: {len(record.claims)} claim(s){cached}")
    return summary
