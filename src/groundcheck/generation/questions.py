"""Questions for candidate generation, from two files, so each question is written once:

- `data/questions/candidate_questions.yaml`: questions written only for generation,
  ten typed slots per labeled company (created as a template if missing);
- `data/eval/retrieval_questions.yaml`: every retrieval-eval entry whose question text
  is filled in (gold evidence isn't needed for generation).

Only companies with `role: labeled` are used: BRK-B and JPM can never enter the
labeled set by accident. IDs must be unique across both files. If the same question
text (whitespace and case ignored) appears in both files for one company, it is used
once, from the retrieval file.
"""

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from groundcheck.config import REPO_ROOT, CompaniesConfig, ConfigError
from groundcheck.retrieval.evaluate import QType, load_questions

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_QUESTIONS_PATH = REPO_ROOT / "data" / "questions" / "candidate_questions.yaml"

# Ten slots per labeled company: the PLAN.md question mix.
TEMPLATE_SLOTS: tuple[str, ...] = (
    "numeric_lookup",
    "numeric_lookup",
    "table",
    "table",
    "multi_step",
    "multi_step",
    "yes_no",
    "temporal",
    "temporal",
    "unanswerable",
)

Source = Literal["candidate_questions", "retrieval_questions"]


class CandidateQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    ticker: str
    qtype: QType
    question: str = ""
    notes: str = ""


class CandidateQuestionFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    questions: list[CandidateQuestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "CandidateQuestionFile":
        ids = [q.id for q in self.questions]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate question ids: {duplicates}")
        return self


class GenerationQuestion(BaseModel):
    """One question to generate a candidate for, with where it came from."""

    id: str
    ticker: str
    qtype: str
    question: str
    source: Source


TEMPLATE_HEADER = """\
# Candidate-generation questions (template). Fill these in yourself.
#
# Every labeled company has ten typed slots (the PLAN.md mix: 2 numeric_lookup,
# 2 table, 2 multi_step, 1 yes_no, 2 temporal, 1 unanswerable). Write the question
# text; leave `question: ""` to skip a slot. qtype may be changed.
#
# You don't need to repeat questions from data/eval/retrieval_questions.yaml:
# scripts/generate_candidates.py also uses every entry there with a question filled in.
#
# qtype is one of: numeric_lookup, table, multi_step, yes_no, temporal, unanswerable
#   numeric_lookup  "What was total net sales in fiscal 2025?"
#   table           a value that only appears in a financial table
#   multi_step      a difference, ratio or growth rate between two reported values
#   yes_no          "Did the company repurchase shares during the year?"
#   temporal        across years or periods, or fiscal vs calendar year
#   unanswerable    a plausible question the filing does not answer
"""


def template_text(companies: CompaniesConfig) -> str:
    entries = [
        {
            "id": f"{c.ticker}-G{i:02d}",
            "ticker": c.ticker,
            "qtype": qtype,
            "question": "",
            "notes": "",
        }
        for c in companies.companies
        if c.role == "labeled"
        for i, qtype in enumerate(TEMPLATE_SLOTS, start=1)
    ]
    body = yaml.safe_dump(
        {"version": 1, "questions": entries}, sort_keys=False, allow_unicode=True, width=100
    )
    return TEMPLATE_HEADER + "\n" + body


def ensure_template(path: Path, companies: CompaniesConfig) -> bool:
    """Write the template if `path` doesn't exist. Returns True if it was created."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template_text(companies), encoding="utf-8")
    return True


def load_candidate_questions(path: Path) -> CandidateQuestionFile:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"question file not found: {path}") from exc
    try:
        return CandidateQuestionFile.model_validate(raw or {})
    except ValidationError as exc:
        raise ConfigError(f"invalid question file {path}:\n{exc}") from exc


def _text_key(ticker: str, question: str) -> tuple[str, str]:
    return ticker, " ".join(question.casefold().split())


def collect_questions(
    companies: CompaniesConfig,
    candidate_path: Path,
    retrieval_path: Path | None,
    tickers: list[str] | None = None,
) -> list[GenerationQuestion]:
    """All filled-in questions for labeled companies, retrieval file first."""
    labeled = {c.ticker for c in companies.companies if c.role == "labeled"}
    if tickers:
        for ticker in tickers:
            if companies.get(ticker).role != "labeled":
                raise ConfigError(f"{ticker} is not a labeled company (see config/companies.yaml)")
    entries: list[GenerationQuestion] = []
    if retrieval_path is not None and Path(retrieval_path).is_file():
        for q in load_questions(retrieval_path).questions:
            entries.append(
                GenerationQuestion(
                    id=q.id,
                    ticker=q.ticker,
                    qtype=q.qtype,
                    question=q.question.strip(),
                    source="retrieval_questions",
                )
            )
    for q in load_candidate_questions(candidate_path).questions:
        entries.append(
            GenerationQuestion(
                id=q.id,
                ticker=q.ticker,
                qtype=q.qtype,
                question=q.question.strip(),
                source="candidate_questions",
            )
        )
    seen_ids: dict[str, Source] = {}
    seen_text: set[tuple[str, str]] = set()
    out = []
    for entry in entries:
        if entry.id in seen_ids:
            raise ConfigError(
                f"question id {entry.id!r} is used in both {seen_ids[entry.id]} and {entry.source}"
            )
        seen_ids[entry.id] = entry.source
        if not entry.question:
            continue
        if entry.ticker not in labeled:
            logger.warning("%s: %s is not a labeled company; skipped", entry.id, entry.ticker)
            continue
        if tickers and entry.ticker not in tickers:
            continue
        key = _text_key(entry.ticker, entry.question)
        if key in seen_text:
            logger.warning("%s: same question text as an earlier entry; skipped", entry.id)
            continue
        seen_text.add(key)
        out.append(entry)
    return out
