"""Reading and writing `data/labels/labels.jsonl`, and loading the candidates it labels.

The label file holds one `LabelRecord` per line, at most one per claim. Every save
rewrites the whole file atomically (write `labels.jsonl.part`, fsync, rename), so an
interrupted session leaves either the old file or the new one, never a torn one. The
file is small (hundreds of lines), so rewriting it on every label costs nothing.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from groundcheck.config import REPO_ROOT, ConfigError
from groundcheck.generation.pipeline import (
    DEFAULT_CANDIDATES_DIR,
    CandidateRecord,
    Claim,
    read_candidates,
)
from groundcheck.labeling.schema import LabelRecord

DEFAULT_LABELS_PATH = REPO_ROOT / "data" / "labels" / "labels.jsonl"
DEFAULT_EXPORT_PATH = REPO_ROOT / "data" / "eval" / "gold_labels.jsonl"


@dataclass
class ClaimItem:
    """One claim to label, with the candidate (question, answer, chunks) it came from."""

    candidate: CandidateRecord
    claim: Claim

    @property
    def chunks(self) -> dict[str, str]:
        return {c.chunk_id: c.text for c in self.candidate.retrieved}


def load_candidates(
    candidates_dir: Path = DEFAULT_CANDIDATES_DIR, tickers: list[str] | None = None
) -> list[CandidateRecord]:
    paths = sorted(Path(candidates_dir).glob("*.jsonl"))
    records = [r for p in paths for r in read_candidates(p)]
    if tickers:
        records = [r for r in records if r.ticker in tickers]
    return records


def claim_items(candidates: list[CandidateRecord]) -> list[ClaimItem]:
    return [ClaimItem(c, claim) for c in candidates for claim in c.claims]


def write_jsonl_atomic(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with tmp.open("w", encoding="utf-8") as f:
        f.write("".join(line + "\n" for line in lines))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def record_line(record: LabelRecord) -> str:
    return json.dumps(record.model_dump(), ensure_ascii=False)


class LabelStore:
    """Labels keyed by claim ID, in file order (new labels go at the end)."""

    def __init__(self, path: Path = DEFAULT_LABELS_PATH):
        self.path = Path(path)
        self.records: dict[str, LabelRecord] = {}
        if self.path.is_file():
            problems = []
            for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = LabelRecord.model_validate_json(line)
                except ValidationError as exc:
                    problems.append(f"line {number}: {exc.errors()[0]['msg']}")
                    continue
                if record.claim_id in self.records:
                    problems.append(f"line {number}: duplicate claim_id {record.claim_id}")
                self.records[record.claim_id] = record
            if problems:
                # Saving would silently drop the bad lines, so refuse to go on.
                raise ConfigError(
                    f"{self.path} has invalid lines; fix them first (`groundcheck label-check`):\n"
                    + "\n".join(problems[:10])
                )

    def get(self, claim_id: str) -> LabelRecord | None:
        return self.records.get(claim_id)

    def put(self, record: LabelRecord) -> None:
        self.records[record.claim_id] = record
        self.save()

    def save(self) -> None:
        write_jsonl_atomic([record_line(r) for r in self.records.values()], self.path)
