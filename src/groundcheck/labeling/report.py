"""`label-stats`, `label-check` and `export-labels`."""

import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from groundcheck.generation.pipeline import CandidateRecord
from groundcheck.labeling.schema import EDGE_CASES, LABELS, LabelRecord
from groundcheck.labeling.store import claim_items, write_jsonl_atomic


def _counter_table(title: str, counter: Counter, order: list[str] | None = None) -> list[str]:
    keys = order if order is not None else sorted(counter)
    lines = [f"{title}:"]
    lines += [f"  {key:<20} {counter.get(key, 0):>5}" for key in keys]
    return lines


def label_stats(all_records: list[LabelRecord], candidates: list[CandidateRecord]) -> str:
    total_claims = len(claim_items(candidates))
    seen = {r.claim_id for r in all_records}
    invalid = [r for r in all_records if r.invalid]
    records = [r for r in all_records if not r.invalid]
    confirmed = sum(r.confirmed for r in records)
    unlabeled = sum(1 for i in claim_items(candidates) if i.claim.claim_id not in seen)
    lines = [
        f"Labels: {len(records)} ({confirmed} confirmed, {len(records) - confirmed} not "
        f"confirmed); claims in candidates: {total_claims}, unlabeled: {unlabeled}",
        f"Invalid claims (marked with x; not labeled, never exported): {len(invalid)}",
    ]
    mode_of = {c.candidate_id: c.mode for c in candidates}

    def mode(r: LabelRecord) -> str:
        return mode_of.get(r.candidate_id, "(unknown)")

    lines += _counter_table("By label", Counter(r.label for r in records), list(LABELS))
    by_label_confirmed = Counter(r.label for r in records if r.confirmed)
    lines += _counter_table("By label (confirmed only)", by_label_confirmed, list(LABELS))
    lines += _counter_table("By company", Counter(r.ticker for r in records))
    lines += _counter_table("By question type", Counter(r.qtype for r in records))
    edge = Counter(r.edge_case or "(none)" for r in records)
    lines += _counter_table("By edge case", edge, ["(none)", *EDGE_CASES])
    lines += _counter_table("By mode", Counter(mode(r) for r in records))
    lines += _cross_table("Mode x label", records, mode, lambda r: r.label, list(LABELS))
    lines += _cross_table("Company x label", records, lambda r: r.ticker, lambda r: r.label,
                          list(LABELS))  # fmt: skip
    lines += _cross_table("Question type x label", records, lambda r: r.qtype,
                          lambda r: r.label, list(LABELS))  # fmt: skip
    return "\n".join(lines)


def _cross_table(title, records, row_of, col_of, columns: list[str]) -> list[str]:
    counter = Counter((row_of(r), col_of(r)) for r in records)
    rows = sorted({row_of(r) for r in records})
    header = f"  {'':<20}" + "".join(f" {c[:12]:>12}" for c in columns) + f" {'total':>7}"
    lines = [f"{title}:", header]
    for row in rows:
        cells = [counter[(row, c)] for c in columns]
        lines.append(f"  {row:<20}" + "".join(f" {n:>12}" for n in cells) + f" {sum(cells):>7}")
    return lines


def check_labels(
    path: Path, candidates: list[CandidateRecord]
) -> tuple[list[LabelRecord], list[str]]:
    """Validate every line of the label file against the schema and the candidates.

    Returns the valid records and a list of problems ("line N (claim id): ...").
    """
    items = {i.claim.claim_id: i for i in claim_items(candidates)}
    records: list[LabelRecord] = []
    problems: list[str] = []
    seen: set[str] = set()
    if not path.is_file():
        return records, [f"{path} not found"]
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {number}: invalid JSON ({exc.msg})")
            continue
        claim_id = data.get("claim_id", "?") if isinstance(data, dict) else "?"
        where = f"line {number} ({claim_id})"
        item = items.get(claim_id)
        context = {"chunks": item.chunks} if item else None
        try:
            record = LabelRecord.model_validate(data, context=context)
        except ValidationError as exc:
            for error in exc.errors():
                loc = ".".join(str(p) for p in error["loc"])
                problems.append(f"{where}: {loc + ': ' if loc else ''}{error['msg']}")
            continue
        if item is None:
            problems.append(f"{where}: no such claim in the candidates")
            continue
        if record.claim != item.claim.text:
            problems.append(f"{where}: claim text differs from the candidate's claim")
        if record.candidate_id != item.candidate.candidate_id:
            problems.append(f"{where}: candidate_id differs from the candidate's")
        if claim_id in seen:
            problems.append(f"{where}: duplicate claim_id")
        seen.add(claim_id)
        records.append(record)
    return records, problems


def export_record(record: LabelRecord, candidate: CandidateRecord) -> dict:
    return {
        "claim_id": record.claim_id,
        "candidate_id": record.candidate_id,
        "question_id": record.question_id,
        "ticker": record.ticker,
        "qtype": record.qtype,
        "mode": candidate.mode,
        "context_source": candidate.context_source,
        "question": candidate.question,
        "answer": candidate.answer,
        "claim": record.claim,
        "label": record.label,
        "evidence": [s.model_dump() for s in record.evidence],
        "edge_case": record.edge_case,
        "note": record.note,
        "passages": [
            {"rank": c.rank, "chunk_id": c.chunk_id, "text": c.text} for c in candidate.retrieved
        ],
        "generator": {
            "provider": candidate.answer_model.provider,
            "model": candidate.answer_model.model,
            "model_digest": candidate.answer_model.model_digest,
            "prompt_version": candidate.prompt_version,
        },
    }


def export_labels(records: list[LabelRecord], candidates: list[CandidateRecord], out: Path) -> int:
    """Write confirmed labels, sorted by claim ID, with their passages. Returns the count."""
    by_id = {c.candidate_id: c for c in candidates}
    confirmed = sorted(
        (r for r in records if r.confirmed and not r.invalid), key=lambda r: r.claim_id
    )
    lines = [
        json.dumps(export_record(r, by_id[r.candidate_id]), ensure_ascii=False) for r in confirmed
    ]
    write_jsonl_atomic(lines, out)
    return len(lines)
