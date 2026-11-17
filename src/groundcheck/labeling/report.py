"""`label-stats`, `label-check` and `export-labels`."""

import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from groundcheck.generation.modes import MODES
from groundcheck.generation.pipeline import CandidateRecord
from groundcheck.labeling.schema import EDGE_CASES, LABELS, LabelRecord
from groundcheck.labeling.store import claim_items, write_jsonl_atomic

DEFAULT_TARGET = 300  # claims in the balanced labeled set (docs/labeling_guide.md)
# Unanswerable questions end in an abstention, which has no claims to label.
ANSWERABLE_QTYPES = ("numeric_lookup", "table", "multi_step", "yes_no", "temporal")


def progress_to_target(
    all_records: list[LabelRecord],
    candidates: list[CandidateRecord],
    target: int = DEFAULT_TARGET,
    companies: list[str] | None = None,
) -> str:
    """A one-screen summary: confirmed labels per label, company, mode and question
    type, against an even share of `target` claims, and how many more each needs.

    Only confirmed, valid labels count: those are what `export-labels` writes. An even
    share is `target` divided by the number of values of that dimension (rounded up):
    the three labels, the three generation modes, the answerable question types, and
    `companies` (the labeled companies of config/companies.yaml; without it, the
    companies that occur in the candidates or the labels). Labels cannot be planned (they depend on
    what the model answered), so their "need" says which kind of claim to look for,
    e.g. by labeling more no-context or wrong-context candidates when contradicted and
    unsupported are short."""
    done = [r for r in all_records if r.confirmed and not r.invalid]
    unconfirmed = sum(1 for r in all_records if not r.confirmed and not r.invalid)
    mode_of = {c.candidate_id: c.mode for c in candidates}
    available: dict[str, Counter] = {name: Counter() for name in ("company", "mode", "qtype")}
    labeled_ids = {r.claim_id for r in all_records}
    for c in candidates:
        left = sum(claim.claim_id not in labeled_ids for claim in c.claims)
        available["company"][c.ticker] += left
        available["mode"][c.mode] += left
        available["qtype"][c.qtype] += left
    dimensions = [
        ("Label", "label", lambda r: r.label, list(LABELS)),
        ("Company", "company", lambda r: r.ticker, companies),
        ("Mode", "mode", lambda r: mode_of.get(r.candidate_id, "(unknown)"), list(MODES)),
        ("Question type", "qtype", lambda r: r.qtype, list(ANSWERABLE_QTYPES)),
    ]
    n = len(done)
    lines = [
        f"PROGRESS TO TARGET: {n} of {target} confirmed claims ({min(n / target, 1):.0%}); "
        f"{max(target - n, 0)} to go"
        + (f"; {unconfirmed} labeled but not confirmed (not counted)" if unconfirmed else ""),
    ]
    short: list[str] = []
    for title, key, value_of, fixed in dimensions:
        have = Counter(value_of(r) for r in done)
        seen = sorted(set(have) | {v for v, n in available.get(key, {}).items() if n})
        values = [*fixed, *(v for v in seen if v not in fixed)] if fixed else seen
        if not values:
            continue
        share = -(-target // len(values))  # ceiling
        lines.append(f"{title} (even share: {share} each)")
        lines.append(
            f"  {'':<16} {'have':>5} {'need':>5}" + ("  unlabeled" if key != "label" else "")
        )
        for value in values:
            need = max(share - have[value], 0)
            row = f"  {value[:16]:<16} {have[value]:>5} {need:>5}"
            if key != "label":
                left = available[key][value]
                row += f"  {left:>9}"
                if need > left:
                    short.append(f"{value} (needs {need}, {left} unlabeled)")
            lines.append(row)
    if n >= target:
        lines.append("Target size reached; the need columns show what is still uneven.")
    if short:
        lines.append("Too few unlabeled claims for an even share of: " + "; ".join(short) + ".")
        lines.append("  Generate more candidates for those (scripts/generate_candidates.py).")
    if n < target:
        plan = f"groundcheck label-plan --claims {target - n}"
        lines.append(f"Next: {plan} && groundcheck label --plan data/labels/label_plan.json")
    else:
        lines.append("Next: groundcheck label-check && groundcheck export-labels")
    return "\n".join(lines)


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
    lines += assistance_stats(records)
    return "\n".join(lines)


def _shown(r: LabelRecord) -> bool:
    """A label suggestion was shown for this claim at any time (not only the last pass)."""
    return bool(r.suggestions_seen)


def assistance_stats(records: list[LabelRecord]) -> list[str]:
    """Agreement with label suggestions, and labels made with vs without them."""
    with_s = [r for r in records if _shown(r)]
    without = [r for r in records if not _shown(r)]
    lines = ["Label suggestions (--suggest):"]
    if with_s:
        agreed = sum(bool(r.suggestions_seen[-1].agreed) for r in with_s)
        lines.append(
            f"  shown for {len(with_s)} label(s); you agreed with {agreed} "
            f"({agreed / len(with_s):.0%})"
        )
        kept = sum(any(a.kept_existing for a in r.suggestions_seen) for r in with_s)
        if kept:
            lines.append(f"  of those, {kept} label(s) were kept with Enter after a suggestion")
        by_verifier = Counter(v for r in with_s for v in r.suggestion_verifiers)
        lines.append(
            "  by verifier: " + ", ".join(f"{k} {v}" for k, v in sorted(by_verifier.items()))
        )
    else:
        lines.append("  none shown")
    lines.append(
        f"  label distribution, with suggestion (n={len(with_s)}) vs without (n={len(without)}):"
    )
    for label in LABELS:
        a = sum(r.label == label for r in with_s)
        b = sum(r.label == label for r in without)
        pa = f"{a / len(with_s):.0%}" if with_s else "n/a"
        pb = f"{b / len(without):.0%}" if without else "n/a"
        lines.append(f"  {label:<20} {a:>5} ({pa:>4})   {b:>5} ({pb:>4})")
    suggested = [r for r in records if r.assistance and r.assistance.evidence_suggested]
    if suggested:
        accepted = sum(r.assistance.evidence_accepted for r in suggested)
        lines.append(
            f"Evidence suggestions: offered for {len(suggested)} label(s), accepted "
            f"{accepted} ({accepted / len(suggested):.0%})"
        )
    return lines


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


def evidence_source(record: LabelRecord) -> str:
    """How the gold evidence was chosen: `suggestion_accepted` (the one-key suggestion,
    picked by the same lexical rule the cross-encoder verifiers cite spans with),
    `hand_pasted`, or `none` (unsupported labels have no evidence)."""
    if not record.evidence:
        return "none"
    accepted = bool(record.assistance and record.assistance.evidence_accepted)
    return "suggestion_accepted" if accepted else "hand_pasted"


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
        # What the annotator was shown (labeling/assist.py): eval-verifiers breaks its
        # results down by these, because a shown label can anchor the annotator towards
        # the suggesting verifier and an accepted evidence suggestion makes span
        # agreement circular for verifiers that pick spans the same way.
        "assistance": record.assistance.model_dump() if record.assistance else None,
        "suggestion_history": [a.model_dump() for a in record.suggestion_history],
        "label_suggestion_shown": bool(record.suggestions_seen),
        "suggestion_verifiers": record.suggestion_verifiers,
        "evidence_source": evidence_source(record),
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
