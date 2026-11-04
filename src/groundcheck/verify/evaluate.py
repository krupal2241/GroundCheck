"""`groundcheck eval-verifiers`: run verifiers over the confirmed gold labels and score them.

Input: `data/eval/gold_labels.jsonl` from `export-labels` (confirmed labels only, each
with its retrieved passages). Every claim is checked against exactly those passages.

Per verifier the report has: accuracy and macro-F1 over the three labels, per-class
precision / recall / F1, the confusion matrix (rows = gold, columns = predicted), a
binary view (supported vs not supported, where binary verifiers compete fairly),
breakdowns by edge-case tag and by company, latency p50 / p95 per check, throughput,
evidence agreement with the gold span, and cost per 1,000 checks ($0 locally; hosted
models only as an ESTIMATE from config/prices.yaml).

Outputs, under `results/` (committed when you want the numbers in the README):

    results/verifier_eval.json            every metric, one entry per verifier
    results/verifier_eval.md              README-ready tables
    results/predictions/<verifier>.jsonl  one prediction per claim (input to error-report)

Re-running with a subset of verifiers updates their entries and keeps the others, so
you can run one model at a time on a Mac with 8 GB RAM. Each entry records the gold
file's sha256; the Markdown flags entries computed on a different gold file.
"""

import datetime as dt
import gc
import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from groundcheck import __version__
from groundcheck.config import REPO_ROOT, ConfigError
from groundcheck.labeling.schema import LABELS, EdgeCase, EvidenceSpan, Label
from groundcheck.labeling.store import write_jsonl_atomic
from groundcheck.verify.base import EvidenceChunk, Verifier
from groundcheck.verify.costs import CostEstimate, Prices, approx_tokens, estimate_costs

DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
NO_TAG = "(none)"
BINARY_VERIFIERS = ("minicheck", "hhem")


# --- Gold records -----------------------------------------------------------------------


class GoldPassage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rank: int
    chunk_id: str
    text: str


class GoldRecord(BaseModel):
    """One line of gold_labels.jsonl (the fields the evaluation uses)."""

    model_config = ConfigDict(extra="ignore")

    claim_id: str
    ticker: str
    qtype: str = ""
    question: str = ""
    claim: str
    label: Label
    evidence: list[EvidenceSpan] = []
    edge_case: EdgeCase | None = None
    note: str = ""
    passages: list[GoldPassage]

    @property
    def chunks(self) -> list[EvidenceChunk]:
        ordered = sorted(self.passages, key=lambda p: p.rank)
        return [EvidenceChunk(p.chunk_id, p.text) for p in ordered]

    @property
    def tag(self) -> str:
        return self.edge_case or NO_TAG


def load_gold(path: Path) -> list[GoldRecord]:
    if not Path(path).is_file():
        raise ConfigError(
            f"no gold labels at {path}; run `groundcheck export-labels` after labeling"
        )
    records, problems = [], []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(GoldRecord.model_validate_json(line))
        except ValidationError as exc:
            problems.append(f"line {number}: {exc.errors()[0]['msg']}")
    if problems:
        raise ConfigError(f"{path} has invalid lines:\n" + "\n".join(problems[:10]))
    if not records:
        raise ConfigError(f"{path} has no gold labels")
    return records


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- Running a verifier -----------------------------------------------------------------


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    ticker: str
    edge_case: str
    claim: str
    gold: Label
    predicted: Label
    score: float
    evidence: EvidenceSpan | None
    latency_ms: float
    parse_error: bool = False
    details: dict[str, Any] = {}


def run_verifier(
    verifier: Verifier,
    gold: list[GoldRecord],
    echo: Callable[[str], None] = print,
    progress_every: int = 25,
) -> tuple[list[Prediction], float]:
    """Verify every gold claim against its passages. Returns predictions and wall seconds."""
    predictions = []
    started = time.perf_counter()
    for number, record in enumerate(gold, start=1):
        result = verifier.verify(record.claim, record.chunks)
        predictions.append(
            Prediction(
                claim_id=record.claim_id,
                ticker=record.ticker,
                edge_case=record.tag,
                claim=record.claim,
                gold=record.label,
                predicted=result.label,
                score=result.score,
                evidence=result.evidence,
                latency_ms=result.latency_ms,
                parse_error="parse_error" in result.details,
                details=_small(result.details),
            )
        )
        if progress_every and number % progress_every == 0:
            echo(f"  {verifier.name}: {number}/{len(gold)}")
    return predictions, time.perf_counter() - started


def _small(details: dict[str, Any]) -> dict[str, Any]:
    """Keep predictions files readable: drop long raw text, keep scalars and short maps."""
    keep = {}
    for key, value in details.items():
        if isinstance(value, str) and len(value) > 300:
            value = value[:300] + "..."
        keep[key] = value
    return keep


# --- Metrics ----------------------------------------------------------------------------


class ClassMetrics(BaseModel):
    precision: float | None  # None: the verifier never predicted this label
    recall: float | None  # None: no gold claim has this label
    f1: float
    support: int  # gold count
    predicted: int


class GroupMetrics(BaseModel):
    n: int
    correct: int
    accuracy: float


class BinaryMetrics(BaseModel):
    """Supported vs not supported (contradicted and unsupported merged)."""

    accuracy: float
    precision: float | None
    recall: float | None
    f1: float


class LatencyMetrics(BaseModel):
    p50_ms: float
    p95_ms: float
    mean_ms: float
    wall_seconds: float
    throughput_per_s: float


class EvidenceAgreement(BaseModel):
    """Correct supported/contradicted verdicts whose cited span overlaps a gold span."""

    eligible: int
    overlapping: int
    rate: float | None


class VerifierReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verifier: str
    identity: dict[str, Any]
    binary_model: bool
    n: int
    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassMetrics]
    confusion: dict[str, dict[str, int]]  # gold -> predicted -> count
    binary: BinaryMetrics
    by_edge_case: dict[str, GroupMetrics]
    by_company: dict[str, GroupMetrics]
    latency: LatencyMetrics
    evidence_agreement: EvidenceAgreement
    parse_errors: int
    local: bool
    cost_per_1000_checks_usd: float | None  # 0.0 when local; None when not local
    gold_file: str
    gold_sha256: str
    generated_at: str
    groundcheck_version: str = __version__


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _f1(p: float | None, r: float | None) -> float:
    return 2 * p * r / (p + r) if p and r else 0.0


def class_metrics(gold: list[str], pred: list[str], label: str) -> ClassMetrics:
    tp = sum(g == p == label for g, p in zip(gold, pred, strict=True))
    support, predicted = gold.count(label), pred.count(label)
    precision, recall = _ratio(tp, predicted), _ratio(tp, support)
    return ClassMetrics(
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        support=support,
        predicted=predicted,
    )


def macro_f1(per_class: dict[str, ClassMetrics]) -> float:
    """Mean F1 over the labels present in gold or predictions (sklearn's default set).
    A label that is in gold but never predicted contributes F1 = 0."""
    present = [m.f1 for m in per_class.values() if m.support or m.predicted]
    return sum(present) / len(present) if present else 0.0


def group_metrics(predictions: list[Prediction], key: Callable[[Prediction], str]):
    groups: dict[str, list[Prediction]] = defaultdict(list)
    for p in predictions:
        groups[key(p)].append(p)
    out = {}
    for name in sorted(groups):
        items = groups[name]
        correct = sum(p.gold == p.predicted for p in items)
        out[name] = GroupMetrics(n=len(items), correct=correct, accuracy=correct / len(items))
    return out


def binary_metrics(gold: list[str], pred: list[str]) -> BinaryMetrics:
    g = [x == "supported" for x in gold]
    p = [x == "supported" for x in pred]
    tp = sum(a and b for a, b in zip(g, p, strict=True))
    precision, recall = _ratio(tp, sum(p)), _ratio(tp, sum(g))
    accuracy = sum(a == b for a, b in zip(g, p, strict=True)) / len(g)
    return BinaryMetrics(
        accuracy=accuracy, precision=precision, recall=recall, f1=_f1(precision, recall)
    )


def latency_metrics(latencies_ms: list[float], wall_seconds: float) -> LatencyMetrics:
    values = np.asarray(latencies_ms, dtype=float)
    return LatencyMetrics(
        p50_ms=round(float(np.percentile(values, 50)), 3),
        p95_ms=round(float(np.percentile(values, 95)), 3),
        mean_ms=round(float(values.mean()), 3),
        wall_seconds=round(wall_seconds, 3),
        throughput_per_s=round(len(values) / wall_seconds, 3) if wall_seconds > 0 else 0.0,
    )


def spans_overlap(a: EvidenceSpan, b: EvidenceSpan) -> bool:
    return a.chunk_id == b.chunk_id and a.start < b.end and b.start < a.end


def evidence_agreement(predictions: list[Prediction], gold: list[GoldRecord]) -> EvidenceAgreement:
    by_id = {g.claim_id: g for g in gold}
    eligible = overlapping = 0
    for p in predictions:
        record = by_id[p.claim_id]
        if p.predicted != p.gold or p.gold == "unsupported" or not record.evidence:
            continue
        eligible += 1
        if p.evidence and any(spans_overlap(p.evidence, s) for s in record.evidence):
            overlapping += 1
    return EvidenceAgreement(
        eligible=eligible, overlapping=overlapping, rate=_ratio(overlapping, eligible)
    )


def build_report(
    verifier: Verifier,
    predictions: list[Prediction],
    wall_seconds: float,
    gold: list[GoldRecord],
    gold_path: Path,
) -> VerifierReport:
    golds = [p.gold for p in predictions]
    preds = [p.predicted for p in predictions]
    per_class = {label: class_metrics(golds, preds, label) for label in LABELS}
    pairs = Counter(zip(golds, preds, strict=True))
    confusion = {g: {p: pairs.get((g, p), 0) for p in LABELS} for g in LABELS}
    local = bool(getattr(verifier, "local", True))
    return VerifierReport(
        verifier=verifier.name,
        identity=verifier.identity(),
        binary_model=verifier.name in BINARY_VERIFIERS,
        n=len(predictions),
        accuracy=sum(g == p for g, p in zip(golds, preds, strict=True)) / len(predictions),
        macro_f1=macro_f1(per_class),
        per_class=per_class,
        confusion=confusion,
        binary=binary_metrics(golds, preds),
        by_edge_case=group_metrics(predictions, lambda p: p.edge_case),
        by_company=group_metrics(predictions, lambda p: p.ticker),
        latency=latency_metrics([p.latency_ms for p in predictions], wall_seconds),
        evidence_agreement=evidence_agreement(predictions, gold),
        parse_errors=sum(p.parse_error for p in predictions),
        local=local,
        cost_per_1000_checks_usd=0.0 if local else None,
        gold_file=str(gold_path),
        gold_sha256=file_sha256(gold_path),
        generated_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    )


# --- The whole run, and the files it writes --------------------------------------------


class EvalResults(BaseModel):
    """The contents of results/verifier_eval.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    verifiers: dict[str, VerifierReport] = {}
    hosted_cost_estimates: list[CostEstimate] = []
    hosted_cost_note: str = ""


def prompt_tokens_per_check(gold: list[GoldRecord]) -> float:
    """Approximate input tokens a prompted judge would read per claim (see costs.py)."""
    from groundcheck.verify.prompted import VERIFY_SYSTEM, verify_prompt

    counts = [approx_tokens(VERIFY_SYSTEM + verify_prompt(g.claim, g.chunks)) for g in gold]
    return sum(counts) / len(counts)


def measured_output_tokens(predictions: list[Prediction]) -> float | None:
    counts = [
        p.details["completion_tokens"]
        for p in predictions
        if isinstance(p.details.get("completion_tokens"), int | float)
    ]
    return sum(counts) / len(counts) if counts else None


def predictions_path(results_dir: Path, name: str) -> Path:
    return Path(results_dir) / "predictions" / f"{name}.jsonl"


def load_results(path: Path) -> EvalResults:
    if not path.is_file():
        return EvalResults()
    try:
        return EvalResults.model_validate_json(path.read_text())
    except ValidationError as exc:
        raise ConfigError(
            f"{path} is not a valid results file; move it away first:\n{exc}"
        ) from exc


def evaluate_verifiers(
    names: list[str],
    build: Callable[[str], Verifier],
    gold: list[GoldRecord],
    gold_path: Path,
    results_dir: Path,
    prices: Prices,
    echo: Callable[[str], None] = print,
) -> EvalResults:
    """Run each verifier in turn (built, run, freed), write every output, return results."""
    results_dir = Path(results_dir)
    json_path = results_dir / "verifier_eval.json"
    results = load_results(json_path)
    output_tokens = None
    for name in names:
        echo(f"{name}: building")
        verifier = build(name)
        echo(f"{name}: verifying {len(gold)} claim(s)")
        predictions, wall = run_verifier(verifier, gold, echo)
        report = build_report(verifier, predictions, wall, gold, gold_path)
        results.verifiers[name] = report
        write_jsonl_atomic(
            [json.dumps(p.model_dump(), ensure_ascii=False) for p in predictions],
            predictions_path(results_dir, name),
        )
        if name == "prompted":
            output_tokens = measured_output_tokens(predictions)
        echo(
            f"{name}: accuracy {report.accuracy:.3f}, macro-F1 {report.macro_f1:.3f}, "
            f"p50 {report.latency.p50_ms:.1f} ms"
        )
        del verifier
        gc.collect()  # one model in memory at a time (8 GB Mac)
    if prices.models:
        results.hosted_cost_estimates = estimate_costs(
            prices, prompt_tokens_per_check(gold), output_tokens
        )
        results.hosted_cost_note = (
            f"ESTIMATE ONLY: computed from config/prices.yaml over {len(gold)} gold claim(s); "
            "no hosted model was called."
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(results.model_dump_json(indent=2) + "\n")
    (results_dir / "verifier_eval.md").write_text(format_markdown(results))
    return results


# --- Markdown -----------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _ms(value: float) -> str:
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}"


def format_markdown(results: EvalResults) -> str:
    reports = list(results.verifiers.values())
    lines = ["# Verifier benchmark", ""]
    if not reports:
        return "\n".join([*lines, "No verifier has been evaluated yet.", ""])
    shas = {r.gold_sha256 for r in reports}
    first = reports[0]
    lines += [
        f"Gold set: `{Path(first.gold_file).name}`, {first.n} confirmed claim(s) "
        f"(sha256 `{first.gold_sha256[:12]}`).",
        "",
    ]
    if len(shas) > 1:
        lines += [
            "> **Warning:** these rows were computed on different gold files "
            "(see `gold_sha256` in verifier_eval.json). Re-run all verifiers before "
            "comparing them.",
            "",
        ]
    lines += [
        "| Verifier | Model | Accuracy | Macro-F1 | Binary acc. | Latency p50 | Latency p95 "
        "| Throughput | Cost / 1,000 checks |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        cost = "$0 (local)" if r.local else "n/a"
        lines.append(
            f"| {r.verifier} | `{r.identity.get('model', '')}` | {_pct(r.accuracy)} | "
            f"{r.macro_f1:.3f} | {_pct(r.binary.accuracy)} | {_ms(r.latency.p50_ms)} ms | "
            f"{_ms(r.latency.p95_ms)} ms | {r.latency.throughput_per_s:.2f} checks/s | {cost} |"
        )
    lines += [
        "",
        "Accuracy and macro-F1 are over the three labels. *Binary acc.* merges contradicted "
        'and unsupported into "not supported". MiniCheck and HHEM are binary models: '
        "they never predict contradicted (see docs/DESIGN.md), so their three-way scores "
        "count every gold-contradicted claim as an error; compare them on the binary "
        "column. Latency is per claim (all retrieved passages), measured on this machine.",
        "",
    ]
    if results.hosted_cost_estimates:
        lines += [
            "### Hosted-model cost: ESTIMATE",
            "",
            f"> {results.hosted_cost_note} Tokens are approximated as characters / 4.",
            "",
            "| Hosted model (ESTIMATE) | Input tokens / check | Output tokens / check "
            "| Estimated cost / 1,000 checks |",
            "|---|---|---|---|",
        ]
        for e in results.hosted_cost_estimates:
            lines.append(
                f"| {e.model} | {e.input_tokens_per_check:,.0f} | "
                f"{e.output_tokens_per_check:,.0f} ({e.output_tokens_source}) | "
                f"~{e.cost_per_1000_checks:,.4f} {e.currency} (estimate) |"
            )
        lines.append("")
    for r in reports:
        lines += _verifier_section(r)
    return "\n".join(lines)


def _verifier_section(r: VerifierReport) -> list[str]:
    lines = [f"## {r.verifier}", "", f"Model: `{r.identity.get('model', '')}`. "]
    extras = {k: v for k, v in r.identity.items() if k not in {"verifier", "model"} and v}
    if extras:
        lines[-1] += "; ".join(f"{k}: `{v}`" for k, v in extras.items()) + "."
    if r.binary_model:
        lines[-1] += " Binary model: never predicts contradicted."
    if r.parse_errors:
        lines[-1] += f" {r.parse_errors} claim(s) had no parseable JSON (counted unsupported)."
    ev = r.evidence_agreement
    lines += [
        "",
        f"Evidence agreement: {ev.overlapping} of {ev.eligible} correct supported/contradicted "
        f"verdicts cite a span overlapping the gold span ({_pct(ev.rate)}).",
        "",
        "| Label | Precision | Recall | F1 | Gold | Predicted |",
        "|---|---|---|---|---|---|",
    ]
    for label, m in r.per_class.items():
        lines.append(
            f"| {label} | {_pct(m.precision)} | {_pct(m.recall)} | {m.f1:.3f} | "
            f"{m.support} | {m.predicted} |"
        )
    lines += ["", "Confusion matrix (rows: gold, columns: predicted):", ""]
    lines.append("| gold \\ predicted | " + " | ".join(LABELS) + " |")
    lines.append("|---" * (len(LABELS) + 1) + "|")
    for g in LABELS:
        lines.append(f"| {g} | " + " | ".join(str(r.confusion[g][p]) for p in LABELS) + " |")
    for title, groups in (("edge-case tag", r.by_edge_case), ("company", r.by_company)):
        lines += ["", f"By {title}:", "", f"| {title.capitalize()} | n | Accuracy |"]
        lines.append("|---|---|---|")
        lines += [f"| {k} | {g.n} | {_pct(g.accuracy)} |" for k, g in groups.items()]
    lines.append("")
    return lines
