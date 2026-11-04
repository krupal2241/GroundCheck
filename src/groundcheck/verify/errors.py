"""`groundcheck error-report`: every misclassified claim, per verifier, by edge-case tag.

It reads what `eval-verifiers` wrote (`results/predictions/<verifier>.jsonl`) and the
gold file, so no model runs again. For each verifier: a summary table of errors per
edge-case tag, the most common confusions, then each misclassified claim with the
claim text, the gold label and gold evidence span(s), the predicted label, score and
cited span, and the labeler's note. Written as Markdown (`results/error_report.md`),
meant for reading, not for parsing.
"""

from collections import Counter, defaultdict
from pathlib import Path

from pydantic import ValidationError

from groundcheck.config import ConfigError
from groundcheck.verify.evaluate import (
    NO_TAG,
    GoldRecord,
    Prediction,
    file_sha256,
    load_results,
)


def load_predictions(path: Path) -> list[Prediction]:
    predictions = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            predictions.append(Prediction.model_validate_json(line))
        except ValidationError as exc:
            raise ConfigError(f"{path}:{number}: not a prediction record: {exc}") from exc
    return predictions


def available_verifiers(results_dir: Path) -> list[str]:
    return sorted(p.stem for p in (Path(results_dir) / "predictions").glob("*.jsonl"))


def _quote(text: str) -> str:
    """Inline code that survives backticks and pipes in table rows."""
    return "`` " + text.replace("`", "'") + " ``"


def _tag_order(tag: str) -> tuple[int, str]:
    return (1 if tag == NO_TAG else 0, tag)


def verifier_section(name: str, predictions: list[Prediction], gold: dict[str, GoldRecord]):
    errors = [p for p in predictions if p.predicted != p.gold]
    lines = [
        f"## {name}",
        "",
        f"{len(errors)} of {len(predictions)} claim(s) misclassified "
        f"({100 * len(errors) / max(len(predictions), 1):.1f}%).",
        "",
        "| Edge-case tag | Claims | Errors | Error rate |",
        "|---|---|---|---|",
    ]
    by_tag_all = Counter(p.edge_case for p in predictions)
    by_tag_err = Counter(p.edge_case for p in errors)
    for tag in sorted(by_tag_all, key=_tag_order):
        n, e = by_tag_all[tag], by_tag_err.get(tag, 0)
        lines.append(f"| {tag} | {n} | {e} | {100 * e / n:.1f}% |")
    confusions = Counter((p.gold, p.predicted) for p in errors)
    if confusions:
        common = ", ".join(f"{g} -> {p}: {n}" for (g, p), n in confusions.most_common())
        lines += ["", f"Confusions (gold -> predicted): {common}."]
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for p in errors:
        grouped[p.edge_case].append(p)
    for tag in sorted(grouped, key=_tag_order):
        lines += ["", f"### {name} / {tag} ({len(grouped[tag])} error(s))", ""]
        for p in sorted(grouped[tag], key=lambda p: p.claim_id):
            lines += claim_block(p, gold.get(p.claim_id))
    lines.append("")
    return lines


def claim_block(p: Prediction, record: GoldRecord | None) -> list[str]:
    lines = [
        f"#### `{p.claim_id}`: gold **{p.gold}**, predicted **{p.predicted}** "
        f"(score {p.score:.3f})",
        "",
        f"- Claim: {p.claim}",
    ]
    if record is None:
        lines.append("- Gold evidence: (claim not in the current gold file)")
    elif record.evidence:
        for span in record.evidence:
            lines.append(f"- Gold evidence ({span.chunk_id}): {_quote(span.text)}")
    else:
        lines.append("- Gold evidence: none (unsupported: the passages don't address it)")
    if p.evidence is not None:
        lines.append(f"- Predicted evidence ({p.evidence.chunk_id}): {_quote(p.evidence.text)}")
    else:
        lines.append("- Predicted evidence: none cited")
    if p.parse_error:
        lines.append("- The prompted verifier returned no parseable JSON for this claim.")
    if p.details.get("quote_found") is False:
        quote = _quote(p.details.get("quote", ""))
        lines.append(f"- The LLM quoted text not found in any passage: {quote}")
    if record is not None and record.note:
        lines.append(f"- Labeler's note: {record.note}")
    if record is not None:
        lines.append(f"- Company: {record.ticker}; question type: {record.qtype or 'n/a'}")
    lines.append("")
    return lines


def error_report(
    results_dir: Path, gold: list[GoldRecord], gold_path: Path, verifiers: list[str] | None
) -> str:
    names = verifiers or available_verifiers(results_dir)
    if not names:
        raise ConfigError(
            f"no predictions in {Path(results_dir) / 'predictions'}; run eval-verifiers first"
        )
    by_id = {g.claim_id: g for g in gold}
    sha = file_sha256(gold_path)
    evaluated = load_results(Path(results_dir) / "verifier_eval.json").verifiers
    lines = ["# Verifier error report", ""]
    lines += [
        f"Misclassified claims per verifier, grouped by edge-case tag. Gold file: "
        f"`{Path(gold_path).name}` (sha256 `{sha[:12]}`).",
        "",
    ]
    for name in names:
        path = Path(results_dir) / "predictions" / f"{name}.jsonl"
        if not path.is_file():
            raise ConfigError(f"no predictions for {name!r} at {path}; run eval-verifiers")
        report = evaluated.get(name)
        if report is not None and report.gold_sha256 != sha:
            lines += [
                f"> **Warning:** {name}'s predictions were made on a different gold file; "
                "re-run eval-verifiers.",
                "",
            ]
        lines += verifier_section(name, load_predictions(path), by_id)
    return "\n".join(lines)
