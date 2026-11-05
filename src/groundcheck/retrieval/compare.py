"""`groundcheck eval-retrieval --compare`: one table over every configured variant.

Per variant: recall@k, section-hit@k, MRR and mean query latency over the same
questions, plus recall@5 by question type and by company (the weak spots: multi_step,
table, AMZN). Written as JSON (machine-readable, every number) and Markdown (pasteable
into docs/DESIGN.md) under `data/eval/results/`.

The recommendation applies the rule in `variants.recommend_default`: best recall@5,
but the fastest variant within 0.05 of the best. With a 30-question eval that an AI
drafted, a difference under about 0.1 is noise; the table says so.
"""

import json
from pathlib import Path

from groundcheck.retrieval.evaluate import RetrievalReport
from groundcheck.retrieval.variants import recommend_default

NOISE_NOTE = (
    "Only {n} questions, drafted by an AI from the filings and not yet reviewed by a human: "
    "one question moves recall by 1/{n} = {step:.2f}, so differences under about 0.1 are noise."
)


def comparison_rows(
    reports: dict[str, RetrievalReport], diagnostic: set[str] | frozenset = frozenset()
) -> list[dict]:
    rows = []
    for name, report in reports.items():
        s = report.overall
        rows.append(
            {
                "variant": name,
                "diagnostic": name in diagnostic,
                "retriever": report.embedder,
                "n": s.n,
                "recall": dict(s.recall),
                "section_hit": dict(s.section_hit),
                "mrr": s.mrr,
                "latency_ms": s.latency_ms,
                "recall_at_5_by_qtype": {q: v.recall.get(5) for q, v in report.by_qtype.items()},
                "recall_at_5_by_ticker": {t: v.recall.get(5) for t, v in report.by_ticker.items()},
            }
        )
    return rows


def format_comparison(rows: list[dict], ks: list[int]) -> str:
    width = max([len("variant"), *(len(r["variant"]) for r in rows)])
    header = f"{'variant':<{width}} {'n':>3}  " + "  ".join(f"R@{k:<3}" for k in ks)
    header += "  " + "  ".join(f"S@{k:<3}" for k in ks) + "  MRR     ms/q"
    lines = [header]
    for r in rows:
        cells = [f"{r['recall'][k]:.2f} " for k in ks]
        cells += [f"{r['section_hit'][k]:.2f} " for k in ks]
        cells += [f"{r['mrr']:.2f} ", f"{r['latency_ms']:7.1f}"]
        mark = "  (diagnostic)" if r.get("diagnostic") else ""
        lines.append(f"{r['variant']:<{width}} {r['n']:>3}  " + "  ".join(cells) + mark)
    return "\n".join(lines)


def comparison_markdown(rows: list[dict], ks: list[int], recommendation: str, note: str) -> str:
    head = ["Variant", *[f"R@{k}" for k in ks], *[f"S@{k}" for k in ks], "MRR", "ms/query"]
    lines = [
        "# Retrieval variant comparison",
        "",
        note,
        "",
        "| " + " | ".join(head) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(head))) + "|",
    ]
    for r in rows:
        cells = [f"`{r['variant']}`" + (" (diagnostic)" if r["diagnostic"] else "")]
        cells += [f"{r['recall'][k]:.2f}" for k in ks]
        cells += [f"{r['section_hit'][k]:.2f}" for k in ks]
        cells += [f"{r['mrr']:.2f}", f"{r['latency_ms']:.1f}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"**Recommended default:** {recommendation}", ""]

    for key, title in (("recall_at_5_by_qtype", "type"), ("recall_at_5_by_ticker", "company")):
        groups = sorted({g for r in rows for g in r[key]})
        lines += [
            f"## Recall@5 by question {title}" if title == "type" else "## Recall@5 by company",
            "",
            "| Variant | " + " | ".join(groups) + " |",
            "|---|" + "---:|" * len(groups),
        ]
        for r in rows:
            values = [r[key].get(g) for g in groups]
            cells = ["-" if v is None else f"{v:.2f}" for v in values]
            lines.append(f"| `{r['variant']}` | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Retrievers", ""]
    lines += [f"- `{r['variant']}`: {r['retriever']}" for r in rows]
    return "\n".join(lines) + "\n"


def write_comparison(
    reports: dict[str, RetrievalReport],
    ks: list[int],
    results_dir: Path,
    diagnostic: set[str] | frozenset = frozenset(),
) -> tuple[str, str, Path, Path]:
    """Returns (printable table, recommendation line, json path, markdown path).
    Variants in `diagnostic` are shown but never recommended."""
    rows = comparison_rows(reports, diagnostic)
    n = max((r["n"] for r in rows), default=0)
    note = NOISE_NOTE.format(n=n, step=1 / n if n else 0.0)
    picked = recommend_default(rows) if 5 in ks else None
    recommendation = (
        f"`{picked[0]}`: {picked[1]}" if picked else "n/a (needs recall@5: pass -k ... 5 ...)"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = results_dir / "compare.json", results_dir / "compare.md"
    payload = {
        "note": note,
        "ks": ks,
        "recommended_default": picked[0] if picked else None,
        "recommendation": recommendation,
        "variants": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path.write_text(comparison_markdown(rows, ks, recommendation, note))
    return format_comparison(rows, ks), recommendation, json_path, md_path
