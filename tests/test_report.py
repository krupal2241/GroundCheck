"""`groundcheck report`: README markers, docs/RESULTS.md, charts, and never inventing numbers."""

import json
from pathlib import Path

import pytest

from groundcheck.cli import main
from groundcheck.report.build import (
    BALANCE_END,
    BALANCE_START,
    RESULTS_END,
    RESULTS_START,
    build_report,
    replace_between,
)

README = f"# Project\n\nIntro.\n\n## Results\n\n{RESULTS_START}\nTBD\n{RESULTS_END}\n\n## After\n"


class StandInPlot:
    """Records matplotlib calls; savefig writes a placeholder file."""

    def __init__(self):
        self.calls = []
        self.saved = []
        plot = self

        class Axes:
            def __getattr__(self, name):
                def record(*args, **kwargs):
                    plot.calls.append((name, args, kwargs))
                    return self

                return record

            spines = {s: type("S", (), {"set_visible": lambda *a: None,
                                        "set_color": lambda *a: None})()
                      for s in ("top", "right", "left", "bottom")}  # fmt: skip

        class Figure:
            def tight_layout(self):
                pass

            def savefig(self, path, **kwargs):
                plot.saved.append(str(path))
                Path(path).write_bytes(b"PNG")

        self.Axes, self.Figure = Axes, Figure

    def subplots(self, **kwargs):
        return self.Figure(), self.Axes()

    def close(self, fig):
        pass


@pytest.fixture
def results(gold_path, tmp_path, capsys):
    """Real eval-verifiers output (fake verifier) plus a retrieval report."""
    out = tmp_path / "results"
    assert main(["eval-verifiers", "--verifier", "fake", "--gold", str(gold_path),
                 "--results-dir", str(out)]) == 0  # fmt: skip
    retrieval = tmp_path / "eval-results"
    retrieval.mkdir()
    report = {
        "embedder": "hashing-256", "chunker": "fixed-p250-o50-t120", "ks": [1, 5],
        "overall": {"n": 4, "recall": {"1": 0.25, "5": 0.75}, "section_hit": {"1": 0.5, "5": 1.0}},
        "by_ticker": {}, "by_qtype": {}, "skipped": {}, "questions": [],
    }  # fmt: skip
    (retrieval / "retrieval_hashing-256.json").write_text(json.dumps(report))
    capsys.readouterr()
    return {"verifier": out, "retrieval": retrieval, "root": tmp_path}


def test_full_report_updates_readme_results_and_charts(results):
    readme = results["root"] / "README.md"
    readme.write_text(README)
    docs = results["root"] / "docs"
    plot = StandInPlot()
    outcome = build_report(results["verifier"], results["retrieval"], readme, docs, plt=plot)
    assert outcome.readme_updated and not outcome.missing
    text = readme.read_text()
    assert text.startswith("# Project\n\nIntro.") and text.endswith("## After\n")
    section = text.split(RESULTS_START)[1].split(RESULTS_END)[0]
    assert "TBD" not in section and "| fake |" in section and "$0 (local)" in section
    assert "Recall@5" in section and "75.0%" in section and "docs/RESULTS.md" in section
    md = (docs / "RESULTS.md").read_text()
    for heading in (
        "## Retrieval",
        "## Verifier comparison",
        "## Confusion matrices",
        "## Accuracy by edge case",
        "## Accuracy by generation mode",
        "## Charts",
    ):
        assert heading in md  # fmt: skip
    assert "| gold \\ predicted | supported | contradicted | unsupported | total |" in md
    assert "Latency p50" in md and "Latency p95" in md and "Supported vs not" in md
    assert sorted(p.rsplit("/", 1)[1] for p in plot.saved) == [
        "accuracy_vs_latency.png",
        "edge_case_accuracy.png",
    ]
    assert (docs / "figures" / "accuracy_vs_latency.png").is_file()
    assert any(name == "set_xscale" and args == ("log",) for name, args, _ in plot.calls)
    # Running again gives the same README (idempotent between the markers).
    build_report(results["verifier"], results["retrieval"], readme, docs, plt=plot)
    assert readme.read_text() == text


def test_missing_results_leave_the_readme_untouched(results, tmp_path, capsys):
    readme = tmp_path / "README.md"
    readme.write_text(README)
    docs = tmp_path / "docs2"
    args = ["report", "--results-dir", str(tmp_path / "nothing"), "--retrieval-results-dir",
            str(results["retrieval"]), "--readme", str(readme), "--docs-dir", str(docs),
            "--labels", str(tmp_path / "no-labels.jsonl")]  # fmt: skip
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "MISSING: verifier results" in out and "NOT changed" in out
    assert readme.read_text() == README
    md = (docs / "RESULTS.md").read_text()
    assert "Not run yet" in md and "| fake |" not in md
    assert "%" not in md.split("## Verifier comparison")[1]  # no numbers invented


def test_no_matplotlib_still_writes_tables(results, tmp_path, monkeypatch):
    monkeypatch.setattr("groundcheck.report.build._pyplot", lambda: None)
    outcome = build_report(results["verifier"], results["retrieval"], tmp_path / "R.md",
                           tmp_path / "d")  # fmt: skip
    assert outcome.figures == [] and any(
        "matplotlib is not installed" in m for m in outcome.messages
    )
    assert "## Charts" not in (tmp_path / "d" / "RESULTS.md").read_text()


def test_readme_without_markers_is_not_changed(results, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# No markers\n")
    outcome = build_report(results["verifier"], results["retrieval"], readme, tmp_path / "d",
                           plt=StandInPlot())  # fmt: skip
    assert not outcome.readme_updated and readme.read_text() == "# No markers\n"
    assert replace_between("a", "<x>", "</x>", "b") is None


def test_dataset_card_balance_table_from_confirmed_labels(results, tiny_corpus, capsys):
    from groundcheck.generation import runner
    from groundcheck.labeling.schema import LabelRecord
    from groundcheck.labeling.store import LabelStore, load_candidates

    assert runner.main(tiny_corpus["generate_args"]("--dry-run")) == 0
    candidates = load_candidates(tiny_corpus["out"])
    labels = tiny_corpus["root"] / "labels.jsonl"
    store = LabelStore(labels)
    for c, label, confirmed in zip(candidates, ["unsupported", "unsupported", "unsupported"],
                                   [True, True, False], strict=True):  # fmt: skip
        record = LabelRecord(
            claim_id=c.claims[0].claim_id, candidate_id=c.candidate_id,
            question_id=c.question_id, ticker=c.ticker, qtype=c.qtype,
            claim=c.claims[0].text, label=label, confirmed=confirmed,
        )  # fmt: skip
        store.put(record)
    docs = tiny_corpus["root"] / "docs"
    docs.mkdir()
    card = docs / "DATASET_CARD.md"
    card.write_text(f"# Card\n\n{BALANCE_START}\nTBD\n{BALANCE_END}\n")
    build_report(results["verifier"], results["retrieval"], tiny_corpus["root"] / "R.md", docs,
                 labels, tiny_corpus["out"], plt=StandInPlot())  # fmt: skip
    text = card.read_text()
    assert "| rag | 0 | 0 | 2 | 2 |" in text and "TBD" not in text
    assert "| **all** | 0 (0%) | 0 (0%) | 2 (100%) | 2 |" in text
