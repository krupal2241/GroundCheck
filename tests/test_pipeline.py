"""`groundcheck pipeline`: stage selection, skipping unchanged stages, the offline dry run."""

import json
from pathlib import Path

import pytest

from groundcheck.cli import main
from groundcheck.config import ConfigError, load_companies
from groundcheck.labeling.schema import LabelRecord
from groundcheck.labeling.store import LabelStore, load_candidates
from groundcheck.pipeline import runner
from groundcheck.pipeline.fake import FakeSecClient, synthetic_filing_html
from groundcheck.pipeline.runner import (
    STAGES,
    Context,
    Options,
    Paths,
    fingerprint,
    run_pipeline,
    select,
)

REPO = Path(__file__).resolve().parents[1]


def statuses(result):
    return {r.name: r.status for r in result.results}


@pytest.fixture
def ctx(tmp_path):
    lines: list[str] = []
    context = Context(Paths.dry_run(tmp_path / "scratch"), Options(dry_run=True), lines.append)
    context.lines = lines
    return context


def test_select_and_stage_order():
    assert STAGES[0] == "download" and STAGES[-1] == "report" and len(STAGES) == 10
    assert select("index", "generate") == ["index", "eval-retrieval", "generate"]
    assert select(None, "parse") == ["download", "parse"]
    with pytest.raises(ConfigError, match="comes after"):
        select("generate", "index")
    with pytest.raises(ConfigError, match="unknown stage"):
        select("nope", None)


def test_fingerprint_changes_with_inputs_and_settings(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "a.txt").write_text("one")
    first = fingerprint("chunk", [folder], {"tokenizer": "none"})
    assert first == fingerprint("chunk", [folder], {"tokenizer": "none"})
    assert first != fingerprint("chunk", [folder], {"tokenizer": "bge-small"})
    (folder / "a.txt").write_text("two")
    assert first != fingerprint("chunk", [folder], {"tokenizer": "none"})
    assert fingerprint("x", [tmp_path / "missing"], {}) != fingerprint("x", [], {})


def test_fake_sec_client_serves_synthetic_filings_from_the_committed_chunks():
    companies = load_companies().companies
    client = FakeSecClient(companies, REPO / "data" / "chunks")
    apple = next(c for c in companies if c.ticker == "AAPL")
    html = client.get(f"https://www.sec.gov/Archives/edgar/data/1/2/{apple.primary_document}")
    assert b"Item 7." in html and b"synthetic dry-run filing" in html
    assert html == synthetic_filing_html(apple, REPO / "data" / "chunks")
    with pytest.raises(KeyError):
        client.get("https://www.sec.gov/unknown.htm")


def test_dry_run_runs_offline_then_skips_unchanged_stages(ctx):
    first = run_pipeline(ctx, end="error-report")
    assert statuses(first) == {
        "download": "ran",
        "parse": "ran",
        "chunk": "ran",
        "index": "ran",
        "eval-retrieval": "ran",
        "generate": "ran",
        "export-labels": "nothing to do",  # no labels: never invented
        "eval-verifiers": "nothing to do",
        "error-report": "nothing to do",
    }
    root = ctx.paths.raw.parent
    candidates = load_candidates(ctx.paths.candidates)
    assert {c.mode for c in candidates} == {"rag", "no-context", "wrong-context"}
    assert all(c.answer_model.provider == "fake" for c in candidates)
    assert all(str(p).startswith(str(root)) for p in (ctx.paths.chunks, ctx.paths.state))

    second = run_pipeline(ctx, end="generate")
    assert set(statuses(second).values()) == {"unchanged"}
    assert any("skipped (inputs unchanged" in line for line in ctx.lines)

    # Touch a parse output that chunking reads: chunk re-runs; its output is identical,
    # so index and everything after it stay skipped.
    parsed = sorted(ctx.paths.parsed.glob("*.txt"))[0]
    parsed.write_text(parsed.read_text() + "\n")
    third = statuses(run_pipeline(ctx, start="chunk", end="generate"))
    assert third == {
        "chunk": "ran",
        "index": "unchanged",
        "eval-retrieval": "unchanged",
        "generate": "unchanged",
    }
    forced = Context(ctx.paths, Options(dry_run=True, force=True), ctx.echo)
    assert statuses(run_pipeline(forced, start="index", end="index")) == {"index": "ran"}


def test_dry_run_with_a_label_runs_the_evaluation_stages(ctx):
    run_pipeline(ctx, end="generate")
    candidate = next(c for c in load_candidates(ctx.paths.candidates) if c.claims)
    claim = candidate.claims[0]
    LabelStore(ctx.paths.labels).put(
        LabelRecord(
            claim_id=claim.claim_id,
            candidate_id=candidate.candidate_id,
            question_id=candidate.question_id,
            ticker=candidate.ticker,
            qtype=candidate.qtype,
            claim=claim.text,
            label="unsupported",
            confirmed=True,
        )
    )
    result = statuses(run_pipeline(ctx, start="export-labels"))
    assert result == {
        "export-labels": "ran",
        "eval-verifiers": "ran",
        "error-report": "ran",
        "report": "ran",
    }
    assert "| fake |" in (ctx.paths.docs / "RESULTS.md").read_text()
    report = json.loads((ctx.paths.results / "verifier_eval.json").read_text())
    assert "fake" in json.dumps(report)
    assert (ctx.paths.results / "error_report.md").is_file()


def test_a_failed_stage_stops_the_pipeline_and_is_not_stored(ctx, monkeypatch):
    stages = runner.build_stages()
    stages["parse"].run = lambda c: 1
    monkeypatch.setattr(runner, "build_stages", lambda: stages)
    result = run_pipeline(ctx, end="chunk")
    assert statuses(result) == {"download": "ran", "parse": "failed", "chunk": "not run"}
    assert not result.ok and not (ctx.paths.state / "parse.json").exists()


def test_pipeline_command(tmp_path, capsys):
    assert main(["pipeline", "--list"]) == 0
    assert "download -> parse -> chunk" in capsys.readouterr().out
    scratch = tmp_path / "s"
    args = ["pipeline", "--dry-run", "--scratch-dir", str(scratch), "--to", "index"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "DRY RUN, fakes only" in out and "Pipeline summary:" in out
    assert (scratch / "index").is_dir() and (scratch / "state" / "index.json").is_file()
    assert main(["pipeline", "--dry-run", "--scratch-dir", str(scratch), "--from", "parse",
                 "--to", "download"]) == 2  # fmt: skip


def test_retrieval_stages_use_the_configured_variant(ctx):
    """index, eval-retrieval and generate open the variant from config/retrieval.yaml
    (the repo's default, with the offline fakes swapped in for its models)."""
    from groundcheck.retrieval.variants import load_retrieval_config

    default = load_retrieval_config(ctx.paths.retrieval_config).default
    run_pipeline(ctx, end="generate")
    report = json.loads((ctx.paths.retrieval_results / f"retrieval_{default}.json").read_text())
    assert report["variant"] == default
    candidates = load_candidates(ctx.paths.candidates)
    assert {c.retrieval.variant for c in candidates} == {default}
    assert {c.mode for c in candidates} == {"rag", "no-context", "wrong-context"}


def test_variant_option_reruns_retrieval_and_compare_feeds_the_report(ctx, tmp_path):
    run_pipeline(ctx, end="eval-retrieval")
    # Another variant: different settings, so index and eval-retrieval run again.
    bm25 = Context(ctx.paths, Options(dry_run=True, variant="bm25"), ctx.echo)
    assert statuses(run_pipeline(bm25, start="index", end="eval-retrieval")) == {
        "index": "ran",
        "eval-retrieval": "ran",
    }
    assert (ctx.paths.retrieval_results / "retrieval_bm25.json").is_file()
    # --compare: every variant indexed and compared; the report shows the comparison.
    compare = Context(ctx.paths, Options(dry_run=True, compare=True, modes=("rag",)), ctx.echo)
    result = statuses(run_pipeline(compare, start="index", end="eval-retrieval"))
    assert result == {"index": "ran", "eval-retrieval": "ran"}
    data = json.loads((ctx.paths.retrieval_results / "compare.json").read_text())
    names = {row["variant"] for row in data["variants"]}
    assert {"dense", "hybrid", "bm25", "hybrid-headers-rerank"} <= names
    assert statuses(run_pipeline(compare, start="report")) == {"report": "ran"}
    results_md = (ctx.paths.docs / "RESULTS.md").read_text()
    assert (
        "## Retrieval variant comparison" in results_md and "`hybrid-headers-rerank`" in results_md
    )
    with pytest.raises(ConfigError, match="drop --variant"):
        both = Options(dry_run=True, compare=True, variant="bm25", force=True)
        runner.run_eval_retrieval(Context(ctx.paths, both, ctx.echo))
