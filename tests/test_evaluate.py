import pytest

from groundcheck.chunking.base import Chunk, chunks_path, read_chunks
from groundcheck.config import REPO_ROOT, ConfigError
from groundcheck.retrieval.evaluate import (
    RetrievalQuestion,
    check_questions,
    contains,
    evaluate,
    format_report,
    load_questions,
    normalize_for_match,
)


def chunk(cid, item, text, ticker="EXCO"):
    return Chunk(
        chunk_id=cid,
        ticker=ticker,
        accession="0000000001-25-000001",
        item=item,
        section_title="",
        kind="table" if text.startswith("[TABLE]") else "prose",
        index=0,
        text=text,
        word_count=len(text.split()),
        chunker="fixed-p250-o50-t120",
    )


CORPUS = [
    chunk("c1", "1", "Example Corp designs widgets in Ohio."),
    chunk("c2", "1A", "Competition could reduce our margins."),
    chunk("c3", "7", "[TABLE]\n| 2025 | 2024\nTotal net sales | $12,345 | $11,000\n[/TABLE]"),
    chunk("c4", "7", "Operating income was $2,000 million in 2025."),
    chunk("c5", "8", "The company repurchased 10 million shares."),
]


def q(qid="Q1", qtype="numeric_lookup", question="What?", gold=("$2,000 million",), items=()):
    return RetrievalQuestion(
        id=qid,
        ticker="EXCO",
        qtype=qtype,
        question=question,
        gold_evidence=list(gold),
        gold_items=list(items),
    )


def fixed_retriever(order):
    """A fake retriever that always returns these chunk IDs in this order."""
    by_id = {c.chunk_id: c for c in CORPUS}
    return lambda ticker, question, k: [by_id[i] for i in order][:k]


def run(questions, order, ks=(1, 3, 5)):
    return evaluate(questions, {"EXCO": CORPUS}, fixed_retriever(order), "fake", ks)


# --- matching --------------------------------------------------------------------------


def test_matching_normalizes_whitespace_on_both_sides():
    assert normalize_for_match("  a\n\tb   c ") == "a b c"
    table_chunk = CORPUS[2]
    assert contains(table_chunk, "Total net sales | $12,345")
    assert contains(table_chunk, "| 2024 Total net sales")  # row boundary is whitespace
    assert not contains(table_chunk, "total net sales | $12,345")  # case matters


# --- metrics ---------------------------------------------------------------------------


def test_recall_at_k():
    report = run([q()], ["c1", "c2", "c4", "c3", "c5"])
    result = report.questions[0]
    assert result.status == "evaluated"
    assert result.recall == {1: 0.0, 3: 1.0, 5: 1.0}
    assert result.first_gold_rank == 3
    assert result.retrieved == ["c1", "c2", "c4", "c3", "c5"]


def test_multi_step_partial_credit():
    question = q(qtype="multi_step", gold=("$12,345", "$2,000 million"))
    result = run([question], ["c3", "c1", "c2", "c5", "c4"]).questions[0]
    assert result.recall == {1: 0.5, 3: 0.5, 5: 1.0}


def test_section_hit_uses_inferred_items():
    result = run([q()], ["c3", "c1"]).questions[0]
    assert result.gold_items == ["7"]
    assert result.recall[1] == 0.0  # the right section, but not the right chunk
    assert result.section_hit[1] == 1.0


def test_section_hit_uses_explicit_items():
    result = run([q(items=["8"])], ["c4", "c5"]).questions[0]
    assert result.section_hit == {1: 0.0, 3: 1.0, 5: 1.0}


def test_summary_means_and_breakdowns():
    questions = [
        q("Q1", gold=("$2,000 million",)),
        q("Q2", qtype="table", gold=("Total net sales | $12,345",)),
    ]
    report = run(questions, ["c4", "c1", "c3"])
    assert report.overall.n == 2
    assert report.overall.recall == {1: 0.5, 3: 1.0, 5: 1.0}
    assert report.by_qtype["table"].recall[1] == 0.0
    assert report.by_ticker["EXCO"].n == 2
    assert report.chunker == "fixed-p250-o50-t120"
    text = format_report(report)
    assert "overall" in text and "R@1" in text and "S@5" in text


def test_mrr_and_latency():
    questions = [
        q("Q1", gold=("$2,000 million",)),  # c4 at rank 2 -> 1/2
        q("Q2", qtype="table", gold=("Total net sales | $12,345",)),  # c3 at rank 3 -> 1/3
        q("Q3", gold=("10 million shares",)),  # c5 not retrieved -> 0
    ]
    report = run(questions, ["c1", "c4", "c3"])
    assert [r.reciprocal_rank for r in report.questions] == [0.5, pytest.approx(1 / 3), 0.0]
    assert report.overall.mrr == pytest.approx((0.5 + 1 / 3 + 0) / 3)
    assert all(r.latency_ms is not None and r.latency_ms >= 0 for r in report.questions)
    assert report.overall.latency_ms == pytest.approx(
        sum(r.latency_ms for r in report.questions) / 3
    )
    assert "MRR" in format_report(report) and "ms/q" in format_report(report)


def test_comparison_table_json_and_markdown(tmp_path):
    import json

    from groundcheck.retrieval.compare import write_comparison

    questions = [q("Q1", gold=("$2,000 million",)), q("Q2", qtype="table", gold=("$12,345",))]
    reports = {
        "slow-good": run(questions, ["c4", "c3"], ks=(1, 3, 5, 10)),
        "fast-bad": run(questions, ["c1", "c2", "c5"], ks=(1, 3, 5, 10)),
        "lexical": run(questions, ["c3", "c4"], ks=(1, 3, 5, 10)),
    }
    reports["slow-good"].overall.latency_ms = 50.0
    reports["fast-bad"].overall.latency_ms = 1.0
    reports["lexical"].overall.latency_ms = 0.1
    table, recommendation, json_path, md_path = write_comparison(
        reports, [1, 3, 5, 10], tmp_path, diagnostic={"lexical"}
    )
    lines = table.splitlines()
    assert lines[0].split()[:3] == ["variant", "n", "R@1"] and "MRR" in lines[0]
    assert [line.split()[0] for line in lines[1:]] == ["slow-good", "fast-bad", "lexical"]
    assert lines[3].endswith("(diagnostic)")
    # lexical ties slow-good and is faster, but it is diagnostic: never recommended.
    assert recommendation.startswith("`slow-good`")
    payload = json.loads(json_path.read_text())
    assert payload["recommended_default"] == "slow-good"
    assert "2 questions" in payload["note"] and "noise" in payload["note"]
    assert payload["variants"][0]["recall"]["5"] == 1.0
    assert payload["variants"][0]["recall_at_5_by_qtype"] == {"numeric_lookup": 1.0, "table": 1.0}
    md = md_path.read_text()
    assert "| `slow-good` | 1.00 | 1.00 |" in md
    assert "## Recall@5 by question type" in md and "## Recall@5 by company" in md
    assert "**Recommended default:** `slow-good`" in md


def test_skipped_questions_are_counted_not_scored():
    questions = [
        q("filled"),
        q("blank", question="", gold=()),
        q("no-gold", gold=()),
        q("unans", qtype="unanswerable", gold=()),
        q("typo", gold=("$2,000 milion",)),
    ]
    report = run(questions, ["c4"])
    assert report.overall.n == 1
    assert report.skipped == {"unfilled": 2, "unanswerable": 1, "gold_not_in_corpus": 1}
    typo = next(r for r in report.questions if r.id == "typo")
    assert typo.missing_gold == ["$2,000 milion"]
    assert "WARNING: typo: gold evidence not found" in format_report(report)


def test_no_scored_questions_gives_zeros_not_a_crash():
    report = run([q("blank", question="")], [])
    assert report.overall.n == 0 and report.overall.recall[1] == 0.0


def test_unknown_ticker_is_an_error():
    question = RetrievalQuestion(
        id="X", ticker="NOPE", qtype="table", question="?", gold_evidence=["a"]
    )
    with pytest.raises(ConfigError, match="NOPE"):
        evaluate([question], {"EXCO": CORPUS}, fixed_retriever([]), "fake")


# --- offline check -----------------------------------------------------------------------


def test_check_questions_finds_problems():
    problems = check_questions(
        [
            q("ok"),
            q("typo", gold=("nope",)),
            q("stray", items=["9A"]),
            q("wrong-section", items=["8"]),
            q("blank", question=""),
        ],
        {"EXCO": CORPUS},
    )
    assert len(problems) == 3
    assert problems[0].startswith("typo:")
    assert "9A" in problems[1]
    assert "found only in Items ['7']" in problems[2]


# --- question file -----------------------------------------------------------------------


def test_committed_question_file_is_filled_and_gold_is_in_the_chunks():
    path = REPO_ROOT / "data" / "eval" / "retrieval_questions.yaml"
    questions = load_questions(path).questions
    assert len(questions) == 30
    assert {x.ticker for x in questions} == {"AAPL", "MSFT", "AMZN", "WMT", "XOM", "JNJ"}
    assert all(x.is_filled and x.qtype != "unanswerable" for x in questions)
    corpora = {
        t: read_chunks(chunks_path(t, REPO_ROOT / "data" / "chunks"))
        for t in {x.ticker for x in questions}
    }
    assert check_questions(questions, corpora) == []


def test_question_file_rejects_typos_and_duplicates(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text("questions:\n  - {id: A, ticker: EXCO, qtype: table, gold_evidnce: [x]}\n")
    with pytest.raises(ConfigError, match="gold_evidnce"):
        load_questions(path)
    path.write_text(
        "questions:\n"
        "  - {id: A, ticker: EXCO, qtype: table}\n"
        "  - {id: A, ticker: EXCO, qtype: table}\n"
    )
    with pytest.raises(ConfigError, match="duplicate question ids"):
        load_questions(path)
    path.write_text("questions:\n  - {id: A, ticker: EXCO, qtype: trivia}\n")
    with pytest.raises(ConfigError, match="qtype"):
        load_questions(path)


def test_missing_question_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_questions(tmp_path / "nope.yaml")
