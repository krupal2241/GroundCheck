import pytest

from groundcheck.chunking.base import Chunk
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


def test_committed_template_is_valid_and_unfilled():
    path = REPO_ROOT / "data" / "eval" / "retrieval_questions.yaml"
    questions = load_questions(path).questions
    assert len(questions) == 30
    assert {x.ticker for x in questions} == {"AAPL", "MSFT", "AMZN", "WMT", "XOM", "JNJ"}
    # No gold label is ever written for you: the template ships empty.
    assert not any(x.gold_evidence for x in questions)


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
