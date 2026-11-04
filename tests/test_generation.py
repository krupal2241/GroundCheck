"""Candidate generation: question sources, prompts, the pipeline and the script."""

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from groundcheck.config import ConfigError, LLMConfig, load_companies
from groundcheck.generation import runner
from groundcheck.generation.pipeline import (
    CandidateRecord,
    Generator,
    RetrievalInfo,
    append_record,
    candidates_path,
    read_candidates,
    run,
)
from groundcheck.generation.prompts import (
    ABSTAIN_TEXT,
    ANSWER_SYSTEM,
    PROMPT_VERSION,
    answer_prompt,
    claims_prompt,
    fake_responder,
    make_request,
)
from groundcheck.generation.questions import (
    TEMPLATE_SLOTS,
    GenerationQuestion,
    collect_questions,
    ensure_template,
    load_candidate_questions,
)
from groundcheck.llm.fake import FakeProvider
from groundcheck.retrieval.embedder import HashingEmbedder
from groundcheck.retrieval.index import CompanyIndex

REPO = Path(__file__).resolve().parents[1]


# --- Questions --------------------------------------------------------------------------


def test_committed_questions_have_ten_typed_slots_per_labeled_company():
    companies = load_companies()
    labeled = [c.ticker for c in companies.companies if c.role == "labeled"]
    path = REPO / "data" / "questions" / "candidate_questions.yaml"
    questions = load_candidate_questions(path).questions
    assert len(questions) == 10 * len(labeled) == 60
    for ticker in labeled:
        mine = [q for q in questions if q.ticker == ticker]
        assert tuple(q.qtype for q in mine) == TEMPLATE_SLOTS
    # With the retrieval questions, ~10 questions per labeled company in the PLAN.md mix.
    collected = collect_questions(
        companies,
        path,
        REPO / "data" / "eval" / "retrieval_questions.yaml",
    )
    for ticker in labeled:
        mine = sorted(q.qtype for q in collected if q.ticker == ticker)
        assert mine == sorted(TEMPLATE_SLOTS)
    assert {"BRK-B", "JPM"}.isdisjoint(q.ticker for q in questions)
    assert {"numeric_lookup", "table", "multi_step", "yes_no", "temporal", "unanswerable"} == set(
        TEMPLATE_SLOTS
    )


def test_ensure_template_creates_once_and_never_overwrites(tmp_path):
    path = tmp_path / "q" / "candidate_questions.yaml"
    companies = load_companies()
    assert ensure_template(path, companies) is True
    path.write_text(path.read_text().replace("question: ''", "question: 'Mine?'", 1))
    assert ensure_template(path, companies) is False
    assert "Mine?" in path.read_text()


def test_questions_come_from_both_files(tiny_corpus, caplog):
    companies = load_companies(tiny_corpus["config"])
    with caplog.at_level(logging.WARNING):
        questions = collect_questions(companies, tiny_corpus["questions"], tiny_corpus["retrieval"])
    assert [(q.id, q.source) for q in questions] == [
        ("EXCO-01", "retrieval_questions"),
        ("EXCO-G01", "candidate_questions"),
        ("EXCO-G02", "candidate_questions"),
    ]
    assert "OTHR is not a labeled company" in caplog.text


def test_same_question_text_is_used_once(tiny_corpus):
    path = tiny_corpus["questions"]
    path.write_text(
        path.read_text().replace(
            "What were total net sales in fiscal 2025?", "what was  OPERATING income in 2025?"
        )
    )
    companies = load_companies(tiny_corpus["config"])
    ids = [q.id for q in collect_questions(companies, path, tiny_corpus["retrieval"])]
    assert ids == ["EXCO-01", "EXCO-G02"]


def test_duplicate_ids_across_files_are_an_error(tiny_corpus):
    path = tiny_corpus["questions"]
    path.write_text(path.read_text().replace("EXCO-G02", "EXCO-01"))
    with pytest.raises(ConfigError, match="EXCO-01"):
        collect_questions(load_companies(tiny_corpus["config"]), path, tiny_corpus["retrieval"])


def test_company_filter_and_unlabeled_company(tiny_corpus):
    companies = load_companies(tiny_corpus["config"])
    with pytest.raises(ConfigError, match="not a labeled company"):
        collect_questions(companies, tiny_corpus["questions"], None, tickers=["OTHR"])
    only = collect_questions(companies, tiny_corpus["questions"], None, tickers=["EXCO"])
    assert [q.id for q in only] == ["EXCO-G01", "EXCO-G02"]


# --- Prompts ----------------------------------------------------------------------------


def test_answer_prompt_numbers_passages_with_ids():
    prompt = answer_prompt("EXCO", "Q?", [("EXCO-7-p0000", " A. "), ("EXCO-8-p0000", "B")])
    assert "[1] (EXCO-7-p0000)\nA." in prompt and "[2] (EXCO-8-p0000)\nB" in prompt
    assert prompt.index("Passages") < prompt.index("Question: Q?")
    assert "ONLY" in ANSWER_SYSTEM and ABSTAIN_TEXT in ANSWER_SYSTEM


def test_fake_responder_answers_and_splits():
    prompt = answer_prompt("EXCO", "Q?", [("X-1", "Sales rose to $5 million. Costs fell.")])
    answer = fake_responder(make_request("s", prompt, 0, 1, 10, 512))
    assert answer == "According to the filing, Sales rose to $5 million."
    split = make_request("s", claims_prompt("Q?", "One fact. Two facts."), 0, 1, 10, 512)
    split = type(split)(**{**split.__dict__, "json_mode": True})
    assert json.loads(fake_responder(split)) == {"claims": ["One fact.", "Two facts."]}
    abstain = type(split)(**{**split.__dict__, "prompt": claims_prompt("Q?", ABSTAIN_TEXT)})
    assert json.loads(fake_responder(abstain)) == {"claims": []}


# --- Pipeline ---------------------------------------------------------------------------


def make_generator(corpus, provider, **config):
    index = CompanyIndex.load("EXCO", corpus["chunks"], corpus["index"])
    embedder = HashingEmbedder()
    info = RetrievalInfo(embedder=embedder.name, chunker="c", top_k=3, chunks_sha256="s")
    return Generator(
        provider,
        lambda t, q, k: index.search(embedder, q, k),
        lambda t: info,
        LLMConfig(top_k=3, **config),
    )


QUESTION = GenerationQuestion(
    id="EXCO-G01",
    ticker="EXCO",
    qtype="numeric_lookup",
    question="What were total net sales in fiscal 2025?",
    source="candidate_questions",
)


def test_generator_record_has_everything(tiny_corpus):
    fake = FakeProvider(
        script=[
            "Total net sales were $12,345 million in fiscal 2025, up 6%.",
            '```json\n{"claims": ["Total net sales were $12,345 million in fiscal 2025.",'
            ' "Total net sales rose 6% in fiscal 2025."]}\n```',
        ]
    )
    record = make_generator(tiny_corpus, fake).generate(QUESTION)
    assert record.candidate_id == "EXCO-G01@rag@fake:fake-model"
    assert [c.claim_id for c in record.claims] == [
        "EXCO-G01@rag@fake:fake-model#c01",
        "EXCO-G01@rag@fake:fake-model#c02",
    ]
    assert len(record.retrieved) == 3 and record.retrieved[0].rank == 1
    assert record.retrieved[0].chunk_id == "EXCO-7-p0000"  # best word overlap
    assert all(r.text for r in record.retrieved)
    # The answer prompt contains exactly the retrieved chunks, in rank order.
    answer_call, claims_call = fake.calls
    positions = [answer_call.prompt.index(r.chunk_id) for r in record.retrieved]
    assert positions == sorted(positions)
    assert answer_call.temperature == 0 and answer_call.seed == 42
    assert answer_call.context_window == 8192 and not answer_call.json_mode
    assert claims_call.json_mode and record.answer in claims_call.prompt
    assert record.answer_model.model_digest == "fake-digest"
    assert record.claims_model.attempts == 1
    assert record.prompt_version == PROMPT_VERSION and record.question_source.startswith("cand")


def test_claims_retry_then_failure_is_reported(tiny_corpus):
    fake = FakeProvider(script=["An answer.", "nope", "nope", "nope"])
    summary = run(make_generator(tiny_corpus, fake), [QUESTION], tiny_corpus["out"], echo=print)
    assert summary.written == 0 and summary.failed[0][0] == "EXCO-G01"
    assert not candidates_path("EXCO", tiny_corpus["out"]).exists()  # retried next run


def test_run_is_resumable_batched_and_limited(tiny_corpus):
    questions = [
        QUESTION.model_copy(update={"id": f"EXCO-G0{i}", "question": f"Net sales {i}?"})
        for i in range(1, 6)
    ]
    sleeps, lines = [], []
    generator = make_generator(tiny_corpus, FakeProvider(responder=fake_responder))
    first = run(
        generator,
        questions,
        tiny_corpus["out"],
        batch_size=2,
        limit=3,
        sleep_between_batches=30,
        sleep=sleeps.append,
        echo=lines.append,
    )
    assert (first.written, first.already_done) == (3, 0)
    assert sleeps == [30]  # batches of 2 and 1: one pause between them
    path = candidates_path("EXCO", tiny_corpus["out"])
    before = path.read_text()
    second = run(generator, questions, tiny_corpus["out"], sleep=sleeps.append, echo=lines.append)
    assert (second.written, second.already_done) == (2, 3)
    assert path.read_text().startswith(before)  # append-only
    assert [r.question_id for r in read_candidates(path)] == [q.id for q in questions]
    third = run(generator, questions, tiny_corpus["out"], echo=lines.append)
    assert third.written == 0 and third.already_done == 5


def test_another_model_appends_its_own_records(tiny_corpus):
    out = tiny_corpus["out"]
    run(make_generator(tiny_corpus, FakeProvider(responder=fake_responder)), [QUESTION], out)
    other = FakeProvider(responder=fake_responder, model="other-model")
    assert run(make_generator(tiny_corpus, other), [QUESTION], out).written == 1
    ids = [r.candidate_id for r in read_candidates(candidates_path("EXCO", out))]
    assert ids == ["EXCO-G01@rag@fake:fake-model", "EXCO-G01@rag@fake:other-model"]


def test_partial_last_line_is_skipped_and_appending_still_works(tiny_corpus, caplog):
    out = tiny_corpus["out"]
    generator = make_generator(tiny_corpus, FakeProvider(responder=fake_responder))
    run(generator, [QUESTION], out)
    path = candidates_path("EXCO", out)
    with path.open("a") as f:
        f.write('{"schema_version": 1, "candidate_id": "EXCO-G0')  # a crash mid-write
    second = QUESTION.model_copy(update={"id": "EXCO-G02"})
    with caplog.at_level(logging.WARNING):
        assert run(generator, [QUESTION, second], out).written == 1
    assert "not a valid candidate record" in caplog.text
    assert [r.question_id for r in read_candidates(path)] == ["EXCO-G01", "EXCO-G02"]


def test_append_record_round_trips(tmp_path, tiny_corpus):
    record = make_generator(tiny_corpus, FakeProvider(responder=fake_responder)).generate(QUESTION)
    path = tmp_path / "x.jsonl"
    append_record(record, path)
    append_record(record, path)
    assert read_candidates(path) == [record, record]
    assert CandidateRecord.model_validate_json(path.read_text().splitlines()[0]) == record


# --- The script -------------------------------------------------------------------------


def test_dry_run_end_to_end(tiny_corpus, capsys):
    assert runner.main(tiny_corpus["generate_args"]("--dry-run", "--batch-size", "2")) == 0
    out = capsys.readouterr().out
    assert "Model: fake fake-model (digest fake-digest" in out
    assert "Wrote 3 candidate(s)" in out
    records = read_candidates(candidates_path("EXCO", tiny_corpus["out"]))
    assert [r.question_id for r in records] == ["EXCO-01", "EXCO-G01", "EXCO-G02"]
    assert all(r.claims and r.retrieval.embedder == "hashing-256" for r in records)
    assert all(len(r.retrieved) == 3 for r in records)  # top_k from the llm config
    # Resume: nothing left to do.
    assert runner.main(tiny_corpus["generate_args"]("--dry-run")) == 0
    assert "Wrote 0 candidate(s)" in capsys.readouterr().out


def test_script_flags(tiny_corpus, capsys):
    args = tiny_corpus["generate_args"]("--dry-run", "--company", "EXCO", "--limit", "1")
    assert runner.main([*args, "--top-k", "2"]) == 0
    records = read_candidates(candidates_path("EXCO", tiny_corpus["out"]))
    assert len(records) == 1 and len(records[0].retrieved) == 2
    assert runner.main(tiny_corpus["generate_args"]("--dry-run", "--company", "OTHR")) == 2
    assert "not a labeled company" in capsys.readouterr().err


def test_dry_run_default_output_is_outside_committed_data(tiny_corpus, monkeypatch):
    monkeypatch.setattr(runner, "DRY_RUN_CANDIDATES_DIR", tiny_corpus["root"] / "dry")
    args = [a for a in tiny_corpus["generate_args"]("--dry-run", "--limit", "1")]
    i = args.index("--out-dir")
    del args[i : i + 2]
    assert runner.main(args) == 0
    assert (tiny_corpus["root"] / "dry" / "EXCO.jsonl").is_file()


def test_missing_index_fails_before_any_model_call(tiny_corpus, capsys):
    args = tiny_corpus["generate_args"]("--dry-run")
    args[args.index("--index-dir") + 1] = str(tiny_corpus["root"] / "no-index")
    assert runner.main(args) == 2
    assert "run `groundcheck index`" in capsys.readouterr().err


def test_no_questions_yet(tiny_corpus, capsys):
    tiny_corpus["questions"].write_text("version: 1\nquestions: []\n")
    tiny_corpus["retrieval"].write_text("version: 1\nquestions: []\n")
    assert runner.main(tiny_corpus["generate_args"]("--dry-run")) == 1
    assert "nothing to generate" in capsys.readouterr().out


def test_missing_question_file_creates_the_template(tiny_corpus, capsys):
    tiny_corpus["questions"].unlink()
    runner.main(tiny_corpus["generate_args"]("--dry-run"))
    assert "Created the question template" in capsys.readouterr().out
    assert "EXCO-G10" in tiny_corpus["questions"].read_text()


def test_ollama_without_server_is_a_clean_error(tiny_corpus, capsys, monkeypatch):
    class Down:
        def request(self, *a, **k):
            import requests

            raise requests.ConnectionError("refused")

    monkeypatch.setattr("groundcheck.llm.ollama.requests.Session", Down)
    assert runner.main(tiny_corpus["generate_args"]("--provider", "ollama")) == 2
    assert "cannot reach Ollama" in capsys.readouterr().err


def test_script_help_runs_as_a_file():
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "generate_candidates.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    for flag in ("--company", "--batch-size", "--limit", "--sleep-between-batches", "--model"):
        assert flag in result.stdout
    assert "--dry-run" in result.stdout and "--allow-paid" in result.stdout


def test_colab_notebook_uses_the_hf_provider_and_the_same_script():
    notebook = json.loads((REPO / "notebooks" / "generate_candidates_colab.ipynb").read_text())
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'pip install -q -e ".[embed,hf]"' in source
    assert "scripts/generate_candidates.py --provider hf-local" in source
    assert "Qwen/Qwen2.5-3B-Instruct" in source
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])  # committed clean
