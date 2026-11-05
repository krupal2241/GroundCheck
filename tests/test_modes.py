"""Generation modes (rag, no-context, wrong-context), label-plan and label --plan."""

import json

import pytest

from groundcheck.chunking.base import Chunk, write_chunks
from groundcheck.cli import main
from groundcheck.config import ConfigError
from groundcheck.generation import runner
from groundcheck.generation.modes import WrongContextPicker
from groundcheck.generation.pipeline import candidates_path, read_candidates
from groundcheck.generation.prompts import ANSWER_SYSTEM, CLOSED_BOOK_SYSTEM
from groundcheck.labeling.plan import load_plan, make_plan
from groundcheck.labeling.store import LabelStore, load_candidates
from groundcheck.retrieval.embedder import HashingEmbedder
from groundcheck.retrieval.retriever import build_variant_index
from groundcheck.retrieval.variants import load_retrieval_config
from groundcheck.training.data import candidate_chunk_ids

OTHER_TEXT = (
    "Other Corp total net sales increased 9% to $55,000 million in fiscal 2025, and Other "
    "Corp repurchased 12 million shares."
)


@pytest.fixture
def two_companies(tiny_corpus):
    """The tiny corpus plus an indexed second company, OTHR."""
    chunk = Chunk(
        chunk_id="OTHR-7-p0000",
        ticker="OTHR",
        accession="0000000002-25-000001",
        item="7",
        section_title="T",
        kind="prose",
        index=0,
        text=OTHER_TEXT,
        word_count=len(OTHER_TEXT.split()),
        chunker="fixed-p250-o50-t120",
    )
    write_chunks([chunk], tiny_corpus["chunks"] / "OTHR.jsonl")
    config = load_retrieval_config(tiny_corpus["retrieval_config"])
    for variant in ("dense", "hybrid-rerank"):  # plain and header-embedded indexes
        build_variant_index(
            variant, config.variants[variant], "OTHR", HashingEmbedder(),
            tiny_corpus["chunks"], tiny_corpus["index"],
        )  # fmt: skip
        build_variant_index(
            variant, config.variants[variant], "EXCO", HashingEmbedder(),
            tiny_corpus["chunks"], tiny_corpus["index"],
        )  # fmt: skip
    return tiny_corpus


def generate(corpus, *extra):
    return runner.main(corpus["generate_args"]("--dry-run", *extra))


def records(corpus):
    return read_candidates(candidates_path("EXCO", corpus["out"]))


def test_modes_append_side_by_side_with_mode_in_id_and_record(two_companies, capsys):
    for mode in ("rag", "no-context", "wrong-context"):
        assert generate(two_companies, "--mode", mode) == 0
    assert "mode wrong-context" in capsys.readouterr().out
    by_mode = {}
    for r in records(two_companies):
        by_mode.setdefault(r.mode, []).append(r)
    assert sorted(by_mode) == ["no-context", "rag", "wrong-context"]
    assert all(len(v) == 3 for v in by_mode.values())
    rag, closed, wrong = by_mode["rag"][0], by_mode["no-context"][0], by_mode["wrong-context"][0]
    assert rag.candidate_id == "EXCO-01@rag@fake:fake-model"
    assert closed.candidate_id == "EXCO-01@no-context@fake:fake-model"
    assert rag.context is None and rag.context_source is None
    # The evidence for labeling is always the passages retrieved for this question.
    assert closed.retrieved == rag.retrieved == wrong.retrieved
    assert closed.context == [] and closed.context_source == "none"
    assert closed.claims and closed.claims[0].claim_id.startswith(closed.candidate_id + "#c")
    for r in by_mode["wrong-context"]:
        assert r.context and r.context_source.startswith(("other-company:", "other-item:"))
        rag_ids = {c.chunk_id for c in r.retrieved}
        assert not rag_ids & {c.chunk_id for c in r.context}
    # Resume works per mode.
    assert generate(two_companies, "--mode", "no-context") == 0
    assert "Wrote 0 candidate(s)" in capsys.readouterr().out


@pytest.mark.parametrize("variant", ["hybrid-rerank", "bm25"])
def test_all_modes_work_with_a_hybrid_or_lexical_retriever(two_companies, variant, capsys):
    """hybrid-rerank: hashing dense on header text + BM25, RRF, then a reranker; bm25:
    no dense stage at all. Every mode retrieves with the configured variant, and
    wrong-context finds the other company through the same retriever."""
    for mode in ("rag", "no-context", "wrong-context"):
        args = ["--variant", variant, "--mode", mode, "--wrong-context-source", "other-company"]
        assert generate(two_companies, *args) == 0
    out = capsys.readouterr().out
    assert f"retriever {variant} (" in out
    found = records(two_companies)
    assert sorted({r.mode for r in found}) == ["no-context", "rag", "wrong-context"]
    assert all(r.retrieval.variant == variant for r in found)
    assert all(r.retrieved for r in found)
    rag = {r.question_id: r for r in found if r.mode == "rag"}
    for r in found:
        assert r.retrieved == rag[r.question_id].retrieved  # same evidence in every mode
        if r.mode == "wrong-context":
            assert r.context_source == "other-company:OTHR"
            assert {c.chunk_id for c in r.context} == {"OTHR-7-p0000"}
    if variant == "hybrid-rerank":
        assert "bm25" in found[0].retrieval.retriever and "rerank=" in found[0].retrieval.retriever


def make_index_retriever(corpus, tickers, variant="dense"):
    return corpus["retriever"](variant, tickers).search


def test_wrong_context_picker_sources(two_companies):
    retrieve = make_index_retriever(two_companies, ["EXCO", "OTHR"])
    question = "What were total net sales in fiscal 2025?"
    correct = retrieve("EXCO", question, 2)
    company = WrongContextPicker(retrieve, ["EXCO", "OTHR"], "other-company")
    hits, source = company.pick("EXCO", question, "EXCO-G01", correct, 2)
    assert source == "other-company:OTHR" and [h.chunk.ticker for h in hits] == ["OTHR"]
    item = WrongContextPicker(retrieve, ["EXCO", "OTHR"], "other-item")
    hits, source = item.pick("EXCO", question, "EXCO-G01", correct, 2)
    correct_items = {h.chunk.item for h in correct}
    assert hits and not {h.chunk.item for h in hits} & correct_items
    assert source.startswith("other-item:EXCO Item ") and [h.rank for h in hits] == [1, 2]
    # Mixed is deterministic per question ID; with no other company it falls back.
    mixed = WrongContextPicker(retrieve, ["EXCO", "OTHR"])
    assert mixed.kind_for("EXCO-G01") == mixed.kind_for("EXCO-G01")
    alone = WrongContextPicker(retrieve, ["EXCO"], "other-company")
    assert alone.pick("EXCO", question, "EXCO-G01", correct, 2)[1].startswith("other-item:")
    nothing = WrongContextPicker(lambda t, q, k: [], ["EXCO"], "other-item")
    with pytest.raises(ConfigError, match="no mismatched passages"):
        nothing.pick("EXCO", question, "EXCO-G01", correct, 2)


def test_prompts_per_mode(two_companies, monkeypatch):
    from groundcheck.llm.fake import FakeProvider

    seen = []
    real = FakeProvider.generate

    def spy(self, request):
        seen.append(request)
        return real(self, request)

    monkeypatch.setattr(FakeProvider, "generate", spy)
    generate(two_companies, "--mode", "no-context", "--limit", "1")
    assert seen[0].system == CLOSED_BOOK_SYSTEM and "[1] (" not in seen[0].prompt
    assert "Company: Example (EXCO)" in seen[0].prompt
    seen.clear()
    generate(two_companies, "--mode", "wrong-context", "--wrong-context-source",
             "other-company", "--limit", "1")  # fmt: skip
    assert seen[0].system == ANSWER_SYSTEM and "(OTHR-7-p0000)" in seen[0].prompt
    assert "EXCO-" not in seen[0].prompt.split("Question:")[0].replace("Company: EXCO", "")


def test_training_excludes_wrong_context_passages(two_companies):
    generate(two_companies, "--mode", "wrong-context", "--wrong-context-source", "other-company")
    assert "OTHR-7-p0000" in candidate_chunk_ids(two_companies["out"])


# --- label-plan and label --plan ---------------------------------------------------------


@pytest.fixture
def all_modes(two_companies, capsys):
    for mode in ("rag", "no-context", "wrong-context"):
        assert generate(two_companies, "--mode", mode) == 0
    capsys.readouterr()
    return two_companies


def plan_args(corpus, *extra):
    return [
        "label-plan",
        "--candidates-dir", str(corpus["out"]),
        "--labels", str(corpus["root"] / "labels.jsonl"),
        "--out", str(corpus["root"] / "plan.json"),
        *extra,
    ]  # fmt: skip


def test_plan_is_balanced_and_deterministic(all_modes):
    candidates = load_candidates(all_modes["out"])
    plan = make_plan(candidates, questions=3)
    chosen = [c for i in plan.candidate_ids for c in candidates if c.candidate_id == i]
    assert sorted(c.mode for c in chosen) == ["no-context", "rag", "wrong-context"]
    assert len({c.qtype for c in chosen}) == 3  # one per question type too
    assert plan.reached and make_plan(candidates, questions=3).candidate_ids == plan.candidate_ids
    assert make_plan(candidates, questions=3, seed=1).candidate_ids  # another tie-break
    by_claims = make_plan(candidates, claims=2)
    assert sum(len(c.claims) for c in candidates if c.candidate_id in by_claims.candidate_ids) >= 2
    big = make_plan(candidates, questions=100)
    assert not big.reached and len(big.candidate_ids) == 9
    # Labeled candidates are not planned again, and they count towards the balance.
    done = {cl.claim_id for c in chosen for cl in c.claims}
    again = make_plan(candidates, questions=3, already_labeled=done)
    assert not set(again.candidate_ids) & set(plan.candidate_ids)
    with pytest.raises(ConfigError, match="exactly one"):
        make_plan(candidates)


def test_label_plan_command_and_label_walks_the_plan(all_modes, capsys, monkeypatch):
    assert main(plan_args(all_modes, "--questions", "3", "--dry-run")) == 0
    assert "plan not written" in capsys.readouterr().out
    assert main(plan_args(all_modes, "--questions", "3")) == 0
    out = capsys.readouterr().out
    assert "By mode (claims):" in out and "Wrote" in out
    plan_path = all_modes["root"] / "plan.json"
    ids = load_plan(plan_path)
    assert json.loads(plan_path.read_text())["target"] == {"questions": 3}

    answers = iter(["k"] * 20)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    shown = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: shown.append(" ".join(map(str, a))))
    label = [
        "label", "--candidates-dir", str(all_modes["out"]),
        "--labels", str(all_modes["root"] / "labels.jsonl"), "--plan", str(plan_path),
    ]  # fmt: skip
    assert main(label) == 0
    headers = [line for line in shown if line.startswith("[")]
    assert [h.split()[2] for h in headers] == [i.split("@")[0] for i in ids]
    assert any("MODE:     no-context" in line for line in shown)


def test_label_plan_errors(all_modes, capsys):
    plan_path = all_modes["root"] / "plan.json"
    plan_path.write_text(json.dumps({"candidate_ids": ["NOPE@rag@fake:fake-model"]}))
    label = [
        "label", "--candidates-dir", str(all_modes["out"]),
        "--labels", str(all_modes["root"] / "labels.jsonl"), "--plan", str(plan_path),
    ]  # fmt: skip
    assert main(label) == 2
    assert "not in" in capsys.readouterr().err
    plan_path.write_text("{oops")
    assert main(label) == 2


def test_label_stats_by_mode(all_modes, capsys):
    from groundcheck.labeling.report import label_stats
    from groundcheck.labeling.schema import LabelRecord

    candidates = load_candidates(all_modes["out"])
    store = LabelStore(all_modes["root"] / "labels.jsonl")
    for c in candidates[:4]:
        claim = c.claims[0]
        store.put(
            LabelRecord(
                claim_id=claim.claim_id, candidate_id=c.candidate_id, question_id=c.question_id,
                ticker=c.ticker, qtype=c.qtype, claim=claim.text, label="unsupported",
                confirmed=True,
            )
        )  # fmt: skip
    text = label_stats(list(store.records.values()), candidates)
    assert "By mode:" in text and "Mode x label:" in text
    assert "Company x label:" in text and "Question type x label:" in text
    mode_rows = text.split("Mode x label:")[1].split("Company x label:")[0]
    assert "rag" in mode_rows and "unsupported" in mode_rows
