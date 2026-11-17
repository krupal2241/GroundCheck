"""Regression tests for the audit's before-labeling findings (#5, #8, #9, #10, #2).

Each test reproduces the failing example given in the audit report.
"""

import json

import pytest

from groundcheck.config import ConfigError, load_companies
from groundcheck.generation import runner
from groundcheck.generation.claims import (
    claim_problems,
    is_abstention,
    is_meta,
    near_duplicate,
    split_claims,
    validate_claims,
)
from groundcheck.generation.pipeline import read_candidates
from groundcheck.generation.prompts import ABSTAIN_TEXT, CLAIMS_SYSTEM, claims_prompt, make_request
from groundcheck.labeling.assist import LabelSuggestion
from groundcheck.labeling.report import check_labels, export_labels, label_stats
from groundcheck.labeling.session import Labeler
from groundcheck.labeling.store import LabelStore, claim_items, load_candidates
from groundcheck.llm.fake import FakeProvider

# --- #5: label-suggestion anchoring survives re-labeling, Enter-keep and export ----------


@pytest.fixture
def candidates(tiny_corpus, capsys):
    assert runner.main(tiny_corpus["generate_args"]("--dry-run")) == 0
    capsys.readouterr()
    return load_candidates(tiny_corpus["out"])


class Script:
    def __init__(self, *answers):
        self.answers = list(answers)

    def __call__(self, prompt):
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


def labeler(path, script, **kw):
    return Labeler(LabelStore(path), script, lambda text: None, lambda: "2026-11-16T00:00:00", **kw)


def suggester(label, name="minicheck"):
    return lambda claim, chunks: LabelSuggestion(label, 0.9, name)


LABEL_SUPPORTED = ("s", "a", "", "", "", "", "")  # accept the evidence suggestion, confirm


def saved(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_relabeling_without_suggest_never_erases_a_shown_suggestion(candidates, tmp_path):
    path = tmp_path / "labels.jsonl"
    item = claim_items(candidates)[:1]
    labeler(path, Script(*LABEL_SUPPORTED), label_suggester=suggester("supported")).run(item)
    assert saved(path)[0]["assistance"]["label_suggestion_shown"] is True
    # Second pass, no --suggest: the audit's example. The new `assistance` says nothing
    # was shown, but the history keeps what the annotator saw the first time.
    labeler(path, Script("u", "", "", "")).run(item)
    record = saved(path)[0]
    assert record["label"] == "unsupported"
    assert record["assistance"]["label_suggestion_shown"] is False
    assert [h["suggestion_verifier"] for h in record["suggestion_history"]] == ["minicheck"]
    assert record["suggestion_history"][0]["label_suggestion_shown"] is True
    stats = label_stats(list(LabelStore(path).records.values()), candidates)
    assert "shown for 1 label(s)" in stats and "none shown" not in stats
    # A third pass keeps the history too: marking the claim invalid does not drop it.
    labeler(path, Script("x", "duplicate")).run(item)
    assert len(saved(path)[0]["suggestion_history"]) == 1


def test_enter_keeps_the_label_but_records_the_suggestion_seen(candidates, tmp_path):
    path = tmp_path / "labels.jsonl"
    item = claim_items(candidates)[:1]
    labeler(path, Script(*LABEL_SUPPORTED)).run(item)  # labeled with no suggestion
    before = saved(path)[0]
    assert before["suggestion_history"] == []
    result = labeler(path, Script(""), label_suggester=suggester("unsupported", "hhem")).run(item)
    assert result.kept == 1
    after = saved(path)[0]
    assert after["label"] == before["label"] and after["assistance"] == before["assistance"]
    (seen,) = after["suggestion_history"]
    assert seen["kept_existing"] is True and seen["suggestion_verifier"] == "hhem"
    assert seen["suggested_label"] == "unsupported" and seen["agreed"] is False
    # Enter with no suggestion on screen changes nothing.
    labeler(path, Script("")).run(item)
    assert saved(path)[0] == after
    stats = label_stats(list(LabelStore(path).records.values()), candidates)
    assert "shown for 1 label(s)" in stats and "kept with Enter after a suggestion" in stats


def test_export_carries_assistance_for_eval_verifiers(candidates, tiny_corpus, tmp_path):
    path = tmp_path / "labels.jsonl"
    items = claim_items(candidates)[:2]
    script = Script(*LABEL_SUPPORTED, "u", "", "", "")
    labeler(path, script, label_suggester=suggester("supported")).run(items[:1])
    labeler(path, script).run(items[1:])
    records, problems = check_labels(path, candidates)
    assert not problems
    out = tmp_path / "gold.jsonl"
    assert export_labels(records, candidates, out) == 2
    by_label = {e["label"]: e for e in saved(out)}
    shown, plain = by_label["supported"], by_label["unsupported"]
    assert shown["label_suggestion_shown"] is True and shown["suggestion_verifiers"] == [
        "minicheck"
    ]
    assert shown["assistance"]["suggested_label"] == "supported"
    assert shown["evidence_source"] == "suggestion_accepted"
    assert plain["label_suggestion_shown"] is False and plain["suggestion_verifiers"] == []
    assert plain["evidence_source"] == "none" and plain["suggestion_history"] == []


def test_hand_pasted_evidence_is_exported_as_such(candidates, tmp_path):
    path = tmp_path / "labels.jsonl"
    script = Script("s", "$12,345 million", "", "", "", "", "")
    labeler(path, script).run(claim_items(candidates)[:1])
    records, _ = check_labels(path, candidates)
    out = tmp_path / "gold.jsonl"
    export_labels(records, candidates, out)
    assert saved(out)[0]["evidence_source"] == "hand_pasted"


# --- #8: abstentions and claims about the passages or the model --------------------------

GREATER_CHINA = (
    "The passages do not contain information about Apple's net sales in the Greater China region."
)
NO_CONTEXT = "Apple does not have information about its cash spent on share repurchases."


@pytest.mark.parametrize(
    "answer",
    [
        ABSTAIN_TEXT,
        GREATER_CHINA,  # the real AAPL candidate from the audit
        NO_CONTEXT,
        NO_CONTEXT[:-1] + " in fiscal 2025.",
        "The provided context doesn't mention Apple's dividend.",
        "I cannot determine Apple's revenue from memory. No information is available.",
    ],
)
def test_reworded_abstentions_are_abstentions(answer):
    assert is_abstention(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "The passages do not contain the total. However, net sales were $5 billion in 2025.",
        "Apple did not pay a dividend in fiscal 2025.",
        "Apple does not have any outstanding borrowings under the facility.",
        "Amazon does not disclose the number of Prime members.",
        "",
    ],
)
def test_answers_with_a_fact_are_not_abstentions(answer):
    assert not is_abstention(answer)


@pytest.mark.parametrize(
    "claim",
    [
        "The passages do not contain information about Apple's net sales in Japan.",
        "Apple does not have information about its cash spent on share repurchases.",
        "Apple's dividend per share is not mentioned in the passages.",
        "I am unable to find Apple's research and development expense.",
    ],
)
def test_meta_claims_are_rejected(claim):
    assert is_meta(claim)
    reasons = claim_problems(claim, claim, [])
    assert any("not the company" in r for r in reasons)


def test_company_negations_are_still_claims():
    for claim in (
        "Apple did not pay a dividend in fiscal 2025.",
        "Apple has no employees covered by collective bargaining agreements.",
        "Walmart was not profitable in the first quarter of fiscal 2026.",
        "The product was not available in China during fiscal 2025.",
    ):
        assert not is_meta(claim) and claim_problems(claim, claim, []) == []


def request_for(answer):
    return make_request(CLAIMS_SYSTEM, claims_prompt("Q?", answer, "Apple"), 0.0, 42, 256, 4096)


def as_json(claims):
    return json.dumps({"claims": claims})


def test_a_reworded_abstention_is_not_pushed_to_invent_claims():
    # Before: an empty split for this answer was retried three times with "You returned
    # no claims, but the ANSWER states facts".
    fake = FakeProvider(script=[as_json([])])
    claims, _, stats = split_claims(fake, request_for(GREATER_CHINA), GREATER_CHINA)
    assert claims == [] and stats.retries == 0 and len(fake.calls) == 1
    # And if the model returns the abstention as a "claim", it is dropped, not kept.
    fake = FakeProvider(script=[as_json([NO_CONTEXT])])
    claims, _, stats = split_claims(fake, request_for(NO_CONTEXT), NO_CONTEXT)
    assert claims == [] and stats.rejected == 1 and len(fake.calls) == 1


# --- #9: opposite claims are not near-duplicates ----------------------------------------

UP = (
    "Apple's total net sales in the Americas segment increased 4% in fiscal 2025 compared "
    "with fiscal 2024."
)


def test_direction_and_negation_pairs_are_kept_apart():
    down = UP.replace("increased", "decreased")
    assert not near_duplicate(UP, down)
    assert not near_duplicate(
        "Apple did not pay a dividend in 2025.", "Apple did pay a dividend in 2025."
    )
    assert not near_duplicate(
        "Apple's gross margin was higher in fiscal 2025 than in fiscal 2024 overall.",
        "Apple's gross margin was lower in fiscal 2025 than in fiscal 2024 overall.",
    )
    kept, rejected = validate_claims([UP, down], f"{UP} {down}")
    assert kept == [UP, down] and not rejected
    # A plain rewording with the same numbers is still a near-duplicate.
    assert near_duplicate(UP, UP.replace("compared with", "compared to"))


# --- #10: the resume guard covers prompt version, top_k, embedder and retriever -----------


def test_resume_refuses_another_prompt_version_or_top_k(tiny_corpus, capsys, monkeypatch):
    assert runner.main(tiny_corpus["generate_args"]("--dry-run", "--limit", "1")) == 0
    capsys.readouterr()
    assert runner.main(tiny_corpus["generate_args"]("--dry-run", "--top-k", "2")) == 2
    err = capsys.readouterr().err
    assert "made with top_k" in err and "--out-dir" in err
    monkeypatch.setattr(runner, "PROMPT_VERSION", "v99")
    assert runner.main(tiny_corpus["generate_args"]("--dry-run")) == 2
    assert "made with prompt version 'v2', not 'v99'" in capsys.readouterr().err
    monkeypatch.undo()
    assert runner.main(tiny_corpus["generate_args"]("--dry-run")) == 0  # same settings resume


def test_resume_refuses_another_embedder_or_retriever(tiny_corpus, capsys):
    """The audit's example: a smoke run with --embedder hashing under the variant name
    `hybrid`, then the real run skipping those records."""
    assert runner.main(tiny_corpus["generate_args"]("--dry-run", "--limit", "1")) == 0
    capsys.readouterr()
    record = read_candidates(tiny_corpus["out"] / "EXCO.jsonl")[0]
    questions = [type("Q", (), {"id": record.question_id, "ticker": "EXCO"})()]
    provider = type("P", (), {"name": "fake", "model": record.answer_model.model})()
    same = {
        "variant": record.retrieval.variant,
        "prompt_version": record.prompt_version,
        "top_k": record.retrieval.top_k,
        "embedder": record.retrieval.embedder,
        "retriever": record.retrieval.retriever,
    }
    runner._refuse_mixed_settings(questions, provider, tiny_corpus["out"], "rag", **same)
    for field, value, text in (
        ("embedder", "BAAI/bge-small-en-v1.5", "made with embedder 'hashing-256'"),
        ("retriever", "dense=hashing-256 bm25 rrf_k=10 pool=30", "made with retriever"),
    ):
        with pytest.raises(ConfigError, match=text):
            runner._refuse_mixed_settings(
                questions, provider, tiny_corpus["out"], "rag", **(same | {field: value})
            )


# --- #2: wrong-context never borrows an unlabeled or stress-test company's passages -------


def test_wrong_context_companies_are_the_labeled_ones(tiny_corpus):
    from groundcheck.config import DEFAULT_COMPANIES_PATH

    assert runner.wrong_context_companies(load_companies(tiny_corpus["config"])) == {"EXCO"}
    real = runner.wrong_context_companies(load_companies(DEFAULT_COMPANIES_PATH))
    assert "JPM" not in real and "BRK-B" not in real and "AAPL" in real


def test_wrong_context_never_uses_unlabeled(tiny_corpus, capsys):
    """OTHR (role: unlabeled, like JPM) has chunks and BM25 can search it, but its
    passages are never shown as another company's."""
    other = read_chunk(tiny_corpus).model_copy(
        update={"chunk_id": "OTHR-7-p0000", "ticker": "OTHR"}
    )
    from groundcheck.chunking.base import write_chunks

    write_chunks([other], tiny_corpus["chunks"] / "OTHR.jsonl")
    args = tiny_corpus["generate_args"]("--dry-run", "--variant", "bm25", "--mode", "wrong-context")
    # The tiny filing has no chunk from another Item either, so every question fails
    # loudly instead of quietly borrowing OTHR's passages (before the fix: 3 records with
    # context_source "other-company:OTHR").
    assert runner.main(args) == 1
    out = capsys.readouterr().out
    assert out.count("no mismatched passages possible (no other labeled company") == 3
    assert not read_candidates(tiny_corpus["out"] / "EXCO.jsonl")


def read_chunk(corpus):
    from groundcheck.chunking.base import read_chunks

    return read_chunks(corpus["chunks"] / "EXCO.jsonl")[0]
