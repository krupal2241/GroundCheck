"""Labeling assistance: evidence-span suggestion and the optional label suggestion."""

import json

import pytest

from groundcheck.cli import main
from groundcheck.generation import runner
from groundcheck.labeling.assist import LabelSuggestion, suggest_span, verifier_suggester
from groundcheck.labeling.report import label_stats
from groundcheck.labeling.session import Labeler
from groundcheck.labeling.store import LabelStore, claim_items, load_candidates
from groundcheck.retrieval.evaluate import normalize_for_match
from groundcheck.verify.base import VerifierError
from groundcheck.verify.factory import make_verifier


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


def labeler(path, script, lines, **kw):
    return Labeler(LabelStore(path), script, lines.append, lambda: "2026-11-04T00:00:00", **kw)


def test_span_suggestion_picks_the_unit_with_most_numbers_and_words(candidates):
    retrieved = candidates[0].retrieved
    s = suggest_span("Total net sales were $12,345 million in fiscal 2025.", retrieved)
    assert s.span.chunk_id == "EXCO-7-p0000"
    assert s.span.text.startswith("Total net sales increased 6% to $12,345 million")
    chunk = next(c for c in retrieved if c.chunk_id == s.span.chunk_id)
    assert normalize_for_match(chunk.text)[s.span.start : s.span.end] == s.span.text
    row = suggest_span("Operating income was $3,210 million.", retrieved)
    assert row.span.text == "Operating income | $3,210 | $2,987" and 0 < row.score <= 1
    assert suggest_span("Zebras graze quietly.", retrieved) is None


def test_one_key_accept_records_assistance(candidates, tmp_path):
    path = tmp_path / "labels.jsonl"
    lines = []
    item = [i for i in claim_items(candidates) if i.candidate.question_id == "EXCO-G01"]
    # s, "a" + empty line accepts the suggestion, empty line finishes the evidence,
    # no edge case, no note, confirm.
    result = labeler(path, Script("s", "a", "", "", "", "", ""), lines).run(item)
    assert result.labeled == 1
    saved = json.loads(path.read_text())
    assert saved["assistance"]["evidence_suggested"] and saved["assistance"]["evidence_accepted"]
    assert not saved["assistance"]["label_suggestion_shown"]
    assert saved["evidence"][0]["chunk_id"].startswith("EXCO-")
    shown = "\n".join(lines)
    assert "SUGGESTED EVIDENCE from [" in shown and "SUGGESTED LABEL" not in shown  # default off


def test_evidence_suggestion_can_be_turned_off(candidates, tmp_path):
    lines = []
    script = Script("s", "$12,345 million", "", "", "", "", "")
    labeler(tmp_path / "l.jsonl", script, lines, suggest_evidence=False).run(
        claim_items(candidates)[:1]
    )
    assert "SUGGESTED EVIDENCE" not in "\n".join(lines)
    saved = json.loads((tmp_path / "l.jsonl").read_text())
    assert saved["assistance"]["evidence_suggested"] is False


def test_label_suggestion_is_recorded_with_agreement_and_reported(candidates, tmp_path):
    path = tmp_path / "labels.jsonl"
    lines = []
    fixed = iter(["supported", "unsupported", "supported"])

    def suggester(claim, chunks):
        return LabelSuggestion(next(fixed), 0.9, "scripted")

    items = claim_items(candidates)
    script = Script(
        "s", "a", "", "", "", "", "",  # agrees with "supported"
        "s", "a", "", "", "", "", "",  # disagrees with "unsupported"
        "x", "fragment",  # invalid: no agreement recorded
    )  # fmt: skip
    labeler(path, script, lines, label_suggester=suggester).run(items)
    saved = [json.loads(line)["assistance"] for line in path.read_text().splitlines()]
    assert [a["suggested_label"] for a in saved] == ["supported", "unsupported", "supported"]
    assert [a["agreed"] for a in saved] == [True, False, None]
    assert all(
        a["label_suggestion_shown"] and a["suggestion_verifier"] == "scripted" for a in saved
    )
    assert "SUGGESTED LABEL (scripted): supported" in "\n".join(lines)

    text = label_stats(list(LabelStore(path).records.values()), candidates)
    assert "shown for 2 label(s); you agreed with 1 (50%)" in text
    assert "with suggestion (n=2) vs without (n=0)" in text
    assert "Evidence suggestions: offered for 2 label(s), accepted 2 (100%)" in text


def test_verifier_suggester_and_failure():
    from groundcheck.generation.pipeline import RetrievedChunk

    chunk = RetrievedChunk(
        rank=1, chunk_id="X-1", score=1.0, item="7", kind="prose",
        text="Total net sales were $12,345 million in fiscal 2025.",
    )  # fmt: skip
    suggest = verifier_suggester(make_verifier("fake"))
    result = suggest("Total net sales were $12,345 million in fiscal 2025.", [chunk])
    assert result.label == "supported" and result.verifier == "fake"

    class Broken:
        name = "broken"

        def verify(self, claim, chunks):
            raise VerifierError("down")

    assert verifier_suggester(Broken())("c", [chunk]) is None


def test_label_command_suggest_flag(candidates, tiny_corpus, monkeypatch):
    answers = iter(["k", "k", "k"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    shown = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: shown.append(" ".join(map(str, a))))
    args = [
        "label", "--candidates-dir", str(tiny_corpus["out"]),
        "--labels", str(tiny_corpus["root"] / "labels.jsonl"), "--suggest", "fake",
    ]  # fmt: skip
    assert main(args) == 0
    text = "\n".join(shown)
    assert "Label suggestions ON (fake)" in text and "SUGGESTED LABEL (fake):" in text
