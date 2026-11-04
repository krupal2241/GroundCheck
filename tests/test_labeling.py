"""Label schema, the interactive labeler (scripted input), stats, checks and export."""

import json

import pytest
from pydantic import ValidationError

from groundcheck.cli import main
from groundcheck.config import ConfigError
from groundcheck.generation import runner
from groundcheck.labeling.report import check_labels, label_stats
from groundcheck.labeling.schema import EDGE_CASES, EvidenceSpan, LabelRecord, make_span
from groundcheck.labeling.session import Labeler
from groundcheck.labeling.store import LabelStore, claim_items, load_candidates

CHUNK = "Total net sales increased 6% to\n$12,345 million in fiscal 2025."


def record(**overrides):
    span = make_span("X-7-p0000", CHUNK, "$12,345 million")
    data = {
        "claim_id": "Q1@fake:m#c01",
        "candidate_id": "Q1@fake:m",
        "question_id": "Q1",
        "ticker": "EXCO",
        "qtype": "numeric_lookup",
        "claim": "Net sales were $12,345 million.",
        "label": "supported",
        "evidence": [span.model_dump()],
        "confirmed": True,
    }
    data.update(overrides)
    return data


# --- Schema -----------------------------------------------------------------------------


def test_valid_record_and_context_check():
    chunks = {"X-7-p0000": CHUNK}
    parsed = LabelRecord.model_validate(record(), context={"chunks": chunks})
    assert parsed.evidence[0].text == "$12,345 million"
    assert parsed.evidence[0].start == "Total net sales increased 6% to $12,345".index("$")


def test_edge_case_list_matches_the_guide():
    assert EDGE_CASES == (
        "negation",
        "numeric_mismatch",
        "rounding",
        "units_scale",
        "table_misread",
        "temporal",
        "partial_support",
        "hedging",
        "absent_evidence",
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"evidence": []}, "needs at least one evidence span"),
        ({"label": "contradicted", "evidence": []}, "needs at least one evidence span"),
        ({"label": "unsupported"}, "must not have evidence"),
        ({"label": "maybe"}, "Input should be"),
        ({"edge_case": "sarcasm"}, "Input should be"),
        ({"edge_case": "absent_evidence"}, "absent_evidence goes with the unsupported"),
        ({"claim": ""}, "at least 1 character"),
        ({"labelled": "x"}, "Extra inputs"),
    ],
)
def test_schema_rules(overrides, message):
    with pytest.raises(ValidationError, match=message):
        LabelRecord.model_validate(record(**overrides))


def test_unsupported_without_evidence_is_valid():
    parsed = LabelRecord.model_validate(
        record(label="unsupported", evidence=[], edge_case="absent_evidence")
    )
    assert parsed.evidence == [] and parsed.edge_case == "absent_evidence"


@pytest.mark.parametrize(
    ("span", "chunks", "message"),
    [
        ({"text": "$12,345  million"}, None, "whitespace-normalized"),
        ({"end": 3}, None, "end - start"),
        ({"chunk_id": "X-8-p0009"}, {"X-7-p0000": CHUNK}, "not one of the candidate's"),
        ({"start": 0, "end": 15}, {"X-7-p0000": CHUNK}, "it occurs at 32"),
        ({"text": "$99,999 million"}, {"X-7-p0000": CHUNK}, "is not at offsets"),
    ],
)
def test_evidence_span_rules(span, chunks, message):
    data = record()
    data["evidence"][0].update(span)
    with pytest.raises(ValidationError, match=message):
        LabelRecord.model_validate(data, context={"chunks": chunks} if chunks else None)


def test_make_span_normalizes_whitespace():
    span = make_span("c", CHUNK, "  6% to \n\n $12,345 ")
    assert span.text == "6% to $12,345" and isinstance(span, EvidenceSpan)
    assert make_span("c", CHUNK, "not there") is None and make_span("c", CHUNK, "   ") is None


# --- Interactive labeler ---------------------------------------------------------------


@pytest.fixture
def candidates(tiny_corpus, capsys):
    """Three dry-run candidates (FakeProvider), one claim each: EXCO-01, -G01, -G02."""
    assert runner.main(tiny_corpus["generate_args"]("--dry-run")) == 0
    capsys.readouterr()
    return load_candidates(tiny_corpus["out"])


class Script:
    """Feeds scripted answers to input(); records prompts; EOF when exhausted."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


def labeler(store, script, lines):
    return Labeler(store, input_fn=script, output=lines.append, now=lambda: "2026-11-04T00:00:00")


def test_scripted_session_quit_and_resume(candidates, tmp_path):
    items = claim_items(candidates)
    assert [i.candidate.question_id for i in items] == ["EXCO-01", "EXCO-G01", "EXCO-G02"]
    path = tmp_path / "labels" / "labels.jsonl"
    lines = []
    script = Script(
        # claim 1: supported, evidence pasted across two lines, no edge case, confirmed
        "s", "Total net sales increased 6% to", "$12,345 million in fiscal 2025", "", "",
        "", "", "",
        # claim 2: contradicted, two table spans, numeric_mismatch, a note, not confirmed
        "c", "Total net sales | $12,345", "", "Operating income | $3,210", "", "",
        "2", "made-up test note", "n",
        # claim 3: quit
        "q",
    )  # fmt: skip
    result = labeler(LabelStore(path), script, lines).run(items)
    assert (result.labeled, result.quit, result.remaining) == (2, True, 1)
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["label"] for r in saved] == ["supported", "contradicted"]
    assert saved[0]["confirmed"] is True and saved[1]["confirmed"] is False
    assert saved[0]["evidence"][0]["chunk_id"] == "EXCO-7-p0000"
    assert (
        saved[0]["evidence"][0]["text"]
        == "Total net sales increased 6% to $12,345 million in fiscal 2025"
    )
    assert [s["chunk_id"] for s in saved[1]["evidence"]] == ["EXCO-7-t0000", "EXCO-7-t0000"]
    assert saved[1]["edge_case"] == "numeric_mismatch" and saved[1]["note"] == "made-up test note"
    assert not path.with_name("labels.jsonl.part").exists()
    shown = "\n".join(lines)
    assert "QUESTION: What was operating income in 2025?" in shown
    assert "[1] EXCO-7-p0000 (Item 7, prose" in shown and "CLAIM:" in shown
    assert "      Total net sales | $12,345 | $11,646" in shown  # table rows shown as rows

    # Resume: only the unlabeled claim is shown.
    lines.clear()
    script = Script("u", "9", "", "y")
    result = labeler(LabelStore(path), script, lines).run(items, only_unlabeled=True)
    assert (result.labeled, result.quit) == (1, False)
    assert "[1/1] EXCO EXCO-G02" in "\n".join(lines)
    third = json.loads(path.read_text().splitlines()[2])
    assert (third["label"], third["evidence"], third["edge_case"]) == (
        "unsupported",
        [],
        "absent_evidence",
    )
    records, problems = check_labels(path, candidates)
    assert problems == [] and len(records) == 3


def test_bad_answers_are_asked_again(candidates, tmp_path):
    item = claim_items(candidates)[:1]
    lines = []
    script = Script(
        "x",  # not a label key
        "s",
        "",  # no span yet: required
        "net sales were $99", "",  # not in any chunk
        "$12,345", "",  # in two chunks: asks which one
        "7",  # not a listed rank
        "2",  # rank 2 = the table
        "$12,345", "", "2",  # the same span again
        "",  # finish
        "9",  # absent_evidence doesn't go with supported
        "banana",  # not an edge case
        "rounding",  # names work too
        "", "",
    )  # fmt: skip
    path = tmp_path / "labels.jsonl"
    assert labeler(LabelStore(path), script, lines).run(item).labeled == 1
    shown = "\n".join(lines)
    assert "Type s, c, u, k or q." in shown
    assert "At least one evidence span is required" in shown
    assert "Not found in any retrieved chunk" in shown
    assert "That span is already cited." in shown
    assert "absent_evidence goes with unsupported" in shown
    assert any("Which one?" in p for p in script.prompts)
    saved = json.loads(path.read_text())
    assert saved["evidence"][0]["chunk_id"] == "EXCO-7-t0000" and saved["edge_case"] == "rounding"


def test_skip_keep_and_end_of_input(candidates, tmp_path):
    items = claim_items(candidates)
    path = tmp_path / "labels.jsonl"
    labeler(LabelStore(path), Script("u", "", "", ""), []).run(items[:1])
    lines = []
    # claim 1 has a label: Enter keeps it; claim 2 skipped; input ends at claim 3.
    result = labeler(LabelStore(path), Script("", "k"), lines).run(items)
    assert (result.kept, result.skipped, result.quit, result.remaining) == (1, 1, True, 1)
    assert "CURRENT LABEL: unsupported (confirmed)" in "\n".join(lines)
    assert len(path.read_text().splitlines()) == 1


def test_relabeling_replaces_the_record(candidates, tmp_path):
    items = claim_items(candidates)[:1]
    path = tmp_path / "labels.jsonl"
    labeler(LabelStore(path), Script("u", "", "", "n"), []).run(items)
    labeler(LabelStore(path), Script("s", "$12,345 million", "", "", "", "", ""), []).run(items)
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(saved) == 1 and saved[0]["label"] == "supported" and saved[0]["confirmed"]


def test_store_refuses_a_file_with_invalid_lines(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(record()) + "\n" + json.dumps(record(evidence=[])) + "\n")
    with pytest.raises(ConfigError, match="line 2"):
        LabelStore(path)


def test_label_command_with_scripted_input(candidates, tiny_corpus, monkeypatch, capsys):
    script = Script("u", "", "made-up", "")
    monkeypatch.setattr("builtins.input", script)
    labels = tiny_corpus["root"] / "labels.jsonl"
    args = ["label", "--candidates-dir", str(tiny_corpus["out"]), "--labels", str(labels)]
    assert main([*args, "--company", "EXCO", "--only-unlabeled"]) == 0
    out = capsys.readouterr().out
    assert "3 claim(s) in 3 candidate(s); 0 already labeled." in out
    assert "Session: 1 labeled" in out and "quit with 2 left" in out
    assert main([*args, "--company", "OTHR"]) == 2
    assert "no candidates for OTHR" in capsys.readouterr().err


# --- Stats, check, export ---------------------------------------------------------------


@pytest.fixture
def labeled(candidates, tiny_corpus):
    """Labels for all three claims: supported (confirmed), contradicted (draft),
    unsupported (confirmed)."""
    path = tiny_corpus["root"] / "labels.jsonl"
    script = Script(
        "s", "$12,345 million", "", "", "", "", "y",
        "c", "Total net sales | $12,345", "", "", "2", "", "n",
        "u", "9", "", "y",
    )  # fmt: skip
    labeler(LabelStore(path), script, []).run(claim_items(candidates))
    return path


def cli_args(tiny_corpus, labels, command):
    return [command, "--candidates-dir", str(tiny_corpus["out"]), "--labels", str(labels)]


def test_label_stats(labeled, candidates, tiny_corpus, capsys):
    records = list(LabelStore(labeled).records.values())
    text = label_stats(records, candidates)
    assert "Labels: 3 (2 confirmed, 1 not confirmed); claims in candidates: 3, unlabeled: 0" in text
    assert "  supported                1" in text and "  EXCO                     3" in text
    assert "  numeric_lookup           1" in text and "  absent_evidence          1" in text
    assert main(cli_args(tiny_corpus, labeled, "label-stats")) == 0
    assert "By question type:" in capsys.readouterr().out


def test_label_check_passes_and_catches_problems(labeled, tiny_corpus, capsys):
    assert main(cli_args(tiny_corpus, labeled, "label-check")) == 0
    assert "3 valid label record(s)" in capsys.readouterr().out
    lines = labeled.read_text().splitlines()
    tampered = json.loads(lines[0])
    tampered["evidence"][0]["text"] = "$12,346 million"  # no longer in the chunk
    ghost = json.loads(lines[2]) | {"claim_id": "EXCO-99@fake:fake-model#c01"}
    changed = json.loads(lines[2]) | {"claim": "Something else."}
    labeled.write_text(
        "\n".join([json.dumps(tampered), lines[1], json.dumps(ghost), json.dumps(changed),
                   lines[1], "{oops"]) + "\n"
    )  # fmt: skip
    assert main(cli_args(tiny_corpus, labeled, "label-check")) == 1
    out = capsys.readouterr().out
    assert "line 1 (EXCO-01@fake:fake-model#c01): evidence.0: Value error, evidence is not" in out
    assert "line 3 (EXCO-99@fake:fake-model#c01): no such claim" in out
    assert "line 4 (EXCO-G02@fake:fake-model#c01): claim text differs" in out
    assert "line 5 (EXCO-G01@fake:fake-model#c01): duplicate claim_id" in out
    assert "line 6: invalid JSON" in out


def test_export_keeps_only_confirmed(labeled, tiny_corpus, capsys):
    out_path = tiny_corpus["root"] / "eval" / "gold_labels.jsonl"
    args = [*cli_args(tiny_corpus, labeled, "export-labels"), "--out", str(out_path)]
    assert main(args) == 0
    assert "Exported 2 confirmed label(s)" in capsys.readouterr().out
    exported = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert [e["label"] for e in exported] == ["supported", "unsupported"]
    assert [e["claim_id"] for e in exported] == sorted(e["claim_id"] for e in exported)
    first = exported[0]
    assert first["question"] and first["answer"] and first["claim"]
    assert [p["chunk_id"] for p in first["passages"]][0] == "EXCO-7-p0000"
    assert first["generator"]["model_digest"] == "fake-digest"
    assert "confirmed" not in first  # every exported label is confirmed
    # One invalid record blocks the export entirely.
    labeled.write_text(labeled.read_text() + "{oops\n")
    before = out_path.read_text()
    assert main(args) == 1
    assert "Nothing exported" in capsys.readouterr().out
    assert out_path.read_text() == before
