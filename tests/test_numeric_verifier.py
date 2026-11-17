"""`NumericVerifier`: the rule-based number-matching baseline, and its place in
`eval-verifiers`."""

import json

import pytest

from groundcheck.cli import main
from groundcheck.verify.base import EvidenceChunk
from groundcheck.verify.factory import VERIFIER_NAMES, make_verifier
from groundcheck.verify.numeric import (
    NumericVerifier,
    chunk_facts,
    direction,
    nearest_year,
    split_quantities,
)

TABLE = EvidenceChunk(
    "A-7-t0000",
    "[TABLE]\n | 2025 | Change | 2024 | Change | 2023\n"
    "Greater China | 64,377 | (4)% | 66,952 | (8)% | 72,559\n"
    "Total net sales | $416,161 | 6% | $391,035 | 2% | $383,285\n[/TABLE]\n(in millions)",
)
PROSE = EvidenceChunk(
    "A-7-p0001",
    "During 2025, the Company repurchased $89.3 billion of its common stock. Total net "
    "sales increased 6% to $12,345 million in fiscal 2025, compared with $11,646 million "
    "in fiscal 2024.",
)
CHUNKS = [TABLE, PROSE]


@pytest.fixture
def verifier():
    return NumericVerifier()


@pytest.mark.parametrize(
    ("claim", "label"),
    [
        # units and scale: billions against a table in millions, with rounding
        ("Apple's net sales in Greater China were $64.4 billion in 2025.", "supported"),
        ("Apple repurchased $89,300 million of its common stock during 2025.", "supported"),
        ("Total net sales were about $12.3 billion in fiscal 2025.", "supported"),
        ("Total net sales were $416.2 billion in 2025 and $391.0 billion in 2024.", "supported"),
        # a table's "(4)%" is a decrease
        ("Apple's net sales in Greater China decreased 4% in 2025.", "supported"),
        ("Apple's net sales in Greater China increased 4% in 2025.", "contradicted"),
        # the right row, the wrong year's value (temporal)
        ("Apple's net sales in Greater China were $66,952 million in 2025.", "contradicted"),
        ("Total net sales were $11,646 million in fiscal 2025.", "contradicted"),
        # a scale error against a table that states its scale (units_scale)
        ("Apple's net sales in Greater China were $64,377 billion in 2025.", "contradicted"),
        # a different number in the same context (numeric_mismatch)
        ("Apple repurchased $95 billion of its common stock during 2025.", "contradicted"),
        # nothing in the passages is about this
        ("Apple opened 40 new stores in Brazil in 2025.", "unsupported"),
    ],
)
def test_verdicts(verifier, claim, label):
    result = verifier.verify(claim, CHUNKS)
    assert result.label == label, result.details
    assert (result.evidence is None) == (label == "unsupported")
    assert 0 <= result.score <= 1 and (result.score >= 0.5) == (label == "supported")


def test_evidence_is_the_row_or_sentence_the_verdict_rests_on(verifier):
    result = verifier.verify(
        "Apple's net sales in Greater China were $64.4 billion in 2025.", CHUNKS
    )
    assert result.evidence.chunk_id == "A-7-t0000"
    assert result.evidence.text == "Greater China | 64,377 | (4)% | 66,952 | (8)% | 72,559"
    assert result.details["matched"] == "64,377"
    prose = verifier.verify("Apple repurchased $95 billion of its common stock in 2025.", CHUNKS)
    assert prose.evidence.text.startswith("During 2025, the Company repurchased $89.3 billion")


def test_claims_without_numbers_get_no_opinion(verifier):
    for claim in ("Apple faces intense competition.", "Net sales rose in fiscal 2025."):
        result = verifier.verify(claim, CHUNKS)  # a year alone is not a number to check
        assert result.label == "unsupported" and result.details["no_numbers"] is True
    assert verifier.verify("Sales were $5 million.", []).details == {"no_chunks": True}


def test_known_limits_are_what_the_docs_say(verifier):
    # Without a scale note, a scale error cannot be seen against a bare table cell.
    bare = EvidenceChunk("B-7-t0000", TABLE.text.replace("\n(in millions)", ""))
    wrong_scale = "Apple's net sales in Greater China were $64,377 billion in 2025."
    assert verifier.verify(wrong_scale, [bare]).label == "supported"
    # A derived number the passage does not print is not computed.
    derived = "Total net sales grew by $25,126 million from 2024 to 2025."
    assert verifier.verify(derived, [TABLE]).label != "supported"


def test_helpers():
    values, years = split_quantities("sales of $5 million in 2025, up from $4 million in 2024")
    assert [nearest_year(v, years) for v in values] == [2025, 2024]
    assert direction("sales increased") == 1 and direction("sales fell") == -1
    assert direction("sales were flat") == 0 and direction("the gain offset the decline") == 0
    facts = chunk_facts(TABLE)
    china = [f for f in facts if "china" in f.context]
    assert [(f.quantity.raw, f.period) for f in china] == [
        (64377, 2025), (-4, None), (66952, 2024), (-8, None), (72559, 2023),
    ]  # fmt: skip
    assert all(f.hint == 1e6 and f.table for f in china)


def test_it_is_a_registered_local_verifier_and_runs_in_eval(gold_path, tmp_path, capsys):
    assert "numeric" in VERIFIER_NAMES
    verifier = make_verifier("numeric")
    assert isinstance(verifier, NumericVerifier) and verifier.local
    assert "no model" in verifier.identity()["model"]
    results = tmp_path / "results"
    args = ["eval-verifiers", "--gold", str(gold_path), "--results-dir", str(results)]
    assert main([*args, "--verifier", "numeric", "fake"]) == 0
    capsys.readouterr()
    data = json.loads((results / "verifier_eval.json").read_text())["verifiers"]
    report = data["numeric"]
    assert report["n"] == 6 and report["local"] and not report["binary_model"]
    assert report["cost_per_1000_checks_usd"] == 0.0 and report["parse_errors"] == 0
    assert report["claims_sha256"] == data["fake"]["claims_sha256"]
    md = (results / "verifier_eval.md").read_text()
    assert "| numeric | `rules: number matching (no model)` | 6 |" in md
    # The tiny gold set: both amounts in the table are found, and the wrong one is caught.
    predictions = [
        json.loads(line)
        for line in (results / "predictions" / "numeric.jsonl").read_text().splitlines()
    ]
    by_id = {p["claim_id"]: p["predicted"] for p in predictions}
    assert by_id["EXCO-G01#c01"] == "supported" and by_id["EXCO-G01#c03"] == "contradicted"
    assert by_id["OTHR-G01#c01"] == "unsupported"
