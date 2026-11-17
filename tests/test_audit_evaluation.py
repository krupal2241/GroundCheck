"""Regression tests for the audit's evaluation findings (#1, #4, #5, #6, #7, #13).

Each test reproduces the failing example given in the audit report.
"""

import json

import pytest

from groundcheck.cli import main
from groundcheck.labeling.schema import make_span
from groundcheck.verify.base import BaseVerifier, EvidenceChunk, Verification
from groundcheck.verify.costs import Prices
from groundcheck.verify.evaluate import (
    NOT_SHOWN,
    GoldRecord,
    claims_sha256,
    evaluate_verifiers,
    format_markdown,
    load_gold,
)
from groundcheck.verify.hf import count_truncated

CHUNK = EvidenceChunk("X-7-p0000", "Net sales were $900 million in 2025. Costs rose 3% in 2025.")


def gold(claim_id, label, *, tag=None, ticker="EXCO", quote=None, **extra):
    spans = []
    if quote:
        spans = [make_span(CHUNK.chunk_id, CHUNK.text, quote).model_dump()]
    return GoldRecord.model_validate(
        {
            "claim_id": claim_id,
            "ticker": ticker,
            "claim": f"Claim {claim_id}.",
            "label": label,
            "evidence": spans,
            "edge_case": tag,
            "passages": [{"rank": 1, "chunk_id": CHUNK.chunk_id, "text": CHUNK.text}],
            **extra,
        }
    )


class Scripted(BaseVerifier):
    """Returns a fixed verdict per claim: {claim text: Verification}."""

    def __init__(self, name, verdicts, default="unsupported"):
        self.name, self.model = name, "scripted"
        self.verdicts, self.default = verdicts, default

    def _verify(self, claim, chunks):
        return self.verdicts.get(claim) or Verification(label=self.default, score=0.1)


def run(tmp_path, records, verifiers, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "gold.jsonl"
    path.write_text("".join(json.dumps(r.model_dump()) + "\n" for r in records))
    return evaluate_verifiers(
        list(verifiers),
        lambda name: verifiers[name],
        records,
        path,
        tmp_path / "results",
        Prices(),
        echo=lambda s: None,
        **kw,
    )


# --- #1: binary verifiers are not scored three-way inside the breakdowns -----------------


def test_binary_verifier_on_an_all_contradicted_tag_is_not_zero(tmp_path):
    """The audit's example: every numeric_mismatch claim is contradicted, MiniCheck
    rejects all of them (correctly, for a binary model), and the tag showed 0%."""
    records = [gold(f"c{i:02d}", "contradicted", tag="numeric_mismatch") for i in range(20)]
    results = run(tmp_path, records, {"minicheck": Scripted("minicheck", {})})
    group = results.verifiers["minicheck"].by_edge_case["numeric_mismatch"]
    assert group.n == 20 and group.correct == 0 and group.accuracy == 0.0
    assert group.binary_correct == 20 and group.binary_accuracy == 1.0
    md = (tmp_path / "results" / "verifier_eval.md").read_text()
    assert "| numeric_mismatch | 20 | 0.0% | 100.0% |" in md
    assert "not meaningful for a binary model" in md

    from groundcheck.report.build import BINARY_NOTE, fair_accuracy, group_table

    report = results.verifiers["minicheck"]
    assert fair_accuracy(report, group) == 1.0
    table = "\n".join(group_table(results, "by_edge_case", "Edge case"))
    assert "| numeric_mismatch | 100.0% (n=20) |" in table and BINARY_NOTE in table
    # A three-way verifier keeps three-way accuracy in the same table.
    three = run(tmp_path, records, {"prompted": Scripted("prompted", {})})
    cell = "\n".join(group_table(three, "by_edge_case", "Edge case"))
    assert "| numeric_mismatch | 100.0% (n=20) | 0.0% (n=20) |" in cell


# --- #4: evidence agreement, by how the gold span was chosen ----------------------------


def test_evidence_agreement_is_split_by_gold_span_source(tmp_path):
    span = make_span(CHUNK.chunk_id, CHUNK.text, "Net sales were $900 million in 2025.")
    other = make_span(CHUNK.chunk_id, CHUNK.text, "Costs rose 3% in 2025.")
    records = [
        gold("a1", "supported", quote=span.text, evidence_source="suggestion_accepted"),
        gold("a2", "supported", quote=span.text, evidence_source="suggestion_accepted"),
        gold("p1", "supported", quote=span.text, evidence_source="hand_pasted"),
        gold("p2", "supported", quote=other.text, evidence_source="hand_pasted"),
    ]
    hit = Verification(label="supported", score=0.9, evidence=span)
    verifier = Scripted("fake", {r.claim: hit for r in records})
    report = run(tmp_path, records, {"fake": verifier}).verifiers["fake"]
    assert (report.evidence_agreement.eligible, report.evidence_agreement.overlapping) == (4, 3)
    accepted = report.evidence_agreement_by_source["suggestion_accepted"]
    pasted = report.evidence_agreement_by_source["hand_pasted"]
    assert (accepted.eligible, accepted.overlapping, accepted.rate) == (2, 2, 1.0)
    assert (pasted.eligible, pasted.overlapping, pasted.rate) == (2, 1, 0.5)
    md = (tmp_path / "results" / "verifier_eval.md").read_text()
    assert "hand-pasted by the annotator 1 of 2 (50.0%)" in md
    assert "one-key suggestion accepted 2 of 2 (100.0%)" in md


# --- #5: breakdown by label suggestion shown, and by suggesting verifier -----------------


def test_results_are_broken_down_by_suggestion_and_verifier(tmp_path):
    shown = {"label_suggestion_shown": True, "suggestion_verifiers": ["minicheck"]}
    records = [gold(f"s{i}", "supported", **shown) for i in range(3)]
    records += [gold(f"n{i}", "supported") for i in range(2)]
    agree = Verification(label="supported", score=0.9)
    verdicts = {r.claim: agree for r in records[:3]}  # right only where it was shown
    results = run(tmp_path, records, {"minicheck": Scripted("minicheck", verdicts)})
    by = results.verifiers["minicheck"].by_suggestion
    assert set(by) == {NOT_SHOWN, "shown: minicheck"}
    assert by["shown: minicheck"].n == 3 and by["shown: minicheck"].binary_accuracy == 1.0
    assert by[NOT_SHOWN].n == 2 and by[NOT_SHOWN].binary_accuracy == 0.0
    md = (tmp_path / "results" / "verifier_eval.md").read_text()
    assert "By label suggestion shown to the annotator:" in md
    assert "**Anchoring:** 3 of these labels were made while minicheck's own suggestion" in md

    from groundcheck.report.build import suggestion_section

    assert "| shown: minicheck | 100.0% (n=3) |" in "\n".join(suggestion_section(results))
    plain = run(tmp_path / "plain", records[3:], {"fake": Scripted("fake", {})})
    assert suggestion_section(plain) == []


# --- #6: parse errors counted as wrong ---------------------------------------------------


def test_parse_errors_are_also_reported_as_wrong(tmp_path):
    """Ten parse errors on ten gold-unsupported claims used to be ten correct answers."""
    records = [gold(f"u{i:02d}", "unsupported") for i in range(10)]
    records += [gold(f"s{i:02d}", "supported") for i in range(10)]
    failed = Verification(label="unsupported", score=0.5, details={"parse_error": "no JSON"})
    verdicts = {r.claim: failed for r in records[:10]}
    verdicts |= {r.claim: Verification(label="supported", score=0.9) for r in records[10:]}
    report = run(tmp_path, records, {"prompted": Scripted("prompted", verdicts)}).verifiers[
        "prompted"
    ]
    assert report.parse_errors == 10 and report.accuracy == 1.0
    assert report.accuracy_parse_errors_wrong == 0.5
    assert report.binary_accuracy_parse_errors_wrong == 0.5
    md = (tmp_path / "results" / "verifier_eval.md").read_text()
    assert "| prompted | 10 | 50.0% (was 100.0%) | 50.0% (was 100.0%) |" in md


# --- #7: subset runs are flagged ---------------------------------------------------------


def test_a_limit_run_next_to_a_full_run_is_flagged(gold_path, tmp_path, capsys):
    results = tmp_path / "results"
    base = ["eval-verifiers", "--gold", str(gold_path), "--results-dir", str(results)]
    assert main([*base, "--verifier", "fake"]) == 0
    md = (results / "verifier_eval.md").read_text()
    assert "Warning" not in md and "6 confirmed claim(s) evaluated per verifier" in md
    capsys.readouterr()
    assert main([*base, "--verifier", "prompted", "--provider", "fake", "--limit", "4"]) == 0
    assert "NOTE: evaluating a subset (4 of 6 gold claims)" in capsys.readouterr().out
    data = json.loads((results / "verifier_eval.json").read_text())["verifiers"]
    assert data["fake"]["gold_sha256"] == data["prompted"]["gold_sha256"]  # same file...
    assert data["fake"]["claims_sha256"] != data["prompted"]["claims_sha256"]  # ...other claims
    assert data["prompted"]["gold_total"] == 6 and data["prompted"]["n"] == 4
    md = (results / "verifier_eval.md").read_text()
    assert "computed on different sets of claims (fake n=6, prompted n=4" in md
    assert "| 4 of 6 |" in md

    from groundcheck.report.build import verifier_table
    from groundcheck.verify.evaluate import load_results

    table = "\n".join(verifier_table(load_results(results / "verifier_eval.json")))
    assert "different sets of claims" in table


def test_same_size_but_different_claims_is_flagged_too(tmp_path):
    a = [gold("a1", "supported"), gold("a2", "unsupported")]
    b = [gold("b1", "supported"), gold("b2", "unsupported")]
    first = run(tmp_path, a, {"fake": Scripted("fake", {})})
    second = run(tmp_path, b, {"hhem": Scripted("hhem", {})})
    assert set(second.verifiers) == {"fake", "hhem"}
    assert first.verifiers["fake"].claims_sha256 == claims_sha256(["a2", "a1"])
    assert "different sets of claims" in format_markdown(second)


# --- #13: truncated (chunk, claim) pairs are counted -------------------------------------


class WordTokenizer:
    """One token per word plus 3 special tokens; refuses nothing."""

    def __call__(self, texts, claims, truncation=False, **kwargs):
        ids = [
            [0] * (len(t.split()) + len(c.split()) + 3) for t, c in zip(texts, claims, strict=True)
        ]
        return {"input_ids": ids}


def test_count_truncated_pairs():
    texts = ["word " * 600, "short chunk", "word " * 505]
    assert count_truncated(WordTokenizer(), texts, "a five word claim here", 512) == 2
    assert count_truncated(object(), texts, "claim", 512) is None  # no count, no crash


class Truncating(BaseVerifier):
    name, model, max_length = "minicheck", "scripted", 512
    tokenizer = WordTokenizer()

    def _verify(self, claim, chunks):
        return Verification(label="unsupported", score=0.1)

    def _untimed_details(self, claim, chunks):
        from groundcheck.verify.hf import truncation_details

        return truncation_details(self.tokenizer, chunks, claim, self.max_length)


def test_truncated_pairs_are_reported_and_not_timed(tmp_path):
    long_chunk = {"rank": 2, "chunk_id": "X-7-p0001", "text": "word " * 600}
    record = gold("t1", "unsupported")
    record = record.model_copy(
        update={"passages": [*record.passages, type(record.passages[0])(**long_chunk)]}
    )
    results = run(tmp_path, [record, gold("t2", "unsupported")], {"minicheck": Truncating()})
    stats = results.verifiers["minicheck"].truncation
    assert (stats.pairs, stats.truncated_pairs, stats.claims_affected) == (3, 1, 1)
    md = (tmp_path / "results" / "verifier_eval.md").read_text()
    assert "Truncation: 1 of 3 (chunk, claim) input(s)" in md and "(1 of 2 claim(s) affected)" in md
    # A verifier that reports nothing has no truncation entry at all.
    plain = run(tmp_path / "p", [gold("t2", "unsupported")], {"fake": Scripted("fake", {})})
    assert plain.verifiers["fake"].truncation is None


def test_minicheck_and_finetuned_count_through_their_tokenizer(monkeypatch):
    import hf_standins
    from groundcheck.verify.hf import MiniCheckVerifier

    hf_standins.install(monkeypatch)
    verifier = MiniCheckVerifier()
    verifier.tokenizer = WordTokenizer.__call__.__get__(WordTokenizer())  # counting tokenizer
    details = verifier._untimed_details("a claim", [CHUNK, EvidenceChunk("X-2", "w " * 600)])
    assert details == {"pairs": 2, "truncated_pairs": 1}


@pytest.mark.parametrize("name", ["by_edge_case", "by_company", "by_mode", "by_suggestion"])
def test_every_breakdown_has_binary_accuracy(gold_path, tmp_path, name):
    from groundcheck.verify.fake import FakeVerifier

    records = load_gold(gold_path)
    results = evaluate_verifiers(
        ["fake"], lambda n: FakeVerifier(), records, gold_path, tmp_path / "r", Prices(),
        echo=lambda s: None,
    )  # fmt: skip
    groups = getattr(results.verifiers["fake"], name)
    assert groups and all(g.binary_accuracy is not None for g in groups.values())
    assert sum(g.n for g in groups.values()) == len(records)
