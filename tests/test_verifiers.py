"""Verifiers: the protocol, evidence spans, the fake, the prompted LLM verifier (over a
fake provider), and MiniCheck / HHEM over stand-in torch and transformers modules."""

import json
import sys

import pytest

import hf_standins
from groundcheck.config import LLMConfig
from groundcheck.llm import factory as llm_factory
from groundcheck.llm.base import PaidProviderRefused
from groundcheck.llm.fake import FakeProvider
from groundcheck.verify.base import EvidenceChunk, Verification, VerifierError
from groundcheck.verify.factory import VERIFIER_NAMES, VerifierOptions, make_verifier
from groundcheck.verify.fake import FakeVerifier
from groundcheck.verify.hf import (
    HHEM_MODEL,
    MINICHECK_MODEL,
    HHEMVerifier,
    MiniCheckVerifier,
    pick_device,
    support_index,
)
from groundcheck.verify.prompted import PromptedLLMVerifier, verify_prompt
from groundcheck.verify.spans import best_unit_span, find_quote, units

TABLE = (
    "[TABLE]\n | 2025 | 2024\nTotal net sales | $12,345 | $11,646\n"
    "Operating income | $3,210 | $2,987\n[/TABLE]"
)
PROSE = (
    "Total net sales increased 6% to $12,345 million in fiscal 2025. "
    "The increase was partially offset by lower services revenue."
)
RISK = "The Company faces intense competition and may not maintain its gross margin."
CHUNKS = [
    EvidenceChunk("EXCO-7-t0000", TABLE),
    EvidenceChunk("EXCO-7-p0000", PROSE),
    EvidenceChunk("EXCO-1A-p0000", RISK),
]
SUPPORTED = "Total net sales increased 6% to $12,345 million in fiscal 2025."
CONTRADICTED = "Total net sales increased 9% to $13,000 million in fiscal 2025."
UNSUPPORTED = "The Company opened 40 new stores in Brazil."


# --- Spans ------------------------------------------------------------------------------


def test_units_are_table_rows_and_sentences():
    assert units(TABLE) == [
        "| 2025 | 2024",
        "Total net sales | $12,345 | $11,646",
        "Operating income | $3,210 | $2,987",
    ]
    assert units(PROSE)[1] == "The increase was partially offset by lower services revenue."


def test_best_unit_span_picks_the_row_with_the_numbers():
    span = best_unit_span("Operating income was $3,210 million in 2025.", "T", TABLE)
    assert span.text == "Operating income | $3,210 | $2,987"
    normalized = " ".join(TABLE.split())
    assert normalized[span.start : span.end] == span.text


def test_find_quote_tolerates_whitespace_and_reports_misses():
    pairs = [(c.chunk_id, c.text) for c in CHUNKS]
    span = find_quote("Total net sales |  $12,345 | $11,646", pairs)
    assert span.chunk_id == "EXCO-7-t0000"
    assert find_quote("not in any chunk", pairs) is None


# --- Base and fake ----------------------------------------------------------------------


def test_fake_verifier_rule_gives_all_three_labels_with_latency():
    fake = FakeVerifier()
    supported = fake.verify(SUPPORTED, CHUNKS)
    assert supported.label == "supported" and supported.evidence.chunk_id == "EXCO-7-p0000"
    assert supported.latency_ms >= 0 and 0 <= supported.score <= 1
    contradicted = fake.verify(CONTRADICTED, CHUNKS)
    assert contradicted.label == "contradicted" and contradicted.evidence is not None
    unsupported = fake.verify(UNSUPPORTED, CHUNKS)
    assert unsupported.label == "unsupported" and unsupported.evidence is None


def test_fake_verifier_scripted_labels_and_edge_inputs():
    fake = FakeVerifier(labels={"A claim.": "contradicted"})
    assert fake.verify("A   claim.", CHUNKS).label == "contradicted"  # whitespace-normalized
    assert fake.verify("Another.", CHUNKS).label == "unsupported"
    no_chunks = fake.verify("A claim.", [])
    assert no_chunks.label == "unsupported" and no_chunks.details == {"no_chunks": True}
    with pytest.raises(VerifierError, match="empty"):
        fake.verify("   ", CHUNKS)


def test_verification_score_is_bounded():
    with pytest.raises(ValueError):
        Verification(label="supported", score=1.5)


# --- Prompted LLM verifier --------------------------------------------------------------


def prompted(*replies, config=None):
    provider = FakeProvider(script=list(replies))
    return PromptedLLMVerifier(provider, config or LLMConfig()), provider


def test_prompt_is_few_shot_with_numbered_passages():
    prompt = verify_prompt(SUPPORTED, CHUNKS)
    assert prompt.count('"label": "supported"') >= 1 and "EXAMPLES:" in prompt
    assert "[2] (EXCO-7-p0000)\n" + PROSE in prompt
    assert prompt.rstrip().endswith('"confidence": <0 to 1>}')
    assert prompt.index("EXAMPLES:") < prompt.index("NOW THE REAL TASK")


def test_prompted_maps_json_to_a_verification():
    reply = {
        "label": "Contradicted",
        "passage": 1,
        "evidence": "Total net sales | $12,345 | $11,646",
        "confidence": 0.8,
    }
    verifier, provider = prompted("Sure:\n```json\n" + json.dumps(reply) + "\n```")
    result = verifier.verify(CONTRADICTED, CHUNKS)
    assert result.label == "contradicted" and result.score == pytest.approx(0.2)
    assert result.evidence.chunk_id == "EXCO-7-t0000"
    assert result.details["quote_found"] is True and result.details["attempts"] == 1
    request = provider.calls[0]
    assert request.json_mode and request.temperature == 0 and request.seed == 42


def test_prompted_wrong_passage_number_still_finds_the_quote_elsewhere():
    reply = {"label": "supported", "passage": 3, "evidence": SUPPORTED, "confidence": 0.9}
    verifier, _ = prompted(json.dumps(reply))
    result = verifier.verify(SUPPORTED, CHUNKS)
    assert result.evidence.chunk_id == "EXCO-7-p0000" and result.score == pytest.approx(0.9)


def test_prompted_hallucinated_quote_gives_no_span():
    reply = {"label": "supported", "passage": 2, "evidence": "Sales were great.", "confidence": 1}
    verifier, _ = prompted(json.dumps(reply))
    result = verifier.verify(SUPPORTED, CHUNKS)
    assert result.label == "supported" and result.evidence is None
    assert result.details["quote_found"] is False and result.details["quote"] == "Sales were great."


def test_prompted_retries_then_reports_a_parse_error():
    verifier, provider = prompted("nope", '{"label": "maybe"}', "still nope")
    result = verifier.verify(SUPPORTED, CHUNKS)
    assert result.label == "unsupported" and "parse_error" in result.details
    assert len(provider.calls) == 3
    assert "label" in provider.calls[2].prompt.split("could not be used:")[1]


def test_prompted_unsupported_cites_nothing():
    reply = {"label": "unsupported", "passage": None, "evidence": None, "confidence": 0.7}
    verifier, _ = prompted(json.dumps(reply))
    result = verifier.verify(UNSUPPORTED, CHUNKS)
    assert result.label == "unsupported" and result.evidence is None
    assert result.score == pytest.approx(0.3)


def test_make_prompted_verifier_goes_through_the_paid_guard(monkeypatch, tmp_path):
    monkeypatch.setitem(
        llm_factory.PROVIDERS,
        "pricey",
        (lambda config, model, **kw: FakeProvider(paid=True), True),
    )
    with pytest.raises(PaidProviderRefused):
        make_verifier("prompted", VerifierOptions(provider="pricey"))
    verifier = make_verifier("prompted", VerifierOptions(provider="fake"))
    assert verifier.local and verifier.verify(SUPPORTED, CHUNKS).label == "unsupported"
    assert verifier.identity()["model_digest"] == "fake-digest"


# --- MiniCheck and HHEM (stand-in modules) ------------------------------------------------


def test_minicheck_is_binary_and_scores_each_chunk(monkeypatch):
    recorder = hf_standins.install(monkeypatch)
    verifier = MiniCheckVerifier()
    assert recorder.loads[0] == ("AutoTokenizer", MINICHECK_MODEL, {"revision": None})
    assert verifier.device == "cpu" and verifier.revision == "commit123"
    supported = verifier.verify(SUPPORTED, CHUNKS)
    assert supported.label == "supported" and supported.score > 0.99
    assert supported.evidence.chunk_id == "EXCO-7-p0000"
    assert supported.details["best_chunk"] == "EXCO-7-p0000"
    texts, claims, kwargs = recorder.tokenized[-1]
    assert texts == [c.text for c in CHUNKS] and claims == [SUPPORTED] * 3
    assert kwargs["truncation"] == "only_first" and kwargs["max_length"] == 512
    # A binary model never says contradicted: a wrong number is just "not supported".
    contradicted = verifier.verify(CONTRADICTED, CHUNKS)
    assert contradicted.label == "unsupported" and contradicted.evidence is None
    assert contradicted.score < 0.01


def test_minicheck_threshold_and_named_labels(monkeypatch):
    hf_standins.install(monkeypatch, id2label={0: "supported", 1: "unsupported"})
    verifier = MiniCheckVerifier(threshold=0.999999)
    assert verifier.support_index == 0
    assert verifier.verify(SUPPORTED, CHUNKS).label == "unsupported"  # 0.9975 < threshold


def test_support_index_rules():
    assert support_index({0: "LABEL_0", 1: "LABEL_1"}) == 1
    assert support_index({0: "entailment", 1: "not_entailment"}) == 0
    assert support_index(None) == 1
    with pytest.raises(VerifierError, match="binary"):
        support_index({0: "a", 1: "b", 2: "c"})


def test_minicheck_refuses_generative_variants():
    with pytest.raises(VerifierError, match="sequence-classification"):
        MiniCheckVerifier("lytang/MiniCheck-Flan-T5-Large")


def test_hhem_passes_premise_then_claim_and_trusts_remote_code(monkeypatch):
    recorder = hf_standins.install(monkeypatch, mps=True)
    verifier = HHEMVerifier(revision="abc")
    name, kwargs = recorder.loads[0][1:]
    assert name == HHEM_MODEL and kwargs == {"trust_remote_code": True, "revision": "abc"}
    assert verifier.device == "mps" and recorder.devices == ["mps"]
    result = verifier.verify(SUPPORTED, CHUNKS)
    assert result.label == "supported" and result.score == pytest.approx(0.9)
    assert recorder.predicted[-1][1] == (PROSE, SUPPORTED)
    assert verifier.verify(UNSUPPORTED, CHUNKS).label == "unsupported"
    assert verifier.identity()["model_revision"] == "abc"


@pytest.mark.parametrize(
    ("cuda", "mps", "requested", "expected"),
    [
        (False, False, None, "cpu"),
        (False, True, None, "mps"),
        (True, True, None, "cuda"),
        (False, True, "cpu", "cpu"),
    ],
)
def test_device_auto_select(cuda, mps, requested, expected):
    assert pick_device(hf_standins.make_torch(cuda=cuda, mps=mps), requested) == expected


@pytest.mark.parametrize("name", ["minicheck", "hhem"])
def test_hf_verifiers_without_the_extra_are_a_clean_error(monkeypatch, name):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(VerifierError, match=r"\[verifiers\] extra"):
        make_verifier(name)


def test_factory_names_and_unknown():
    assert VERIFIER_NAMES == ("fake", "prompted", "minicheck", "hhem")
    assert isinstance(make_verifier("fake"), FakeVerifier)
    with pytest.raises(VerifierError, match="unknown verifier"):
        make_verifier("oracle")
