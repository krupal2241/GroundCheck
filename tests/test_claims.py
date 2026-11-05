"""Claim splitting: the few-shot prompt, the deterministic validator and the retries.

The scripted responses reproduce what llama3.2:3b did on the Mac: eight "claims" for
the share-repurchase answer, five of them fragments, and three clean claims for the
Greater China answer.
"""

import json

import pytest

from groundcheck.config import LLMConfig
from groundcheck.generation.claims import (
    claim_problems,
    has_verb,
    near_duplicate,
    numbers_in,
    split_claims,
    validate_claims,
)
from groundcheck.generation.pipeline import Generator, RetrievalInfo, short_company_name
from groundcheck.generation.prompts import (
    CLAIM_EXAMPLES,
    CLAIMS_SYSTEM,
    PROMPT_VERSION,
    claims_prompt,
    make_request,
)
from groundcheck.generation.questions import GenerationQuestion
from groundcheck.llm.base import ModelIdentity
from groundcheck.llm.cache import cache_key
from groundcheck.llm.fake import FakeProvider
from groundcheck.llm.structured import StructuredOutputError
from groundcheck.retrieval.embedder import HashingEmbedder

BAD_ANSWER = (
    "According to passage [1], the Company repurchased 402 million shares of its common "
    "stock for $89.3 billion in fiscal 2025."
)
# What llama3.2:3b returned on the Mac: 8 claims, 5 of them fragments.
BAD_SPLIT = [
    "The Company repurchased 402 million shares of its common stock in fiscal 2025.",
    "The Company",
    "its common stock",
    "402 million",
    "its common stock",
    "billion",
    "The Company paid $89.3 billion for the repurchased shares in fiscal 2025.",
    "According to passage [1], the Company repurchased shares of its common stock.",
]
GOOD_SPLIT = [
    "Apple repurchased 402 million shares of Apple common stock in fiscal 2025.",
    "Apple paid $89.3 billion for the shares of Apple common stock it repurchased in fiscal 2025.",
]
CHINA_ANSWER = (
    "Apple's net sales in Greater China were $64,377 million in fiscal 2025, compared with "
    "$66,952 million in fiscal 2024, a decrease of 4%."
)
CHINA_SPLIT = [
    "Apple's net sales in Greater China were $64,377 million in fiscal 2025.",
    "Apple's net sales in Greater China were $66,952 million in fiscal 2024.",
    "Apple's net sales in Greater China decreased 4% from fiscal 2024 to fiscal 2025.",
]


def as_json(claims):
    return json.dumps({"claims": claims})


def request(answer, company="Apple"):
    prompt = claims_prompt("How many shares did Apple repurchase?", answer, company)
    return make_request(CLAIMS_SYSTEM, prompt, 0.0, 42, 768, 8192)


# --- The validator ----------------------------------------------------------------------


def test_bad_case_fragments_are_rejected_with_reasons():
    kept, rejected = validate_claims(BAD_SPLIT, BAD_ANSWER)
    assert kept == [BAD_SPLIT[0], BAD_SPLIT[6]]
    reasons = dict(rejected)
    assert set(reasons) == {
        "The Company",
        "its common stock",
        "402 million",
        "billion",
        BAD_SPLIT[7],
    }
    assert len(rejected) == 6  # "its common stock" twice
    assert "a fragment" in reasons["402 million"][0]
    assert any("no verb" in r for r in reasons["billion"])
    assert any("passage citation" in r for r in reasons[BAD_SPLIT[7]])


def test_good_case_passes_untouched():
    assert validate_claims(CHINA_SPLIT, CHINA_ANSWER) == (CHINA_SPLIT, [])
    assert validate_claims(GOOD_SPLIT, BAD_ANSWER) == (GOOD_SPLIT, [])


@pytest.mark.parametrize(
    "claim, earlier, reason",
    [
        ("Apple repurchased 410 million shares in fiscal 2025.", [], "not in the answer: 410"),
        ("Apple repurchased 402 million shares in fiscal 2024.", [], "not in the answer: 2024"),
        ("Apple stock buybacks, 402 million shares, fiscal 2025.", [], "no verb"),
        ("Apple bought back shares (passage 1) in 2025.", [], "passage citation"),
        ("Apple repurchased shares [2] in fiscal 2025.", [], "passage citation"),
        (GOOD_SPLIT[0], [GOOD_SPLIT[0]], "duplicate of claim 1"),
        (
            "Apple repurchased 402 million shares of its common stock in fiscal 2025.",
            [GOOD_SPLIT[0]],
            "near-duplicate of claim 1",
        ),
    ],
)
def test_each_rule(claim, earlier, reason):
    assert any(reason in r for r in claim_problems(claim, BAD_ANSWER, earlier))


def test_helpers():
    assert numbers_in("$12,345 million, 4%, $89.30 and (8)% in 2025.") == {
        "12345",
        "4",
        "89.3",
        "8",
        "2025",
    }
    assert has_verb("Apple's sales decreased.") and has_verb("Net income was $5 million.")
    assert not has_verb("Shares of its common stock") and not has_verb("operating income")
    # Same words but different numbers: two different facts, not a near-duplicate.
    assert not near_duplicate(CHINA_SPLIT[0], CHINA_SPLIT[1])
    assert short_company_name("Apple Inc.") == "Apple"
    assert short_company_name("Amazon.com, Inc.") == "Amazon"
    assert short_company_name("JPMorgan Chase & Co.") == "JPMorgan Chase"
    assert short_company_name("Johnson & Johnson") == "Johnson & Johnson"
    assert short_company_name("Exxon Mobil Corporation") == "Exxon Mobil"


# --- The prompt -------------------------------------------------------------------------


def test_prompt_has_three_examples_rules_and_the_company():
    prompt = claims_prompt("Q?", BAD_ANSWER, "Microsoft")
    assert len(CLAIM_EXAMPLES) == 3
    assert prompt.count("OUTPUT: {") == 3
    assert "COMPANY: Microsoft" in prompt and prompt.endswith('"claims": ["...", "..."]}')
    assert '"the Company"' in prompt and "passage [1]" in prompt
    # The bad case's corrected output is two claims; every example passes the validator.
    assert len(CLAIM_EXAMPLES[0][3]) == 2 and CLAIM_EXAMPLES[0][2] == BAD_ANSWER
    for _, _, answer, claims in CLAIM_EXAMPLES:
        assert validate_claims(claims, answer) == (claims, [])
    assert any("%" in answer for _, _, answer, _ in CLAIM_EXAMPLES[2:])
    assert PROMPT_VERSION == "v2"


V1_CLAIMS_PROMPT = (  # the v1 prompt, verbatim, for the cache-key test
    "QUESTION:\nQ?\n\nANSWER:\n{answer}\n\n"
    "Split the ANSWER into atomic claims. Each claim:\n"
    "- is one standalone factual statement that can be checked on its own;\n"
    "- keeps its number, unit, scale and period exactly as the answer states them;\n"
    "- names the company or item instead of using a pronoun;\n"
    "- adds nothing that is not in the answer.\n"
    "If the answer makes no factual claim (it only says the information is not "
    'available), return an empty list.\nReturn only: {{"claims": ["...", "..."]}}'
)


def test_cache_key_changes_with_the_prompt():
    identity = ModelIdentity("ollama", "llama3.2:3b", "sha256:x", "ollama", "0.5")
    new = request(BAD_ANSWER)
    old = make_request(
        "You split an answer into atomic claims. Reply with JSON only.",
        V1_CLAIMS_PROMPT.format(answer=BAD_ANSWER),
        0.0,
        42,
        768,
        8192,
    )
    assert cache_key(identity, new) != cache_key(identity, old)
    other_company = request(BAD_ANSWER, company="Microsoft")
    assert cache_key(identity, new) != cache_key(identity, other_company)


# --- Retries ----------------------------------------------------------------------------


def test_bad_case_retries_with_the_problems_listed_then_keeps_the_clean_split():
    fake = FakeProvider(script=[as_json(BAD_SPLIT), as_json(GOOD_SPLIT)])
    claims, response, stats = split_claims(fake, request(BAD_ANSWER), BAD_ANSWER)
    assert claims == GOOD_SPLIT
    assert len(fake.calls) == 2
    retry = fake.calls[1].prompt
    assert fake.calls[0].prompt in retry  # the stricter message extends the original
    assert "'402 million': a fragment" in retry and "'billion'" in retry
    assert "passage citation" in retry and "at least 5 words" in retry
    assert (stats.proposed, stats.rejected, stats.kept, stats.retries) == (2, 0, 2, 1)
    assert [(a.proposed, a.rejected, a.kept) for a in stats.per_attempt] == [(8, 6, 2), (2, 0, 2)]
    assert stats.chosen_attempt == 1 and len(stats.rejections) == 6
    assert all(r.attempt == 0 for r in stats.rejections)
    assert response.metadata.attempts == 1


def test_good_case_needs_one_call():
    fake = FakeProvider(script=[as_json(CHINA_SPLIT)])
    claims, _, stats = split_claims(fake, request(CHINA_ANSWER), CHINA_ANSWER)
    assert claims == CHINA_SPLIT and len(fake.calls) == 1
    assert (stats.proposed, stats.rejected, stats.kept, stats.retries) == (3, 0, 3, 0)


def test_gives_up_after_three_retries_and_keeps_only_valid_claims():
    fake = FakeProvider(script=[as_json(BAD_SPLIT)] * 4)
    claims, _, stats = split_claims(fake, request(BAD_ANSWER), BAD_ANSWER)
    assert len(fake.calls) == 4  # first request + 3 retries
    assert claims == [BAD_SPLIT[0], BAD_SPLIT[6]]
    assert (stats.proposed, stats.rejected, stats.kept, stats.retries) == (8, 6, 2, 3)
    assert stats.chosen_attempt == 3


def test_best_attempt_wins_and_late_json_failure_keeps_it():
    fake = FakeProvider(script=[as_json(GOOD_SPLIT + ["billion"]), "no", "no", "no"])
    claims, _, stats = split_claims(fake, request(BAD_ANSWER), BAD_ANSWER)
    assert claims == GOOD_SPLIT and stats.retries == 0 and stats.rejected == 1


def test_first_request_without_json_is_still_an_error():
    fake = FakeProvider(script=["no", "no", "no"])
    with pytest.raises(StructuredOutputError):
        split_claims(fake, request(BAD_ANSWER), BAD_ANSWER)


def test_empty_list_is_retried_unless_the_answer_abstains():
    abstain = "The passages do not contain this information."
    fake = FakeProvider(script=[as_json([])])
    assert split_claims(fake, request(abstain), abstain)[0] == []
    assert len(fake.calls) == 1
    fake = FakeProvider(script=[as_json([]), as_json(CHINA_SPLIT)])
    assert split_claims(fake, request(CHINA_ANSWER), CHINA_ANSWER)[0] == CHINA_SPLIT
    assert "You returned no claims" in fake.calls[1].prompt


# --- In the generator -------------------------------------------------------------------


def test_generator_records_filter_stats_and_uses_the_company_name(tiny_corpus):
    retriever = tiny_corpus["retriever"]()
    embedder = HashingEmbedder()
    info = RetrievalInfo(embedder=embedder.name, chunker="c", top_k=3, chunks_sha256="s")
    fake = FakeProvider(script=[BAD_ANSWER, as_json(BAD_SPLIT), as_json(GOOD_SPLIT)])
    generator = Generator(
        fake,
        retriever.search,
        lambda t: info,
        LLMConfig(top_k=3),
        {"EXCO": "Example Corp"},
    )
    question = GenerationQuestion(
        id="EXCO-G02",
        ticker="EXCO",
        qtype="yes_no",
        question="Did the company repurchase shares in fiscal 2025?",
        source="candidate_questions",
    )
    record = generator.generate(question)
    assert [c.text for c in record.claims] == GOOD_SPLIT
    assert "COMPANY: Example\n" in fake.calls[1].prompt
    f = record.claim_filter
    assert (f.proposed, f.rejected, f.kept, f.retries) == (2, 0, 2, 1)
    assert record.prompt_version == "v2"
