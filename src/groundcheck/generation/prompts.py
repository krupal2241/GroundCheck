"""Prompts for answering from retrieved chunks and for splitting answers into claims.

`PROMPT_VERSION` is recorded in every candidate; bump it whenever a prompt changes, so
old and new candidates can be told apart (the cache key changes with the prompt text
anyway).
"""

import json
import re

from pydantic import BaseModel, ConfigDict, field_validator

from groundcheck.llm.base import LLMRequest

PROMPT_VERSION = "v2"  # v2: few-shot claim splitting + validator
MAX_CLAIMS = 20
ABSTAIN_TEXT = "The passages do not contain this information."

ANSWER_SYSTEM = f"""\
You answer questions about one company's annual report (Form 10-K) using ONLY the \
numbered passages you are given. Do not use outside knowledge.
- Answer in one to four sentences of plain prose. No lists, no citations.
- Copy numbers, units and scale exactly as the passages write them (e.g. "$12,345 \
million"), and say which fiscal year or period each number belongs to.
- Tables are written as rows with cells separated by " | "; the first rows are headers.
- If the passages do not contain the answer, reply exactly: "{ABSTAIN_TEXT}\""""

CLAIMS_SYSTEM = """\
You split an answer about one company's annual report into claims that a fact-checker \
can verify one at a time. Reply with JSON only."""

ANSWER_MARKER = "ANSWER:\n"


def answer_prompt(ticker: str, question: str, passages: list[tuple[str, str]]) -> str:
    """`passages` is a list of (chunk_id, text) in rank order."""
    blocks = [
        f"[{i}] ({chunk_id})\n{text.strip()}" for i, (chunk_id, text) in enumerate(passages, 1)
    ]
    return (
        f"Company: {ticker}\n\nPassages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n"
        "Answer using only the passages above."
    )


CLAIM_RULES = """\
Rules for every claim:
1. It is one complete, self-contained sentence: a named subject, a verb, and the facts.
2. The subject is the company's name (given above). Replace "the Company", "we", "it" \
and "its" with the name, so the sentence makes sense on its own.
3. Numbers, units, scale and periods are copied exactly as the answer writes them \
("$89.3 billion", "402 million", "4%", "fiscal 2025"). Never add a number, year or \
fact that is not in the answer, and never drop the period a number belongs to.
4. No fragments: "The Company", "its common stock", "402 million" or "billion" on \
their own are NOT claims.
5. No duplicates: each fact appears in exactly one claim.
6. Leave out citations such as "According to passage [1]" or "(passage [2])".
7. If the answer makes no factual claim (it only says the information is not \
available), return an empty list."""

# Three worked examples, shown before the real answer (prompt version v2). The first is
# the failure seen on llama3.2:3b (8 claims, five of them fragments) with the corrected
# output; the second is a clean case; the third shows percentage changes and "its".
CLAIM_EXAMPLES: tuple[tuple[str, str, str, list[str]], ...] = (
    (
        "Apple",
        "How many shares did Apple repurchase in fiscal 2025, and for how much?",
        "According to passage [1], the Company repurchased 402 million shares of its "
        "common stock for $89.3 billion in fiscal 2025.",
        [
            "Apple repurchased 402 million shares of Apple common stock in fiscal 2025.",
            "Apple paid $89.3 billion for the shares of Apple common stock it repurchased "
            "in fiscal 2025.",
        ],
    ),
    (
        "Apple",
        "How did Apple's net sales in Greater China change in fiscal 2025?",
        "Apple's net sales in Greater China were $64,377 million in fiscal 2025, compared "
        "with $66,952 million in fiscal 2024, a decrease of 4%.",
        [
            "Apple's net sales in Greater China were $64,377 million in fiscal 2025.",
            "Apple's net sales in Greater China were $66,952 million in fiscal 2024.",
            "Apple's net sales in Greater China decreased 4% from fiscal 2024 to fiscal 2025.",
        ],
    ),
    (
        "Walmart",
        "How did Sam's Club membership and other income change?",
        "According to passage [2], membership and other income at Sam's Club increased "
        "8.7% in fiscal 2026 and 13.3% in fiscal 2025 compared with the previous fiscal "
        "year, primarily due to growth in its membership base.",
        [
            "Walmart's Sam's Club membership and other income increased 8.7% in fiscal "
            "2026 compared with the previous fiscal year.",
            "Walmart's Sam's Club membership and other income increased 13.3% in fiscal "
            "2025 compared with the previous fiscal year.",
            "The increases in Walmart's Sam's Club membership and other income were "
            "primarily due to growth in the Sam's Club membership base.",
        ],
    ),
)


def _example_block(company: str, question: str, answer: str, claims: list[str]) -> str:
    return (
        f"COMPANY: {company}\nQUESTION: {question}\nANSWER: {answer}\n"
        f"OUTPUT: {json.dumps({'claims': claims}, ensure_ascii=False)}"
    )


def claims_prompt(question: str, answer: str, company: str = "the company") -> str:
    """The claim-splitting prompt: rules, three worked examples, then the real answer.

    `company` is the short company name ("Apple"), used to replace "the Company"/"its".
    """
    examples = "\n\n".join(
        f"Example {i}:\n" + _example_block(*example)
        for i, example in enumerate(CLAIM_EXAMPLES, start=1)
    )
    return (
        "Split the ANSWER into atomic claims.\n\n"
        f"{CLAIM_RULES}\n\n{examples}\n\n"
        "Now the real one.\n"
        f"COMPANY: {company}\nQUESTION: {question}\n\n{ANSWER_MARKER}{answer.strip()}"
        f"{CLAIMS_TAIL}"
    )


CLAIMS_TAIL = '\n\nReturn only: {"claims": ["...", "..."]}'


class ClaimSplit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claims: list[str]

    @field_validator("claims")
    @classmethod
    def _clean(cls, claims: list[str]) -> list[str]:
        cleaned = [" ".join(c.split()) for c in claims]
        if any(not c for c in cleaned):
            raise ValueError("claims must be non-empty strings")
        if len(cleaned) > MAX_CLAIMS:
            raise ValueError(f"at most {MAX_CLAIMS} claims")
        return cleaned


SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def fake_responder(request: LLMRequest) -> str:
    """What `--dry-run` answers: deterministic, shaped like real output, no model.

    The answer is "According to the filing, " plus the first sentence (at most 40
    words) of the top passage; the claim split is one claim per answer sentence.
    """
    if request.json_mode:
        answer = request.prompt.split(ANSWER_MARKER, 1)[1].split(CLAIMS_TAIL, 1)[0]
        if answer.strip() == ABSTAIN_TEXT:
            return '{"claims": []}'
        sentences = [s for s in SENTENCE_END_RE.split(answer.strip()) if s]
        sentences = [_strip_lead(s) for s in sentences]
        return json.dumps({"claims": sentences})
    if "\n[1] (" not in request.prompt:
        return ABSTAIN_TEXT
    first = request.prompt.split("\n[1] (", 1)[1].split("\n", 1)[1].split("\n\n", 1)[0]
    words = " ".join(first.split()).split()[:40]
    sentence = SENTENCE_END_RE.split(" ".join(words))[0].rstrip(".")
    return f"According to the filing, {sentence}."


def _strip_lead(sentence: str) -> str:
    """Drop the fake answer's "According to the filing, " lead-in (as the prompt asks)."""
    lead = "According to the filing, "
    if sentence.startswith(lead):
        rest = sentence[len(lead) :]
        return rest[:1].upper() + rest[1:]
    return sentence


def make_request(
    system: str, prompt: str, temperature: float, seed: int, max_tokens: int, context_window: int
) -> LLMRequest:
    return LLMRequest(
        prompt=prompt,
        system=system,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        context_window=context_window,
    )
