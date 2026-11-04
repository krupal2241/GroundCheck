"""Prompts for answering from retrieved chunks and for splitting answers into claims.

`PROMPT_VERSION` is recorded in every candidate; bump it whenever a prompt changes, so
old and new candidates can be told apart (the cache key changes with the prompt text
anyway).
"""

import json
import re

from pydantic import BaseModel, ConfigDict, field_validator

from groundcheck.llm.base import LLMRequest

PROMPT_VERSION = "v1"
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
You split an answer into atomic claims. Reply with JSON only."""

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


def claims_prompt(question: str, answer: str) -> str:
    return (
        f"QUESTION:\n{question}\n\n{ANSWER_MARKER}{answer.strip()}\n\n"
        "Split the ANSWER into atomic claims. Each claim:\n"
        "- is one standalone factual statement that can be checked on its own;\n"
        "- keeps its number, unit, scale and period exactly as the answer states them;\n"
        "- names the company or item instead of using a pronoun;\n"
        "- adds nothing that is not in the answer.\n"
        "If the answer makes no factual claim (it only says the information is not "
        'available), return an empty list.\nReturn only: {"claims": ["...", "..."]}'
    )


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
        answer = request.prompt.split(ANSWER_MARKER, 1)[1].split("\n\nSplit the ANSWER", 1)[0]
        if answer.strip() == ABSTAIN_TEXT:
            return '{"claims": []}'
        sentences = [s for s in SENTENCE_END_RE.split(answer.strip()) if s]
        return json.dumps({"claims": sentences})
    if "\n[1] (" not in request.prompt:
        return ABSTAIN_TEXT
    first = request.prompt.split("\n[1] (", 1)[1].split("\n", 1)[1].split("\n\n", 1)[0]
    words = " ".join(first.split()).split()[:40]
    sentence = SENTENCE_END_RE.split(" ".join(words))[0].rstrip(".")
    return f"According to the filing, {sentence}."


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
