"""`PromptedLLMVerifier`: a local LLM (Ollama by default) judging one claim at a time.

It goes through the existing provider layer (`llm.factory.make_provider`), so it gets
the paid-call guard, deterministic settings (temperature 0, fixed seed) and, when
asked, the disk cache. The prompt is few-shot: the labeling rules from
docs/labeling_guide.md in short form, then three worked examples (one per label,
including a table and a rounding case), then the numbered passages and the claim.

The model answers in JSON, validated with pydantic and retried with the error appended
(`llm.structured.generate_structured`):

    {"label": "supported" | "contradicted" | "unsupported",
     "passage": <passage number or null>,
     "evidence": "<exact quote from that passage, or empty>",
     "confidence": <0..1>}

Mapping onto a `Verification`: `label` as given; `score` (P(supported)) is
`confidence` for a supported verdict and `1 - confidence` otherwise; the quote is
looked up in the cited passage first, then in all of them, and becomes the evidence
span. A quote that is not in any passage is kept in `details` but gives no span
(`quote_found: false`), so a hallucinated quote is visible, not silently trusted. If
the model never returns valid JSON, the verdict is `unsupported` with `parse_error`
set in `details` (the evaluator counts these).
"""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from groundcheck.config import LLMConfig
from groundcheck.llm.base import LLMProvider, LLMRequest
from groundcheck.llm.structured import StructuredOutputError, generate_structured
from groundcheck.verify.base import BaseVerifier, EvidenceChunk, Verification
from groundcheck.verify.spans import find_quote

VERIFY_PROMPT_VERSION = "verify-v1"

VERIFY_SYSTEM = """\
You check whether a claim is supported by passages from a company's annual report \
(Form 10-K). Judge ONLY against the passages, never against outside knowledge. \
Reply with JSON only."""

RULES = """\
Labels:
- "supported": a passage states the claim (numbers may be rounded correctly to the \
claim's own precision, e.g. $12,345 million -> $12.3 billion).
- "contradicted": a passage shows the claim is false: a different number, the wrong \
unit or scale (million vs billion), the opposite direction (increased vs decreased), \
a value from the wrong year, or a negation.
- "unsupported": the passages neither state nor contradict the claim (the topic or \
period is missing, or the passage only states an expectation or risk).
A claim with several parts takes the label of its weakest part \
(contradicted > unsupported > supported).
Tables are written as rows with cells separated by " | "; the first row is the header.
"evidence" must be copied exactly from the passage you cite; empty for unsupported."""

FEW_SHOT = [
    (
        "[1]\n[TABLE]\n | 2025 | 2024\nTotal net sales | $12,345 | $11,646\n[/TABLE]\n"
        "(in millions)",
        "Total net sales were about $12.3 billion in 2025.",
        {
            "label": "supported",
            "passage": 1,
            "evidence": "Total net sales | $12,345 | $11,646",
            "confidence": 0.9,
        },
    ),
    (
        "[1]\nOperating income decreased 4% to $3,210 million in fiscal 2025.",
        "Operating income increased in fiscal 2025.",
        {
            "label": "contradicted",
            "passage": 1,
            "evidence": "Operating income decreased 4% to $3,210 million in fiscal 2025.",
            "confidence": 0.95,
        },
    ),
    (
        "[1]\nThe Company faces intense competition and may not be able to maintain its "
        "gross margin.",
        "The Company's gross margin will decline next year.",
        {"label": "unsupported", "passage": None, "evidence": "", "confidence": 0.85},
    ),
]


def _example_block(passages: str, claim: str, answer: dict) -> str:
    return f"PASSAGES:\n{passages}\nCLAIM: {claim}\nJSON: {json.dumps(answer)}"


def verify_prompt(claim: str, chunks: list[EvidenceChunk]) -> str:
    examples = "\n\n".join(_example_block(*example) for example in FEW_SHOT)
    passages = "\n\n".join(
        f"[{i}] ({chunk.chunk_id})\n{chunk.text.strip()}" for i, chunk in enumerate(chunks, 1)
    )
    return (
        f"{RULES}\n\nEXAMPLES:\n\n{examples}\n\nNOW THE REAL TASK.\n\n"
        f"PASSAGES:\n{passages}\nCLAIM: {claim}\n"
        'Return only: {"label": "...", "passage": <number or null>, "evidence": "...", '
        '"confidence": <0 to 1>}'
    )


class VerdictJSON(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: Literal["supported", "contradicted", "unsupported"]
    passage: int | None = None
    evidence: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("label", mode="before")
    @classmethod
    def _lower(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("evidence", mode="before")
    @classmethod
    def _none_is_empty(cls, value):
        return "" if value is None else value


class PromptedLLMVerifier(BaseVerifier):
    name = "prompted"
    local = True

    def __init__(self, provider: LLMProvider, config: LLMConfig | None = None):
        self.provider = provider
        self.config = config or LLMConfig()
        self.model = provider.model
        self.local = not bool(getattr(provider, "paid", False))

    def identity(self) -> dict:
        ident = self.provider.identity()
        return {
            "verifier": self.name,
            "model": ident.model,
            "provider": ident.provider,
            "model_digest": ident.model_digest,
            "runtime": f"{ident.runtime} {ident.runtime_version}",
            "prompt_version": VERIFY_PROMPT_VERSION,
        }

    def request(self, claim: str, chunks: list[EvidenceChunk]) -> LLMRequest:
        c = self.config
        return LLMRequest(
            prompt=verify_prompt(claim, chunks),
            system=VERIFY_SYSTEM,
            temperature=c.temperature,
            seed=c.seed,
            max_tokens=min(c.max_tokens, 256),
            context_window=c.context_window,
        )

    def _verify(self, claim: str, chunks: list[EvidenceChunk]) -> Verification:
        try:
            verdict, response = generate_structured(
                self.provider,
                self.request(claim, chunks),
                VerdictJSON,
                max_attempts=self.config.json_max_attempts,
            )
        except StructuredOutputError as exc:
            return Verification(
                label="unsupported",
                score=0.5,
                details={"parse_error": str(exc), "raw": exc.texts[-1][:500]},
            )
        pairs = [(c.chunk_id, c.text) for c in chunks]
        span = None
        if verdict.evidence:
            cited = []
            if verdict.passage and 1 <= verdict.passage <= len(pairs):
                cited = [pairs[verdict.passage - 1]]
            span = find_quote(verdict.evidence, cited) or find_quote(verdict.evidence, pairs)
        score = verdict.confidence if verdict.label == "supported" else 1 - verdict.confidence
        meta = response.metadata
        return Verification(
            label=verdict.label,
            score=round(score, 4),
            evidence=span if verdict.label != "unsupported" else None,
            details={
                "quote": verdict.evidence,
                "quote_found": span is not None if verdict.evidence else None,
                "passage": verdict.passage,
                "confidence": verdict.confidence,
                "attempts": meta.attempts,
                "prompt_tokens": meta.prompt_tokens,
                "completion_tokens": meta.completion_tokens,
                "cached": meta.cached,
            },
        )
