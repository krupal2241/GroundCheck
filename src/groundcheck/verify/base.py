"""The `Verifier` interface: one claim plus its evidence chunks in, one verdict out.

Input: the claim text and the chunks it should be judged against (for the gold set,
the passages retrieved for the question; see docs/labeling_guide.md: judge only
against the retrieved chunks).

Output, a `Verification`:

- `label`: supported, contradicted or unsupported (the gold labels' vocabulary);
- `score`: the verifier's estimate, in [0, 1], that the claim is **supported**. One
  meaning for every verifier, so binary models (MiniCheck, HHEM) and three-way ones can
  be put on the same threshold or ROC plot. How each model's native output is mapped
  onto label and score is documented in docs/DESIGN.md;
- `evidence`: the span the verdict rests on, as an `EvidenceSpan` (chunk ID, text and
  offsets into the whitespace-normalized chunk, exactly like a gold span), or None;
- `latency_ms`: wall-clock time of this one check, measured around the whole call;
- `scores`: optional per-label probabilities, and `details` for anything
  model-specific (raw model scores, the LLM's quote, parse attempts).

Every implementation subclasses `BaseVerifier`, which times the call, so latency is
measured the same way for all of them.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from groundcheck.labeling.schema import LABELS, EvidenceSpan, Label

__all__ = [
    "LABELS",
    "BaseVerifier",
    "EvidenceChunk",
    "Label",
    "Verification",
    "Verifier",
    "VerifierError",
]


class VerifierError(Exception):
    """A verifier could not be built or could not produce a verdict."""


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    text: str


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Label
    score: float = Field(ge=0.0, le=1.0)  # P(supported), see module docs
    evidence: EvidenceSpan | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    scores: dict[str, float] | None = None  # per-label probabilities, when the model has them
    details: dict[str, Any] = {}


class Verifier(Protocol):
    """Anything with these attributes and methods is a verifier."""

    name: str  # registry name, e.g. "hhem"
    model: str  # the model behind it, e.g. "vectara/hallucination_evaluation_model"
    local: bool  # runs on this machine (cost per check is $0)

    def identity(self) -> dict[str, Any]: ...

    def verify(self, claim: str, chunks: Sequence[EvidenceChunk]) -> Verification: ...


class BaseVerifier:
    """Shared plumbing: input checks and latency. Subclasses implement `_verify`."""

    name = "base"
    model = ""
    local = True

    def identity(self) -> dict[str, Any]:
        return {"verifier": self.name, "model": self.model}

    def verify(self, claim: str, chunks: Sequence[EvidenceChunk]) -> Verification:
        claim = " ".join(claim.split())
        if not claim:
            raise VerifierError("the claim is empty")
        started = time.perf_counter()
        if chunks:
            result = self._verify(claim, list(chunks))
        else:
            # Nothing to judge against: by definition there is no supporting evidence.
            result = Verification(label="unsupported", score=0.0, details={"no_chunks": True})
        elapsed_ms = (time.perf_counter() - started) * 1000
        return result.model_copy(update={"latency_ms": round(elapsed_ms, 3)})

    def _verify(self, claim: str, chunks: list[EvidenceChunk]) -> Verification:
        raise NotImplementedError
