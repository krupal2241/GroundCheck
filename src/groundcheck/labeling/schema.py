"""The `LabelRecord` schema: one human label for one claim.

Rules (see docs/labeling_guide.md):

- `label` is `supported`, `contradicted` or `unsupported`.
- `supported` and `contradicted` need at least one evidence span; `unsupported` has
  none ("absent evidence = unsupported").
- An evidence span is an exact substring of one of the candidate's retrieved chunks
  after whitespace normalization (`normalize_for_match`, the same rule as retrieval
  gold strings). It is stored normalized, with `start`/`end` character offsets into
  the normalized chunk text, so it can be checked mechanically.
- `edge_case` is one tag from the labeling guide's list, or None.
- `confirmed` is True only once a human has confirmed the label. Export keeps only
  confirmed labels (PLAN.md: no fabricated gold labels).
- `invalid` marks a claim that should never have been proposed (a fragment, a
  duplicate, a claim with no checkable content): it has no label, no evidence and no
  edge case, and it is never exported. `label-stats` counts these separately.

The schema checks what it can on its own. Pass the candidate's chunks as validation
context (`LabelRecord.model_validate(data, context={"chunks": {chunk_id: text}})`)
and it also checks that every span really is in its cited chunk at its offsets.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from groundcheck.retrieval.evaluate import normalize_for_match

Label = Literal["supported", "contradicted", "unsupported"]
LABELS: tuple[str, ...] = ("supported", "contradicted", "unsupported")

# The edge-case list from docs/labeling_guide.md, in the guide's order.
EdgeCase = Literal[
    "negation",
    "numeric_mismatch",
    "rounding",
    "units_scale",
    "table_misread",
    "temporal",
    "partial_support",
    "hedging",
    "absent_evidence",
]
EDGE_CASES: tuple[str, ...] = EdgeCase.__args__


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str = Field(min_length=1)  # whitespace-normalized
    start: int = Field(ge=0)  # offsets into normalize_for_match(chunk text)
    end: int

    @model_validator(mode="after")
    def _consistent(self, info: ValidationInfo) -> "EvidenceSpan":
        if self.text != normalize_for_match(self.text):
            raise ValueError("evidence text must be whitespace-normalized")
        if self.end - self.start != len(self.text):
            raise ValueError("end - start must equal the length of the evidence text")
        chunks = (info.context or {}).get("chunks")
        if chunks is not None:
            if self.chunk_id not in chunks:
                raise ValueError(f"{self.chunk_id} is not one of the candidate's retrieved chunks")
            normalized = normalize_for_match(chunks[self.chunk_id])
            if normalized[self.start : self.end] != self.text:
                where = normalized.find(self.text)
                hint = f" (it occurs at {where})" if where >= 0 else ""
                raise ValueError(
                    f"evidence is not at offsets {self.start}:{self.end} of {self.chunk_id}{hint}"
                )
        return self


def make_span(chunk_id: str, chunk_text: str, evidence: str) -> EvidenceSpan | None:
    """The span for `evidence` in `chunk_text` (first occurrence), or None if absent."""
    needle = normalize_for_match(evidence)
    if not needle:
        return None
    start = normalize_for_match(chunk_text).find(needle)
    if start < 0:
        return None
    return EvidenceSpan(chunk_id=chunk_id, text=needle, start=start, end=start + len(needle))


class Assistance(BaseModel):
    """What assistance was shown for one label (stored in `LabelRecord.assistance`)."""

    model_config = ConfigDict(extra="forbid")

    evidence_suggested: bool = False  # an evidence span was offered
    evidence_accepted: bool = False  # ... and accepted with one key
    label_suggestion_shown: bool = False
    suggested_label: Label | None = None
    suggestion_verifier: str | None = None
    suggestion_score: float | None = None  # the verifier's P(supported)
    agreed: bool | None = None  # final label == suggested label (None: none shown)


class LabelRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    claim_id: str
    candidate_id: str
    question_id: str
    ticker: str
    qtype: str
    claim: str = Field(min_length=1)
    label: Label | None = None  # None only for an invalid claim
    invalid: bool = False  # the `x` key: a malformed claim, not labeled, never exported
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    edge_case: EdgeCase | None = None
    confirmed: bool = False
    note: str = ""
    labeled_at: str = ""  # UTC, ISO 8601
    # Which suggestions were shown (labeling/assist.py). None: labeled before
    # assistance existed, i.e. nothing was suggested.
    assistance: Assistance | None = None

    @field_validator("note")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _label_rules(self) -> "LabelRecord":
        if self.invalid:
            if self.label is not None or self.evidence or self.edge_case is not None:
                raise ValueError("an invalid claim has no label, evidence or edge case")
            return self
        if self.label is None:
            raise ValueError("a label is required unless the claim is marked invalid")
        if self.label in ("supported", "contradicted") and not self.evidence:
            raise ValueError(f"a {self.label} label needs at least one evidence span")
        if self.label == "unsupported" and self.evidence:
            raise ValueError("an unsupported label must not have evidence spans")
        if self.edge_case == "absent_evidence" and self.label != "unsupported":
            raise ValueError("edge case absent_evidence goes with the unsupported label")
        return self
