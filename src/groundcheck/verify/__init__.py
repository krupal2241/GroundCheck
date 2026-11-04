"""Claim verifiers: one claim plus evidence chunks in, a label, score and span out."""

from groundcheck.verify.base import (
    BaseVerifier,
    EvidenceChunk,
    Verification,
    Verifier,
    VerifierError,
)
from groundcheck.verify.fake import FakeVerifier

__all__ = [
    "BaseVerifier",
    "EvidenceChunk",
    "FakeVerifier",
    "Verification",
    "Verifier",
    "VerifierError",
]
