"""Build a verifier by name.

    fake       lexical rule, no model (tests, dry runs, the offline demo)
    prompted   a local LLM through the provider layer (Ollama by default)
    minicheck  lytang/MiniCheck-DeBERTa-v3-Large    ([verifiers] extra)
    hhem       vectara/hallucination_evaluation_model ([verifiers] extra)
    finetuned  a model trained by scripts/train_verifier.py, from a local path

The prompted verifier's provider is built by `llm.factory.make_provider`, so the
paid-call guard applies to it as to everything else.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from groundcheck.config import LLMConfig, load_llm_config
from groundcheck.llm.base import LLMRequest
from groundcheck.verify.base import Verifier, VerifierError

VERIFIER_NAMES = ("fake", "prompted", "minicheck", "hhem", "finetuned")


@dataclass
class VerifierOptions:
    """Everything a verifier might need; each one reads only its own fields."""

    llm_config: Path | None = None  # prompted: default config/llm.yaml
    provider: str | None = None  # prompted: ollama (default), hf-local, fake
    llm_model: str | None = None  # prompted: e.g. llama3.2:3b
    allow_paid: bool = False
    use_cache: bool = False  # prompted: off by default, so latency is real
    model: str | None = None  # minicheck / hhem: override the Hub model ID
    revision: str | None = None  # minicheck / hhem: pin a Hub commit
    finetuned_path: Path | None = None
    device: str | None = None  # cpu, mps or cuda; default: auto
    threshold: float = 0.5  # binary verifiers: P(supported) needed for "supported"


def fake_verdict_responder(request: LLMRequest) -> str:
    """What the fake LLM provider answers to a verification prompt."""
    return json.dumps({"label": "unsupported", "passage": None, "evidence": "", "confidence": 0.5})


def make_verifier(name: str, options: VerifierOptions | None = None) -> Verifier:
    options = options or VerifierOptions()
    if name == "fake":
        from groundcheck.verify.fake import FakeVerifier

        return FakeVerifier()
    if name == "prompted":
        from groundcheck.llm.factory import make_provider
        from groundcheck.verify.prompted import PromptedLLMVerifier

        config: LLMConfig = load_llm_config(options.llm_config)
        provider_name = options.provider or config.provider
        kwargs = {"responder": fake_verdict_responder} if provider_name == "fake" else {}
        provider = make_provider(
            config,
            provider_name,
            options.llm_model,
            allow_paid=options.allow_paid,
            use_cache=options.use_cache,
            **kwargs,
        )
        return PromptedLLMVerifier(provider, config)
    if name == "minicheck":
        from groundcheck.verify.hf import MINICHECK_MODEL, MiniCheckVerifier

        return MiniCheckVerifier(
            options.model or MINICHECK_MODEL, options.device, options.threshold, options.revision
        )
    if name == "hhem":
        from groundcheck.verify.hf import HHEM_MODEL, HHEMVerifier

        return HHEMVerifier(
            options.model or HHEM_MODEL, options.device, options.threshold, options.revision
        )
    if name == "finetuned":
        if options.finetuned_path is None:
            raise VerifierError("the finetuned verifier needs --finetuned-path <model dir>")
        from groundcheck.verify.finetuned import FineTunedVerifier

        return FineTunedVerifier(options.finetuned_path, options.device)
    raise VerifierError(f"unknown verifier {name!r}; choose from {', '.join(VERIFIER_NAMES)}")
