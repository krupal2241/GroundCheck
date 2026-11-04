"""Open Hugging Face verifiers: MiniCheck and HHEM (binary), and the fine-tuned model.

All of them are cross-encoders: they read one (evidence, claim) pair at a time. The
claim is scored against each retrieved chunk separately (one batch per claim) and the
chunk-level results are aggregated. This is how MiniCheck itself handles long
documents (split into chunks, take the maximum support probability), and it keeps
every input under the 512-token limit, since our chunks are built to fit bge-small's
512 tokens.

**Binary models (MiniCheck, HHEM)** output one number per pair: the probability that
the evidence supports (MiniCheck) or is consistent with (HHEM) the claim. Mapping:

- `score` = the maximum over chunks (P(supported) by the best chunk);
- `label` = supported if `score >= threshold` (default 0.5), else **unsupported**;
- they **never predict contradicted**: a low score means "this evidence does not back
  the claim", which covers both "says otherwise" and "says nothing", and the models
  were not trained to tell the two apart. Guessing contradicted from a low score would
  invent a distinction the model does not make. The evaluator therefore also reports a
  binary view (supported vs not supported) where all verifiers compete on equal terms;
- evidence = the best-matching unit (sentence or table row) of the best chunk, for
  supported verdicts only (an unsupported verdict cites nothing, like a gold label).

**Three-way models**: the fine-tuned verifier (`verify/finetuned.py`) reuses
`import_torch_stack` and `pick_device` from here.

`torch` and `transformers` come from the `[verifiers]` extra and are imported only when
a verifier is built, so nothing here is needed (or imported) by CI.
"""

from typing import Any

from groundcheck.verify.base import BaseVerifier, EvidenceChunk, Verification, VerifierError
from groundcheck.verify.spans import best_unit_span

MINICHECK_MODEL = "lytang/MiniCheck-DeBERTa-v3-Large"
HHEM_MODEL = "vectara/hallucination_evaluation_model"
MAX_LENGTH = 512


def import_torch_stack(extra: str = "verifiers") -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise VerifierError(
            f'this verifier needs the [{extra}] extra: pip install -e ".[{extra}]"'
        ) from exc
    return torch, transformers


def pick_device(torch: Any, requested: str | None = None) -> str:
    """`requested` if given, else cuda (Colab), else mps (Apple silicon), else cpu."""
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def runtime_version(torch: Any, transformers: Any) -> str:
    return f"transformers {transformers.__version__}, torch {torch.__version__}"


def support_index(id2label: dict[int, str] | None) -> int:
    """Which output class means 'supported'. MiniCheck's DeBERTa/RoBERTa checkpoints use
    the generic LABEL_0/LABEL_1 names with 1 = supported; a named label wins if present."""
    names = {int(i): str(n).lower() for i, n in (id2label or {}).items()}
    for index, name in names.items():
        if name in {"supported", "support", "entailment", "consistent", "1"}:
            return index
    if len(names) not in (0, 2):
        raise VerifierError(f"expected a binary classifier, got labels {sorted(names.values())}")
    return 1


class ChunkScoringVerifier(BaseVerifier):
    """Scores (chunk, claim) pairs with a binary model and aggregates by the maximum."""

    threshold = 0.5

    def support_probs(self, claim: str, texts: list[str]) -> list[float]:
        raise NotImplementedError

    def identity(self) -> dict[str, Any]:
        return {
            "verifier": self.name,
            "model": self.model,
            "model_revision": getattr(self, "revision", None),
            "device": getattr(self, "device", None),
            "runtime": getattr(self, "runtime", None),
            "threshold": self.threshold,
        }

    def _verify(self, claim: str, chunks: list[EvidenceChunk]) -> Verification:
        raw = self.support_probs(claim, [c.text for c in chunks])
        probs = [min(max(float(p), 0.0), 1.0) for p in raw]
        best = max(range(len(chunks)), key=lambda i: probs[i])
        score = probs[best]
        label = "supported" if score >= self.threshold else "unsupported"
        evidence = None
        if label == "supported":
            evidence = best_unit_span(claim, chunks[best].chunk_id, chunks[best].text)
        return Verification(
            label=label,
            score=round(score, 6),
            evidence=evidence,
            details={
                "binary": True,
                "best_chunk": chunks[best].chunk_id,
                "chunk_scores": {
                    c.chunk_id: round(p, 6) for c, p in zip(chunks, probs, strict=True)
                },
            },
        )


class MiniCheckVerifier(ChunkScoringVerifier):
    """MiniCheck (Tang et al., 2024) sequence-classification checkpoints.

    Default `lytang/MiniCheck-DeBERTa-v3-Large`; `lytang/MiniCheck-RoBERTa-Large` also
    works. Input: (chunk, claim), truncating the chunk if needed; output: softmax over
    two classes, class 1 = supported. The Flan-T5 and 7B MiniCheck variants need the
    `minicheck` package and a different head, so they are refused here.
    """

    name = "minicheck"

    def __init__(
        self,
        model: str = MINICHECK_MODEL,
        device: str | None = None,
        threshold: float = 0.5,
        revision: str | None = None,
        max_length: int = MAX_LENGTH,
    ):
        if "flan-t5" in model.lower() or "7b" in model.lower():
            raise VerifierError(
                f"{model} is not a sequence-classification checkpoint; use "
                f"{MINICHECK_MODEL} or lytang/MiniCheck-RoBERTa-Large"
            )
        torch, transformers = import_torch_stack()
        self._torch = torch
        self.model, self.revision, self.threshold = model, revision, threshold
        self.max_length = max_length
        self.device = pick_device(torch, device)
        self.runtime = runtime_version(torch, transformers)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, revision=revision)
        self.lm = transformers.AutoModelForSequenceClassification.from_pretrained(
            model, revision=revision
        )
        self.lm.to(self.device)
        self.lm.eval()
        self.support_index = support_index(getattr(self.lm.config, "id2label", None))
        self.revision = revision or getattr(self.lm.config, "_commit_hash", None)

    def support_probs(self, claim: str, texts: list[str]) -> list[float]:
        torch = self._torch
        batch = self.tokenizer(
            texts,
            [claim] * len(texts),
            truncation="only_first",
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.lm(**batch).logits
        probs = torch.softmax(logits.float(), dim=-1)[:, self.support_index]
        return [float(p) for p in probs.cpu().tolist()]


class HHEMVerifier(ChunkScoringVerifier):
    """Vectara's HHEM-2.1-Open (`vectara/hallucination_evaluation_model`, Flan-T5-base).

    `model.predict([(premise, hypothesis), ...])` returns a consistency score in [0, 1]
    per pair (1 = the hypothesis is consistent with the premise). Premise = the chunk,
    hypothesis = the claim. The checkpoint ships its own modeling code, so loading it
    needs `trust_remote_code=True`: pin `revision` to a commit you have read to make
    that code fixed (the commit actually loaded is recorded either way).
    """

    name = "hhem"

    def __init__(
        self,
        model: str = HHEM_MODEL,
        device: str | None = None,
        threshold: float = 0.5,
        revision: str | None = None,
    ):
        torch, transformers = import_torch_stack()
        self._torch = torch
        self.model, self.threshold = model, threshold
        self.device = pick_device(torch, device)
        self.runtime = runtime_version(torch, transformers)
        self.lm = transformers.AutoModelForSequenceClassification.from_pretrained(
            model, trust_remote_code=True, revision=revision
        )
        self.lm.to(self.device)
        self.lm.eval()
        self.revision = revision or getattr(self.lm.config, "_commit_hash", None)

    def support_probs(self, claim: str, texts: list[str]) -> list[float]:
        with self._torch.no_grad():
            scores = self.lm.predict([(text, claim) for text in texts])
        if hasattr(scores, "cpu"):
            scores = scores.float().cpu().tolist()
        return [float(s) for s in scores]
