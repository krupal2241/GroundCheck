"""`FineTunedVerifier`: the three-way classifier trained by scripts/train_verifier.py.

It loads from a **local path only** (`local_files_only=True`), so it never downloads
anything; `torch`/`transformers` come from the `[verifiers]` extra and are imported when
it is built. Like MiniCheck and HHEM it scores the claim against each chunk separately
(input order (chunk, claim), the same as in training) and aggregates; see the class.
"""

from pathlib import Path
from typing import Any

from groundcheck.verify.base import LABELS, BaseVerifier, EvidenceChunk, Verification, VerifierError
from groundcheck.verify.hf import MAX_LENGTH, import_torch_stack, pick_device, runtime_version
from groundcheck.verify.spans import best_unit_span


class FineTunedVerifier(BaseVerifier):
    """A three-way classifier fine-tuned by scripts/train_verifier.py, from a local path.

    Input per chunk: (chunk, claim), exactly as in training (`training.data.encode_pair`).
    The model's `id2label` must name the three labels. Aggregation over chunks:

    - **supported** if some chunk's most likely class is supported (of those, the one
      with the highest P(supported) is cited): one passage stating the claim is enough;
    - else **contradicted** if some chunk's most likely class is contradicted (of those,
      the one with the highest P(contradicted) is cited);
    - else **unsupported**, citing nothing.

    `score` = the highest P(supported) over chunks; `scores` = the per-label
    probabilities of the chunk the verdict rests on (the best-support chunk when
    unsupported).
    """

    name = "finetuned"

    def __init__(self, path: str | Path, device: str | None = None, max_length: int = MAX_LENGTH):
        path = Path(path).expanduser()
        if not (path / "config.json").is_file():
            raise VerifierError(
                f"no fine-tuned model at {path} (expected config.json); train one with "
                "scripts/train_verifier.py or notebooks/finetune_verifier_colab.ipynb"
            )
        torch, transformers = import_torch_stack()
        self._torch = torch
        self.path, self.max_length = path, max_length
        self.model = str(path)
        self.device = pick_device(torch, device)
        self.runtime = runtime_version(torch, transformers)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(path), local_files_only=True
        )
        self.lm = transformers.AutoModelForSequenceClassification.from_pretrained(
            str(path), local_files_only=True
        )
        self.lm.to(self.device)
        self.lm.eval()
        self.label_index = label_indices(getattr(self.lm.config, "id2label", None))

    def identity(self) -> dict[str, Any]:
        return {
            "verifier": self.name,
            "model": self.model,
            "device": self.device,
            "runtime": self.runtime,
        }

    def chunk_probs(self, claim: str, texts: list[str]) -> list[dict[str, float]]:
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
        rows = torch.softmax(logits.float(), dim=-1).cpu().tolist()
        return [{label: float(row[i]) for label, i in self.label_index.items()} for row in rows]

    def _verify(self, claim: str, chunks: list[EvidenceChunk]) -> Verification:
        probs = self.chunk_probs(claim, [c.text for c in chunks])
        return aggregate_three_way(claim, chunks, probs)


def label_indices(id2label: dict[int, str] | None) -> dict[str, int]:
    names = {str(n).lower(): int(i) for i, n in (id2label or {}).items()}
    missing = [label for label in LABELS if label not in names]
    if missing:
        raise VerifierError(
            f"the model's id2label {sorted(names)} does not name the labels {missing}; "
            "was it trained with scripts/train_verifier.py?"
        )
    return {label: names[label] for label in LABELS}


def aggregate_three_way(
    claim: str, chunks: list[EvidenceChunk], probs: list[dict[str, float]]
) -> Verification:
    """Per-chunk three-way probabilities -> one verdict (rules in FineTunedVerifier)."""

    def top(row: dict[str, float]) -> str:
        return max(LABELS, key=lambda label: row[label])

    best_support = max(range(len(chunks)), key=lambda i: probs[i]["supported"])
    label, rest = "unsupported", best_support
    for candidate in ("supported", "contradicted"):
        winners = [i for i, row in enumerate(probs) if top(row) == candidate]
        if winners:
            label, rest = candidate, max(winners, key=lambda i: probs[i][candidate])
            break
    evidence = None
    if label != "unsupported":
        evidence = best_unit_span(claim, chunks[rest].chunk_id, chunks[rest].text)
    return Verification(
        label=label,
        score=round(probs[best_support]["supported"], 6),
        evidence=evidence,
        scores={k: round(v, 6) for k, v in probs[rest].items()},
        details={"verdict_chunk": chunks[rest].chunk_id},
    )
