"""Fine-tune a small encoder as a three-way claim verifier (scripts/train_verifier.py).

Model: `microsoft/deberta-v3-small` by default (~140M parameters with its embedding
matrix; trains on a free Colab T4 in well under an hour at the default data size).
Input: the (evidence chunk, claim) sentence pair, truncating only the evidence at 512
tokens (`training.data.encode_pair`), which is exactly what `FineTunedVerifier` feeds
it at inference. Labels: supported = 0, contradicted = 1, unsupported = 2, saved in
the model's `id2label`.

Reproducibility: `transformers.set_seed(seed)` seeds Python, numpy and torch before the
model is built; the Trainer gets the same `seed` and `data_seed`; `--full-determinism`
also turns on PyTorch's deterministic algorithms (slower; some GPU kernels stay
nondeterministic without it). The best epoch is chosen on dev macro-F1.

Written to `--out` (default `models/verifier-deberta-v3-small/`, gitignored): the model
and tokenizer, `metrics.json` (dev and synthetic-test metrics, train loss, settings),
`training_manifest.json` (from the data build) and `README.md`, a model card. The gold
set is not touched here; evaluate on it with `groundcheck eval-verifiers --verifier
finetuned --finetuned-path <out>`.

`torch`, `transformers` and `accelerate` come from the `[train]` extra and are imported
only in `train()`.
"""

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from groundcheck import __version__
from groundcheck.config import REPO_ROOT, ConfigError
from groundcheck.labeling.schema import LABELS
from groundcheck.training.data import (
    DEFAULT_TRAINING_DIR,
    LABEL_IDS,
    TrainingExample,
    encode_pair,
    read_examples,
)
from groundcheck.verify.evaluate import class_metrics, macro_f1

DEFAULT_BASE_MODEL = "microsoft/deberta-v3-small"
DEFAULT_OUT = REPO_ROOT / "models" / "verifier-deberta-v3-small"
ID2LABEL = {i: label for label, i in LABEL_IDS.items()}


class EncodedDataset:
    """A list of tokenized examples; enough for the Trainer (len + getitem)."""

    def __init__(self, tokenizer, examples: list[TrainingExample], max_length: int):
        self.rows = []
        for e in examples:
            row = dict(encode_pair(tokenizer, e.evidence, e.claim, max_length))
            row["labels"] = LABEL_IDS[e.label]
            self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


MIN_TRAINING_EXAMPLES = 500  # below this, warn: see config/training.yaml


def metrics_from_logits(logits: Any, label_ids: Any) -> dict[str, float]:
    """Accuracy, macro-F1 and per-class F1 / recall, from raw logits."""
    predicted = np.asarray(logits).argmax(axis=-1).tolist()
    gold = [ID2LABEL[int(i)] for i in np.asarray(label_ids).tolist()]
    pred = [ID2LABEL[int(i)] for i in predicted]
    per = {label: class_metrics(gold, pred, label) for label in LABELS}
    out = {
        "accuracy": sum(g == p for g, p in zip(gold, pred, strict=True)) / max(len(gold), 1),
        "macro_f1": macro_f1(per),
    }
    for label, m in per.items():
        out[f"f1_{label}"] = m.f1
        out[f"recall_{label}"] = m.recall if m.recall is not None else 0.0
    return out


def compute_metrics(eval_prediction) -> dict[str, float]:
    logits, label_ids = eval_prediction.predictions, eval_prediction.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]
    return metrics_from_logits(logits, label_ids)


def model_card(settings: dict, metrics: dict, manifest: dict) -> str:
    counts = manifest.get("counts_by_split_and_label", {})
    splits = manifest.get("config", {}).get("splits", {})
    lines = [
        f"# GroundCheck verifier ({settings['base_model']}, fine-tuned)",
        "",
        "A three-way claim verifier for RAG answers over SEC 10-K filings: given an "
        "evidence passage and a claim, it predicts **supported**, **contradicted** or "
        "**unsupported**.",
        "",
        "## Training data",
        "",
        "Synthetic, built by `groundcheck build-training-data` from the filings' own "
        "chunks: supported claims are sentences and table cells taken from a chunk; "
        "contradicted claims are rule-based perturbations of them (numbers, scale, "
        "negation, year swap, entity swap); unsupported claims are paired with a chunk "
        "from a different section or company. **No hand-labeled gold example was used**: "
        "every gold claim and every chunk retrieved for a candidate answer was excluded "
        f"({manifest.get('gold_claims_excluded', 0)} gold claims, "
        f"{manifest.get('gold_chunks_excluded', 0)} gold chunks, "
        f"{manifest.get('candidate_chunks_excluded', 0)} further candidate chunks).",
        "",
        "Splits are by company:",
        "",
    ]
    for split in ("train", "dev", "test"):
        tickers = ", ".join(splits.get(split, [])) or "none"
        by_label = counts.get(split, {})
        total = sum(by_label.values())
        detail = ", ".join(f"{k} {v}" for k, v in sorted(by_label.items()))
        lines.append(f"- **{split}**: {tickers} ({total} examples: {detail or 'none'})")
    lines += [
        "",
        "## Input format",
        "",
        "A sentence pair: the evidence chunk first, the claim second, truncating only the "
        f"evidence at {settings['max_length']} tokens. Labels: "
        + ", ".join(f"{i} = {label}" for i, label in ID2LABEL.items())
        + ". Use it through GroundCheck's `FineTunedVerifier`, which scores each "
        "retrieved chunk and aggregates (supported if any chunk supports the claim).",
        "",
        "## Metrics (synthetic splits, not the hand-labeled gold set)",
        "",
        "| Split | Accuracy | Macro-F1 | F1 supported | F1 contradicted | F1 unsupported |",
        "|---|---|---|---|---|---|",
    ]
    for split in ("dev", "test"):
        m = metrics.get(split)
        if m:
            lines.append(
                f"| {split} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
                f"{m['f1_supported']:.3f} | {m['f1_contradicted']:.3f} | "
                f"{m['f1_unsupported']:.3f} |"
            )
    lines += [
        "",
        "Synthetic-split scores measure how well the model learned the perturbation "
        "rules, not real-world accuracy. The real benchmark is "
        "`groundcheck eval-verifiers --verifier finetuned` on the hand-labeled gold set.",
        "",
        "## Training settings",
        "",
        *[f"- {k}: `{v}`" for k, v in settings.items()],
        "",
        "## Limitations",
        "",
        "- Rule-based labels are noisy (e.g. a numeric change that happens to match "
        "another value in the table is still labeled contradicted).",
        "- Supported claims are mostly near-verbatim sentences, so paraphrased LLM claims "
        "are harder for the model than the synthetic dev/test sets suggest.",
        "- Each chunk is judged on its own: a claim needing two chunks can't be supported.",
        "- English 10-K filings of eight large US companies only.",
        "",
    ]
    return "\n".join(lines)


def train(args: argparse.Namespace) -> dict:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ConfigError('training needs the [train] extra: pip install -e ".[train]"') from exc
    data_dir = Path(args.data_dir)
    train_examples = read_examples(data_dir / "train.jsonl")
    dev_examples = read_examples(data_dir / "dev.jsonl")
    test_examples = read_examples(data_dir / "test.jsonl")
    if not train_examples or not dev_examples:
        raise ConfigError(
            f"{data_dir} needs non-empty train.jsonl and dev.jsonl; "
            "run `groundcheck build-training-data` first"
        )
    if args.limit_train:
        train_examples = train_examples[: args.limit_train]
    if len(train_examples) < MIN_TRAINING_EXAMPLES and not args.limit_train:
        print(
            f"WARNING: only {len(train_examples)} training example(s) (fewer than "
            f"{MIN_TRAINING_EXAMPLES}); the fine-tuned verifier will be weak."
        )
    transformers.set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(LABEL_IDS), id2label=ID2LABEL, label2id=LABEL_IDS
    )
    splits = {"train": train_examples, "dev": dev_examples, "test": test_examples}
    datasets = {
        name: EncodedDataset(tokenizer, examples, args.max_length)
        for name, examples in splits.items()
    }
    fp16 = bool(args.fp16) and torch.cuda.is_available()
    training_args = transformers.TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        data_seed=args.seed,
        fp16=fp16,
        full_determinism=args.full_determinism,
        report_to=[],
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        data_collator=transformers.DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    train_output = trainer.train()
    metrics: dict[str, Any] = {
        "train_loss": float(getattr(train_output, "training_loss", float("nan")))
    }
    for split in ("dev", "test"):
        if len(datasets[split]):
            raw = trainer.evaluate(datasets[split], metric_key_prefix=split)
            metrics[split] = {k.removeprefix(f"{split}_"): v for k, v in raw.items()}
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    shutil.rmtree(out / "checkpoints", ignore_errors=True)
    settings = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "fp16": fp16,
        "full_determinism": args.full_determinism,
        "train_examples": len(datasets["train"]),
        "dev_examples": len(datasets["dev"]),
        "test_examples": len(datasets["test"]),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "groundcheck": __version__,
        "trained_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    (out / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "metrics.json").write_text(
        json.dumps({"settings": settings, "metrics": metrics}, indent=2) + "\n"
    )
    (out / "README.md").write_text(model_card(settings, metrics, manifest))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a small encoder as a three-way claim verifier."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fp16", action="store_true", help="mixed precision (CUDA only; ignored on CPU/MPS)"
    )
    parser.add_argument(
        "--full-determinism",
        action="store_true",
        help="PyTorch deterministic algorithms (slower, fully repeatable on one machine)",
    )
    parser.add_argument(
        "--limit-train", type=int, default=None, help="first N training examples (smoke test)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metrics = train(args)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2
    for split in ("dev", "test"):
        if split in metrics:
            m = metrics[split]
            print(f"{split}: accuracy {m['accuracy']:.3f}, macro-F1 {m['macro_f1']:.3f}")
    print(f"Saved the model, metrics.json and README.md (model card) to {args.out}")
    return 0
