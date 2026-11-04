"""Synthetic training data for the fine-tuned verifier.

Seeds are **supported claim/evidence pairs taken from the chunks themselves**:

- a prose sentence (8 to 45 words) is supported by the chunk it comes from;
- a table cell becomes a templated claim, "<row label> was <value> for <column
  header>.", supported by its table chunk.

From each seed, at most one example per label is made:

- **supported**: the seed, or a meaning-preserving rewrite of it (`synonym`, `rescale`);
- **contradicted**: one rule-based perturbation of the claim, same evidence
  (`numeric`, `scale`, `negation`, `year_swap`, `entity_swap`; see `perturb.py`);
- **unsupported**: the unchanged claim paired with a *different* chunk, from another
  Item of the same company (`other_section`) or from another company in the same split
  (`other_company`). The partner is the most word-similar candidate that shares none of
  the claim's numbers, so it is on topic but cannot back the claim.

Leakage rules, enforced in code:

1. **The hand-labeled gold set never enters training.** Every chunk that appears as a
   passage in `data/eval/gold_labels.jsonl` (and in the label file, if given) is
   excluded as seed and as evidence, and any example whose claim matches a gold claim
   (after whitespace normalization, case-insensitive) is dropped. The counts are in
   the manifest.
2. **Splits are by company.** Each company's examples go to exactly one split
   (`config/training.yaml`), and `other_company` partners come from the same split.

Output (`data/training/`, gitignored, re-creatable): `train.jsonl`, `dev.jsonl`,
`test.jsonl` and `manifest.json` (settings, seed, input hashes, counts by split,
label and rule, exclusion counts).
"""

import datetime as dt
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from groundcheck import __version__
from groundcheck.chunking.base import Chunk, read_chunks
from groundcheck.config import REPO_ROOT, ConfigError
from groundcheck.labeling.schema import Label
from groundcheck.retrieval.evaluate import normalize_for_match
from groundcheck.training.perturb import CONTRADICTING_REWRITES, SUPPORTED_REWRITES, table_rows
from groundcheck.verify.spans import SENTENCE_SPLIT_RE, content_words, numbers

DEFAULT_TRAINING_CONFIG = REPO_ROOT / "config" / "training.yaml"
DEFAULT_TRAINING_DIR = REPO_ROOT / "data" / "training"
LABEL_IDS: dict[str, int] = {"supported": 0, "contradicted": 1, "unsupported": 2}
SPLITS = ("train", "dev", "test")


def encode_pair(tokenizer, evidence: str, claim: str, max_length: int):
    """The one input format, shared by training and `FineTunedVerifier`: (evidence,
    claim) as a sentence pair, truncating only the evidence."""
    return tokenizer(evidence, claim, truncation="only_first", max_length=max_length)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 13
    splits: dict[Literal["train", "dev", "test"], list[str]]
    max_seeds_per_company: int = Field(default=1500, ge=1)
    supported_rewrite_rate: float = Field(default=0.5, ge=0, le=1)
    other_company_rate: float = Field(default=0.5, ge=0, le=1)
    partner_candidates: int = Field(default=60, ge=1)
    min_words: int = 8
    max_words: int = 45

    @model_validator(mode="after")
    def _disjoint(self) -> "TrainingConfig":
        seen: dict[str, str] = {}
        for split, tickers in self.splits.items():
            for ticker in tickers:
                if ticker in seen:
                    raise ValueError(f"{ticker} is in both {seen[ticker]} and {split}")
                seen[ticker] = split
        if not self.splits.get("train"):
            raise ValueError("the train split needs at least one company")
        return self

    def split_of(self, ticker: str) -> str | None:
        return next((s for s, tickers in self.splits.items() if ticker in tickers), None)


def load_training_config(path: Path | None = None) -> TrainingConfig:
    """config/training.yaml (or `path`)."""
    target = Path(path) if path is not None else DEFAULT_TRAINING_CONFIG
    if not target.is_file():
        raise ConfigError(f"training config not found: {target}")
    try:
        return TrainingConfig.model_validate(yaml.safe_load(target.read_text()) or {})
    except ValidationError as exc:
        raise ConfigError(f"invalid training config {target}:\n{exc}") from exc


class TrainingExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    split: Literal["train", "dev", "test"]
    ticker: str
    label: Label
    claim: str
    evidence: str
    evidence_chunk_id: str
    seed_chunk_id: str
    seed_kind: Literal["sentence", "table_cell"]
    rule: str  # "seed", a rewrite name, "other_section" or "other_company"


@dataclass
class Seed:
    chunk: Chunk
    claim: str
    kind: Literal["sentence", "table_cell"]


# --- Gold exclusion -------------------------------------------------------------------


def claim_key(text: str) -> str:
    return normalize_for_match(text).lower()


def gold_exclusions(paths: list[Path]) -> tuple[set[str], set[str]]:
    """(chunk IDs, normalized claims) found in the gold export and/or the label file."""
    chunk_ids: set[str] = set()
    claims: set[str] = set()
    for path in paths:
        if not Path(path).is_file():
            continue
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("claim"):
                claims.add(claim_key(record["claim"]))
            chunk_ids |= {p["chunk_id"] for p in record.get("passages", [])}
            chunk_ids |= {s["chunk_id"] for s in record.get("evidence", [])}
    return chunk_ids, claims


def candidate_chunk_ids(candidates_dir: Path | None) -> set[str]:
    """Every chunk retrieved for any candidate: the passages the gold claims came from,
    including candidates not (yet) labeled. Excluded too, to be safe."""
    if candidates_dir is None or not Path(candidates_dir).is_dir():
        return set()
    ids = set()
    for path in sorted(Path(candidates_dir).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            ids |= {c["chunk_id"] for c in record.get("retrieved", [])}
    return ids


# --- Seeds ----------------------------------------------------------------------------


def sentence_seeds(chunk: Chunk, min_words: int, max_words: int) -> list[Seed]:
    seeds = []
    for line in chunk.text.splitlines():
        line = line.strip()
        if not line or " | " in line or line.startswith("[") or line.isupper():
            continue
        for sentence in SENTENCE_SPLIT_RE.split(line):
            sentence = normalize_for_match(sentence)
            words = sentence.split()
            if not (min_words <= len(words) <= max_words) or sentence[-1] not in ".!":
                continue
            if not sentence[0].isupper() or sentence.lower().startswith(("item ", "see ")):
                continue
            seeds.append(Seed(chunk=chunk, claim=sentence, kind="sentence"))
    return seeds


def table_seeds(chunk: Chunk) -> list[Seed]:
    rows = table_rows(chunk.text)
    if len(rows) < 2:
        return []
    header = rows[0]
    seeds = []
    for row in rows[1:]:
        label = row[0]
        if not label or not re.search(r"[A-Za-z]{3}", label) or len(label) > 80:
            continue
        for column, value in zip(header[1:], row[1:], strict=False):
            if column and value and re.search(r"\d", value):
                claim = f"{label} was {value} for {column}."
                seeds.append(Seed(chunk=chunk, claim=normalize_for_match(claim), kind="table_cell"))
                break  # one cell per row keeps tables from dominating
    return seeds


def company_seeds(chunks: list[Chunk], config: TrainingConfig, rng: random.Random) -> list[Seed]:
    seeds = []
    for chunk in chunks:
        if chunk.kind == "table":
            seeds += table_seeds(chunk)
        else:
            seeds += sentence_seeds(chunk, config.min_words, config.max_words)
    rng.shuffle(seeds)
    return seeds[: config.max_seeds_per_company]


# --- Examples -------------------------------------------------------------------------


def partner_for(
    seed: Seed, pool: list[Chunk], rng: random.Random, n_candidates: int
) -> Chunk | None:
    """The most word-similar chunk from `pool` that shares none of the claim's numbers."""
    claim_numbers, claim_words = numbers(seed.claim), content_words(seed.claim)
    sample = rng.sample(pool, min(n_candidates, len(pool)))
    best, best_score = None, -1.0
    for chunk in sample:
        if claim_numbers & numbers(chunk.text):
            continue
        if normalize_for_match(seed.claim) in normalize_for_match(chunk.text):
            continue
        words = content_words(chunk.text)
        score = len(claim_words & words) / (len(claim_words | words) or 1)
        if score > best_score:
            best, best_score = chunk, score
    return best


def apply_first(
    rules: dict[str, Callable], claim: str, evidence: str, rng: random.Random
) -> tuple[str, str] | None:
    names = list(rules)
    rng.shuffle(names)
    for name in names:
        new = rules[name](claim, evidence, rng)
        if new and normalize_for_match(new) != claim:
            return name, normalize_for_match(new)
    return None


def build_examples(
    chunks_by_ticker: dict[str, list[Chunk]],
    config: TrainingConfig,
    excluded_chunks: set[str],
    gold_claims: set[str],
) -> tuple[list[TrainingExample], Counter]:
    rng = random.Random(config.seed)
    stats: Counter = Counter()
    usable = {}
    for ticker in sorted(chunks_by_ticker):
        if config.split_of(ticker) is None:
            stats["companies_not_in_any_split"] += 1
            continue
        kept = [c for c in chunks_by_ticker[ticker] if c.chunk_id not in excluded_chunks]
        stats["chunks_excluded_as_gold"] += len(chunks_by_ticker[ticker]) - len(kept)
        usable[ticker] = kept
    examples: list[TrainingExample] = []
    seen: set[tuple[str, str]] = set()

    def add(seed: Seed, label: str, claim: str, evidence: Chunk, rule: str) -> None:
        if claim_key(claim) in gold_claims:
            stats["examples_dropped_gold_claim"] += 1
            return
        if (claim, evidence.chunk_id) in seen:
            stats["examples_dropped_duplicate"] += 1
            return
        seen.add((claim, evidence.chunk_id))
        examples.append(
            TrainingExample(
                example_id=f"{seed.chunk.ticker}-{len(examples):06d}",
                split=config.split_of(seed.chunk.ticker),
                ticker=seed.chunk.ticker,
                label=label,
                claim=claim,
                evidence=evidence.text,
                evidence_chunk_id=evidence.chunk_id,
                seed_chunk_id=seed.chunk.chunk_id,
                seed_kind=seed.kind,
                rule=rule,
            )
        )

    for ticker, chunks in usable.items():
        split = config.split_of(ticker)
        same_split = [t for t in usable if t != ticker and config.split_of(t) == split]
        for seed in company_seeds(chunks, config, rng):
            claim, evidence = seed.claim, seed.chunk
            rewrite = None
            if rng.random() < config.supported_rewrite_rate:
                rewrite = apply_first(SUPPORTED_REWRITES, claim, evidence.text, rng)
            if rewrite:
                add(seed, "supported", rewrite[1], evidence, rewrite[0])
            else:
                add(seed, "supported", claim, evidence, "seed")
            contradicted = apply_first(CONTRADICTING_REWRITES, claim, evidence.text, rng)
            if contradicted:
                add(seed, "contradicted", contradicted[1], evidence, contradicted[0])
            else:
                stats["seeds_without_contradiction"] += 1
            use_other_company = same_split and rng.random() < config.other_company_rate
            if use_other_company:
                pool = [c for t in same_split for c in usable[t]]
                rule = "other_company"
            else:
                pool = [c for c in chunks if c.item != seed.chunk.item]
                rule = "other_section"
            partner = partner_for(seed, pool, rng, config.partner_candidates) if pool else None
            if partner is not None:
                add(seed, "unsupported", claim, partner, rule)
            else:
                stats["seeds_without_partner"] += 1
    return examples, stats


# --- Files ----------------------------------------------------------------------------


def sha256_or_none(path: Path) -> str | None:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None


def build_training_data(
    chunks_dir: Path,
    out_dir: Path,
    config: TrainingConfig,
    gold_paths: list[Path],
    candidates_dir: Path | None = None,
) -> dict:
    """Read chunks, build examples, write the split files and the manifest; return it."""
    chunks_by_ticker = {}
    for path in sorted(Path(chunks_dir).glob("*.jsonl")):
        chunks = read_chunks(path)
        if chunks:
            chunks_by_ticker[chunks[0].ticker] = chunks
    if not chunks_by_ticker:
        raise ConfigError(f"no chunks in {chunks_dir}; run `groundcheck chunk` first")
    configured = {t for tickers in config.splits.values() for t in tickers}
    missing = sorted(configured - set(chunks_by_ticker))
    if not set(config.splits["train"]) & set(chunks_by_ticker):
        raise ConfigError(f"none of the train companies {config.splits['train']} has chunks")
    gold_chunks, gold_claims = gold_exclusions(gold_paths)
    excluded = gold_chunks | candidate_chunk_ids(candidates_dir)
    examples, stats = build_examples(chunks_by_ticker, config, excluded, gold_claims)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        lines = [e.model_dump_json() for e in examples if e.split == split]
        (out_dir / f"{split}.jsonl").write_text("".join(line + "\n" for line in lines))
    counts = {
        split: dict(Counter(e.label for e in examples if e.split == split)) for split in SPLITS
    }
    manifest = {
        "groundcheck_version": __version__,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config": config.model_dump(),
        "companies_with_chunks": sorted(chunks_by_ticker),
        "companies_in_config_without_chunks": missing,
        "chunk_files_sha256": {
            t: sha256_or_none(Path(chunks_dir) / f"{t}.jsonl") for t in sorted(chunks_by_ticker)
        },
        "gold_files": {str(p): sha256_or_none(p) for p in gold_paths},
        "gold_claims_excluded": len(gold_claims),
        "gold_chunks_excluded": len(gold_chunks),
        "candidate_chunks_excluded": len(excluded - gold_chunks),
        "counts_by_split_and_label": counts,
        "counts_by_rule": dict(sorted(Counter(e.rule for e in examples).items())),
        "stats": dict(sorted(stats.items())),
        "label_ids": LABEL_IDS,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_examples(path: Path) -> list[TrainingExample]:
    if not Path(path).is_file():
        return []
    return [
        TrainingExample.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
