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
   (after whitespace normalization, case-insensitive) is dropped. Chunk IDs are not
   enough: prose windows overlap by 50 words, a window split to fit the token limit
   has two IDs, and Items 7 and 8 repeat the same table rows. So every other chunk
   whose text **contains a gold evidence span** (whitespace-normalized, at least
   `MIN_SPAN_CHARS` characters) is excluded too. The counts are in the manifest.
2. **Splits are by company, and inside a training company by Item.** A company in
   `splits.train` trains, except for the Items listed for it under `dev_items`, which
   go to dev (model selection) instead. Prose windows never cross an Item, so train and
   dev share no chunk and no overlapping window; `check_item_disjoint` verifies that no
   (company, Item) pair is used by two splits, counting the evidence chunk of every
   example, and the build fails otherwise. `other_company` and `other_section` partners
   always come from the same split. Test companies are never in train or dev.
3. **A missing gold file is reported**, per path (`gold_files_missing` in the manifest;
   the command prints a warning for each), never skipped silently.

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
from groundcheck.training.perturb import (
    CONTRADICTING_REWRITES,
    SUPPORTED_REWRITES,
    YEAR_RE,
    table_rows,
)
from groundcheck.verify.spans import SENTENCE_SPLIT_RE, content_words, numbers

DEFAULT_TRAINING_CONFIG = REPO_ROOT / "config" / "training.yaml"
DEFAULT_TRAINING_DIR = REPO_ROOT / "data" / "training"
LABEL_IDS: dict[str, int] = {"supported": 0, "contradicted": 1, "unsupported": 2}
SPLITS = ("train", "dev", "test")
# Gold evidence spans shorter than this ("4%", "2025") occur in too many chunks to be
# evidence of anything; they are not used for the span-based exclusion.
MIN_SPAN_CHARS = 12


def encode_pair(tokenizer, evidence: str, claim: str, max_length: int):
    """The one input format, shared by training and `FineTunedVerifier`: (evidence,
    claim) as a sentence pair, truncating only the evidence."""
    return tokenizer(evidence, claim, truncation="only_first", max_length=max_length)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 13
    splits: dict[Literal["train", "dev", "test"], list[str]]
    # Items of a TRAIN company held out for dev: {ticker: [item, ...]}.
    dev_items: dict[str, list[str]] = Field(default_factory=dict)
    max_seeds_per_company: int = Field(default=1500, ge=1)
    max_dev_seeds_per_company: int = Field(default=400, ge=1)
    numeric_seed_share: float = Field(default=0.7, ge=0, le=1)
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
        for ticker, items in self.dev_items.items():
            if seen.get(ticker) != "train":
                raise ValueError(f"dev_items: {ticker} is not a company of the train split")
            if not items or len(set(items)) != len(items):
                raise ValueError(f"dev_items: {ticker} needs a list of distinct Items")
        if not self.splits.get("dev") and not self.dev_items:
            raise ValueError(
                "no dev set: list companies under splits.dev or held-out Items under dev_items"
            )
        return self

    def split_of(self, ticker: str, item: str | None = None) -> str | None:
        """The split of a company, or of one of its Items when `item` is given (a train
        company's `dev_items` go to dev)."""
        split = next((s for s, tickers in self.splits.items() if ticker in tickers), None)
        if split == "train" and item is not None and item in self.dev_items.get(ticker, []):
            return "dev"
        return split


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
    # Where the seed and the evidence come from, for the Item-disjointness check.
    seed_item: str = ""
    evidence_ticker: str = ""
    evidence_item: str = ""


@dataclass
class Seed:
    chunk: Chunk
    claim: str
    kind: Literal["sentence", "table_cell"]


# --- Gold exclusion -------------------------------------------------------------------


def claim_key(text: str) -> str:
    return normalize_for_match(text).lower()


@dataclass
class GoldExclusions:
    chunk_ids: set[str]
    claims: set[str]  # normalized (claim_key)
    spans: set[str]  # normalized gold evidence spans of at least MIN_SPAN_CHARS
    short_spans: int  # spans too short to match on
    missing: list[str]  # paths that do not exist

    def __iter__(self):  # (chunk IDs, claims): the original two-part result
        return iter((self.chunk_ids, self.claims))


def gold_exclusions(paths: list[Path]) -> GoldExclusions:
    """Chunk IDs, normalized claims and evidence spans found in the gold export and/or
    the label file; paths that do not exist are listed in `missing`."""
    chunk_ids: set[str] = set()
    claims: set[str] = set()
    spans: set[str] = set()
    short: set[str] = set()
    missing: list[str] = []
    for path in paths:
        if not Path(path).is_file():
            missing.append(str(path))
            continue
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("claim"):
                claims.add(claim_key(record["claim"]))
            chunk_ids |= {p["chunk_id"] for p in record.get("passages", [])}
            chunk_ids |= {s["chunk_id"] for s in record.get("evidence", [])}
            for span in record.get("evidence", []):
                text = normalize_for_match(span.get("text", ""))
                (spans if len(text) >= MIN_SPAN_CHARS else short).add(text)
    return GoldExclusions(chunk_ids, claims, spans, len(short - {""}), missing)


def chunks_sharing_gold_spans(
    chunks_by_ticker: dict[str, list[Chunk]], spans: set[str], already: set[str]
) -> set[str]:
    """IDs of chunks (not already excluded) whose text contains a gold evidence span:
    overlap neighbours, the other part of a split window, the same row in another Item."""
    if not spans:
        return set()
    found = set()
    for chunks in chunks_by_ticker.values():
        for chunk in chunks:
            if chunk.chunk_id in already:
                continue
            text = normalize_for_match(chunk.text)
            if any(span in text for span in spans):
                found.add(chunk.chunk_id)
    return found


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
            # wrong-context mode: the mismatched passages the model answered from.
            ids |= {c["chunk_id"] for c in record.get("context") or []}
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
            if any(ch.isdigit() for ch in words[0]):
                continue  # page furniture glued to a sentence: "K-106 (31) Business ..."
            seeds.append(Seed(chunk=chunk, claim=sentence, kind="sentence"))
    return seeds


def header_index(rows: list[list[str]]) -> int:
    """The header row: the first row with a year in a value column (multi-row headers
    put "Year ended December 31," above the years), else the first row."""
    for index, row in enumerate(rows[:4]):
        if any(YEAR_RE.search(cell) for cell in row[1:]):
            return index
    return 0


def table_seeds(chunk: Chunk) -> list[Seed]:
    rows = table_rows(chunk.text)
    if len(rows) < 2:
        return []
    start = header_index(rows)
    header = rows[start]
    seeds = []
    for row in rows[start + 1 :]:
        label = row[0]
        if not label or not re.search(r"[A-Za-z]{3}", label) or len(label) > 80:
            continue
        for column, value in zip(header[1:], row[1:], strict=False):
            if column.lower().startswith("page"):
                continue  # a table of contents: the "values" are page numbers
            if column and value and re.search(r"\d", value):
                claim = f"{label} was {value} for {column}."
                seeds.append(Seed(chunk=chunk, claim=normalize_for_match(claim), kind="table_cell"))
                break  # one cell per row keeps tables from dominating
    return seeds


def company_seeds(
    chunks: list[Chunk], config: TrainingConfig, rng: random.Random, limit: int | None = None
) -> list[Seed]:
    seeds = []
    for chunk in chunks:
        if chunk.kind == "table":
            seeds += table_seeds(chunk)
        else:
            seeds += sentence_seeds(chunk, config.min_words, config.max_words)
    rng.shuffle(seeds)
    # Gold claims are mostly about numbers, so seeds with a number are preferred: up to
    # `numeric_seed_share` of the quota, the rest filled with any remaining seeds.
    limit = config.max_seeds_per_company if limit is None else limit
    with_numbers = [s for s in seeds if numbers(s.claim)]
    chosen = with_numbers[: round(limit * config.numeric_seed_share)]
    taken = {id(s) for s in chosen}
    chosen += [s for s in seeds if id(s) not in taken][: limit - len(chosen)]
    rng.shuffle(chosen)
    return chosen


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
    # (ticker, split) -> usable chunks. A train company with `dev_items` has two groups.
    groups: dict[tuple[str, str], list[Chunk]] = {}
    for ticker in sorted(chunks_by_ticker):
        if config.split_of(ticker) is None:
            stats["companies_not_in_any_split"] += 1
            continue
        kept = [c for c in chunks_by_ticker[ticker] if c.chunk_id not in excluded_chunks]
        stats["chunks_excluded_as_gold"] += len(chunks_by_ticker[ticker]) - len(kept)
        for split in SPLITS:
            part = [c for c in kept if config.split_of(ticker, c.item) == split]
            if part:
                groups[(ticker, split)] = part
    examples: list[TrainingExample] = []
    seen: set[tuple[str, str]] = set()

    def add(seed: Seed, split: str, label: str, claim: str, evidence: Chunk, rule: str) -> None:
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
                split=split,
                ticker=seed.chunk.ticker,
                label=label,
                claim=claim,
                evidence=evidence.text,
                evidence_chunk_id=evidence.chunk_id,
                seed_chunk_id=seed.chunk.chunk_id,
                seed_kind=seed.kind,
                rule=rule,
                seed_item=seed.chunk.item,
                evidence_ticker=evidence.ticker,
                evidence_item=evidence.item,
            )
        )

    for (ticker, split), chunks in groups.items():
        # Chunks of the other companies in the same split (same split: no leakage).
        others = [c for (t, s), part in groups.items() if s == split and t != ticker for c in part]
        held_out = split == "dev" and ticker in config.dev_items
        limit = config.max_dev_seeds_per_company if held_out else config.max_seeds_per_company
        for seed in company_seeds(chunks, config, rng, limit):
            claim, evidence = seed.claim, seed.chunk
            rewrite = None
            if rng.random() < config.supported_rewrite_rate:
                rewrite = apply_first(SUPPORTED_REWRITES, claim, evidence.text, rng)
            if rewrite:
                add(seed, split, "supported", rewrite[1], evidence, rewrite[0])
            else:
                add(seed, split, "supported", claim, evidence, "seed")
            contradicted = apply_first(CONTRADICTING_REWRITES, claim, evidence.text, rng)
            if contradicted:
                add(seed, split, "contradicted", contradicted[1], evidence, contradicted[0])
            else:
                stats["seeds_without_contradiction"] += 1
            pools = {
                "other_company": others,
                "other_section": [c for c in chunks if c.item != seed.chunk.item],
            }
            first = "other_company" if rng.random() < config.other_company_rate else "other_section"
            second = "other_section" if first == "other_company" else "other_company"
            partner, rule = None, first
            for rule in (first, second):  # the other kind if the first has no candidate
                if pools[rule]:
                    partner = partner_for(seed, pools[rule], rng, config.partner_candidates)
                if partner is not None:
                    break
            if partner is not None:
                add(seed, split, "unsupported", claim, partner, rule)
            else:
                stats["seeds_without_partner"] += 1
    return examples, stats


def items_by_split(examples: list[TrainingExample]) -> dict[str, dict[str, list[str]]]:
    """{split: {ticker: [Items used, as seed or as evidence]}}."""
    used: dict[str, dict[str, set[str]]] = {split: {} for split in SPLITS}
    for e in examples:
        used[e.split].setdefault(e.ticker, set()).add(e.seed_item)
        used[e.split].setdefault(e.evidence_ticker or e.ticker, set()).add(e.evidence_item)
    return {
        split: {t: sorted(items) for t, items in sorted(by_ticker.items())}
        for split, by_ticker in used.items()
    }


def check_item_disjoint(examples: list[TrainingExample]) -> None:
    """No (company, Item) pair may be used by two splits, as seed or as evidence."""
    owner: dict[tuple[str, str], str] = {}
    for split, by_ticker in items_by_split(examples).items():
        for ticker, items in by_ticker.items():
            for item in items:
                other = owner.setdefault((ticker, item), split)
                if other != split:
                    raise ConfigError(
                        f"split leak: {ticker} Item {item} is used by both {other} and {split}"
                    )


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
    gold = gold_exclusions(gold_paths)
    gold_chunks, gold_claims = gold.chunk_ids, gold.claims
    by_id = gold_chunks | candidate_chunk_ids(candidates_dir)
    by_span = chunks_sharing_gold_spans(chunks_by_ticker, gold.spans, by_id)
    excluded = by_id | by_span
    examples, stats = build_examples(chunks_by_ticker, config, excluded, gold_claims)
    check_item_disjoint(examples)
    if not any(e.split == "dev" for e in examples):
        raise ConfigError(
            "the dev split is empty: the dev companies or the dev_items of the train "
            "companies have no usable chunks (check the Item names in the training config)"
        )
    have = {(c.ticker, c.item) for chunks in chunks_by_ticker.values() for c in chunks}
    dev_items_missing = sorted(
        f"{t} Item {i}"
        for t, items in config.dev_items.items()
        if t in chunks_by_ticker
        for i in items
        if (t, i) not in have
    )
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
        "gold_files_missing": gold.missing,
        "gold_claims_excluded": len(gold_claims),
        "gold_chunks_excluded": len(gold_chunks),
        "candidate_chunks_excluded": len(by_id - gold_chunks),
        "gold_spans_matched_on": len(gold.spans),
        "gold_spans_too_short_to_match": gold.short_spans,
        "chunks_excluded_sharing_gold_span": len(by_span),
        "dev_items_without_chunks": dev_items_missing,
        "items_by_split": items_by_split(examples),
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
