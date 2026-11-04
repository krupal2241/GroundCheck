"""`groundcheck label-plan`: a balanced sample of candidates to label.

Labeling every candidate of every mode is more work than one person can do well, and
labeling them in file order gives whatever balance the files happen to have. A plan is
an ordered list of candidate IDs chosen greedily so that companies, question types and
generation modes stay as even as possible:

    repeat until the target is reached:
        pick the remaining candidate whose (company, qtype, mode) are the least used
        so far, i.e. the lowest  n_company/companies_share + n_qtype/... + n_mode/...
        ties: a seeded hash of the candidate ID (deterministic, not file order)

Counts are in candidates for `--questions N`, in claims for `--claims N`. Candidates
with no claims (abstentions) are never planned: there is nothing to label. Already
labeled claims count towards the balance (so a plan made after a first session fills
the gaps), and `--only-unlabeled` style resumption works inside a plan.

The plan is written as JSON (`data/labels/label_plan.json` by default) and
`groundcheck label --plan` walks through exactly those candidates, in plan order.
"""

import datetime as dt
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from groundcheck.config import REPO_ROOT, ConfigError
from groundcheck.generation.modes import stable_int
from groundcheck.generation.pipeline import CandidateRecord

DEFAULT_PLAN_PATH = REPO_ROOT / "data" / "labels" / "label_plan.json"
DIMENSIONS = ("ticker", "qtype", "mode")


@dataclass
class Plan:
    candidate_ids: list[str]
    target: dict  # {"questions": N} or {"claims": N}
    seed: int
    reached: bool

    def to_json(self, candidates: list[CandidateRecord]) -> str:
        by_id = {c.candidate_id: c for c in candidates}
        chosen = [by_id[i] for i in self.candidate_ids]
        return (
            json.dumps(
                {
                    "version": 1,
                    "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                    "target": self.target,
                    "seed": self.seed,
                    "reached": self.reached,
                    "candidates": len(chosen),
                    "claims": sum(len(c.claims) for c in chosen),
                    "candidate_ids": self.candidate_ids,
                },
                indent=2,
            )
            + "\n"
        )


def _key(c: CandidateRecord, dim: str) -> str:
    return getattr(c, dim)


def make_plan(
    candidates: list[CandidateRecord],
    questions: int | None = None,
    claims: int | None = None,
    seed: int = 0,
    already_labeled: set[str] | None = None,
) -> Plan:
    if (questions is None) == (claims is None):
        raise ConfigError("give exactly one of --questions or --claims")
    target = questions if questions is not None else claims
    if target is None or target < 1:
        raise ConfigError("the target must be at least 1")
    pool = [c for c in candidates if c.claims]
    if not pool:
        raise ConfigError("no candidate has claims to label")
    labeled = already_labeled or set()
    # Every value of each dimension that exists in the pool gets an equal share.
    values = {dim: sorted({_key(c, dim) for c in pool}) for dim in DIMENSIONS}
    counts = {dim: Counter() for dim in DIMENSIONS}

    def weight(c: CandidateRecord) -> int:
        return len(c.claims) if claims is not None else 1

    # Labeled claims already in the set count towards the balance.
    for c in pool:
        done = sum(claim.claim_id in labeled for claim in c.claims)
        if done:
            for dim in DIMENSIONS:
                counts[dim][_key(c, dim)] += done if claims is not None else 1
    remaining = [c for c in pool if not all(cl.claim_id in labeled for cl in c.claims)]
    chosen: list[CandidateRecord] = []
    total = 0
    while remaining and total < target:

        def cost(c: CandidateRecord) -> tuple[float, int]:
            imbalance = sum(counts[d][_key(c, d)] * len(values[d]) for d in DIMENSIONS)
            return imbalance, stable_int(f"{seed}:{c.candidate_id}")

        best = min(remaining, key=cost)
        remaining.remove(best)
        chosen.append(best)
        total += weight(best)
        for dim in DIMENSIONS:
            counts[dim][_key(best, dim)] += weight(best)
    kind = {"questions": questions} if questions is not None else {"claims": claims}
    return Plan([c.candidate_id for c in chosen], kind, seed, total >= target)


def balance_table(candidates: list[CandidateRecord], unit: str = "claims") -> str:
    """Counts by company, qtype and mode for a set of candidates."""
    lines = []
    for dim, title in zip(DIMENSIONS, ("By company", "By question type", "By mode"), strict=True):
        counter: Counter = Counter()
        for c in candidates:
            counter[_key(c, dim)] += len(c.claims) if unit == "claims" else 1
        lines.append(f"{title} ({unit}):")
        lines += [f"  {key:<20} {counter[key]:>5}" for key in sorted(counter)]
    return "\n".join(lines)


def load_plan(path: Path) -> list[str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"plan file not found: {path}; run `groundcheck label-plan`") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc.msg}") from exc
    ids = data.get("candidate_ids") if isinstance(data, dict) else None
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ConfigError(f"{path} has no candidate_ids list")
    return ids
