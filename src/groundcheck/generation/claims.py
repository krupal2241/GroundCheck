"""Claim splitting with a deterministic validator and bounded, problem-specific retries.

Why a validator: on the Mac, llama3.2:3b split "According to passage [1], the Company
repurchased 402 million shares of its common stock for $89.3 billion in fiscal 2025."
into eight "claims", five of them fragments ("The Company", "its common stock" twice,
"402 million", "billion"). A better prompt (few-shot, `prompts.claims_prompt`) fixes
most of that, but a 3B model still slips, and a fragment in the labeled set is worse
than a missing claim. So every proposed claim is checked by plain rules:

- at least `MIN_WORDS` words;
- at least one verb-like word (a known verb or auxiliary, or a word ending in -ed);
- not a duplicate or near-duplicate of an earlier claim (same numbers and >= 80% of
  the same content words);
- no "passage [n]" or bare "[n]" citation;
- every number in it also occurs in the answer (no added facts).

If any claim breaks a rule, the model is asked again (at most `MAX_RETRIES` times) with
a stricter message listing each rejected claim and why. The attempt with the most valid
claims wins (ties: the later one) and only its valid claims are kept. The counts go
into the candidate record (`ClaimFilterStats`), so the filter's effect is measurable.

The retry prompt differs from the first one, and the LLM cache key covers the full
prompt text, so retries and prompt changes never reuse a stale cached reply.
"""

import re
from dataclasses import replace

from pydantic import BaseModel, ConfigDict, Field

from groundcheck.generation.prompts import ABSTAIN_TEXT, ClaimSplit
from groundcheck.llm.base import LLMProvider, LLMRequest, LLMResponse
from groundcheck.llm.structured import StructuredOutputError, generate_structured

VALIDATOR_VERSION = "claims-validator-v1"
MIN_WORDS = 5
MAX_RETRIES = 3
NEAR_DUPLICATE_JACCARD = 0.8

# Verbs and auxiliaries that appear in answers about 10-K filings. A word ending in
# "-ed" also counts (increased, repurchased, reported, ...). Deliberately not "-s" or
# "-ing" endings: "shares", "sales" and "operating income" are not verbs.
VERBS = frozenset(
    """
    is are was were be been being am has have had do does did will would shall should
    can could may might must
    rose rise rises fell fall falls grew grow grows paid pay pays made make makes spent
    spend spends sold sell sells held hold holds bought buy buys earned earn earns lost
    lose loses won win wins took take takes gave give gives saw see sees met meet meets
    became become becomes remained remain remains totaled totalled total totals
    represent represents include includes reported report reports expect expects
    employ employs operate operates own owns lead leads led reflect reflects
    account accounts generate generates exceed exceeds amount amounts compare compares
    decline declines increase increases decrease decreases cost costs returned return
    returns issue issues repurchase repurchases recognize recognizes recorded record
    records drove drive drives contributed contribute contributes continue continues
    face faces depend depends rely relies plan plans intend intends announced announce
    """.split()  # noqa: SIM905
)

STOPWORDS = frozenset(
    """
    a an the of in on at to for from by with and or as its it it's their this that these
    those which who whom whose than then also during compared previous prior year years
    fiscal company company's
    """.split()  # noqa: SIM905
)

NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
CITATION_RE = re.compile(r"\bpassages?\s*\[?\s*\d+\s*\]?|\[\s*\d+\s*\]", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9$%][A-Za-z0-9$%,.'&-]*")


class RejectedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt: int  # 0 = the first request, 1..MAX_RETRIES = retries
    text: str
    reasons: list[str]


class AttemptStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposed: int
    rejected: int
    kept: int


class ClaimFilterStats(BaseModel):
    """Recorded in each candidate (`CandidateRecord.claim_filter`).

    `proposed`, `rejected` and `kept` describe the attempt whose claims were kept
    (kept = proposed - rejected); `per_attempt` has the same counts for every attempt.
    """

    model_config = ConfigDict(extra="forbid")

    validator_version: str = VALIDATOR_VERSION
    proposed: int
    rejected: int
    kept: int
    retries: int
    chosen_attempt: int
    per_attempt: list[AttemptStats] = Field(default_factory=list)
    rejections: list[RejectedClaim] = Field(default_factory=list)


def normalize_number(text: str) -> str:
    """'12,345' -> '12345', '89.30' -> '89.3', '5.0' -> '5'."""
    value = text.replace(",", "").rstrip(".")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def numbers_in(text: str) -> set[str]:
    return {normalize_number(m.group(0)) for m in NUMBER_RE.finditer(text)}


def words_of(text: str) -> list[str]:
    return [w.strip(".,;:!?\"'()").lower() for w in text.split() if w.strip(".,;:!?\"'()")]


def has_verb(text: str) -> bool:
    for word in words_of(text):
        word = word.removesuffix("'s")
        if word in VERBS or (len(word) >= 5 and word.endswith("ed") and word.isalpha()):
            return True
    return False


def content_words(text: str) -> set[str]:
    return {w.removesuffix("'s") for w in words_of(text)} - STOPWORDS


def near_duplicate(a: str, b: str) -> bool:
    if numbers_in(a) != numbers_in(b):
        return False
    wa, wb = content_words(a), content_words(b)
    if not wa or not wb:
        return wa == wb
    return len(wa & wb) / len(wa | wb) >= NEAR_DUPLICATE_JACCARD


def claim_problems(claim: str, answer: str, earlier: list[str]) -> list[str]:
    """Why `claim` is not acceptable (empty list = valid). `earlier` = claims kept so far."""
    reasons = []
    n_words = len(claim.split())
    if n_words < MIN_WORDS:
        reasons.append(f"a fragment: only {n_words} word(s), a claim needs at least {MIN_WORDS}")
    if not has_verb(claim):
        reasons.append("no verb: it is not a complete sentence")
    if CITATION_RE.search(claim):
        reasons.append('contains a passage citation such as "passage [1]"; leave citations out')
    missing = sorted(numbers_in(claim) - numbers_in(answer))
    if missing:
        reasons.append(f"contains number(s) not in the answer: {', '.join(missing)}")
    normalized = " ".join(words_of(claim))
    for i, other in enumerate(earlier, start=1):
        if normalized == " ".join(words_of(other)):
            reasons.append(f"duplicate of claim {i}")
            break
        if near_duplicate(claim, other):
            reasons.append(f"near-duplicate of claim {i} ({other!r})")
            break
    return reasons


Rejection = tuple[str, list[str]]  # (claim, reasons)


def validate_claims(claims: list[str], answer: str) -> tuple[list[str], list[Rejection]]:
    """(valid claims in order, [(rejected claim, reasons)])."""
    kept: list[str] = []
    rejected: list[Rejection] = []
    for claim in claims:
        reasons = claim_problems(claim, answer, kept)
        if reasons:
            rejected.append((claim, reasons))
        else:
            kept.append(claim)
    return kept, rejected


def is_abstention(answer: str) -> bool:
    return " ".join(answer.split()).strip() == ABSTAIN_TEXT


def stricter_prompt(base_prompt: str, rejected: list[Rejection], empty: bool) -> str:
    """The retry prompt: the original prompt, then each problem, then the hard rules."""
    lines = []
    if empty:
        lines.append("- You returned no claims, but the ANSWER states facts.")
    for claim, reasons in rejected:
        lines.append(f"- {claim!r}: " + "; ".join(reasons))
    return (
        f"{base_prompt}\n\nYour previous list was rejected. Problems:\n"
        + "\n".join(lines)
        + "\n\nWrite the whole list again. Every claim MUST be a complete sentence of at "
        f"least {MIN_WORDS} words with the company name as its subject and a verb. Use "
        "ONLY numbers that appear in the ANSWER, never repeat a fact, and never mention "
        'passages. Return only: {"claims": ["...", "..."]}'
    )


def split_claims(
    provider: LLMProvider,
    request: LLMRequest,
    answer: str,
    max_json_attempts: int = 3,
    max_retries: int = MAX_RETRIES,
) -> tuple[list[str], LLMResponse, ClaimFilterStats]:
    """Ask for claims, validate, retry with the problems listed; see the module docs.

    Raises `StructuredOutputError` only if the *first* request never yields valid JSON
    (a later failure keeps the best earlier attempt).
    """
    attempts: list[tuple[list[str], list[Rejection], LLMResponse]] = []
    current = request
    abstained = is_abstention(answer)
    for _ in range(max_retries + 1):
        try:
            split, response = generate_structured(
                provider, current, ClaimSplit, max_attempts=max_json_attempts
            )
        except StructuredOutputError:
            if not attempts:
                raise
            break
        kept, rejected = validate_claims(split.claims, answer)
        attempts.append((kept, rejected, response))
        empty = not split.claims and not abstained
        if not rejected and not empty:
            break
        current = replace(request, prompt=stricter_prompt(request.prompt, rejected, empty))
    # Most valid claims wins; ties go to the later (stricter) attempt.
    chosen = max(range(len(attempts)), key=lambda i: (len(attempts[i][0]), i))
    kept, rejected, response = attempts[chosen]
    stats = ClaimFilterStats(
        proposed=len(kept) + len(rejected),
        rejected=len(rejected),
        kept=len(kept),
        retries=len(attempts) - 1,
        chosen_attempt=chosen,
        per_attempt=[
            AttemptStats(proposed=len(k) + len(r), rejected=len(r), kept=len(k))
            for k, r, _ in attempts
        ],
        rejections=[
            RejectedClaim(attempt=i, text=text, reasons=reasons)
            for i, (_, r, _) in enumerate(attempts)
            for text, reasons in r
        ],
    )
    return kept, response, stats
