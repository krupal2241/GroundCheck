"""Cost per 1,000 checks: $0 for local models, and an ESTIMATE for hosted ones.

Local verifiers (Ollama, MiniCheck, HHEM, the fine-tuned model) run on your own
machine, so their cost per check is $0 (electricity aside). For a hosted model there is
no measurement at all: `config/prices.yaml` holds prices *you* type in, and the
estimate is arithmetic on token counts. No API is called and no key is read.

Token counts are approximate: 1 token ~ 4 characters of English (`CHARS_PER_TOKEN`).
Real tokenizers differ by model (numbers and table pipes often cost more), which is one
more reason the result is always labeled an estimate.
"""

import math
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from groundcheck.config import REPO_ROOT, ConfigError

DEFAULT_PRICES_PATH = REPO_ROOT / "config" / "prices.yaml"
CHARS_PER_TOKEN = 4
ESTIMATE_LABEL = "ESTIMATE (arithmetic from config/prices.yaml; no hosted model was called)"


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    input_per_million_tokens: float = Field(ge=0)
    output_per_million_tokens: float = Field(ge=0)


class Prices(BaseModel):
    model_config = ConfigDict(extra="forbid")

    currency: str = "USD"
    assumed_output_tokens: int = Field(default=60, ge=0)
    models: list[ModelPrice] = []


class CostEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = ESTIMATE_LABEL
    model: str
    currency: str
    input_tokens_per_check: float
    output_tokens_per_check: float
    output_tokens_source: str  # "measured from the local prompted verifier" or "assumed ..."
    cost_per_1000_checks: float


def load_prices(path: Path | None = None) -> Prices:
    """config/prices.yaml (or `path`). A missing default file means no hosted models."""
    target = Path(path) if path is not None else DEFAULT_PRICES_PATH
    if not target.is_file():
        if path is not None:
            raise ConfigError(f"prices file not found: {target}")
        return Prices()
    try:
        return Prices.model_validate(yaml.safe_load(target.read_text()) or {})
    except ValidationError as exc:
        raise ConfigError(f"invalid prices file {target}:\n{exc}") from exc


def approx_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_costs(
    prices: Prices, input_tokens: float, measured_output_tokens: float | None
) -> list[CostEstimate]:
    """One estimate per priced model, for checks averaging `input_tokens` in."""
    if measured_output_tokens is not None:
        out_tokens, source = measured_output_tokens, "measured from the local prompted verifier"
    else:
        out_tokens, source = float(prices.assumed_output_tokens), "assumed in prices.yaml"
    return [
        CostEstimate(
            model=m.name,
            currency=prices.currency,
            input_tokens_per_check=round(input_tokens, 1),
            output_tokens_per_check=round(out_tokens, 1),
            output_tokens_source=source,
            cost_per_1000_checks=round(
                1000
                * (
                    input_tokens * m.input_per_million_tokens
                    + out_tokens * m.output_per_million_tokens
                )
                / 1_000_000,
                6,
            ),
        )
        for m in prices.models
    ]
