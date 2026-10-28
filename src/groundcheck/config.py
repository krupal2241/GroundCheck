"""Load project configuration and secrets.

Two kinds of settings live in two places:

- **Config files** (`config/*.yaml`, committed): which companies to use, which filing
  to pin for each one. These are validated with pydantic so a typo fails loudly with a
  clear message instead of silently producing wrong data later.
- **Secrets and per-machine values** (environment variables or a gitignored `.env`):
  the SEC user agent. These never go in the repo.
"""

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPANIES_PATH = REPO_ROOT / "config" / "companies.yaml"

# SEC accession numbers look like 0000320193-24-000123:
# 10-digit filer ID, 2-digit year, 6-digit sequence.
ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")

# "Name email@example.com": at least one word, then something that looks like an email.
USER_AGENT_RE = re.compile(r"^\S.*\s+\S+@\S+\.\S+$")


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


class Company(BaseModel):
    """One company and the 10-K filing pinned for it."""

    ticker: str
    name: str
    cik: int = Field(gt=0, description="SEC Central Index Key, without leading zeros.")
    # labeled:     questions for the hand-labeled test set come from this company
    # unlabeled:   parsed and indexed, but not used for the labeled set
    # stress_test: only used to stress-test the parser
    role: Literal["labeled", "unlabeled", "stress_test"]
    # Filled in by you from `groundcheck resolve-filings`. None means "not pinned yet".
    accession: str | None = None
    primary_document: str | None = None
    filing_date: str | None = None
    # As reported by SEC, e.g. "0928" for a fiscal year ending September 28.
    fiscal_year_end: str | None = None

    @field_validator("accession")
    @classmethod
    def _check_accession(cls, value: str | None) -> str | None:
        if value is not None and not ACCESSION_RE.match(value):
            raise ValueError(f"accession {value!r} must look like 0000320193-24-000123")
        return value

    @model_validator(mode="after")
    def _pinned_together(self) -> "Company":
        # An accession number is useless without the document name inside it, and the
        # reverse; requiring both prevents half-pinned entries.
        if (self.accession is None) != (self.primary_document is None):
            raise ValueError(f"{self.ticker}: set both accession and primary_document, or neither")
        return self

    @property
    def is_pinned(self) -> bool:
        return self.accession is not None


class CompaniesConfig(BaseModel):
    companies: list[Company]

    @model_validator(mode="after")
    def _unique_tickers(self) -> "CompaniesConfig":
        tickers = [c.ticker for c in self.companies]
        duplicates = sorted({t for t in tickers if tickers.count(t) > 1})
        if duplicates:
            raise ValueError(f"duplicate tickers: {duplicates}")
        return self

    def get(self, ticker: str) -> Company:
        for company in self.companies:
            if company.ticker == ticker:
                return company
        raise ConfigError(f"unknown ticker {ticker!r}")


def load_companies(path: Path = DEFAULT_COMPANIES_PATH) -> CompaniesConfig:
    """Read and validate the companies YAML file."""
    try:
        raw = yaml.safe_load(Path(path).read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    return CompaniesConfig.model_validate(raw)


def get_sec_user_agent() -> str:
    """Return SEC_USER_AGENT from the environment (or .env), checking its format.

    SEC's fair-access policy asks every automated client to identify itself with a
    name and contact email. We refuse to make requests without one.
    """
    load_dotenv()  # no-op if there is no .env file; never overrides real env vars
    value = os.environ.get("SEC_USER_AGENT", "").strip().strip('"')
    if not value:
        raise ConfigError(
            'SEC_USER_AGENT is not set. Add a line like SEC_USER_AGENT="Your Name '
            'you@example.com" to your .env file (see .env.example).'
        )
    if not USER_AGENT_RE.match(value):
        raise ConfigError(f'SEC_USER_AGENT={value!r} should look like "Your Name you@example.com".')
    return value
