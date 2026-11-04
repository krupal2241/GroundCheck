"""Load project configuration and secrets.

Two kinds of settings live in two places:

- **Config files** (`config/*.yaml`, committed): which companies to use, which filing
  to pin for each one. These are validated with pydantic so a typo fails loudly with a
  clear message instead of silently producing wrong data later.
- **Secrets and per-machine values** (environment variables or a gitignored `.env`):
  the SEC user agent. These never go in the repo.
"""

import datetime as dt
import os
import re
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

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

    # A misspelled key (e.g. `acession:`) must fail loudly instead of being dropped.
    model_config = ConfigDict(extra="forbid")

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
    # Period of report: the actual fiscal year-end date of the 10-K (YYYY-MM-DD), added
    # by hand because SEC's fiscal_year_end is only approximate (JNJ reports "0103" for
    # a year ending 2025-12-28).
    period: str | None = None

    @field_validator("filing_date", "period")
    @classmethod
    def _check_iso_date(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                dt.date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"{value!r} must be a date like 2025-09-27") from exc
            if len(value) != 10:
                raise ValueError(f"{value!r} must be a date like 2025-09-27")
        return value

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
    model_config = ConfigDict(extra="forbid")

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
    try:
        return CompaniesConfig.model_validate(raw)
    except ValidationError as exc:
        # Shown by the CLI as a clean one-screen error instead of a traceback.
        raise ConfigError(f"invalid config {path}:\n{exc}") from exc


def load_env_file(env_file: Path | None = None) -> bool:
    """Load variables from a .env file into os.environ, without overriding real ones.

    The file is `env_file` if given, else `.env` in the *current working directory*.
    There is deliberately no upward search: python-dotenv's default (`find_dotenv`)
    walks up from the calling module's own location, so it would always find the
    repo's `.env` no matter where you run from, including inside tests.

    Returns True if a file was loaded. A missing default `.env` is fine (values can
    come from the shell); a missing explicitly requested file is an error.
    """
    path = Path(env_file) if env_file is not None else Path.cwd() / ".env"
    if not path.is_file():
        if env_file is not None:
            raise ConfigError(f"env file not found: {path}")
        return False
    load_dotenv(dotenv_path=path, override=False)
    return True


def get_sec_user_agent(env_file: Path | None = None) -> str:
    """Return SEC_USER_AGENT from the environment (or a .env file), checking its format.

    See `load_env_file` for which .env file is read. SEC's fair-access policy asks every
    automated client to identify itself with a name and contact email. We refuse to make
    requests without one.
    """
    load_env_file(env_file)
    value = os.environ.get("SEC_USER_AGENT", "").strip().strip('"')
    if not value:
        raise ConfigError(
            'SEC_USER_AGENT is not set. Add a line like SEC_USER_AGENT="Your Name '
            'you@example.com" to the .env file in the directory you run from '
            "(see .env.example), or pass --env-file."
        )
    if not USER_AGENT_RE.match(value):
        raise ConfigError(f'SEC_USER_AGENT={value!r} should look like "Your Name you@example.com".')
    return value


# --- LLM settings (config/llm.yaml) ---------------------------------------------------

DEFAULT_LLM_CONFIG_PATH = REPO_ROOT / "config" / "llm.yaml"


class LLMConfig(BaseModel):
    """Which local model generates candidates, and how. `--model` overrides `model`."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama", "hf-local", "fake"] = "ollama"
    model: str = "llama3.2:3b"  # the Ollama model; 3B fits an 8 GB MacBook Air
    hf_model: str = "Qwen/Qwen2.5-3B-Instruct"  # used when provider is hf-local
    ollama_url: str = "http://localhost:11434"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 42
    max_tokens: int = Field(default=768, gt=0)
    context_window: int = Field(default=8192, ge=512)  # Ollama num_ctx
    timeout_s: float = Field(default=300.0, gt=0)
    json_max_attempts: int = Field(default=3, ge=1, le=10)
    cache_dir: str = "data/cache/llm"  # relative to the repo root
    top_k: int = Field(default=5, ge=1, le=50)  # chunks retrieved per question

    @property
    def cache_path(self) -> Path:
        path = Path(self.cache_dir)
        return path if path.is_absolute() else REPO_ROOT / path


def load_llm_config(path: Path | None = None) -> LLMConfig:
    """Read config/llm.yaml (or `path`). A missing default file means all defaults."""
    target = Path(path) if path is not None else DEFAULT_LLM_CONFIG_PATH
    if not target.is_file():
        if path is not None:
            raise ConfigError(f"LLM config file not found: {target}")
        return LLMConfig()
    try:
        return LLMConfig.model_validate(yaml.safe_load(target.read_text()) or {})
    except ValidationError as exc:
        raise ConfigError(f"invalid LLM config {target}:\n{exc}") from exc
