"""Parse a downloaded 10-K end to end: HTML → clean text → Item sections → JSON on disk.

Output, per company, in `data/parsed/` (gitignored, re-creatable):

- `<TICKER>.json`: a `ParsedFiling` (metadata, sections, warnings);
- `<TICKER>.txt`: the full cleaned text, for eyeballing and for writing gold strings.
"""

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from groundcheck.config import REPO_ROOT, Company, ConfigError
from groundcheck.edgar.download import DEFAULT_RAW_DIR, filing_path
from groundcheck.parsing.cleanup import remove_page_furniture
from groundcheck.parsing.html_text import html_to_text
from groundcheck.parsing.sections import split_sections

DEFAULT_PARSED_DIR = REPO_ROOT / "data" / "parsed"


class ParsedSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: str
    title: str
    text: str
    covers: list[str] = []


class ParsedFiling(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    accession: str
    primary_document: str
    period: str | None
    source_sha256: str
    removed_furniture_blocks: int
    front_matter_chars: int
    sections: list[ParsedSection]
    warnings: list[str]


def parse_html(company: Company, html: bytes) -> tuple[ParsedFiling, str]:
    """Parse one filing's HTML. Returns the parsed filing and the full cleaned text."""
    text, removed = remove_page_furniture(html_to_text(html))
    result = split_sections(text, label=company.ticker)
    parsed = ParsedFiling(
        ticker=company.ticker,
        accession=company.accession or "",
        primary_document=company.primary_document or "",
        period=company.period,
        source_sha256=hashlib.sha256(html).hexdigest(),
        removed_furniture_blocks=removed,
        front_matter_chars=result.front_matter_chars,
        sections=[
            ParsedSection(item=s.item, title=s.title, text=s.text, covers=s.covers)
            for s in result.sections
        ],
        warnings=result.warnings,
    )
    return parsed, text


def parse_company(
    company: Company, raw_dir: Path = DEFAULT_RAW_DIR, parsed_dir: Path = DEFAULT_PARSED_DIR
) -> ParsedFiling:
    source = filing_path(company, raw_dir)
    if not source.is_file():
        raise ConfigError(f"{company.ticker}: {source} not found; run `groundcheck download`")
    parsed, text = parse_html(company, source.read_bytes())
    parsed_dir.mkdir(parents=True, exist_ok=True)
    (parsed_dir / f"{company.ticker}.json").write_text(parsed.model_dump_json(indent=2) + "\n")
    (parsed_dir / f"{company.ticker}.txt").write_text(text + "\n")
    return parsed


def load_parsed(ticker: str, parsed_dir: Path = DEFAULT_PARSED_DIR) -> ParsedFiling:
    path = parsed_dir / f"{ticker}.json"
    if not path.is_file():
        raise ConfigError(f"{path} not found; run `groundcheck parse`")
    return ParsedFiling.model_validate_json(path.read_text())
