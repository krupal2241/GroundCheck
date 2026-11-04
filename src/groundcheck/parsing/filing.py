"""Parse a downloaded 10-K end to end: HTML → clean text → Item sections → JSON on disk.

Output, per company, in `data/parsed/` (gitignored, re-creatable):

- `<TICKER>.json`: a `ParsedFiling` (metadata, sections, warnings);
- `<TICKER>.txt`: the full cleaned text, for eyeballing and for writing gold strings.
"""

import hashlib
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from groundcheck.config import REPO_ROOT, Company, ConfigError
from groundcheck.edgar.download import DEFAULT_RAW_DIR, filing_path
from groundcheck.parsing.cleanup import remove_page_furniture, remove_repeated_lines
from groundcheck.parsing.html_text import html_to_text
from groundcheck.parsing.recovery import recover_cross_reference_layout
from groundcheck.parsing.sections import check_section_sizes, split_sections

logger = logging.getLogger(__name__)

DEFAULT_PARSED_DIR = REPO_ROOT / "data" / "parsed"


class ParsedSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: str
    title: str
    text: str
    covers: list[str] = []
    recovered: bool = False  # relabeled by cross-reference layout recovery


class ParsedFiling(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    accession: str
    primary_document: str
    period: str | None
    source_sha256: str
    removed_furniture_blocks: int
    removed_header_lines: int = 0  # running headers repeated MIN_REPEATS+ times
    front_matter_chars: int
    sections: list[ParsedSection]
    warnings: list[str]
    notes: list[str] = []  # informational, e.g. which sections were recovered


def parse_html(company: Company, html: bytes) -> tuple[ParsedFiling, str]:
    """Parse one filing's HTML. Returns the parsed filing and the full cleaned text.

    Steps: HTML → text, page furniture and repeated running headers removed, Item
    split, size checks, then cross-reference layout recovery. Size warnings describe
    the split as found; after a recovery the sizes are checked again and any new
    problem is added, so a warning is never silently dropped.
    """
    label = company.ticker
    text, removed = remove_page_furniture(html_to_text(html))
    text, header_lines = remove_repeated_lines(text, label=label)
    result = split_sections(text, label=label)
    warnings = list(result.warnings)
    for message in check_section_sizes(result.sections):
        logger.warning("%s: %s", label, message)
        warnings.append(message)
    recovery = recover_cross_reference_layout(result.sections, label=label, full_text=text)
    warnings.extend(recovery.warnings)
    if recovery.notes:
        for message in check_section_sizes(recovery.sections):
            if message not in warnings:
                logger.warning("%s: %s", label, message)
                warnings.append(message)
    parsed = ParsedFiling(
        ticker=company.ticker,
        accession=company.accession or "",
        primary_document=company.primary_document or "",
        period=company.period,
        source_sha256=hashlib.sha256(html).hexdigest(),
        removed_furniture_blocks=removed,
        removed_header_lines=header_lines,
        front_matter_chars=result.front_matter_chars,
        sections=[
            ParsedSection(
                item=s.item, title=s.title, text=s.text, covers=s.covers, recovered=s.recovered
            )
            for s in recovery.sections
        ],
        warnings=warnings,
        notes=recovery.notes,
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
