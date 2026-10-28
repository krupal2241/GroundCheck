"""Find and download each company's pinned 10-K, caching everything on disk.

Two steps, on purpose:

1. `resolve_latest_10k` reads SEC's submissions index for a company and reports its
   latest 10-K (accession number, primary document, filing date, fiscal year end).
   You paste those values into config/companies.yaml. Pinning them means the corpus
   never changes underneath you when a company files a new 10-K.
2. `download_filing` fetches exactly the pinned document from the EDGAR Archives.

Cache layout (under data/raw/, which is gitignored):

    data/raw/_submissions/CIK0000320193.json        submissions index (per company)
    data/raw/AAPL/0000320193-24-000123/aapl.htm     the filing itself
    data/raw/AAPL/0000320193-24-000123/meta.json    URL, sha256, size, download time

A file that is already on disk is never downloaded again.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from groundcheck.config import REPO_ROOT, Company, ConfigError
from groundcheck.edgar.client import SecClient

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{document}"


@dataclass(frozen=True)
class FilingInfo:
    """What SEC reports about one company's latest 10-K."""

    ticker: str
    sec_name: str  # the name SEC has on file for this CIK, to catch a wrong CIK
    accession: str
    primary_document: str
    filing_date: str
    report_date: str
    fiscal_year_end: str


def submissions_url(cik: int) -> str:
    return SUBMISSIONS_URL.format(cik=cik)


def archive_url(cik: int, accession: str, document: str) -> str:
    return ARCHIVES_URL.format(
        cik=cik, accession_no_dashes=accession.replace("-", ""), document=document
    )


def find_latest_10k(ticker: str, submissions: dict) -> FilingInfo:
    """Pick the most recent original 10-K from a submissions JSON document.

    SEC lists recent filings as parallel arrays (form[i], accessionNumber[i], ...).
    We only accept form "10-K": amendments ("10-K/A") often contain just a few
    changed pages, not the full report.
    """
    recent = submissions["filings"]["recent"]
    candidates = [i for i, form in enumerate(recent["form"]) if form == "10-K"]
    if not candidates:
        raise ConfigError(
            f"{ticker}: no 10-K in SEC's recent filings list. Older filings are listed in "
            "separate files, which this tool does not read yet."
        )
    # Don't rely on SEC's ordering; pick the latest filing date explicitly.
    latest = max(candidates, key=lambda i: recent["filingDate"][i])
    return FilingInfo(
        ticker=ticker,
        sec_name=submissions.get("name", ""),
        accession=recent["accessionNumber"][latest],
        primary_document=recent["primaryDocument"][latest],
        filing_date=recent["filingDate"][latest],
        report_date=recent["reportDate"][latest],
        fiscal_year_end=submissions.get("fiscalYearEnd", ""),
    )


def load_submissions(
    client: SecClient, company: Company, raw_dir: Path = DEFAULT_RAW_DIR, refresh: bool = False
) -> dict:
    """Return the submissions JSON for a company, from the disk cache if present.

    The index changes whenever the company files something, so `refresh=True`
    re-fetches it on request. Filings themselves never change and are never refetched.
    """
    path = raw_dir / "_submissions" / f"CIK{company.cik:010d}.json"
    if path.exists() and not refresh:
        logger.info("%s: using cached submissions index %s", company.ticker, path)
    else:
        _atomic_write(path, client.get(submissions_url(company.cik)))
    return json.loads(path.read_text())


def resolve_latest_10k(
    client: SecClient, company: Company, raw_dir: Path = DEFAULT_RAW_DIR, refresh: bool = False
) -> FilingInfo:
    return find_latest_10k(company.ticker, load_submissions(client, company, raw_dir, refresh))


def filing_path(company: Company, raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    if not company.is_pinned:
        raise ConfigError(f"{company.ticker}: no accession pinned in config/companies.yaml")
    return raw_dir / company.ticker / company.accession / company.primary_document


def download_filing(client: SecClient, company: Company, raw_dir: Path = DEFAULT_RAW_DIR) -> bool:
    """Download the pinned 10-K for one company. Returns False if it was already cached.

    If the file exists, it is not downloaded again. If meta.json recorded a sha256 and
    the file no longer matches it, we stop with an error rather than silently using (or
    silently replacing) a changed file.
    """
    path = filing_path(company, raw_dir)
    meta_path = path.parent / "meta.json"

    if path.exists():
        if meta_path.exists():
            expected = json.loads(meta_path.read_text())["sha256"]
            actual = _sha256(path.read_bytes())
            if actual != expected:
                raise ConfigError(
                    f"{path} does not match the sha256 in {meta_path}. The cached file was "
                    "modified; delete the folder to download it again."
                )
        logger.info("%s: already downloaded, skipping (%s)", company.ticker, path)
        return False

    url = archive_url(company.cik, company.accession, company.primary_document)
    content = client.get(url)
    _atomic_write(path, content)
    meta = {
        "ticker": company.ticker,
        "cik": company.cik,
        "accession": company.accession,
        "primary_document": company.primary_document,
        "url": url,
        "sha256": _sha256(content),
        "bytes": len(content),
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _atomic_write(meta_path, (json.dumps(meta, indent=2) + "\n").encode())
    logger.info("%s: downloaded %s (%d bytes)", company.ticker, url, len(content))
    return True


def download_all(
    client: SecClient, companies: list[Company], raw_dir: Path = DEFAULT_RAW_DIR
) -> dict[str, bool]:
    """Download every company's pinned filing. Refuses to start if any are unpinned."""
    unpinned = [c.ticker for c in companies if not c.is_pinned]
    if unpinned:
        raise ConfigError(
            f"these companies have no pinned accession number: {', '.join(unpinned)}. "
            "Run `groundcheck resolve-filings` and paste the values into "
            "config/companies.yaml first."
        )
    return {c.ticker: download_filing(client, c, raw_dir) for c in companies}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    """Write to a temporary file, then rename it into place.

    A rename is atomic on the same filesystem, so an interrupted download (Ctrl-C,
    lost connection) never leaves a half-written file that later looks "cached".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)
