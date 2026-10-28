import json
from pathlib import Path

import pytest

from groundcheck.config import Company, ConfigError
from groundcheck.edgar.download import (
    archive_url,
    download_all,
    download_filing,
    find_latest_10k,
    resolve_latest_10k,
    submissions_url,
)

FIXTURES = Path(__file__).parent / "fixtures" / "edgar"


class FakeSecClient:
    """Returns canned bodies by URL and records every URL requested."""

    def __init__(self, bodies):
        self.bodies = bodies
        self.urls = []

    def get(self, url):
        self.urls.append(url)
        return self.bodies[url]


def pinned_company(**overrides):
    fields = {
        "ticker": "EXM",
        "name": "Example Corp",
        "cik": 1234567,
        "role": "labeled",
        "accession": "0001234567-25-000010",
        "primary_document": "ex-20241231.htm",
    }
    fields.update(overrides)
    return Company(**fields)


def load_fixture():
    return json.loads((FIXTURES / "submissions_synthetic.json").read_text())


# --- URLs --------------------------------------------------------------------------


def test_submissions_url_pads_cik_to_ten_digits():
    assert submissions_url(320193) == "https://data.sec.gov/submissions/CIK0000320193.json"


def test_archive_url_strips_dashes_from_accession():
    assert archive_url(1234567, "0001234567-25-000010", "ex.htm") == (
        "https://www.sec.gov/Archives/edgar/data/1234567/000123456725000010/ex.htm"
    )


# --- Resolving the latest 10-K ------------------------------------------------------


def test_find_latest_10k_skips_amendments_and_other_forms():
    info = find_latest_10k("EXM", load_fixture())
    assert info.accession == "0001234567-25-000010"  # not the newer 10-K/A
    assert info.primary_document == "ex-20241231.htm"
    assert info.filing_date == "2025-02-20"
    assert info.report_date == "2024-12-31"
    assert info.fiscal_year_end == "1231"
    assert info.sec_name == "EXAMPLE CORP"


def test_find_latest_10k_errors_when_none_present():
    data = load_fixture()
    data["filings"]["recent"]["form"] = ["10-Q"] * 4
    with pytest.raises(ConfigError, match="no 10-K"):
        find_latest_10k("EXM", data)


def test_resolve_caches_submissions_index(tmp_path):
    company = pinned_company(accession=None, primary_document=None)
    url = submissions_url(company.cik)
    client = FakeSecClient({url: (FIXTURES / "submissions_synthetic.json").read_bytes()})

    first = resolve_latest_10k(client, company, raw_dir=tmp_path)
    second = resolve_latest_10k(client, company, raw_dir=tmp_path)

    assert first == second
    assert client.urls == [url]  # second call came from disk
    resolve_latest_10k(client, company, raw_dir=tmp_path, refresh=True)
    assert client.urls == [url, url]


# --- Downloading -------------------------------------------------------------------


def test_download_writes_file_and_meta(tmp_path):
    company = pinned_company()
    url = archive_url(company.cik, company.accession, company.primary_document)
    client = FakeSecClient({url: b"<html>10-K</html>"})

    assert download_filing(client, company, raw_dir=tmp_path) is True

    folder = tmp_path / "EXM" / "0001234567-25-000010"
    assert (folder / "ex-20241231.htm").read_bytes() == b"<html>10-K</html>"
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["url"] == url
    assert meta["bytes"] == len(b"<html>10-K</html>")
    assert len(meta["sha256"]) == 64
    assert not list(folder.glob("*.part"))  # no temp files left behind


def test_download_never_refetches_existing_file(tmp_path):
    company = pinned_company()
    url = archive_url(company.cik, company.accession, company.primary_document)
    client = FakeSecClient({url: b"<html>10-K</html>"})

    download_filing(client, company, raw_dir=tmp_path)
    assert download_filing(client, company, raw_dir=tmp_path) is False
    assert client.urls == [url]


def test_download_detects_modified_cached_file(tmp_path):
    company = pinned_company()
    url = archive_url(company.cik, company.accession, company.primary_document)
    client = FakeSecClient({url: b"<html>10-K</html>"})
    download_filing(client, company, raw_dir=tmp_path)

    (tmp_path / "EXM" / company.accession / company.primary_document).write_bytes(b"edited")
    with pytest.raises(ConfigError, match="sha256"):
        download_filing(client, company, raw_dir=tmp_path)


def test_download_all_refuses_when_any_company_unpinned(tmp_path):
    client = FakeSecClient({})
    companies = [
        pinned_company(),
        pinned_company(ticker="UNP", accession=None, primary_document=None),
    ]
    with pytest.raises(ConfigError, match="UNP"):
        download_all(client, companies, raw_dir=tmp_path)
    assert client.urls == []  # nothing fetched, not even the pinned one
