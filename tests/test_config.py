import pytest
from pydantic import ValidationError

from groundcheck.config import (
    DEFAULT_COMPANIES_PATH,
    CompaniesConfig,
    Company,
    ConfigError,
    get_sec_user_agent,
    load_companies,
)


def make_company(**overrides):
    fields = {"ticker": "AAPL", "name": "Apple Inc.", "cik": 320193, "role": "labeled"}
    fields.update(overrides)
    return Company(**fields)


def test_committed_companies_file_is_valid():
    config = load_companies(DEFAULT_COMPANIES_PATH)
    tickers = [c.ticker for c in config.companies]
    assert tickers == ["AAPL", "MSFT", "AMZN", "JPM", "XOM", "JNJ", "WMT", "BRK-B"]
    assert config.get("BRK-B").role == "stress_test"
    labeled = {c.ticker for c in config.companies if c.role == "labeled"}
    assert labeled == {"AAPL", "MSFT", "AMZN", "WMT", "XOM", "JNJ"}


def test_unpinned_company_is_allowed():
    assert not make_company().is_pinned


def test_pinned_company():
    company = make_company(accession="0000320193-24-000123", primary_document="aapl.htm")
    assert company.is_pinned


def test_bad_accession_format_rejected():
    with pytest.raises(ValidationError, match="accession"):
        make_company(accession="320193-24-123", primary_document="aapl.htm")


def test_half_pinned_company_rejected():
    with pytest.raises(ValidationError, match="both accession and primary_document"):
        make_company(accession="0000320193-24-000123")


def test_unknown_role_rejected():
    with pytest.raises(ValidationError):
        make_company(role="favorite")


def test_duplicate_tickers_rejected():
    with pytest.raises(ValidationError, match="duplicate tickers"):
        CompaniesConfig(companies=[make_company(), make_company()])


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_companies(tmp_path / "nope.yaml")


@pytest.fixture
def no_dotenv(monkeypatch, tmp_path):
    # Run from an empty directory so a developer's real .env is never picked up.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)


def test_user_agent_missing(no_dotenv):
    with pytest.raises(ConfigError, match="not set"):
        get_sec_user_agent()


def test_user_agent_without_email_rejected(no_dotenv, monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "JustAName")
    with pytest.raises(ConfigError, match="should look like"):
        get_sec_user_agent()


def test_user_agent_valid(no_dotenv, monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Jane Doe jane@example.com")
    assert get_sec_user_agent() == "Jane Doe jane@example.com"
