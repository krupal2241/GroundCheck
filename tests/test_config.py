import os

import dotenv
import dotenv.main
import pytest
from pydantic import ValidationError

from groundcheck.config import (
    DEFAULT_COMPANIES_PATH,
    REPO_ROOT,
    CompaniesConfig,
    Company,
    ConfigError,
    get_sec_user_agent,
    load_companies,
    load_env_file,
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
    # conftest.py already does this for every test; repeated here because these tests
    # depend on it.
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


# --- .env loading (regression: a developer's real .env leaked into tests) -----------


def _write_env(path, value="Env File env@example.com"):
    path.write_text(f'SEC_USER_AGENT="{value}"\n')
    return path


def test_env_loading_never_searches_upward(no_dotenv, monkeypatch):
    # python-dotenv's find_dotenv() walks up from the *calling module's* directory, so a
    # bare load_dotenv() in config.py always found the repo-root .env, even after a
    # test chdir'd away. Fail loudly if anything falls back to that search again.
    def forbidden(*args, **kwargs):
        raise AssertionError("find_dotenv must not be used; .env loading must be explicit")

    monkeypatch.setattr(dotenv, "find_dotenv", forbidden)
    monkeypatch.setattr(dotenv.main, "find_dotenv", forbidden)
    with pytest.raises(ConfigError, match="not set"):
        get_sec_user_agent()


def test_env_in_parent_directory_is_ignored(no_dotenv, tmp_path, monkeypatch):
    # Mirrors the bug: a .env exists higher up (like the repo root) but not in the
    # directory we run from. It must not be loaded.
    _write_env(tmp_path / ".env")
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    assert load_env_file() is False
    with pytest.raises(ConfigError, match="not set"):
        get_sec_user_agent()


def test_repo_root_env_is_never_the_default(no_dotenv, tmp_path):
    # Even if the developer has a real .env in the repo root, the default lookup
    # only checks the current directory (a temp folder here).
    assert tmp_path.resolve() != REPO_ROOT
    assert load_env_file() is False
    assert "SEC_USER_AGENT" not in os.environ


def test_env_in_current_directory_is_loaded(no_dotenv, tmp_path):
    _write_env(tmp_path / ".env")
    assert get_sec_user_agent() == "Env File env@example.com"


def test_explicit_env_file_is_loaded(no_dotenv, tmp_path):
    (tmp_path / "elsewhere").mkdir()
    env_file = _write_env(tmp_path / "elsewhere" / "custom.env", "Custom Name c@example.com")
    assert get_sec_user_agent(env_file) == "Custom Name c@example.com"


def test_missing_explicit_env_file_is_an_error(no_dotenv, tmp_path):
    with pytest.raises(ConfigError, match="env file not found"):
        get_sec_user_agent(tmp_path / "missing.env")


def test_real_env_var_wins_over_env_file(no_dotenv, tmp_path, monkeypatch):
    _write_env(tmp_path / ".env", "From File f@example.com")
    monkeypatch.setenv("SEC_USER_AGENT", "From Shell s@example.com")
    assert get_sec_user_agent() == "From Shell s@example.com"
