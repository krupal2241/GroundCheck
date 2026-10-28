import pytest

from groundcheck import __version__
from groundcheck.cli import main


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "usage: groundcheck" in capsys.readouterr().out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_download_refuses_unpinned_config(capsys, monkeypatch, tmp_path):
    # A config with no accession number: download must refuse before making any request.
    config = tmp_path / "companies.yaml"
    config.write_text(
        "companies:\n  - {ticker: AAPL, name: Apple Inc., cik: 320193, role: labeled}\n"
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    monkeypatch.setenv("SEC_USER_AGENT", "Jane Doe jane@example.com")
    assert main(["download", "--config", str(config), "--raw-dir", str(raw_dir)]) == 2
    err = capsys.readouterr().err
    assert "no pinned accession number" in err
    assert "AAPL" in err
    assert not any(raw_dir.iterdir())


def test_missing_user_agent_is_a_clean_error(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # keep a developer's real .env out of the test
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    assert main(["resolve-filings", "--raw-dir", str(tmp_path)]) == 2
    assert "SEC_USER_AGENT is not set" in capsys.readouterr().err
