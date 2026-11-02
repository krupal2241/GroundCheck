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


def test_env_file_flag_is_used(capsys, monkeypatch, tmp_path):
    # --env-file is read; with a pinned-but-empty config nothing is downloaded, so the
    # command gets past the user-agent check without any network access.
    env_file = tmp_path / "custom.env"
    env_file.write_text('SEC_USER_AGENT="Jane Doe jane@example.com"\n')
    config = tmp_path / "companies.yaml"
    config.write_text("companies: []\n")
    args = ["download", "--config", str(config), "--raw-dir", str(tmp_path / "raw")]
    assert main([*args, "--env-file", str(env_file)]) == 0
    assert "Downloaded 0 filing(s)" in capsys.readouterr().out


def test_missing_env_file_flag_is_a_clean_error(capsys, tmp_path):
    assert main(["resolve-filings", "--env-file", str(tmp_path / "nope.env")]) == 2
    assert "env file not found" in capsys.readouterr().err
