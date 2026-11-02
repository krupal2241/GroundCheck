from pathlib import Path

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


def _pinned_config(tmp_path):
    config = tmp_path / "companies.yaml"
    config.write_text(
        "companies:\n"
        "  - {ticker: EXCO, name: Example Corp, cik: 1, role: labeled,\n"
        "     accession: '0000000001-25-000001', primary_document: exco.htm,\n"
        "     period: '2025-12-31'}\n"
    )
    return config


def test_parse_command(capsys, tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "parsing" / "sections_10k.htm"
    raw = tmp_path / "raw" / "EXCO" / "0000000001-25-000001"
    raw.mkdir(parents=True)
    (raw / "exco.htm").write_bytes(fixture.read_bytes())
    parsed_dir = tmp_path / "parsed"
    args = ["parse", "--config", str(_pinned_config(tmp_path)), "--raw-dir", str(tmp_path / "raw")]
    assert main([*args, "--parsed-dir", str(parsed_dir)]) == 0
    out = capsys.readouterr().out
    assert "Item 1A" in out and "Risk Factors" in out
    assert "0 warning(s)" in out
    assert (parsed_dir / "EXCO.json").is_file() and (parsed_dir / "EXCO.txt").is_file()


def test_parse_prints_missing_item_warnings(capsys, tmp_path):
    raw = tmp_path / "raw" / "EXCO" / "0000000001-25-000001"
    raw.mkdir(parents=True)
    (raw / "exco.htm").write_text("<p>Item 7. MD&amp;A</p><p>Some text.</p>")
    args = ["parse", "--config", str(_pinned_config(tmp_path)), "--raw-dir", str(tmp_path / "raw")]
    assert main([*args, "--parsed-dir", str(tmp_path / "parsed")]) == 0
    out = capsys.readouterr().out
    assert "WARNING: missing Item 1A" in out and "WARNING: missing Item 8" in out
    assert "3 warning(s)" in out


def test_parse_without_download_is_a_clean_error(capsys, tmp_path):
    args = ["parse", "--config", str(_pinned_config(tmp_path)), "--raw-dir", str(tmp_path)]
    assert main([*args, "--parsed-dir", str(tmp_path / "parsed")]) == 2
    assert "run `groundcheck download`" in capsys.readouterr().err


def test_chunk_command(capsys, tmp_path):
    test_parse_command(capsys, tmp_path)  # writes tmp_path/parsed/EXCO.json
    chunks_dir = tmp_path / "chunks"
    args = ["chunk", "--config", str(_pinned_config(tmp_path)), "--parsed-dir"]
    assert main([*args, str(tmp_path / "parsed"), "--chunks-dir", str(chunks_dir)]) == 0
    out = capsys.readouterr().out
    assert "fixed-p250-o50-t120" in out and "EXCO" in out
    lines = (chunks_dir / "EXCO.jsonl").read_text().splitlines()
    assert lines and '"kind": "table"' in "".join(lines)


def test_chunk_bad_settings_is_a_clean_error(capsys, tmp_path):
    args = ["chunk", "--config", str(_pinned_config(tmp_path)), "--overlap", "300"]
    assert main(args) == 2
    assert "overlap" in capsys.readouterr().err


def test_chunk_before_parse_is_a_clean_error(capsys, tmp_path):
    args = ["chunk", "--config", str(_pinned_config(tmp_path)), "--parsed-dir", str(tmp_path)]
    assert main(args) == 2
    assert "run `groundcheck parse`" in capsys.readouterr().err
