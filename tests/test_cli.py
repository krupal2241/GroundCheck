import sys
from pathlib import Path

import pytest

from groundcheck import __version__, cli
from groundcheck.chunking.base import read_chunks
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
    assert "WARNING: Item 7 is only" in out  # the size check counts too
    assert "4 warning(s)" in out


def test_parse_without_download_is_a_clean_error(capsys, tmp_path):
    args = ["parse", "--config", str(_pinned_config(tmp_path)), "--raw-dir", str(tmp_path)]
    assert main([*args, "--parsed-dir", str(tmp_path / "parsed")]) == 2
    assert "run `groundcheck download`" in capsys.readouterr().err


def test_chunk_command(capsys, tmp_path):
    test_parse_command(capsys, tmp_path)  # writes tmp_path/parsed/EXCO.json
    chunks_dir = tmp_path / "chunks"
    args = ["chunk", "--config", str(_pinned_config(tmp_path)), "--tokenizer", "none"]
    args += ["--parsed-dir", str(tmp_path / "parsed"), "--chunks-dir", str(chunks_dir)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "fixed-p250-o50-t120, no token limit" in out and "EXCO" in out
    lines = (chunks_dir / "EXCO.jsonl").read_text().splitlines()
    assert lines and '"kind": "table"' in "".join(lines)


def test_chunk_command_splits_with_the_token_counter(capsys, monkeypatch, tmp_path):
    # A fake tokenizer (3 tokens per word) instead of bge's, so no download is needed.
    monkeypatch.setattr(cli, "load_bge_token_counter", lambda: lambda t: 3 * len(t.split()))
    test_parse_command(capsys, tmp_path)
    chunks_dir = tmp_path / "chunks"
    args = ["chunk", "--config", str(_pinned_config(tmp_path)), "--max-tokens", "300"]
    args += ["--parsed-dir", str(tmp_path / "parsed"), "--chunks-dir", str(chunks_dir)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "fixed-p250-o50-t120-k300, max 300 tokens" in out
    assert "split part(s)" in out and "WARNING" not in out
    chunks = read_chunks(chunks_dir / "EXCO.jsonl")
    assert all(3 * len(c.text.split()) <= 300 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_without_the_embed_extra_is_a_clean_error(capsys, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "transformers", None)  # import fails
    assert main(["chunk", "--config", str(_pinned_config(tmp_path))]) == 2
    assert "--tokenizer none" in capsys.readouterr().err


def test_chunk_bad_settings_is_a_clean_error(capsys, tmp_path):
    args = ["chunk", "--config", str(_pinned_config(tmp_path)), "--overlap", "300"]
    assert main(args) == 2
    assert "overlap" in capsys.readouterr().err


def test_chunk_before_parse_is_a_clean_error(capsys, tmp_path):
    args = ["chunk", "--config", str(_pinned_config(tmp_path)), "--parsed-dir", str(tmp_path)]
    assert main([*args, "--tokenizer", "none"]) == 2
    assert "run `groundcheck parse`" in capsys.readouterr().err


def test_index_and_retrieve_commands_with_fake_embedder(capsys, tmp_path):
    test_chunk_command(capsys, tmp_path)  # writes tmp_path/chunks/EXCO.jsonl
    common = ["--chunks-dir", str(tmp_path / "chunks"), "--index-dir", str(tmp_path / "index")]
    common += ["--embedder", "hashing"]
    config = ["--config", str(_pinned_config(tmp_path))]
    assert main(["index", *config, *common]) == 0
    assert "hashing-256" in capsys.readouterr().out
    assert main(["retrieve", "EXCO", "What were total net sales?", "-k", "3", *common]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 3 and out[0].startswith(" 1. ")
    assert all("EXCO-" in line for line in out)


def _write_questions(tmp_path, gold):
    path = tmp_path / "questions.yaml"
    path.write_text(
        "version: 1\nquestions:\n"
        "  - id: EXCO-01\n    ticker: EXCO\n    qtype: numeric_lookup\n"
        "    question: What did Example Corp design and sell?\n"
        f"    gold_evidence: ['{gold}']\n"
        "  - {id: EXCO-02, ticker: EXCO, qtype: table}\n"
    )
    return path


def test_eval_retrieval_command(capsys, tmp_path):
    test_index_and_retrieve_commands_with_fake_embedder(capsys, tmp_path)
    questions = _write_questions(tmp_path, "Example Corp designs and sells widgets.")
    common = ["--chunks-dir", str(tmp_path / "chunks"), "--index-dir", str(tmp_path / "index")]
    common += ["--embedder", "hashing", "--questions", str(questions)]
    assert main(["eval-retrieval", *common, "--check"]) == 0
    assert "Gold evidence check: OK" in capsys.readouterr().out
    out_path = tmp_path / "report.json"
    assert main(["eval-retrieval", *common, "--out", str(out_path)]) == 0
    out = capsys.readouterr().out
    assert "1 of 2 question(s) filled" in out
    assert "overall" in out and "R@10" in out and "1 unfilled" in out
    assert '"first_gold_rank"' in out_path.read_text()


def test_eval_retrieval_check_reports_typos(capsys, tmp_path):
    test_chunk_command(capsys, tmp_path)
    questions = _write_questions(tmp_path, "Example Corp designs gadgets.")
    args = ["eval-retrieval", "--chunks-dir", str(tmp_path / "chunks"), "--questions"]
    assert main([*args, str(questions), "--check"]) == 1
    out = capsys.readouterr().out
    assert "WARNING: EXCO-01: gold evidence not found" in out


def test_eval_retrieval_on_empty_template(capsys, tmp_path):
    questions = tmp_path / "retrieval_questions.yaml"
    questions.write_text("version: 1\nquestions:\n  - {id: A-01, ticker: AAPL, qtype: table}\n")
    args = ["eval-retrieval", "--questions", str(questions)]
    assert main([*args, "--chunks-dir", str(tmp_path), "--embedder", "hashing"]) == 1
    assert "Nothing to evaluate yet" in capsys.readouterr().out
