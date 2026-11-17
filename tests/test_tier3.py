"""Tier 3: `groundcheck doctor`, `--cooldown-seconds`, and the label-stats progress
summary."""

import json

import pytest

from groundcheck.cli import main
from groundcheck.config import LLMConfig
from groundcheck.doctor import (
    check_api_keys,
    check_data,
    check_disk,
    check_extras,
    check_ollama,
    check_python,
    check_user_agent,
    format_checks,
    run_checks,
)
from groundcheck.generation import runner
from groundcheck.labeling.report import progress_to_target
from groundcheck.labeling.schema import LabelRecord
from groundcheck.labeling.store import load_candidates
from groundcheck.verify.costs import Prices
from groundcheck.verify.evaluate import evaluate_verifiers, load_gold, run_verifier
from groundcheck.verify.fake import FakeVerifier

# --- doctor ------------------------------------------------------------------------------


def ollama(models=("llama3.2:3b",)):
    def get(url):
        if url.endswith("/api/version"):
            return {"version": "0.12.3"}
        return {"models": [{"name": m} for m in models]}

    return get


def test_python_and_extras():
    assert check_python((3, 12, 7)).status == "ok"
    old = check_python((3, 10, 2))
    assert old.status == "fail" and "3.11" in old.fix
    checks = {
        c.name: c for c in check_extras(lambda name: name in {"pytest", "ruff", "hypothesis"})
    }
    assert checks["extra [dev]"].status == "ok"
    missing = checks["extra [verifiers]"]
    assert missing.status == "warn" and "torch" in missing.detail
    assert missing.fix == 'pip install -e ".[verifiers]"'


def test_ollama_reachable_model_pulled_missing_or_down():
    config = LLMConfig()
    up = check_ollama(config, ollama())
    assert [c.status for c in up] == ["ok", "ok"] and "0.12.3" in up[0].detail
    other = check_ollama(config, ollama(models=("qwen2.5:3b",)))
    assert other[1].status == "warn" and other[1].fix == f"ollama pull {config.model}"
    assert "qwen2.5:3b" in other[1].detail

    def down(url):
        raise ConnectionError("refused")

    (check,) = check_ollama(config, down)
    assert check.status == "warn" and "ollama serve" in check.fix
    hf = check_ollama(config.model_copy(update={"provider": "hf-local"}), down)
    assert hf[0].status == "ok" and "not needed" in hf[0].detail


def test_disk_user_agent_and_data_folders(tmp_path, monkeypatch):
    assert check_disk(tmp_path, free_bytes=50 * 10**9).status == "ok"
    assert check_disk(tmp_path, free_bytes=3 * 10**9).status == "warn"
    assert check_disk(tmp_path, free_bytes=5 * 10**8).status == "fail"
    assert check_disk(tmp_path).status in ("ok", "warn", "fail")  # the real disk

    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    env = tmp_path / ".env"
    env.write_text("")
    missing = check_user_agent(env)
    assert missing.status == "warn" and ".env" in missing.fix
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    assert check_user_agent(env).status == "ok"

    (tmp_path / "data" / "chunks").mkdir(parents=True)
    (tmp_path / "data" / "chunks" / "AAPL.jsonl").write_text("{}\n")
    (tmp_path / "data" / "index").mkdir()
    checks = {c.name: c for c in check_data(tmp_path)}
    assert checks["data/chunks"].status == "ok"
    assert checks["data/index"].status == "warn" and "empty" in checks["data/index"].detail
    assert checks["data/raw"].fix == "run `groundcheck download`"
    assert "missing" in checks["data/labels"].detail


def test_api_keys_are_named_but_never_printed():
    assert check_api_keys({"PATH": "/bin", "HF_TOKEN": "x"}).status == "ok"
    assert check_api_keys({"OPENAI_API_KEY": "  "}).status == "ok"  # set but empty
    environ = {"OPENAI_API_KEY": "sk-secret-123", "anthropic_api_key": "sk-ant-456", "HOME": "/h"}
    check = check_api_keys(environ)
    text = "\n".join(check.lines())
    assert check.status == "warn" and "2 API key variable(s)" in text
    assert "OPENAI_API_KEY" in text and "anthropic_api_key" in text and "$0" in text
    assert "sk-secret-123" not in text and "sk-ant-456" not in text
    assert check.fix.startswith("unset OPENAI_API_KEY anthropic_api_key")


def test_report_and_exit_code(tmp_path, monkeypatch, capsys):
    kwargs = {
        "root": tmp_path, "environ": {}, "has_package": lambda name: True,
        "http_get": ollama(), "free_bytes": 50 * 10**9,
    }  # fmt: skip
    checks = run_checks(LLMConfig(), python_version=(3, 12, 0), **kwargs)
    text = format_checks(checks)
    assert "[ OK ] Python: 3.12.0" in text and "[WARN] data/raw: missing" in text
    assert "fix: run `groundcheck download`" in text
    assert "0 failure(s). Ready, with warnings" in text
    failing = run_checks(LLMConfig(), python_version=(3, 9, 0), **kwargs)
    assert "Not ready: fix the [FAIL] lines first." in format_checks(failing)

    # The command: no network in tests, so Ollama is reported as not reachable.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-do-not-print")
    monkeypatch.setattr("groundcheck.doctor._has_package", lambda name: name == "pytest")
    env = tmp_path / ".env"
    env.write_text("")
    assert main(["doctor", "--env-file", str(env)]) == 0
    out = capsys.readouterr().out
    assert "[WARN] Ollama: not reachable" in out and "[WARN] Paid API keys" in out
    assert "unset OPENAI_API_KEY" in out and "sk-do-not-print" not in out


# --- --cooldown-seconds ----------------------------------------------------------------


def test_eval_verifiers_cools_down_between_claims_and_verifiers(gold_path, tmp_path):
    gold = load_gold(gold_path)
    naps = []
    predictions, wall = run_verifier(
        FakeVerifier(), gold, echo=lambda s: None, cooldown_seconds=7, cooldown_every=2,
        sleep=naps.append,
    )  # fmt: skip
    assert naps == [7, 7] and len(predictions) == 6 and wall >= 0  # after claims 2 and 4
    naps.clear()
    run_verifier(FakeVerifier(), gold, echo=lambda s: None, sleep=naps.append)
    assert naps == []  # the default is no pause
    lines = []
    evaluate_verifiers(
        ["fake", "numeric"],
        lambda name: FakeVerifier(),
        gold, gold_path, tmp_path / "r", Prices(),
        echo=lines.append, cooldown_seconds=30, sleep=naps.append,
    )  # fmt: skip
    assert naps == [30] and "cooling down 30s before numeric" in lines


def test_cooldown_flags_on_the_commands(gold_path, tmp_path, tiny_corpus, capsys, monkeypatch):
    naps = []
    monkeypatch.setattr("time.sleep", naps.append)
    args = ["eval-verifiers", "--gold", str(gold_path), "--results-dir", str(tmp_path / "r")]
    assert main([*args, "--verifier", "fake", "numeric", "--cooldown-seconds", "5"]) == 0
    assert naps == [5]
    assert main([*args, "--verifier", "fake", "--cooldown-seconds", "-1"]) == 2
    assert "must not be negative" in capsys.readouterr().err

    naps.clear()
    generate = tiny_corpus["generate_args"]
    code = runner.main(generate("--dry-run", "--batch-size", "1", "--cooldown-seconds", "9"))
    assert code == 0 and naps == [9, 9]  # three questions, one per batch
    assert runner.main(generate("--dry-run", "--cooldown-seconds", "-1")) == 2


def test_index_cools_down_between_companies(tiny_corpus, capsys, monkeypatch):
    from groundcheck.chunking.base import read_chunks, write_chunks

    naps = []
    monkeypatch.setattr("time.sleep", naps.append)
    chunks = read_chunks(tiny_corpus["chunks"] / "EXCO.jsonl")
    other = [
        c.model_copy(update={"ticker": "OTHR", "chunk_id": "OTHR" + c.chunk_id[4:]}) for c in chunks
    ]
    write_chunks(other, tiny_corpus["chunks"] / "OTHR.jsonl")
    args = [
        "index", "--config", str(tiny_corpus["config"]), "--chunks-dir", str(tiny_corpus["chunks"]),
        "--index-dir", str(tiny_corpus["root"] / "idx"), "--embedder", "hashing",
        "--retrieval-config", str(tiny_corpus["retrieval_config"]),
    ]  # fmt: skip
    assert main([*args, "--cooldown-seconds", "20"]) == 0
    assert naps == [20] and "cooling down 20s before OTHR" in capsys.readouterr().out
    naps.clear()
    assert main(args) == 0 and naps == []


# --- label-stats: progress to target -----------------------------------------------------


@pytest.fixture
def candidates(tiny_corpus, capsys):
    assert runner.main(tiny_corpus["generate_args"]("--dry-run")) == 0
    capsys.readouterr()
    return load_candidates(tiny_corpus["out"])


def label(claim, candidate, label, confirmed=True, invalid=False):
    data = {
        "claim_id": claim.claim_id, "candidate_id": candidate.candidate_id,
        "question_id": candidate.question_id, "ticker": candidate.ticker,
        "qtype": candidate.qtype, "claim": claim.text, "confirmed": confirmed,
    }  # fmt: skip
    if invalid:
        return LabelRecord(**data, invalid=True)
    return LabelRecord(**data, label=label)


def test_progress_counts_confirmed_labels_against_even_shares(candidates):
    first, second = candidates[0], candidates[1]
    records = [
        label(first.claims[0], first, "unsupported"),
        label(second.claims[0], second, "unsupported", confirmed=False),
    ]
    text = progress_to_target(records, candidates, target=300)
    lines = text.splitlines()
    assert lines[0] == (
        "PROGRESS TO TARGET: 1 of 300 confirmed claims (0%); 299 to go; "
        "1 labeled but not confirmed (not counted)"
    )
    assert "Label (even share: 100 each)" in text and "Mode (even share: 100 each)" in text
    assert "  unsupported          1    99" in text and "  supported            0   100" in text
    assert "Company (even share: 300 each)" in text  # one company in the tiny corpus
    assert f"  EXCO                 1   299  {1:>9}" in text  # one claim still unlabeled
    assert "  rag                  1    99" in text and "  no-context           0   100" in text
    assert "Question type (even share: 60 each)" in text  # the five answerable types
    with_config = progress_to_target(records, candidates, 300, ["AAPL", "MSFT", "EXCO"])
    assert "Company (even share: 100 each)" in with_config
    assert f"  MSFT                 0   100  {0:>9}" in with_config  # no candidates yet
    assert "Too few unlabeled claims for an even share of: EXCO (needs 299, 1 unlabeled)" in text
    assert "Next: groundcheck label-plan --claims 299" in text
    assert len(lines) <= 30  # one screen


def test_progress_when_the_target_is_reached_and_invalid_claims_do_not_count(candidates):
    records = [label(c.claims[0], c, "unsupported") for c in candidates]
    records[0] = label(candidates[0].claims[0], candidates[0], None, invalid=True)
    text = progress_to_target(records, candidates, target=2)
    assert text.startswith("PROGRESS TO TARGET: 2 of 2 confirmed claims (100%); 0 to go")
    assert "Target size reached" in text and "Next: groundcheck label-check" in text
    assert progress_to_target([], [], target=300).startswith("PROGRESS TO TARGET: 0 of 300")


def test_label_stats_command_prints_the_summary(candidates, tiny_corpus, capsys):
    labels = tiny_corpus["root"] / "labels.jsonl"
    record = label(candidates[0].claims[0], candidates[0], "unsupported")
    labels.write_text(json.dumps(record.model_dump()) + "\n")
    args = ["label-stats", "--candidates-dir", str(tiny_corpus["out"]), "--labels", str(labels)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "By label:" in out and "PROGRESS TO TARGET: 1 of 300" in out
    assert main([*args, "--summary-only", "--target", "30"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("PROGRESS TO TARGET: 1 of 30") and "By label:" not in out
    assert "Label (even share: 10 each)" in out
    assert main([*args, "--target", "0"]) == 2
