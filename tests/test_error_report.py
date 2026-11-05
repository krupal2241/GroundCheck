"""`error-report`: misclassified claims per verifier, grouped by edge-case tag."""

import json

import pytest

from groundcheck.cli import main
from groundcheck.config import ConfigError
from groundcheck.verify.errors import error_report
from groundcheck.verify.evaluate import load_gold


def evaluate(gold_path, results, *verifiers, extra=()):
    args = ["eval-verifiers", "--verifier", *verifiers, "--gold", str(gold_path)]
    assert main([*args, "--results-dir", str(results), *extra]) == 0


def test_error_report_lists_each_error_by_tag(gold_path, tmp_path, capsys):
    results = tmp_path / "results"
    evaluate(gold_path, results, "fake", "prompted", extra=("--provider", "fake"))
    assert main(["error-report", "--gold", str(gold_path), "--results-dir", str(results)]) == 0
    assert "error_report.md" in capsys.readouterr().out
    md = (results / "error_report.md").read_text()
    assert md.index("## fake") < md.index("## prompted")
    # The fake rule misses the table_misread claim: gold contradicted, predicted otherwise.
    fake = md[md.index("## fake") : md.index("## prompted")]
    assert "| table_misread | 1 | 1 | 100.0% |" in fake
    assert "### fake / table_misread (1 error(s))" in fake
    assert "#### `OTHR-G01#c02`: gold **contradicted**" in fake
    assert "- Claim: Operating income was $3,500 million in 2025." in fake
    assert "- Gold evidence (EXCO-7-t0000): `` Operating income | $3,210 | $2,987 ``" in fake
    # The fake LLM always answers unsupported: every supported/contradicted claim is wrong,
    # and an unsupported gold claim is never listed.
    prompted = md[md.index("## prompted") :]
    assert "5 of 6 claim(s) misclassified" in prompted
    assert "OTHR-G01#c01" not in prompted
    assert "Confusions (gold -> predicted): supported -> unsupported: 3" in prompted
    assert prompted.index("/ numeric_mismatch") < prompted.index("/ (none)")  # untagged last


def test_error_report_subset_custom_out_and_stale_gold(gold_path, tmp_path):
    results = tmp_path / "results"
    evaluate(gold_path, results, "fake")
    lines = gold_path.read_text().splitlines()
    record = json.loads(lines[0])
    record["note"] = "Rounded from the table."
    gold_path.write_text("\n".join([json.dumps(record), *lines[1:]]) + "\n")
    out = tmp_path / "errors.md"
    args = ["error-report", "--verifier", "fake", "--gold", str(gold_path)]
    assert main([*args, "--results-dir", str(results), "--out", str(out)]) == 0
    assert "made on a different gold file" in out.read_text()


def test_error_report_without_predictions_is_a_clean_error(gold_path, tmp_path):
    with pytest.raises(ConfigError, match="run eval-verifiers"):
        error_report(tmp_path, load_gold(gold_path), gold_path, None)
    with pytest.raises(ConfigError, match="no predictions for 'hhem'"):
        error_report(tmp_path, load_gold(gold_path), gold_path, ["hhem"])
