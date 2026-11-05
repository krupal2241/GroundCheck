"""`eval-verifiers`: metrics, latency, reports, merging runs, and the cost estimate."""

import json

import pytest

from groundcheck.cli import main
from groundcheck.config import ConfigError
from groundcheck.verify.costs import Prices, approx_tokens, estimate_costs, load_prices
from groundcheck.verify.evaluate import (
    binary_metrics,
    class_metrics,
    evaluate_verifiers,
    latency_metrics,
    load_gold,
    macro_f1,
)
from groundcheck.verify.fake import FakeVerifier


def test_class_metrics_and_macro_f1_with_a_never_predicted_label():
    gold = ["supported", "supported", "contradicted", "unsupported"]
    pred = ["supported", "unsupported", "unsupported", "unsupported"]
    per = {label: class_metrics(gold, pred, label) for label in set(gold)}
    assert per["supported"].precision == 1 and per["supported"].recall == 0.5
    assert per["contradicted"].precision is None and per["contradicted"].f1 == 0
    assert per["unsupported"].precision == pytest.approx(1 / 3)
    expected = (2 / 3 + 0 + 0.5) / 3
    assert macro_f1(per) == pytest.approx(expected)


def test_macro_f1_ignores_labels_absent_from_gold_and_predictions():
    gold = pred = ["supported", "unsupported"]
    per = {
        label: class_metrics(gold, pred, label)
        for label in ("supported", "contradicted", "unsupported")
    }
    assert macro_f1(per) == 1.0


def test_binary_view_merges_contradicted_and_unsupported():
    m = binary_metrics(["contradicted", "supported"], ["unsupported", "supported"])
    assert m.accuracy == 1.0 and m.f1 == 1.0


def test_latency_percentiles_and_throughput():
    m = latency_metrics([10.0] * 19 + [200.0], wall_seconds=2.0)
    assert m.p50_ms == 10.0 and m.p95_ms == pytest.approx(19.5)
    assert m.throughput_per_s == 10.0


def run_cli(gold_path, tmp_path, *extra):
    results = tmp_path / "results"
    args = ["eval-verifiers", "--gold", str(gold_path), "--results-dir", str(results)]
    return main([*args, *extra]), results


def test_eval_fake_verifier_end_to_end(gold_path, tmp_path, capsys):
    code, results = run_cli(gold_path, tmp_path, "--verifier", "fake")
    assert code == 0
    data = json.loads((results / "verifier_eval.json").read_text())
    report = data["verifiers"]["fake"]
    assert report["n"] == 6 and report["local"] and report["cost_per_1000_checks_usd"] == 0.0
    assert set(report["confusion"]) == {"supported", "contradicted", "unsupported"}
    assert sum(sum(row.values()) for row in report["confusion"].values()) == 6
    assert set(report["by_company"]) == {"EXCO", "OTHR"}
    assert set(report["by_edge_case"]) == {
        "(none)",
        "numeric_mismatch",
        "absent_evidence",
        "table_misread",
    }
    assert report["latency"]["p95_ms"] >= report["latency"]["p50_ms"] >= 0
    assert report["evidence_agreement"]["eligible"] >= 1
    predictions = (results / "predictions" / "fake.jsonl").read_text().splitlines()
    assert len(predictions) == 6 and json.loads(predictions[0])["gold"] == "supported"
    md = (results / "verifier_eval.md").read_text()
    assert "| fake | `lexical-rule` |" in md and "$0 (local)" in md
    assert "Confusion matrix" in md and "By edge-case tag" in md and "ESTIMATE" not in md
    assert "accuracy" in capsys.readouterr().out


def test_runs_merge_and_limits_apply(gold_path, tmp_path):
    run_cli(gold_path, tmp_path, "--verifier", "fake")
    code, results = run_cli(
        gold_path, tmp_path, "--verifier", "prompted", "--provider", "fake", "--company", "EXCO"
    )
    assert code == 0
    data = json.loads((results / "verifier_eval.json").read_text())
    assert set(data["verifiers"]) == {"fake", "prompted"}
    assert data["verifiers"]["prompted"]["n"] == 4 and data["verifiers"]["fake"]["n"] == 6
    md = (results / "verifier_eval.md").read_text()
    assert "| prompted |" in md and "| fake |" in md


class NamedFake(FakeVerifier):
    def __init__(self, name):
        super().__init__()
        self.name = name


def test_binary_models_are_flagged_in_the_report(gold_path, tmp_path):
    gold = load_gold(gold_path)
    results = evaluate_verifiers(
        ["hhem"], NamedFake, gold, gold_path, tmp_path / "r", Prices(), echo=lambda s: None
    )
    assert results.verifiers["hhem"].binary_model
    md = (tmp_path / "r" / "verifier_eval.md").read_text()
    assert "Binary model: never predicts contradicted." in md


def test_cost_estimate_is_arithmetic_and_labeled(gold_path, tmp_path):
    prices = tmp_path / "prices.yaml"
    prices.write_text(
        "assumed_output_tokens: 50\nmodels:\n"
        "  - {name: hosted-x, input_per_million_tokens: 2.0, output_per_million_tokens: 10.0}\n"
    )
    code, results = run_cli(gold_path, tmp_path, "--verifier", "fake", "--prices", str(prices))
    assert code == 0
    data = json.loads((results / "verifier_eval.json").read_text())
    (estimate,) = data["hosted_cost_estimates"]
    assert estimate["label"].startswith("ESTIMATE")
    assert estimate["output_tokens_source"] == "assumed in prices.yaml"
    expected = 1000 * (estimate["input_tokens_per_check"] * 2.0 + 50 * 10.0) / 1e6
    assert estimate["cost_per_1000_checks"] == pytest.approx(expected, rel=1e-3)
    md = (results / "verifier_eval.md").read_text()
    assert "### Hosted-model cost: ESTIMATE" in md and "(estimate)" in md
    assert "no hosted model was called" in md


def test_measured_output_tokens_replace_the_assumption():
    prices = Prices(
        models=[{"name": "m", "input_per_million_tokens": 1, "output_per_million_tokens": 1}]
    )
    (estimate,) = estimate_costs(prices, input_tokens=1000, measured_output_tokens=20)
    assert estimate.output_tokens_source.startswith("measured")
    assert estimate.cost_per_1000_checks == pytest.approx(1.02)
    assert approx_tokens("abcde") == 2


def test_committed_prices_template_is_empty_and_valid():
    prices = load_prices()  # config/prices.yaml
    assert prices.models == [] and prices.currency == "USD"


def test_bad_inputs_are_clean_errors(tmp_path, capsys):
    with pytest.raises(ConfigError, match="export-labels"):
        load_gold(tmp_path / "missing.jsonl")
    bad = tmp_path / "gold.jsonl"
    bad.write_text('{"claim_id": "x"}\n')
    assert main(["eval-verifiers", "--verifier", "fake", "--gold", str(bad)]) == 2
    assert "invalid lines" in capsys.readouterr().err
    prices = tmp_path / "prices.yaml"
    prices.write_text("models:\n  - {name: x, input_per_milion_tokens: 1}\n")
    with pytest.raises(ConfigError, match="invalid prices"):
        load_prices(prices)
