"""Training data (perturbations, gold exclusion, company splits), the training script
over stand-in transformers, and the FineTunedVerifier that loads its output."""

import json
import random

import pytest

import hf_standins
from groundcheck.chunking.base import Chunk, write_chunks
from groundcheck.cli import main
from groundcheck.config import ConfigError, load_companies
from groundcheck.training import perturb
from groundcheck.training.data import (
    TrainingConfig,
    build_training_data,
    load_training_config,
    read_examples,
    sentence_seeds,
    table_seeds,
)
from groundcheck.training.train import main as train_main
from groundcheck.training.train import metrics_from_logits
from groundcheck.verify.base import EvidenceChunk, VerifierError
from groundcheck.verify.factory import VerifierOptions, make_verifier
from groundcheck.verify.finetuned import FineTunedVerifier, aggregate_three_way, label_indices

TABLE = (
    "[TABLE]\n | 2025 | 2024\nTotal net sales | $12,345 | $11,646\n"
    "Operating income | $3,210 | $2,987\nNet income | $2,500 | $2,100\n[/TABLE]"
)
PROSE = (
    "Total net sales increased 6% to $12,345 million in fiscal 2025 compared with fiscal "
    "2024. Services revenue decreased because of lower demand in the Americas segment."
)


def rng():
    return random.Random(0)


# --- Perturbations ----------------------------------------------------------------------


def test_numeric_changes_a_non_year_number_to_one_absent_from_the_evidence():
    claim = "Total net sales were $12,345 million in fiscal 2025."
    new = perturb.numeric(claim, TABLE, rng())
    assert new != claim and "2025" in new and "$12,345" not in new
    assert not (perturb.numbers(new) - perturb.numbers(claim)) & perturb.numbers(TABLE)
    assert perturb.numeric("Sales grew in 2025.", TABLE, rng()) is None  # only a year


def test_format_like_keeps_separators_and_decimals():
    assert perturb.format_like(13579.5, "12,345") == "13,580"
    assert perturb.format_like(1.2345, "1.5") == "1.2"


def test_scale_negation_and_synonym():
    assert perturb.scale("It was $5 million.", "", rng()) == "It was $5 billion."
    assert perturb.negation("Sales increased 6%.", "", rng()) == "Sales decreased 6%."
    assert perturb.negation("The plan was approved.", "", rng()) == "The plan was not approved."
    assert perturb.negation("The plan was not approved.", "", rng()) is None
    assert perturb.synonym("Sales increased 6%.", "", rng()) == "Sales rose 6%."


def test_rescale_rounds_to_the_claims_own_precision():
    assert perturb.rescale("Sales were $12,345 million.", "", rng()) == "Sales were $12.3 billion."
    assert perturb.rescale("Sales were $500 million.", "", rng()) is None


def test_year_swap_only_uses_years_in_the_evidence():
    claim = "Total net sales were $12,345 in 2025."
    assert perturb.year_swap(claim, TABLE, rng()) == "Total net sales were $12,345 in 2024."
    assert perturb.year_swap(claim, "Only 2025 is mentioned.", rng()) is None


def test_entity_swap_uses_another_row_with_different_values():
    claim = "Operating income was $3,210 for 2025."
    new = perturb.entity_swap(claim, TABLE, rng())
    assert new in {"Total net sales was $3,210 for 2025.", "Net income was $3,210 for 2025."}
    prose_claim = "Sales grew in Greater China during the year."
    evidence = "Sales grew in Greater China and fell in Latin America."
    assert perturb.entity_swap(prose_claim, evidence, rng()) == (
        "Sales grew in Latin America during the year."
    )


# --- Seeds ------------------------------------------------------------------------------


def chunk(ticker, item, kind, index, text):
    return Chunk(
        chunk_id=f"{ticker}-{item}-{kind[0]}{index:04d}",
        ticker=ticker,
        accession="0000000001-25-000001",
        item=item,
        section_title="T",
        kind=kind,
        index=index,
        text=text,
        word_count=len(text.split()),
        chunker="fixed-p250-o50-t120",
    )


def test_seeds_from_prose_and_tables():
    sentences = sentence_seeds(chunk("A", "7", "prose", 0, PROSE), 8, 45)
    assert [s.claim for s in sentences][0].startswith("Total net sales increased 6%")
    cells = table_seeds(chunk("A", "8", "table", 0, TABLE))
    assert [s.claim for s in cells] == [
        "Total net sales was $12,345 for 2025.",
        "Operating income was $3,210 for 2025.",
        "Net income was $2,500 for 2025.",
    ]


# --- Building the data set --------------------------------------------------------------


def company_chunks(ticker):
    risk = (
        f"{ticker} Corp faces intense competition in every market and may not be able to "
        "maintain its gross margin or its market share during the next fiscal year."
    )
    other = (
        f"{ticker} Corp employed about 9,000 people at year end, and most of them work in "
        "its retail stores and distribution centers across the country."
    )
    return [
        chunk(ticker, "7", "prose", 0, PROSE),
        chunk(ticker, "8", "table", 0, TABLE),
        chunk(ticker, "1A", "prose", 0, risk),
        chunk(ticker, "1", "prose", 0, other),
    ]


@pytest.fixture
def training_inputs(tmp_path):
    chunks_dir = tmp_path / "chunks"
    for ticker in ("TRN1", "TRN2", "DEVC", "TSTC", "GOLD"):
        write_chunks(company_chunks(ticker), chunks_dir / f"{ticker}.jsonl")
    gold = tmp_path / "gold_labels.jsonl"
    gold_claim = "Services revenue decreased because of lower demand in the Americas segment."
    gold.write_text(
        json.dumps(
            {
                "claim": gold_claim,
                "passages": [{"rank": 1, "chunk_id": "TRN1-7-p0000", "text": PROSE}],
                "evidence": [],
            }
        )
        + "\n"
    )
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / "GOLD.jsonl").write_text(
        json.dumps({"retrieved": [{"chunk_id": "GOLD-8-t0000"}]}) + "\n"
    )
    config = TrainingConfig(
        splits={"train": ["TRN1", "TRN2"], "dev": ["DEVC"], "test": ["TSTC", "GOLD"]},
        partner_candidates=10,
    )
    return {
        "chunks": chunks_dir,
        "gold": gold,
        "gold_claim": gold_claim,
        "candidates": candidates,
        "config": config,
        "out": tmp_path / "training",
    }


def build(inputs, out=None):
    return build_training_data(
        inputs["chunks"], out or inputs["out"], inputs["config"], [inputs["gold"]],
        inputs["candidates"],
    )  # fmt: skip


def test_gold_never_enters_training_and_splits_are_by_company(training_inputs):
    manifest = build(training_inputs)
    out = training_inputs["out"]
    examples = [e for s in ("train", "dev", "test") for e in read_examples(out / f"{s}.jsonl")]
    assert examples
    excluded = {"TRN1-7-p0000", "GOLD-8-t0000"}
    for e in examples:
        assert e.evidence_chunk_id not in excluded and e.seed_chunk_id not in excluded
        assert e.claim.lower() != training_inputs["gold_claim"].lower()
        assert e.split == training_inputs["config"].split_of(e.ticker)
        partner_ticker = e.evidence_chunk_id.split("-")[0]
        assert training_inputs["config"].split_of(partner_ticker) == e.split
    assert manifest["gold_claims_excluded"] == 1 and manifest["gold_chunks_excluded"] == 1
    assert manifest["candidate_chunks_excluded"] == 1
    labels = {e.label for e in examples}
    assert labels == {"supported", "contradicted", "unsupported"}
    rules = manifest["counts_by_rule"]
    assert "other_company" in rules and "other_section" in rules
    assert {"seed", "numeric"} <= set(rules) or {"seed", "year_swap"} <= set(rules)
    for e in examples:
        if e.label == "unsupported":
            assert e.evidence_chunk_id != e.seed_chunk_id
        else:
            assert e.evidence_chunk_id == e.seed_chunk_id


def test_building_is_deterministic(training_inputs, tmp_path):
    build(training_inputs)
    build(training_inputs, tmp_path / "again")
    for split in ("train", "dev", "test"):
        first = (training_inputs["out"] / f"{split}.jsonl").read_text()
        assert first == (tmp_path / "again" / f"{split}.jsonl").read_text()


def test_training_config_rules():
    with pytest.raises(ValueError, match="both"):
        TrainingConfig(splits={"train": ["A"], "dev": ["A"]})
    with pytest.raises(ValueError, match="train split"):
        TrainingConfig(splits={"dev": ["A"]})
    committed = load_training_config()  # config/training.yaml
    seen = committed.splits["train"] + committed.splits["dev"]
    # No company with hand-labeled questions is used for training or model selection.
    labeled = {c.ticker for c in load_companies().companies if c.role == "labeled"}
    assert not labeled & set(seen)
    assert committed.splits["train"] == ["JPM"] and committed.splits["dev"] == ["BRK-B"]
    assert set(committed.splits["test"]) == labeled


def test_build_training_data_command(training_inputs, tmp_path, capsys):
    config = tmp_path / "training.yaml"
    config.write_text(
        "splits:\n  train: [TRN1, TRN2]\n  dev: [DEVC]\n  test: [TSTC, MISSING]\n"
        "partner_candidates: 10\n"
    )
    args = [
        "build-training-data",
        "--chunks-dir", str(training_inputs["chunks"]),
        "--out-dir", str(tmp_path / "cli-out"),
        "--training-config", str(config),
        "--gold", str(training_inputs["gold"]),
        "--labels", str(tmp_path / "no-labels.jsonl"),
        "--candidates-dir", str(training_inputs["candidates"]),
    ]  # fmt: skip
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "WARNING: MISSING" in out and "Excluded 1 gold claim(s)" in out
    assert "WARNING: only" in out and "fewer than 500" in out  # tiny synthetic corpus
    assert (tmp_path / "cli-out" / "manifest.json").is_file()


# --- Training script (stand-in transformers) --------------------------------------------


def test_metrics_from_logits():
    m = metrics_from_logits([[3, 0, 0], [0, 3, 0], [3, 0, 0]], [0, 1, 2])
    assert m["accuracy"] == pytest.approx(2 / 3) and m["recall_unsupported"] == 0


def test_train_script_writes_model_metrics_and_card(training_inputs, monkeypatch, tmp_path):
    build(training_inputs)
    recorder = hf_standins.install_training(monkeypatch, cuda=True)
    out = tmp_path / "model"
    args = ["--data-dir", str(training_inputs["out"]), "--out", str(out), "--fp16", "--seed", "7"]
    assert train_main(args) == 0
    assert recorder.seeds == [7]
    assert recorder.args["seed"] == recorder.args["data_seed"] == 7 and recorder.args["fp16"]
    assert recorder.args["metric_for_best_model"] == "macro_f1"
    assert recorder.args["report_to"] == []
    model_kwargs = recorder.loads[1][2]
    assert model_kwargs["id2label"] == {0: "supported", 1: "contradicted", 2: "unsupported"}
    evidence, claim, kwargs = recorder.tokenized[0]
    assert kwargs == {"truncation": "only_first", "max_length": 512}
    assert len(evidence) > len(claim)  # the evidence chunk comes first
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["metrics"]["dev"]["accuracy"] == 1.0 and metrics["metrics"]["train_loss"] == 0.42
    card = (out / "README.md").read_text()
    assert "No hand-labeled gold example was used" in card
    assert "**train**: TRN1, TRN2" in card and "| dev | 1.000 |" in card
    assert (out / "training_manifest.json").is_file() and not (out / "checkpoints").exists()


def test_train_script_without_data_or_extra(monkeypatch, tmp_path, capsys):
    hf_standins.install_training(monkeypatch)
    assert train_main(["--data-dir", str(tmp_path / "none"), "--out", str(tmp_path / "m")]) == 2
    assert "build-training-data" in capsys.readouterr().out
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    assert train_main(["--data-dir", str(tmp_path)]) == 2
    assert "[train] extra" in capsys.readouterr().out


# --- FineTunedVerifier ------------------------------------------------------------------

CHUNKS = [EvidenceChunk("A-8-t0000", TABLE), EvidenceChunk("A-7-p0000", PROSE)]


def model_dir(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text("{}")
    return path


def test_finetuned_verifier_three_way(monkeypatch, tmp_path):
    recorder = hf_standins.install(monkeypatch, three_way=True)
    verifier = make_verifier("finetuned", VerifierOptions(finetuned_path=model_dir(tmp_path)))
    assert isinstance(verifier, FineTunedVerifier)
    assert all(load[2] == {"local_files_only": True} for load in recorder.loads)
    supported = verifier.verify("Operating income was $3,210 for 2025.", CHUNKS)
    assert supported.label == "supported" and supported.evidence.chunk_id == "A-8-t0000"
    assert supported.evidence.text == "Operating income | $3,210 | $2,987"
    contradicted = verifier.verify("Operating income was $3,900 for 2025.", CHUNKS)
    assert contradicted.label == "contradicted" and contradicted.evidence is not None
    assert set(contradicted.scores) == {"supported", "contradicted", "unsupported"}
    unsupported = verifier.verify("The Company opened stores in Brazil.", CHUNKS)
    assert unsupported.label == "unsupported" and unsupported.evidence is None


def test_finetuned_verifier_errors(monkeypatch, tmp_path):
    with pytest.raises(VerifierError, match="--finetuned-path"):
        make_verifier("finetuned")
    with pytest.raises(VerifierError, match="no fine-tuned model"):
        FineTunedVerifier(tmp_path / "missing")
    hf_standins.install(monkeypatch, three_way=True, id2label={0: "LABEL_0", 1: "a", 2: "b"})
    with pytest.raises(VerifierError, match="does not name the labels"):
        FineTunedVerifier(model_dir(tmp_path))


def test_label_indices_and_aggregation_prefers_support():
    assert label_indices({0: "supported", 1: "contradicted", 2: "unsupported"}) == {
        "supported": 0,
        "contradicted": 1,
        "unsupported": 2,
    }
    probs = [
        {"supported": 0.45, "contradicted": 0.50, "unsupported": 0.05},
        {"supported": 0.40, "contradicted": 0.30, "unsupported": 0.30},
    ]
    result = aggregate_three_way("Total net sales were $12,345.", CHUNKS, probs)
    # Chunk 2's top class is supported, so the claim is supported and chunk 2 is cited,
    # even though chunk 1 has the higher P(supported).
    assert result.label == "supported" and result.details["verdict_chunk"] == "A-7-p0000"
    assert result.score == 0.45


def test_eval_verifiers_accepts_finetuned_path(monkeypatch, tmp_path, gold_path):
    hf_standins.install(monkeypatch, three_way=True)
    args = ["eval-verifiers", "--verifier", "finetuned", "--gold", str(gold_path)]
    args += ["--finetuned-path", str(model_dir(tmp_path)), "--results-dir", str(tmp_path / "r")]
    assert main(args) == 0
    report = json.loads((tmp_path / "r" / "verifier_eval.json").read_text())
    assert report["verifiers"]["finetuned"]["n"] == 6


def test_cli_errors_are_clean(capsys):
    with pytest.raises(ConfigError):
        load_training_config(__import__("pathlib").Path("/nonexistent.yaml"))


def test_colab_notebook_builds_data_trains_and_downloads():
    from pathlib import Path

    notebook_path = Path(__file__).parents[1] / "notebooks" / "finetune_verifier_colab.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "groundcheck build-training-data" in source
    assert "scripts/train_verifier.py" in source and "--seed" in source
    assert '".[train]"' in source and "files.download" in source
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])  # committed clean


def test_rules_found_noisy_on_real_filings_are_tightened():
    # "have" as a main verb is not negated ("could have not an adverse effect").
    assert perturb.negation("It could have an adverse effect.", "", rng()) is None
    # Single capitalized words and sentence starts are not swappable names.
    evidence = "Except as noted, Opinions The Firm reports. Revenue in Greater China grew."
    assert perturb.prose_names(evidence) == ["Greater China"]
    assert perturb.entity_swap("Except for one, sales grew.", evidence, rng()) is None


def test_multi_row_table_headers_use_the_year_row():
    text = (
        "[TABLE]\n | Year ended December 31,\n | 2025 | 2024\nNet revenue | 1,234 | 1,100\n[/TABLE]"
    )
    assert [s.claim for s in table_seeds(chunk("A", "8", "table", 0, text))] == [
        "Net revenue was 1,234 for 2025."
    ]


def test_numeric_skips_references_and_negation_skips_negatives():
    assert perturb.numeric("Refer to Note 14 for a further discussion.", "", rng()) is None
    assert perturb.numeric("The Company adopted ASU 2023-10 in 2025.", "", rng()) is None
    changed = perturb.numeric("See Note 4: sales were $120 million.", "", rng())
    assert "Note 4" in changed and "$120 " not in changed
    assert perturb.negation("There are virtually no barriers to entry.", "", rng()) is None


def test_page_furniture_and_contents_tables_give_no_seeds():
    prose = "K-106 (31) Business segment data We view our insurance segment as two activities."
    assert sentence_seeds(chunk("A", "1", "prose", 0, prose), 8, 45) == []
    toc = "[TABLE]\n | Page\nFirmwide Risk Management | 83\n[/TABLE]"
    assert table_seeds(chunk("A", "1", "table", 0, toc)) == []
