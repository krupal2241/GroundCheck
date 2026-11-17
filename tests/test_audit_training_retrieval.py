"""Regression tests for the audit's training and retrieval findings (#11, #12, #3, #14).

Each test reproduces the failing example given in the audit report.
"""

import json

import pytest

from groundcheck.chunking.base import Chunk, write_chunks
from groundcheck.cli import main
from groundcheck.config import ConfigError
from groundcheck.retrieval.bm25 import BM25, tokenize
from groundcheck.training.data import (
    TrainingConfig,
    TrainingExample,
    build_training_data,
    check_item_disjoint,
    gold_exclusions,
    load_training_config,
    read_examples,
)

SENTENCES = {
    "1": "{t} Corp employed about {n},000 people at year end, and most of them work in its "
    "retail stores and distribution centers across the country.",
    "1A": "{t} Corp faces intense competition in every market and may not be able to maintain "
    "its gross margin or its market share during the next fiscal year.",
    "5": "{t} Corp repurchased {n} million shares of its common stock for $4{n}0 million "
    "during fiscal 2025 under the program authorized by its board.",
    "7": "{t} Corp total net sales increased {n}% to $12,3{n}5 million in fiscal 2025, "
    "compared with $11,646 million in fiscal 2024, driven by higher services revenue.",
    "8": "{t} Corp recorded depreciation expense of $2{n}7 million in fiscal 2025, and its "
    "property and equipment was $3,4{n}1 million at the end of the year.",
}


def chunk(ticker, item, index, text, kind="prose"):
    return Chunk(
        chunk_id=f"{ticker}-{item}-{kind[0]}{index:04d}",
        ticker=ticker,
        accession="0000000000-25-000001",
        item=item,
        section_title="T",
        kind=kind,
        index=index,
        text=text,
        word_count=len(text.split()),
        chunker="fixed-p250-o50-t120",
    )


def company(ticker, n):
    return [chunk(ticker, item, 0, text.format(t=ticker, n=n)) for item, text in SENTENCES.items()]


@pytest.fixture
def corpus(tmp_path):
    chunks = tmp_path / "chunks"
    for n, ticker in enumerate(("BANK", "CONG", "TSTA"), start=2):
        write_chunks(company(ticker, n), chunks / f"{ticker}.jsonl")
    return {"chunks": chunks, "out": tmp_path / "training", "root": tmp_path}


def config(**overrides):
    settings = {
        "splits": {"train": ["BANK", "CONG"], "test": ["TSTA"]},
        "dev_items": {"BANK": ["1", "5"], "CONG": ["7"]},
        "partner_candidates": 10,
    }
    return TrainingConfig(**(settings | overrides))


def examples(out):
    return [e for s in ("train", "dev", "test") for e in read_examples(out / f"{s}.jsonl")]


# --- #11: train on two companies, dev from their held-out Items --------------------------


def test_dev_comes_from_held_out_items_and_is_disjoint_by_item(corpus):
    manifest = build_training_data(corpus["chunks"], corpus["out"], config(), [])
    found = examples(corpus["out"])
    by_split = {s: [e for e in found if e.split == s] for s in ("train", "dev", "test")}
    assert all(by_split.values())
    dev_pairs = {("BANK", "1"), ("BANK", "5"), ("CONG", "7")}
    for e in by_split["dev"]:
        assert (e.ticker, e.seed_item) in dev_pairs
        assert (e.evidence_ticker, e.evidence_item) in dev_pairs  # partners stay in dev
    for e in by_split["train"]:
        assert e.ticker in ("BANK", "CONG") and (e.ticker, e.seed_item) not in dev_pairs
        assert (e.evidence_ticker, e.evidence_item) not in dev_pairs
        assert e.evidence_ticker in ("BANK", "CONG")
    assert {e.ticker for e in by_split["test"]} == {"TSTA"}
    assert manifest["items_by_split"]["dev"] == {"BANK": ["1", "5"], "CONG": ["7"]}
    assert manifest["items_by_split"]["train"] == {
        "BANK": ["1A", "7", "8"],
        "CONG": ["1", "1A", "5", "8"],
    }  # noqa: E501
    assert manifest["dev_items_without_chunks"] == []


def test_two_training_companies_give_cross_company_negatives(corpus):
    """The audit's example: with one company in the train split, `other_company_rate`
    did nothing. With two, the train split has other_company partners."""
    build_training_data(corpus["chunks"], corpus["out"], config(), [])
    train = read_examples(corpus["out"] / "train.jsonl")
    cross = [e for e in train if e.rule == "other_company"]
    assert cross and all(e.evidence_ticker != e.ticker for e in cross)
    single = config(splits={"train": ["BANK"], "test": ["TSTA"]}, dev_items={"BANK": ["1", "5"]})
    build_training_data(corpus["chunks"], corpus["root"] / "single", single, [])
    alone = read_examples(corpus["root"] / "single" / "train.jsonl")
    assert alone and not [e for e in alone if e.rule == "other_company"]


def test_a_leak_between_train_and_dev_fails_the_build():
    def example(split, item, evidence_item):
        return TrainingExample(
            example_id="x", split=split, ticker="BANK", label="unsupported", claim="c",
            evidence="e", evidence_chunk_id="BANK-x", seed_chunk_id="BANK-y",
            seed_kind="sentence", rule="other_section", seed_item=item,
            evidence_ticker="BANK", evidence_item=evidence_item,
        )  # fmt: skip

    check_item_disjoint([example("train", "7", "8"), example("dev", "1", "5")])
    with pytest.raises(ConfigError, match="BANK Item 7 is used by both train and dev"):
        # A dev example borrowing a train Item as evidence is a leak too.
        check_item_disjoint([example("train", "7", "8"), example("dev", "1", "7")])


def test_training_config_validates_dev_items():
    with pytest.raises(ValueError, match="not a company of the train split"):
        config(dev_items={"TSTA": ["7"]})
    with pytest.raises(ValueError, match="distinct Items"):
        config(dev_items={"BANK": []})
    with pytest.raises(ValueError, match="no dev set"):
        config(dev_items={})
    c = config()
    assert c.split_of("BANK") == "train" and c.split_of("BANK", "5") == "dev"
    assert c.split_of("BANK", "7") == "train" and c.split_of("CONG", "7") == "dev"
    assert c.split_of("TSTA", "7") == "test" and c.split_of("NOPE") is None


def test_empty_dev_and_unknown_dev_items_are_reported(corpus):
    bad = config(dev_items={"BANK": ["99"]})
    with pytest.raises(ConfigError, match="the dev split is empty"):
        build_training_data(corpus["chunks"], corpus["out"], bad, [])
    partly = config(dev_items={"BANK": ["1", "99"]})
    manifest = build_training_data(corpus["chunks"], corpus["out"], partly, [])
    assert manifest["dev_items_without_chunks"] == ["BANK Item 99"]


def test_committed_config_trains_on_jpm_and_brk_b_with_item_level_dev():
    committed = load_training_config()
    assert committed.splits["train"] == ["JPM", "BRK-B"] and "dev" not in committed.splits
    assert committed.dev_items == {"JPM": ["1", "5"], "BRK-B": ["7"]}
    assert sorted(committed.splits["test"]) == ["AAPL", "AMZN", "JNJ", "MSFT", "WMT", "XOM"]


def write_config(path):
    path.write_text(
        "splits:\n  train: [BANK, CONG]\n  test: [TSTA]\n"
        'dev_items:\n  BANK: ["1", "5"]\n  CONG: ["7"]\npartner_candidates: 10\n'
    )
    return path


def cli(corpus, *extra):
    return main(
        [
            "build-training-data",
            "--chunks-dir", str(corpus["chunks"]),
            "--out-dir", str(corpus["out"]),
            "--training-config", str(write_config(corpus["root"] / "training.yaml")),
            "--candidates-dir", str(corpus["root"] / "no-candidates"),
            *extra,
        ]
    )  # fmt: skip


def test_under_500_warning_names_the_new_remedies(corpus, capsys):
    gold = corpus["root"] / "gold.jsonl"
    gold.write_text("")
    assert cli(corpus, "--gold", str(gold), "--labels", str(gold)) == 0
    out = capsys.readouterr().out
    assert "fewer than 500" in out and "JPM and BRK-B" in out and "dev_items" in out
    assert "dev   uses BANK: Items 1, 5; CONG: Items 7" in out


# --- #12: a missing gold path is reported, per path -------------------------------------


def test_each_missing_gold_path_gets_its_own_warning(corpus, capsys):
    """The audit's example: a mistyped --gold next to existing candidates gave no
    warning, only a null hash in the manifest."""
    candidates = corpus["root"] / "candidates"
    candidates.mkdir()
    (candidates / "TSTA.jsonl").write_text(
        json.dumps({"retrieved": [{"chunk_id": "TSTA-7-p0000"}]}) + "\n"
    )
    labels = corpus["root"] / "labels.jsonl"
    labels.write_text("")
    typo = corpus["root"] / "gold_lables.jsonl"
    args = ["--gold", str(typo), "--labels", str(labels), "--candidates-dir", str(candidates)]
    assert cli(corpus, *args) == 0
    out = capsys.readouterr().out
    assert f"WARNING: {typo} does not exist" in out
    assert f"WARNING: {labels} does not exist" not in out
    assert "found no gold labels" not in out  # the old all-or-nothing warning stays silent
    manifest = json.loads((corpus["out"] / "manifest.json").read_text())
    assert manifest["gold_files_missing"] == [str(typo)]
    assert gold_exclusions([typo, labels]).missing == [str(typo)]


# --- #3: chunks that share a gold evidence span are excluded too -------------------------


def test_overlap_neighbours_and_repeated_rows_of_a_gold_span_are_excluded(corpus):
    """The audit's example: a gold span in the last 50 words of one window is also in
    the next window, which stayed a seed in the synthetic test split."""
    span = "total net sales increased 9% to $12,999 million in fiscal 2025"
    gold_chunk = chunk("TSTA", "7", 3, f"Earlier text of the window. TSTA Corp {span}.")
    neighbour = chunk("TSTA", "7", 4, f"TSTA Corp {span}. The next window goes on from here.")
    broken = span.replace(" to ", "\nto ")  # the same words, another line break
    split_part = chunk("TSTA", "7", 5, f"TSTA Corp  {broken}, it said.")
    row = "Total net sales | $12,999 | $11,646"
    table7 = chunk("TSTA", "7", 0, f"[TABLE]\n | 2025 | 2024\n{row}\n[/TABLE]", "table")
    rows8 = f"[TABLE]\n | 2025 | 2024\n{row}\nNet income | $9 | $8\n[/TABLE]"
    table8 = chunk("TSTA", "8", 0, rows8, "table")
    extra = [gold_chunk, neighbour, split_part, table7, table8]
    write_chunks([*company("TSTA", 4), *extra], corpus["chunks"] / "TSTA.jsonl")
    gold = corpus["root"] / "gold.jsonl"
    record = {
        "claim": "TSTA's total net sales were $12,999 million in fiscal 2025.",
        "passages": [
            {"rank": 1, "chunk_id": "TSTA-7-p0003", "text": gold_chunk.text},
            {"rank": 2, "chunk_id": "TSTA-7-t0000", "text": table7.text},
        ],
        "evidence": [
            {"chunk_id": "TSTA-7-p0003", "text": span, "start": 0, "end": len(span)},
            {"chunk_id": "TSTA-7-t0000", "text": row, "start": 0, "end": len(row)},
            {"chunk_id": "TSTA-7-t0000", "text": "$12,999", "start": 0, "end": 7},  # too short
        ],
    }
    gold.write_text(json.dumps(record) + "\n")
    manifest = build_training_data(corpus["chunks"], corpus["out"], config(), [gold])
    assert manifest["gold_chunks_excluded"] == 2
    assert manifest["chunks_excluded_sharing_gold_span"] == 3  # p0004, p0005, the Item 8 table
    assert manifest["gold_spans_matched_on"] == 2
    assert manifest["gold_spans_too_short_to_match"] == 1
    used = {e.seed_chunk_id for e in examples(corpus["out"])}
    used |= {e.evidence_chunk_id for e in examples(corpus["out"])}
    assert not used & {c.chunk_id for c in extra}
    assert "TSTA-8-p0000" in used  # an unrelated chunk of the same company is kept
    for e in examples(corpus["out"]):
        assert span not in " ".join(e.evidence.split()) and row not in e.evidence


# --- #14: BM25 tokens for numbers ---------------------------------------------------------


def test_bm25_tokens_from_the_audit():
    assert tokenize("FY2025") == ["fy", "2025"]  # was "fy2025", never matching "2025"
    assert tokenize("2025,2024") == ["2025", "2024"]  # was one token
    assert tokenize("12345") == tokenize("12,345") == tokenize("$12,345") == ["12345"]
    assert tokenize("1,234,567.5 and 89.3%") == ["1234567.5", "89.3"]
    assert tokenize("Q4 10-K U.S.") == ["q", "4", "10", "k", "u", "s"]
    assert tokenize("12,34 1,2345") == ["12", "34", "1", "2345"]  # three digits after a comma


def test_bm25_matches_numbers_across_formats():
    docs = [
        "Total net sales | 12,345 | 11,646",
        "Results for FY2025 improved across the segments.",
        "Property and equipment was unchanged in the period 2023,2022.",
    ]
    bm25 = BM25(docs)
    assert bm25.top("net sales of 12345", 1)[0][0] == 0
    assert bm25.top("what happened in 2025", 1)[0][0] == 1
    assert bm25.top("the year 2022", 1)[0][0] == 2
