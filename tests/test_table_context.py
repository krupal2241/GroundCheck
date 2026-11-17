"""Tier 2 (c) and (d): numeric query aliases and table context for BM25, both behind
config flags, and table captions recorded by the chunker without touching chunk text."""

import json

import pytest

from groundcheck.chunking.base import Chunk, chunks_path, read_chunks, write_chunks
from groundcheck.chunking.fixed import FixedSizeChunker, caption_of, table_captions
from groundcheck.cli import main
from groundcheck.parsing.filing import ParsedFiling, ParsedSection
from groundcheck.retrieval.embedder import HashingEmbedder
from groundcheck.retrieval.retriever import build_variant_index, describe, open_retriever
from groundcheck.retrieval.variants import (
    RetrievalConfig,
    VariantSpec,
    lexical_text,
    load_retrieval_config,
)

HEAD = "[TABLE]\n | 2025 | 2024\n"
SEGMENTS = HEAD + "Americas | 178,353 | 167,045\nEurope | 111,032 | 101,328\n[/TABLE]"
DEBT = HEAD + "Commercial paper | 7,979 | 9,967\nTerm debt | 89,300 | 96,662\n[/TABLE]"
SECTION = "\n\n".join(
    [
        "Segment Operating Performance",
        "The following table shows net sales by reportable segment for 2025 and 2024:\n"
        "(in millions)",
        SEGMENTS,
        "The Company issues debt from time to time. Its outstanding borrowings were as follows.",
        DEBT,
    ]
)


def filing(text=SECTION):
    return ParsedFiling(
        ticker="EXCO", accession="0000000001-25-000001", primary_document="exco.htm",
        period="2025-12-31", source_sha256="0" * 64, removed_furniture_blocks=0,
        front_matter_chars=0, sections=[ParsedSection(item="7", title="MD&A", text=text)],
        warnings=[],
    )  # fmt: skip


def tables(chunker, text=SECTION):
    return [c for c in chunker.chunk(filing(text)) if c.kind == "table"]


# --- (d) captions ----------------------------------------------------------------------


def test_caption_is_the_nearest_preceding_line_with_its_unit_note():
    blocks = SECTION.split("\n\n")
    assert table_captions(blocks) == [
        "The following table shows net sales by reportable segment for 2025 and 2024: "
        "(in millions)",
        "Its outstanding borrowings were as follows.",
    ]
    assert caption_of("Net sales by category") == "Net sales by category"
    assert caption_of("") == "" and table_captions([SEGMENTS, "After"]) == [""]
    # Two tables in a row share the caption above the first.
    assert table_captions(["Quarterly data", SEGMENTS, DEBT]) == ["Quarterly data"] * 2
    long = "First sentence of a long paragraph. " * 12 + "The table below shows term debt."
    assert caption_of(long) == "The table below shows term debt."


def test_captions_are_metadata_and_never_change_chunk_text():
    plain = tables(FixedSizeChunker())
    captioned = tables(FixedSizeChunker(table_captions=True))
    assert [c.text for c in captioned] == [c.text for c in plain]  # gold spans still exact
    assert [c.chunk_id for c in captioned] == [c.chunk_id for c in plain]
    assert all(c.caption == "" for c in plain)
    assert "net sales by reportable segment" in captioned[0].caption
    assert captioned[1].caption == "Its outstanding borrowings were as follows."
    assert FixedSizeChunker(table_captions=True).name == FixedSizeChunker().name
    prose = [c for c in FixedSizeChunker(table_captions=True).chunk(filing()) if c.kind == "prose"]
    assert all(c.caption == "" for c in prose)


def test_every_table_part_repeats_the_column_headers_even_when_split():
    rows = [f"Product line {i} | {1000 + i} | {900 + i}" for i in range(40)]
    big = "\n".join(["[TABLE]", " | 2025 | 2024", *rows, "[/TABLE]"])
    text = "Net sales by product line\n\n" + big

    def count(t):
        return len(t.split())

    chunker = FixedSizeChunker(
        table_words=60, max_tokens=40, count_tokens=count, table_captions=True
    )
    parts = tables(chunker, text)
    assert len(parts) > 4
    for part in parts:
        assert part.text.splitlines()[:2] == ["[TABLE]", " | 2025 | 2024"]
        assert part.caption == "Net sales by product line"
    seen = [line for part in parts for line in part.text.splitlines() if line.startswith("Product")]
    assert seen == rows  # nothing lost, nothing duplicated


def test_caption_in_text_is_a_separate_opt_in_that_renames_the_chunker():
    chunker = FixedSizeChunker(caption_in_text=True)
    parts = tables(chunker)
    assert chunker.name.endswith("-cap")
    assert parts[1].text.splitlines()[0] == "Its outstanding borrowings were as follows."
    assert parts[1].text.splitlines()[1] == "[TABLE]" and parts[1].caption
    assert "Term debt | 89,300 | 96,662" in parts[1].text


def test_chunk_files_without_captions_are_byte_identical_to_before(tmp_path):
    chunks = FixedSizeChunker().chunk(filing())
    write_chunks(chunks, tmp_path / "a.jsonl")
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    assert all("caption" not in json.loads(line) for line in lines)
    assert read_chunks(tmp_path / "a.jsonl") == chunks
    captioned = FixedSizeChunker(table_captions=True).chunk(filing())
    write_chunks(captioned, tmp_path / "b.jsonl")
    assert read_chunks(tmp_path / "b.jsonl") == captioned
    assert any("caption" in json.loads(x) for x in (tmp_path / "b.jsonl").read_text().splitlines())


def test_chunk_command_flags(tmp_path, capsys):
    parsed, chunks_dir = tmp_path / "parsed", tmp_path / "chunks"
    parsed.mkdir()
    (parsed / "EXCO.json").write_text(filing().model_dump_json())
    config = tmp_path / "companies.yaml"
    config.write_text("companies:\n  - {ticker: EXCO, name: Example Corp, cik: 1, role: labeled}\n")
    base = [
        "chunk", "--config", str(config), "--parsed-dir", str(parsed),
        "--chunks-dir", str(chunks_dir), "--tokenizer", "none",
    ]  # fmt: skip
    assert main(base) == 0
    plain = read_chunks(chunks_path("EXCO", chunks_dir))
    assert main([*base, "--table-captions"]) == 0
    captioned = read_chunks(chunks_path("EXCO", chunks_dir))
    assert [c.text for c in captioned] == [c.text for c in plain]
    assert any(c.caption for c in captioned) and not any(c.caption for c in plain)
    capsys.readouterr()


# --- (d) table context in the BM25 text, (c) numeric aliases in the BM25 query ----------

CONFIG = RetrievalConfig(
    default="bm25",
    pool=10,
    variants={
        "bm25": VariantSpec(embedder=None, bm25=True),
        "bm25-tablectx": VariantSpec(embedder=None, bm25=True, table_context=True),
        "bm25-aliases": VariantSpec(embedder=None, bm25=True, number_aliases=True),
        "hybrid": VariantSpec(embedder="hashing", bm25=True),
        "hybrid-numbers": VariantSpec(
            embedder="hashing", bm25=True, number_aliases=True, table_context=True
        ),
    },
)


@pytest.fixture
def corpus(tmp_path):
    chunks = FixedSizeChunker(table_captions=True).chunk(filing())
    filler = [
        Chunk(
            chunk_id=f"EXCO-1-p{i:04d}",
            ticker="EXCO",
            accession="a",
            item="1",
            section_title="Business",
            kind="prose",
            index=i,
            text=f"Note {i}: Example Corp designs and sells widgets and gadgets worldwide.",
            word_count=10,
            chunker="fixed",
        )  # fmt: skip
        for i in range(3)
    ]
    write_chunks([*chunks, *filler], chunks_path("EXCO", tmp_path / "chunks"))
    return tmp_path / "chunks", tmp_path / "index"


def open_variant(name, corpus):
    chunks_dir, index_dir = corpus
    spec = CONFIG.variants[name]
    if spec.embedder:
        build_variant_index(name, spec, "EXCO", HashingEmbedder(), chunks_dir, index_dir)
    return open_retriever(name, spec, CONFIG, ["EXCO"], chunks_dir, index_dir)


def test_lexical_text_adds_the_caption_to_tables_only(corpus):
    chunks = read_chunks(chunks_path("EXCO", corpus[0]))
    table = next(c for c in chunks if c.kind == "table")
    prose = next(c for c in chunks if c.kind == "prose")
    assert lexical_text(table, False) == table.text and lexical_text(prose, True) == prose.text
    with_context = lexical_text(table, True)
    assert with_context == table.caption + "\n" + table.text
    assert "MD&A" not in with_context  # the section title is deliberately left out
    assert read_chunks(chunks_path("EXCO", corpus[0]))[chunks.index(table)].text == table.text


def test_without_a_recorded_caption_table_context_changes_nothing(corpus):
    chunks = read_chunks(chunks_path("EXCO", corpus[0]))
    bare = [c.model_copy(update={"caption": ""}) for c in chunks]
    assert [lexical_text(c, True) for c in bare] == [c.text for c in bare]


def test_table_context_lets_a_question_about_the_table_find_its_rows(corpus):
    question = "net sales by reportable segment"
    without = open_variant("bm25", corpus).search("EXCO", question, 10)
    assert "table" not in [h.chunk.kind for h in without]  # the rows never say what they are
    found = open_variant("bm25-tablectx", corpus).search("EXCO", question, 10)
    table = next(h for h in found if h.chunk.kind == "table")
    assert "Americas | 178,353" in table.chunk.text
    assert "net sales" not in table.chunk.text  # found through the caption, text untouched
    assert table.rank <= 2  # only the sentence that introduces the table can beat it


def test_number_aliases_let_billions_find_a_table_in_millions(corpus):
    question = "$89.3 billion"
    assert open_variant("bm25", corpus).search("EXCO", question, 3) == []
    top = open_variant("bm25-aliases", corpus).search("EXCO", question, 1)[0]
    assert "Term debt | 89,300 | 96,662" in top.chunk.text
    hybrid = open_variant("hybrid-numbers", corpus).search("EXCO", question, 1)[0]
    assert "Term debt | 89,300" in hybrid.chunk.text


def test_flags_are_off_by_default_validated_and_visible_in_the_description(corpus):
    spec = VariantSpec()
    assert not spec.number_aliases and not spec.table_context
    with pytest.raises(ValueError, match="only apply to bm25"):
        VariantSpec(embedder="bge-small", number_aliases=True)
    assert describe(CONFIG.variants["bm25"], None, None, CONFIG) == "bm25"
    assert describe(CONFIG.variants["bm25-aliases"], None, None, CONFIG) == "bm25+aliases"
    retriever = open_variant("hybrid-numbers", corpus)
    assert "bm25+aliases+tablectx" in retriever.description
    assert " bm25 " in open_variant("hybrid", corpus).description  # flags off: plain bm25
    committed = load_retrieval_config()
    assert committed.default == "hybrid" and not committed.variants["hybrid"].number_aliases
    assert committed.variants["hybrid-numbers"].number_aliases
    assert committed.variants["hybrid-numbers"].table_context
    assert committed.variants["bm25-numbers"].diagnostic
    # The A/B variants reuse hybrid's dense index (same embedder, no headers).
    for name in ("hybrid-aliases", "hybrid-tablectx", "hybrid-numbers"):
        v = committed.variants[name]
        assert (v.embedder, v.headers) == ("bge-small", False)
