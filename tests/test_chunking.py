import re

import pytest

from groundcheck.chunking.base import (
    Chunker,
    chunks_path,
    is_split_part,
    part_suffix,
    read_chunks,
    write_chunks,
)
from groundcheck.chunking.fixed import (
    FixedSizeChunker,
    header_row_count,
    is_sentence_end,
    sentence_cut,
)
from groundcheck.config import ConfigError
from groundcheck.parsing.filing import ParsedFiling, ParsedSection


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def filing(*sections):
    return ParsedFiling(
        ticker="EXCO",
        accession="0000000001-25-000001",
        primary_document="exco.htm",
        period="2025-12-31",
        source_sha256="0" * 64,
        removed_furniture_blocks=0,
        front_matter_chars=0,
        sections=[ParsedSection(item=i, title=f"Title {i}", text=t) for i, t in sections],
        warnings=[],
    )


def table(header, n_rows, row_words=4):
    rows = [
        f"Row {r} | " + " | ".join(str(r * 10 + c) for c in range(row_words - 2))
        for r in range(n_rows)
    ]
    return "\n".join(["[TABLE]", *header, *rows, "[/TABLE]"])


# --- prose -----------------------------------------------------------------------------


def test_prose_windows_size_and_overlap():
    chunker = FixedSizeChunker()
    windows = chunker.prose_windows(words(600))
    assert [len(w.split()) for w in windows] == [250, 250, 200]
    first, second = windows[0].split(), windows[1].split()
    assert first[-50:] == second[:50]  # 50 words of overlap
    assert second[0] == "w200"
    assert windows[-1].split()[-1] == "w599"  # nothing lost at the end


def test_short_prose_is_one_chunk_and_exact_fit_has_no_tail():
    chunker = FixedSizeChunker()
    assert chunker.prose_windows(words(10)) == [words(10)]
    assert len(chunker.prose_windows(words(250))) == 1
    assert chunker.prose_windows("") == []


def test_every_word_is_covered():
    chunker = FixedSizeChunker(prose_words=30, overlap=7)
    text = words(1000)
    covered = {w for window in chunker.prose_windows(text) for w in window.split()}
    assert covered == set(text.split())


@pytest.mark.parametrize(("size", "overlap"), [(250, 250), (250, -1), (10, 20)])
def test_bad_settings_rejected(size, overlap):
    with pytest.raises(ValueError):
        FixedSizeChunker(prose_words=size, overlap=overlap)


# --- tables ----------------------------------------------------------------------------


def test_header_rows_detected():
    assert header_row_count(["| Years ended", "| 2025 | 2024", "Net sales | 1 | 2"]) == 2
    assert header_row_count(["Segment | 2025", "A | 1"]) == 1
    assert header_row_count(["| a", "| b"]) == 1  # no labelled row at all: first row only


def test_tables_split_on_row_boundaries_and_repeat_header():
    header = ["| Years ended December 31,", "| 2025 | 2024"]
    t = table(header, n_rows=60)
    pieces = FixedSizeChunker(table_words=40).table_pieces(t)
    assert len(pieces) > 1
    original_rows = t.split("\n")[3:-1]
    seen = []
    for piece in pieces:
        lines = piece.split("\n")
        assert lines[0] == "[TABLE]" and lines[-1] == "[/TABLE]"
        assert lines[1:3] == header  # header repeated in every piece
        assert len(piece.split()) <= 40 + 2  # budget, plus the two marker tokens
        seen.extend(lines[3:-1])
    assert seen == original_rows  # every row exactly once, in order, never cut


def test_oversized_row_becomes_its_own_piece():
    t = "\n".join(["[TABLE]", "| 2025", "Huge | " + words(80), "Small | 1", "[/TABLE]"])
    pieces = FixedSizeChunker(table_words=20).table_pieces(t)
    assert len(pieces) == 2
    assert "Huge" in pieces[0] and "Small" in pieces[1]


def test_header_only_table():
    t = "[TABLE]\n| 2025 | 2024\n[/TABLE]"
    assert FixedSizeChunker().table_pieces(t) == [t]


# --- whole filings ---------------------------------------------------------------------


def test_chunks_never_cross_sections_and_tables_are_separate():
    t = table(["| 2025 | 2024"], n_rows=3)
    f = filing(
        ("1", "Item 1. Business\n\n" + words(300, "a")),
        ("7", "Item 7. MD&A\n\n" + words(100, "b") + "\n\n" + t + "\n\n" + words(20, "c")),
    )
    chunks = FixedSizeChunker().chunk(f)
    for c in chunks:
        if c.item == "1":
            assert "b0" not in c.text
        else:
            assert "a0" not in c.text
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 1 and tables[0].item == "7"
    assert all("[TABLE]" not in c.text for c in chunks if c.kind == "prose")
    mda_prose = [c for c in chunks if c.item == "7" and c.kind == "prose"]
    assert "b99 c0" in mda_prose[0].text  # prose around the table is one stream


def test_chunk_ids_are_deterministic_and_unique():
    f = filing(("1", words(600)), ("1A", words(10) + "\n\n" + table(["| 2025"], 2)))
    first, second = FixedSizeChunker().chunk(f), FixedSizeChunker().chunk(f)
    assert first == second
    ids = [c.chunk_id for c in first]
    assert ids == ["EXCO-1-p0000", "EXCO-1-p0001", "EXCO-1-p0002", "EXCO-1A-p0000", "EXCO-1A-t0000"]
    assert len(set(ids)) == len(ids)
    assert first[0].chunker == "fixed-p250-o50-t120"
    assert first[0].section_title == "Title 1"
    assert first[0].word_count == 250


def test_fixed_chunker_satisfies_protocol():
    chunker: Chunker = FixedSizeChunker()
    assert chunker.name.startswith("fixed-")


# --- token limit (Part D) --------------------------------------------------------------
# A fake tokenizer stands in for bge's, so no model or network is needed. Like a
# WordPiece tokenizer, it charges extra for digits and punctuation, so number-dense
# text costs more tokens per word than plain prose.


def fake_tokens(text: str) -> int:
    """[CLS] + [SEP], one token per word, plus one per digit group and punctuation mark."""
    return 2 + len(text.split()) + len(re.findall(r"\d+|[^\w\s]", text))


def dense(n: int) -> str:
    """Number-dense prose like JNJ Item 1: `$12.3` costs 5 fake tokens per word."""
    sentence = "Sales were $12.3 billion, up 4.5% from $11.8 billion in 2024."
    out = []
    while len(out) < n:
        out.extend(sentence.split())
    return " ".join(out[:n])


def plain(n: int) -> str:
    """Letter-only words: one fake token each."""
    return " ".join((["the", "results", "of", "the", "year", "were", "good"] * n)[:n])


TOKEN_LIMITED = FixedSizeChunker(count_tokens=fake_tokens)


def test_long_windows_are_split_and_ids_get_a_suffix():
    # 250 number-dense words cost ~590 fake tokens, like JNJ-1-p0001 (632 real tokens).
    f = filing(("1", plain(250) + " " + dense(600)))
    assert fake_tokens(TOKEN_LIMITED.prose_windows(dense(250))[0]) > 520
    chunks = TOKEN_LIMITED.chunk(f)
    assert all(fake_tokens(c.text) <= 480 for c in chunks)
    ids = [c.chunk_id for c in chunks]
    assert ids[0] == "EXCO-1-p0000"  # plain words fit: ID unchanged
    assert "EXCO-1-p0001a" in ids and "EXCO-1-p0001b" in ids
    assert len(set(ids)) == len(ids)
    assert [c.index for c in chunks if c.chunk_id.startswith("EXCO-1-p0001")] == [1, 1]


def test_split_keeps_the_overlap_and_loses_nothing():
    text = dense(250)
    parts = TOKEN_LIMITED.fit_prose(text)
    assert len(parts) == 2
    first, second = parts[0].split(), parts[1].split()
    assert first[-50:] == second[:50]  # the configured 50-word overlap
    assert first + second[50:] == text.split()


def test_split_prefers_a_sentence_boundary():
    parts = TOKEN_LIMITED.fit_prose(dense(250))
    assert parts[0].endswith("2024.")  # cut after a sentence, not mid-sentence


def test_without_a_counter_or_when_everything_fits_nothing_changes():
    f = filing(("1", words(600)), ("7", dense(300) + "\n\n" + table(["| 2025"], 5)))
    plain = FixedSizeChunker().chunk(f)
    roomy = FixedSizeChunker(max_tokens=5_000, count_tokens=fake_tokens).chunk(f)
    assert [c.chunk_id for c in roomy] == [c.chunk_id for c in plain]
    assert [c.text for c in roomy] == [c.text for c in plain]
    assert plain[0].chunker == "fixed-p250-o50-t120"
    assert roomy[0].chunker == "fixed-p250-o50-t120-k5000"
    assert not any(is_split_part(c.chunk_id) for c in plain)


def test_unsplit_ids_are_stable_when_another_window_splits():
    f = filing(("1", plain(400) + " " + dense(250) + " " + plain(400)))
    baseline = {c.chunk_id: c.text for c in FixedSizeChunker().chunk(f)}
    limited = {c.chunk_id: c.text for c in TOKEN_LIMITED.chunk(f)}
    for chunk_id, text in limited.items():
        if not is_split_part(chunk_id):
            assert baseline[chunk_id] == text
    split = {i[:-1] for i in limited if is_split_part(i)}
    assert split and split <= set(baseline) and not split & set(limited)


def test_tables_over_the_token_limit_split_between_rows():
    header = ["| 2025 | 2024 | 2023"]
    rows = [f"Segment {r} | $1,{r:03d}.5 | $(2,{r:03d}) | 3.{r}%" for r in range(12)]
    t = "\n".join(["[TABLE]", *header, *rows, "[/TABLE]"])
    chunker = FixedSizeChunker(max_tokens=120, count_tokens=fake_tokens)
    pieces = chunker.table_pieces(t)
    assert len(pieces) > 1
    assert all(fake_tokens(p) <= 120 for p in pieces)
    assert all(p.split("\n")[1] == header[0] for p in pieces)  # header repeated
    assert [r for p in pieces for r in p.split("\n")[2:-1]] == rows  # rows whole, in order
    chunks = chunker.chunk(filing(("8", t)))
    assert [c.chunk_id for c in chunks][:2] == ["EXCO-8-t0000a", "EXCO-8-t0000b"]


def test_one_huge_row_splits_between_words_and_a_giant_word_is_left_alone():
    chunker = FixedSizeChunker(max_tokens=40, count_tokens=fake_tokens)
    pieces = chunker.fit_table(["| 2025"], ["Huge | " + words(100)])
    assert len(pieces) > 1 and all(fake_tokens(p) <= 40 for p in pieces)
    assert chunker.fit_prose("x" * 10 + "." * 50) == ["x" * 10 + "." * 50]


def test_sentence_boundaries():
    assert is_sentence_end("2024.") and is_sentence_end("growth.”") and is_sentence_end("why?")
    for word in ("U.S.", "Inc.", "No.", "e.g.", "J.", "billion,"):
        assert not is_sentence_end(word)
    words_ = ["one", "two", "three.", "Four", "five", "six", "seven", "eight", "nine", "ten"]
    assert sentence_cut(words_) == 3
    assert sentence_cut(["a", "b", "c", "d", "e", "f"]) == 3  # no boundary: the middle
    assert sentence_cut(["a", "b"]) == 1


def test_part_suffixes_sort_and_never_collide():
    suffixes = [part_suffix(i) for i in range(80)]
    assert suffixes[:3] == ["a", "b", "c"] and suffixes[25] == "za"
    assert suffixes == sorted(suffixes) and len(set(suffixes)) == 80


def test_max_tokens_must_be_sensible():
    with pytest.raises(ValueError):
        FixedSizeChunker(max_tokens=5)


# --- JSONL ----------------------------------------------------------------------------


def test_jsonl_round_trip_is_byte_stable(tmp_path):
    chunks = FixedSizeChunker().chunk(filing(("7", "Revenue “grew” 12% — " + words(300))))
    path = chunks_path("EXCO", tmp_path)
    write_chunks(chunks, path)
    content = path.read_bytes()
    assert read_chunks(path) == chunks
    write_chunks(read_chunks(path), path)
    assert path.read_bytes() == content
    assert "“grew”" in content.decode()  # readable unicode, not \u escapes
    assert len(content.decode().splitlines()) == len(chunks)


def test_missing_chunks_file(tmp_path):
    with pytest.raises(ConfigError, match="groundcheck chunk"):
        read_chunks(tmp_path / "nope.jsonl")
