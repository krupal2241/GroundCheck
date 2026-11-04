"""Section-size warnings, repeated running headers, and cross-reference layout recovery.

The real-filing sizes below are the ones measured on the developer's machine; the
fixtures `xref_bank_style.htm` (JPM's layout) and `xref_energy_style.htm` (XOM's) are
synthetic copies of those layouts (see `fixtures/parsing/build_xref_fixtures.py`).
"""

import importlib.util
import json
import logging
from pathlib import Path

import pytest

from groundcheck.chunking.fixed import FixedSizeChunker
from groundcheck.cli import main
from groundcheck.config import Company
from groundcheck.parsing import sections as sections_module
from groundcheck.parsing.cleanup import MIN_REPEATS, remove_repeated_lines
from groundcheck.parsing.filing import parse_html
from groundcheck.parsing.recovery import (
    find_mda_start,
    normalize_line,
    recover_cross_reference_layout,
)
from groundcheck.parsing.sections import (
    MAX_OTHER_SECTION_CHARS,
    MIN_KEY_SECTION_CHARS,
    Section,
    check_section_sizes,
)

FIXTURES = Path(__file__).parent / "fixtures" / "parsing"
PROSE = "This paragraph is ordinary running text about the results of the year. " * 3


def doc(*blocks):
    return "\n\n".join(blocks)


def sized(item, n, covers=()):
    return Section(item=item, title="T", text="x" * n, covers=list(covers))


def company(ticker="EXCO"):
    return Company(ticker=ticker, name="Example", cik=1, role="labeled")


def parse_fixture(name, ticker):
    return parse_html(company(ticker), (FIXTURES / name).read_bytes())


# --- Part A: section-size warnings ---------------------------------------------------


def test_thresholds_are_one_named_block():
    assert (MIN_KEY_SECTION_CHARS, MAX_OTHER_SECTION_CHARS) == (1_500, 60_000)
    assert sections_module.KEY_ITEMS == ("1", "1A", "7", "8")


@pytest.mark.parametrize(
    ("item", "chars", "expected"),
    [
        ("7", 396, "Item 7 is only 396 characters"),  # JPM
        ("8", 370, "Item 8 is only 370 characters"),  # JPM
        ("15", 1_050_977, "Item 15 has 1,050,977 characters"),  # JPM
        ("7", 265, "Item 7 is only 265 characters"),  # XOM
        ("8", 710, "Item 8 is only 710 characters"),  # XOM
        ("16", 342_034, "Item 16 has 342,034 characters"),  # XOM
        ("1A", 1_499, "Item 1A is only 1,499 characters"),
        ("9A", 60_001, "Item 9A has 60,001 characters"),
    ],
)
def test_real_sizes_that_must_warn(item, chars, expected):
    warnings = check_section_sizes([sized(item, chars)])
    assert len(warnings) == 1 and warnings[0].startswith(expected)


@pytest.mark.parametrize(
    ("item", "chars", "covers"),
    [
        ("8", 227_361, ()),  # JNJ
        ("1", 126_647, ()),  # BRK-B
        ("2", 40_860, ()),  # XOM
        ("1", 126_647, ("1", "2")),  # "Items 1 and 2" counts as a key section
        ("7", 1_500, ()),
        ("16", 60_000, ()),
        ("1B", 41, ()),  # a short non-key Item ("None.") is normal
    ],
)
def test_real_sizes_that_must_not_warn(item, chars, covers):
    assert check_section_sizes([sized(item, chars, covers)]) == []


def test_parse_counts_size_warnings_and_shows_recovery(capsys, tmp_path):
    raw = tmp_path / "raw" / "BANK" / "0000000001-25-000001"
    raw.mkdir(parents=True)
    (raw / "bank.htm").write_bytes((FIXTURES / "xref_bank_style.htm").read_bytes())
    config = tmp_path / "companies.yaml"
    config.write_text(
        "companies:\n"
        "  - {ticker: BANK, name: Example Bank, cik: 1, role: labeled,\n"
        "     accession: '0000000001-25-000001', primary_document: bank.htm}\n"
    )
    args = ["parse", "--config", str(config), "--raw-dir", str(tmp_path / "raw")]
    assert main([*args, "--parsed-dir", str(tmp_path / "parsed")]) == 0
    out = capsys.readouterr().out
    assert "repeated header lines removed: 26" in out
    assert "WARNING: Item 7 is only" in out and "WARNING: Item 15 has" in out
    assert "3 warning(s)" in out
    assert out.count("recovered=true") == 2
    assert "NOTE: recovered Item 7" in out and "NOTE: Item 15 shrunk to" in out
    saved = json.loads((tmp_path / "parsed" / "BANK.json").read_text())
    assert {s["item"]: s["recovered"] for s in saved["sections"]}["7"] is True


# --- Part B: repeated running headers ------------------------------------------------


def test_repeated_header_is_removed_but_the_real_heading_kept(caplog):
    header = "Management’s discussion and analysis"
    blocks = [header, PROSE]  # the real heading, followed by prose
    for i in range(MIN_REPEATS - 1):
        blocks += [f"  {header.upper()}  ", f"Page {i} text. " + PROSE]
    with caplog.at_level(logging.INFO):
        text, removed = remove_repeated_lines(doc(*blocks), label="EXCO")
    assert removed == MIN_REPEATS - 1
    assert text.startswith(header + "\n\n")
    assert text.casefold().count(header.casefold()) == 1
    assert any("EXCO: removed 7 repeated header line(s)" in r.getMessage() for r in caplog.records)


def test_fewer_than_min_repeats_are_kept():
    blocks = []
    for _ in range(MIN_REPEATS - 1):
        blocks += ["Running header", PROSE]
    text, removed = remove_repeated_lines(doc(*blocks))
    assert removed == 0 and text.count("Running header") == MIN_REPEATS - 1


def test_long_lines_tables_and_item_headings_are_never_removed():
    long_line = "x" * 121
    table = "[TABLE]\nTotal | 1\n[/TABLE]"
    heading = "Item 8. Financial Statements (continued)"
    blocks = [b for _ in range(10) for b in (long_line, table, heading, PROSE)]
    assert remove_repeated_lines(doc(*blocks)) == (doc(*blocks), 0)


def test_first_occurrence_without_prose_is_dropped_and_a_later_heading_kept():
    header = "MANAGEMENT’S DISCUSSION AND ANALYSIS"
    toc = "[TABLE]\nBusiness Profile | 60\nFinancial Statements | 69\n[/TABLE]"
    blocks = [header, toc]  # a TOC entry: followed by a table of contents, not prose
    blocks += [header, "FUNCTIONAL EARNINGS SUMMARY", "[TABLE]\n | 2025\nUpstream | 1\n[/TABLE]"]
    blocks += [PROSE]  # the real heading: subheading and a data table, then prose
    for _ in range(MIN_REPEATS):
        blocks += [header, PROSE]
    text, removed = remove_repeated_lines(doc(*blocks))
    assert removed == MIN_REPEATS + 1
    kept = text.split("\n\n")
    assert kept.count(header) == 1
    assert kept[kept.index(header) + 1] == "FUNCTIONAL EARNINGS SUMMARY"


def test_header_never_followed_by_prose_is_removed_entirely():
    blocks = [b for _ in range(MIN_REPEATS) for b in ("Example Corp", "[TABLE]\nA | 1\n[/TABLE]")]
    text, removed = remove_repeated_lines(doc(*blocks))
    assert removed == MIN_REPEATS and "Example Corp" not in text


# --- Part C: cross-reference layout recovery -----------------------------------------


def test_bank_style_layout_is_recovered():
    parsed, text = parse_fixture("xref_bank_style.htm", "BANK")
    sections = {s.item: s for s in parsed.sections}
    assert [s.item for s in parsed.sections] == sorted(sections, key=sections_module.item_sort_key)
    mda, fs, host = sections["7"], sections["8"], sections["15"]
    assert mda.recovered and fs.recovered and not host.recovered
    # The stub's own text stays at the top; the real MD&A follows it.
    assert mda.text.startswith("[TABLE]\nItem 7. | Management’s discussion and analysis")
    assert "The following is Management’s discussion and analysis" in mda.text
    assert "Report of Independent Registered" not in mda.text
    assert "Management’s report on internal control" in mda.text
    assert "\n\nReport of Independent Registered Public Accounting Firm\n\n" in fs.text
    assert "Glossary of terms" in fs.text  # Item 8 runs to the end of the old host
    # The host keeps what came before MD&A, including the annual report's TOC rows.
    assert len(host.text) < MAX_OTHER_SECTION_CHARS and "Exhibit 21" in host.text
    assert "Report of Independent Registered Public Accounting Firm | 170" in host.text
    assert "on pages 50–160" in host.text  # a page reference is not the MD&A start
    assert len(text) >= sum(len(s.text) for s in parsed.sections)
    assert any(n.startswith("recovered Item 7 (") and "at line" in n for n in parsed.notes)


def test_energy_style_layout_is_recovered():
    parsed, _ = parse_fixture("xref_energy_style.htm", "ENGY")
    sections = {s.item: s for s in parsed.sections}
    mda, fs, host = sections["7"], sections["8"], sections["16"]
    assert mda.recovered and fs.recovered
    assert mda.text.startswith("ITEM 7. MANAGEMENT’S DISCUSSION")
    assert "FUNCTIONAL EARNINGS SUMMARY" in mda.text
    assert "REPORT OF MANAGEMENT" in mda.text and "REPORT OF INDEPENDENT" not in mda.text
    assert "SUPPLEMENTAL INFORMATION ON OIL AND GAS" in fs.text
    assert "SIGNATURES" in host.text and "FINANCIAL SECTION" in host.text
    assert "Report of Independent Registered Public Accounting Firm | 69" in host.text
    assert sections["1"].covers == ["1", "2"]
    assert parsed.warnings[-1].startswith("Item 16 has")  # as found, before recovery


def test_recovered_filing_chunks_with_unique_ids():
    parsed, _ = parse_fixture("xref_energy_style.htm", "ENGY")
    chunks = FixedSizeChunker().chunk(parsed)
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)
    assert any(i.startswith("ENGY-7-p") for i in ids) and any(i.startswith("ENGY-8-") for i in ids)


def host_with(*blocks):
    filler = PROSE * (MAX_OTHER_SECTION_CHARS // len(PROSE) + 1)
    return Section(item="15", title="Exhibits", text=doc("Item 15. Exhibits", *blocks, filler))


def stubbed(host):
    return [sized("1", 5_000), sized("7", 300), sized("8", 300), host]


def test_missing_mda_start_changes_nothing_and_warns(caplog):
    sections = stubbed(host_with("Report of Independent Registered Public Accounting Firm"))
    with caplog.at_level(logging.WARNING):
        result = recover_cross_reference_layout(sections, label="EXCO")
    assert result.sections == sections and result.notes == []
    assert len(result.warnings) == 1 and "sections left unchanged" in result.warnings[0]
    assert caplog.records[0].getMessage().startswith("EXCO: cross-reference layout suspected")


def test_missing_financial_statements_start_changes_nothing_and_warns():
    sections = stubbed(host_with("Management's discussion and analysis", PROSE))
    result = recover_cross_reference_layout(sections, label="EXCO")
    assert result.sections == sections
    assert "Report of Independent Registered" in result.warnings[0]


def test_table_of_contents_rows_never_count_as_starts():
    toc = (
        "[TABLE]\nManagement’s discussion and analysis | 50\n"
        "Report of Independent Registered Public Accounting Firm | 69\n[/TABLE]"
    )
    one_row = "[TABLE]\nManagement’s discussion and analysis | 50\n[/TABLE]"
    sections = stubbed(host_with(toc, one_row, PROSE, "Notes | 5", PROSE))
    result = recover_cross_reference_layout(sections)
    assert result.sections == sections and result.warnings


def test_straight_and_curly_apostrophes_and_prose_starts():
    assert normalize_line("Management’s Discussion and Analysis.") == normalize_line(
        "MANAGEMENT'S DISCUSSION AND ANALYSIS"
    )
    opener = "The following is Management's discussion and analysis of the results. " + PROSE
    blocks = ["x", "See Management's discussion and analysis on page 50. " + PROSE, opener, PROSE]
    assert find_mda_start(blocks) == 2  # the page reference is skipped


def test_recovery_needs_both_triggers():
    host = host_with("Management's discussion and analysis", PROSE)
    no_stub = [sized("7", 5_000), sized("8", 5_000), host]
    assert recover_cross_reference_layout(no_stub).sections == no_stub
    small_host = [sized("7", 300), sized("8", 300), sized("15", 59_000)]
    assert recover_cross_reference_layout(small_host).sections == small_host


def test_a_full_item_8_is_never_overwritten():
    host = host_with(
        "Management's discussion and analysis",
        PROSE,
        "Report of Independent Registered Public Accounting Firm",
        PROSE,
    )
    sections = [sized("7", 300), sized("8", 90_000), host]
    result = recover_cross_reference_layout(sections)
    assert result.sections == sections and "Item 8 already has" in result.warnings[0]


def test_committed_fixtures_match_their_generator():
    spec = importlib.util.spec_from_file_location("build", FIXTURES / "build_xref_fixtures.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    assert (FIXTURES / "xref_bank_style.htm").read_text(encoding="utf-8") == build.bank_style()
    assert (FIXTURES / "xref_energy_style.htm").read_text(encoding="utf-8") == build.energy_style()
