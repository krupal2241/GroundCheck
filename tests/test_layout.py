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
from groundcheck.parsing.cleanup import MIN_REPEATS, protected_reason, remove_repeated_lines
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
    # Recovery fixed Items 7, 8 and 15, so only its NOTE lines are shown for them.
    assert "WARNING" not in out
    assert "0 warning(s)" in out
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


@pytest.mark.parametrize(
    "line",
    [
        "(millions of dollars)",  # XOM, above each table
        "(In millions, except per share amounts)",
        "Dollars in billions",
        "(in thousands)",
        "partially offset by",  # JPM, a sentence cut by a page break
        "and the impact of lower volumes.",
        "2025 versus 2024",
        "2025 compared with 2024",
        "Fiscal 2026 Compared to Fiscal 2025",
        "2025 vs. 2024",
        "(1)",
        "(a)",
        "(iv)",
        "*",
    ],
)
def test_protected_repeated_lines_keep_every_copy(line, caplog):
    blocks = [b for _ in range(MIN_REPEATS + 2) for b in (line, PROSE)]
    with caplog.at_level(logging.DEBUG, logger="groundcheck.parsing.cleanup"):
        text, removed = remove_repeated_lines(doc(*blocks), label="EXCO")
    assert (text, removed) == (doc(*blocks), 0)
    assert protected_reason(line) is not None
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any(
        "kept repeated line" in r.getMessage() and "protected" in r.getMessage() for r in debug
    )


@pytest.mark.parametrize(
    "line",
    [
        "Exxon Mobil Corporation",
        "JPMorgan Chase & Co.",
        "PART I",
        "PART II",
        "PART IV",
        "Notes to consolidated financial statements",
        "Management’s discussion and analysis",
        "MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
        "________________________________",
        "—————",
        "•",
        "▪",
    ],
)
def test_page_furniture_is_still_removed(line):
    assert protected_reason(line) is None
    blocks = ["Intro", PROSE] + [b for i in range(MIN_REPEATS + 2) for b in (line, f"{i} " + PROSE)]
    text, removed = remove_repeated_lines(doc(*blocks))
    kept = text.split("\n\n")
    assert kept.count(line) <= 1  # at most the copy kept as a real heading
    assert removed >= MIN_REPEATS + 1


def test_one_protected_copy_protects_all_case_variants():
    # "Partially offset by" and "partially offset by" are the same line case-insensitively;
    # the lowercase copy proves it is sentence text, so every copy stays.
    blocks = ["Partially offset by", PROSE]
    blocks += [b for _ in range(MIN_REPEATS) for b in ("partially offset by", PROSE)]
    assert remove_repeated_lines(doc(*blocks)) == (doc(*blocks), 0)


def test_protected_and_furniture_lines_in_one_filing():
    furniture, caption = "Example Corp", "(millions of dollars)"
    blocks = [b for i in range(MIN_REPEATS) for b in (furniture, caption, f"{i} " + PROSE)]
    text, removed = remove_repeated_lines(doc(*blocks))
    assert removed == MIN_REPEATS - 1  # the first "Example Corp" is kept as a heading
    assert text.split("\n\n").count(caption) == MIN_REPEATS


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
    # A successful recovery replaces the size warnings with NOTE lines.
    assert parsed.warnings == []
    assert any(n.startswith("Item 16 shrunk to") for n in parsed.notes)


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


# --- Size warnings and recovery, end to end through `parse` ----------------------------

LONG = "<p>" + PROSE + "</p>"


def _html(*parts):
    return ("<html><body>" + "".join(parts) + "</body></html>").encode()


def _parse_cli(tmp_path, html, capsys):
    raw = tmp_path / "raw" / "EXCO" / "0000000001-25-000001"
    raw.mkdir(parents=True)
    (raw / "exco.htm").write_bytes(html)
    config = tmp_path / "companies.yaml"
    config.write_text(
        "companies:\n"
        "  - {ticker: EXCO, name: Example Corp, cik: 1, role: labeled,\n"
        "     accession: '0000000001-25-000001', primary_document: exco.htm}\n"
    )
    args = ["parse", "--config", str(config), "--raw-dir", str(tmp_path / "raw")]
    assert main([*args, "--parsed-dir", str(tmp_path / "parsed")]) == 0
    out = capsys.readouterr().out
    saved = json.loads((tmp_path / "parsed" / "EXCO.json").read_text())
    return out, saved


def _front(n_host_paragraphs, *host_extra):
    return (
        "<p>Item 1. Business</p>" + LONG * 10,
        "<p>Item 1A. Risk Factors</p>" + LONG * 10,
        "<p>Item 7. Management’s Discussion and Analysis</p><p>See pages 50-160.</p>",
        "<p>Item 8. Financial Statements</p><p>See pages 161-300.</p>",
        "<p>Item 15. Exhibits</p>" + LONG * n_host_paragraphs,
        *host_extra,
    )


def test_failed_recovery_keeps_every_size_warning_and_the_count_matches(tmp_path, capsys):
    out, saved = _parse_cli(tmp_path, _html(*_front(400)), capsys)  # no MD&A start anywhere
    warnings = saved["warnings"]
    assert saved["notes"] == []
    assert any(w.startswith("Item 7 is only") for w in warnings)
    assert any(w.startswith("Item 8 is only") for w in warnings)
    assert any(w.startswith("Item 15 has") for w in warnings)
    assert any("sections left unchanged" in w for w in warnings)
    assert out.count("  WARNING: ") == len(warnings) == 4
    assert f"{len(warnings)} warning(s)" in out


def test_skipped_recovery_keeps_stub_warnings(tmp_path, capsys):
    out, saved = _parse_cli(tmp_path, _html(*_front(5)), capsys)  # no oversized host
    assert saved["notes"] == []
    assert [w.split(" is only")[0] for w in saved["warnings"]] == ["Item 7", "Item 8"]
    assert out.count("  WARNING: ") == 2 and "2 warning(s)" in out


def test_successful_recovery_still_reports_a_leftover_problem(tmp_path, capsys):
    # The host is oversized even before MD&A starts, so after recovery it is still
    # too big: the new size is reported, the stub warnings are not.
    rest = (
        "<p>Management’s discussion and analysis</p>" + LONG * 20,
        "<p>Report of Independent Registered Public Accounting Firm</p>" + LONG * 20,
    )
    out, saved = _parse_cli(tmp_path, _html(*_front(400, *rest)), capsys)
    warnings = saved["warnings"]
    assert any(n.startswith("recovered Item 7") for n in saved["notes"])
    assert len(warnings) == 1 and warnings[0].startswith("Item 15 has")
    shrunk = next(n for n in saved["notes"] if n.startswith("Item 15 shrunk to"))
    assert shrunk.split("to ")[1].split(" ")[0] in warnings[0]
    assert not any("is only" in w for w in warnings)
    assert out.count("  WARNING: ") == 1 and "1 warning(s)" in out
