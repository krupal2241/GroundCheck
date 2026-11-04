import logging
from pathlib import Path

import pytest

from groundcheck.config import Company
from groundcheck.parsing.cleanup import is_page_furniture, remove_page_furniture
from groundcheck.parsing.filing import parse_html
from groundcheck.parsing.html_text import html_to_text
from groundcheck.parsing.sections import item_sort_key, parse_heading, split_sections

FIXTURES = Path(__file__).parent / "fixtures" / "parsing"
BODY = "This paragraph is real section content. " * 10  # > MIN_SECTION_CHARS


def doc(*blocks):
    return "\n\n".join(blocks)


def by_item(result):
    return {s.item: s for s in result.sections}


# --- heading detection -------------------------------------------------------------


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ("Item 1. Business", ("1", [], "Business")),
        ("ITEM 1A. RISK FACTORS", ("1A", [], "RISK FACTORS")),
        ("Item 7 — Management’s Discussion", ("7", [], "Management’s Discussion")),
        ("Item 1A.", ("1A", [], "")),
        ("Item 8. Financial Statements 45", ("8", [], "Financial Statements")),
        ("Items 1 and 2. Business and Properties", ("1", ["1", "2"], "Business and Properties")),
        ("PART II, Item 5. Market", ("5", [], "Market")),
        ("item 9a. Controls", ("9A", [], "Controls")),
        (
            "[TABLE]\nItem 7. | Management’s Discussion\n[/TABLE]",
            ("7", [], "Management’s Discussion"),
        ),
    ],
)
def test_headings_are_recognized(block, expected):
    assert parse_heading(block) == expected


@pytest.mark.parametrize(
    "block",
    [
        "Item 7 of this report discusses results.",  # a sentence, not a heading
        "As described in Item 7, sales grew.",  # doesn't start with Item
        "Item 1. " + "x" * 300,  # too long to be a heading
        "[TABLE]\nItem 1. | Business | 3\nItem 1A. | Risk Factors | 5\n[/TABLE]",  # a TOC table
        "Itemized deductions",
    ],
)
def test_non_headings_are_ignored(block):
    assert parse_heading(block) is None


def test_item_order():
    items = ["8", "1A", "16", "1", "7A", "1C", "7", "1B", "2"]
    assert sorted(items, key=item_sort_key) == ["1", "1A", "1B", "1C", "2", "7", "7A", "8", "16"]


# --- splitting ----------------------------------------------------------------------


def test_prose_toc_duplicates_are_skipped():
    text = doc(
        "Cover page",
        "Item 1. Business 3",
        "Item 1A. Risk Factors 5",
        "Item 7. MD&A 9",
        "Item 8. Financial Statements 12",
        "Item 1. Business",
        "Business " + BODY,
        "Item 1A. Risk Factors",
        "Risk " + BODY,
        "Item 7. MD&A",
        "MDA " + BODY,
        "Item 8. Financial Statements",
        "FS " + BODY,
    )
    result = split_sections(text)
    assert [s.item for s in result.sections] == ["1", "1A", "7", "8"]
    sections = by_item(result)
    assert sections["1"].text.startswith("Item 1. Business\n\nBusiness ")
    assert "Risk " not in sections["1"].text
    assert sections["8"].text.startswith("Item 8. Financial Statements\n\nFS ")
    assert result.warnings == []
    assert result.front_matter_chars > 0


def test_last_toc_entry_with_long_text_after_it_is_not_picked():
    # The last TOC line is followed by a long forward-looking-statements block; ordering
    # still makes the real Item 9 win.
    text = doc(
        "Item 1. Business 3",
        "Item 9. Controls 40",
        "Forward-looking " + BODY,
        "Item 1. Business",
        BODY,
        "Item 9. Controls",
        "Controls " + BODY,
    )
    result = split_sections(text, expected=())
    assert by_item(result)["9"].text.startswith("Item 9. Controls\n\nControls ")
    assert not any("Forward-looking" in s.text for s in result.sections)  # front matter


def test_short_item_body_is_kept():
    text = doc(
        "Item 1B. Unresolved Staff Comments 8",  # TOC line
        "Item 1. Business",
        BODY,
        "Item 1B. Unresolved Staff Comments",
        "None.",
        "Item 2. Properties",
        BODY,
    )
    sections = by_item(split_sections(text, expected=()))
    assert sections["1B"].text == "Item 1B. Unresolved Staff Comments\n\nNone."


def test_continued_heading_stays_inside_its_section():
    text = doc(
        "Item 8. Financial Statements",
        "Part one " + BODY,
        "Item 8. Financial Statements (continued)",
        "Part two " + BODY,
        "Item 9. Changes",
        BODY,
    )
    result = split_sections(text, expected=())
    assert [s.item for s in result.sections] == ["8", "9"]
    assert "Part one" in result.sections[0].text and "Part two" in result.sections[0].text


def test_title_in_next_block():
    text = doc("Item 1A.", "Risk Factors", BODY)
    assert by_item(split_sections(text, expected=()))["1A"].title == "Risk Factors"


def test_combined_items_cover_both():
    text = doc("Items 1 and 2. Business and Properties", BODY, "Item 1A. Risk", BODY)
    result = split_sections(text, expected=("1", "1A"))
    assert result.sections[0].covers == ["1", "2"]
    assert result.warnings == []


def test_part_heading_is_not_left_at_the_end_of_a_section():
    text = doc("Item 1A. Risk", BODY, "PART II", "Item 7. MD&A", BODY)
    assert not by_item(split_sections(text, expected=()))["1A"].text.endswith("PART II")


def test_missing_expected_items_are_warnings(caplog):
    text = doc("Item 1. Business", BODY, "Item 7. MD&A", BODY)
    with caplog.at_level(logging.WARNING):
        result = split_sections(text, label="EXCO")
    assert result.warnings == ["missing Item 1A", "missing Item 8"]
    logged = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert logged == ["EXCO: missing Item 1A", "EXCO: missing Item 8"]


def test_no_headings_at_all(caplog):
    with caplog.at_level(logging.WARNING):
        result = split_sections("Just a cover page.\n\nNothing else.")
    assert result.sections == []
    assert len(result.warnings) == 4


# --- page furniture ------------------------------------------------------------------


@pytest.mark.parametrize(
    "block",
    ["23", "Page 7", "- 13 -", "ii", "Table of Contents", "Example Corp | 2025 Form 10-K | 23"],
)
def test_page_furniture_is_detected(block):
    assert is_page_furniture(block)


@pytest.mark.parametrize(
    "block",
    [
        "Item 16. Form 10-K Summary 95",
        "Net sales were $23 million.",
        "We filed our 2024 Form 10-K with the SEC on February 1, and it is available online. "
        "See page 10 for details about the filing and the related exhibits 23",
        "[TABLE]\n | 2025\nNet sales | 23\n[/TABLE]",
        "2025",
    ],
)
def test_real_content_is_not_furniture(block):
    assert not is_page_furniture(block)


def test_remove_page_furniture_counts():
    text, removed = remove_page_furniture(doc("A", "12", "B", "Table of Contents"))
    assert text == "A\n\nB"
    assert removed == 2


# --- end to end on the synthetic 10-K -------------------------------------------------


def test_synthetic_10k_end_to_end(caplog):
    html = (FIXTURES / "sections_10k.htm").read_bytes()
    company = Company(
        ticker="EXCO",
        name="Example Corp",
        cik=1,
        role="labeled",
        accession="0000000001-25-000001",
        primary_document="exco-20251231.htm",
        period="2025-12-31",
    )
    with caplog.at_level(logging.WARNING):
        parsed, text = parse_html(company, html)
    assert [s.item for s in parsed.sections] == ["1", "1A", "1B", "7", "8", "9"]
    sections = {s.item: s for s in parsed.sections}
    assert sections["1"].title == "Business"
    assert sections["1A"].title == "Risk Factors"
    assert sections["1B"].text.endswith("None.")
    assert sections["7"].title == "Management’s Discussion and Analysis"
    assert "12,345" in sections["7"].text  # the data table stays inside its section
    assert "Notes to the consolidated" in sections["8"].text  # (continued) page kept
    assert "Example Corp designs" in sections["1"].text
    assert "Forward-looking" not in sections["1"].text  # front matter excluded
    assert parsed.warnings == []
    # 3 footers, 2 page numbers, the TOC back-link and the TOC title itself
    assert parsed.removed_furniture_blocks == 7
    assert "2025 Form 10-K |" not in text and "- 13 -" not in text
    assert parsed.period == "2025-12-31"
    assert len(parsed.source_sha256) == 64


def test_existing_fixtures_report_missing_items(caplog):
    # The Day 3 fixtures are fragments: each has one Item heading, so the others warn.
    company = Company(ticker="X", name="X", cik=1, role="labeled")
    with caplog.at_level(logging.WARNING):
        parsed, _ = parse_html(company, (FIXTURES / "workiva_style.htm").read_bytes())
    assert [s.item for s in parsed.sections] == ["7"]
    assert parsed.warnings[:3] == ["missing Item 1", "missing Item 1A", "missing Item 8"]
    assert parsed.warnings[3].startswith("Item 7 is only")  # a fragment, so a stub
    assert "Example Corp | 2025 Form 10-K | 23" in html_to_text(
        (FIXTURES / "workiva_style.htm").read_bytes()
    )
    assert "2025 Form 10-K | 23" not in parsed.sections[0].text  # footer removed
