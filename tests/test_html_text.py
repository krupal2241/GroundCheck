from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from groundcheck.parsing.html_text import html_to_text, table_rows

FIXTURES = Path(__file__).parent / "fixtures" / "parsing"


def rows_of(html):
    return table_rows(BeautifulSoup(html, "lxml").find("table"))


def table_block(text):
    start = text.index("[TABLE]")
    return text[start : text.index("[/TABLE]", start) + len("[/TABLE]")]


# --- prose -----------------------------------------------------------------------


def test_blocks_are_separated_and_inline_tags_are_not():
    html = "<div><span>Net </span><b>sales</b><span> rose.</span></div><p>Next</p>"
    assert html_to_text(html) == "Net sales rose.\n\nNext"


def test_whitespace_and_entities_are_normalized():
    html = "<p>Item&nbsp;7.&#160;&#160; Management&#8217;s\n   Discussion&#8203;</p>"
    assert html_to_text(html) == "Item 7. Management’s Discussion"


def test_br_starts_a_new_block():
    assert html_to_text("<p>one<br>two</p>") == "one\n\ntwo"


def test_hidden_script_style_and_comments_are_dropped():
    html = (
        "<head><title>T</title><style>p{}</style></head><body>"
        "<div style='DISPLAY: none'>hidden</div><p hidden>also hidden</p>"
        "<script>js()</script><!-- comment --><p>visible</p></body>"
    )
    assert html_to_text(html) == "visible"


def test_inline_xbrl_tags_are_unwrapped():
    html = (
        "<div style='display:none'><ix:header>header facts</ix:header></div>"
        "<p>Revenue was $<ix:nonFraction name='us-gaap:Revenues'>12,345</ix:nonFraction>.</p>"
    )
    assert html_to_text(html) == "Revenue was $12,345."


def test_bytes_input_uses_declared_encoding():
    html = '<meta charset="windows-1252"><p>Caf\xe9</p>'.encode("cp1252")
    assert html_to_text(html) == "Café"


# --- tables ----------------------------------------------------------------------


def test_dollar_paren_and_percent_cells_are_merged():
    html = (
        "<table><tr><td>A</td><td>$</td><td>(1,234</td><td>)</td></tr>"
        "<tr><td>B</td><td></td><td>5.6</td><td>%</td></tr></table>"
    )
    assert rows_of(html) == [["A", "$(1,234)"], ["B", "5.6%"]]


def test_dollar_and_open_paren_in_separate_cells():
    html = "<table><tr><td>A</td><td>$</td><td>(</td><td>7)</td></tr></table>"
    assert rows_of(html) == [["A", "$(7)"]]


def test_negative_number_in_one_cell_is_kept_verbatim():
    html = "<table><tr><td>Loss</td><td>(1,234)</td></tr></table>"
    assert rows_of(html) == [["Loss", "(1,234)"]]


def test_colspan_header_lands_on_its_data_column():
    # The 2025 header spans the "$" column and the value column; after the "$" is
    # merged, the header must sit above the value, not above an empty column.
    html = (
        "<table>"
        "<tr><td></td><td colspan='2'>2025</td><td colspan='2'>2024</td></tr>"
        "<tr><td>Sales</td><td>$</td><td>10</td><td>$</td><td>9</td></tr>"
        "</table>"
    )
    assert rows_of(html) == [["", "2025", "2024"], ["Sales", "$10", "$9"]]


def test_empty_cells_keep_columns_aligned():
    html = (
        "<table>"
        "<tr><td>A</td><td>1</td><td>2</td></tr>"
        "<tr><td>B</td><td></td><td>3</td></tr>"
        "<tr><td></td><td></td><td></td></tr>"
        "</table>"
    )
    assert rows_of(html) == [["A", "1", "2"], ["B", "", "3"]]


def test_spanning_cell_without_data_below_keeps_its_own_column():
    html = "<table><tr><td colspan='2'>Title</td></tr><tr><td>a</td><td>b</td></tr></table>"
    assert rows_of(html) == [["Title", ""], ["a", "b"]]


@pytest.mark.parametrize("colspan", ["0", "abc", "-3", "100000"])
def test_malformed_colspan_does_not_crash(colspan):
    html = f"<table><tr><td colspan='{colspan}'>x</td><td>y</td></tr></table>"
    assert rows_of(html) == [["x", "y"]]


def test_hidden_rows_and_cells_are_dropped():
    html = (
        "<table><tr><td>A</td><td>1</td><td style='display:none'>secret</td></tr>"
        "<tr style='display:none'><td>B</td><td>2</td></tr></table>"
    )
    assert rows_of(html) == [["A", "1"]]


def test_table_is_rendered_as_pipe_block():
    html = "<p>Before</p><table><tr><td>A</td><td>1</td></tr></table><p>After</p>"
    assert html_to_text(html) == "Before\n\n[TABLE]\nA | 1\n[/TABLE]\n\nAfter"


def test_single_column_table_becomes_prose():
    html = "<table><tr><td><p>Boxed note.</p></td></tr><tr><td>Second.</td></tr></table>"
    assert html_to_text(html) == "Boxed note.\n\nSecond."


def test_bullet_table_becomes_prose():
    html = "<table><tr><td>&#8226;</td><td>first risk;</td></tr></table>"
    assert html_to_text(html) == "• first risk;"


def test_nested_table_is_flattened_into_its_cell():
    html = (
        "<table><tr><td>Outer<table><tr><td>in</td><td>ner</td></tr></table></td>"
        "<td>1</td></tr></table>"
    )
    assert html_to_text(html) == "[TABLE]\nOuter in ner | 1\n[/TABLE]"


def test_empty_table_produces_nothing():
    assert html_to_text("<p>a</p><table><tr><td> </td></tr></table><p>b</p>") == "a\n\nb"


# --- synthetic filings -----------------------------------------------------------


def test_workiva_style_fixture():
    text = html_to_text((FIXTURES / "workiva_style.htm").read_bytes())
    assert "HIDDEN-HEADER" not in text
    assert "must not appear" not in text
    assert "Item 7. Management’s Discussion and Analysis" in text
    assert (
        "Total net sales were $12,345 million in 2025, compared with $11,000 million in "
        "2024, an increase of 12%."
    ) in text
    assert table_block(text) == (
        "[TABLE]\n"
        "| Years ended December 31, |\n"
        "| 2025 | 2024\n"
        "Net sales | $12,345 | $11,000\n"
        "Restructuring charges | (1,234) | —\n"
        "Gross margin percentage | 45.6% | 44.1%\n"
        "[/TABLE]"
    )


def test_donnelley_style_fixture():
    text = html_to_text((FIXTURES / "donnelley_style.htm").read_bytes())
    assert "must not appear" not in text
    assert "Example Holdings 10-K" not in text  # <title> is not body text
    assert "Item 1A. Risk Factors" in text
    assert "• competition from larger firms;" in text
    assert "• changes in interest rates." in text
    assert "Note: this box is a single-cell layout table." in text
    assert "Café revenue is immaterial." in text
    assert table_block(text) == (
        "[TABLE]\n"
        "| 2025 | 2024\n"
        "Net income (loss) | $(2,500) | $3,100\n"
        "Earnings (loss) per share, diluted | $(0.42) | $0.51\n"
        "Segments nested table | 3 | 2\n"
        "[/TABLE]"
    )
    assert text.count("[TABLE]") == 1
