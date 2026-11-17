"""Financial number normalization (`groundcheck.finnum`) and the places that use it:
the claim filter's number check, BM25 tokens and the evidence-span suggestion."""

import pytest

from groundcheck.finnum import (
    aliases,
    expand_query,
    find_match,
    format_number,
    number_token,
    parse_quantities,
    same_number,
    scale_hint,
)
from groundcheck.generation.claims import claim_problems, near_duplicate, numbers_not_in_answer
from groundcheck.generation.pipeline import RetrievedChunk
from groundcheck.labeling.assist import suggest_span
from groundcheck.retrieval.bm25 import tokenize
from groundcheck.verify.spans import best_unit_span, matched_numbers, overlap_score


def one(text):
    (quantity,) = parse_quantities(text)
    return quantity


@pytest.mark.parametrize(
    ("text", "raw", "scale", "kind"),
    [
        ("$89.3 billion", 89.3, 1e9, "currency"),
        ("89,300 million", 89300, 1e6, "count"),
        ("4%", 4, 1, "percent"),
        ("4 percent", 4, 1, "percent"),
        ("3.5 percentage points", 3.5, 1, "percent"),
        ("(1,234)", -1234, 1, "count"),
        ("$(1,234)", -1234, 1, "currency"),
        ("($5.2) million", -5.2, 1e6, "currency"),
        ("-$5", -5, 1, "currency"),
        ("$391.0", 391, 1, "currency"),
        ("$1.2B", 1.2, 1e9, "currency"),
        ("$5M", 5, 1e6, "currency"),
        ("$3bn", 3, 1e9, "currency"),
        ("402 million", 402, 1e6, "count"),
        ("5 million dollars", 5, 1e6, "currency"),
        (".5%", 0.5, 1, "percent"),
        ("fiscal 2025", 2025, 1, "year"),
        ("fiscal year 2025", 2025, 1, "year"),
        ("FY2025", 2025, 1, "year"),
        ("FY 2025", 2025, 1, "year"),
        ("FY25", 2025, 1, "year"),
        ("2025", 2025, 1, "year"),
    ],
)
def test_parse_one_number(text, raw, scale, kind):
    q = one(text)
    assert (q.raw, q.scale, q.kind) == (raw, scale, kind)
    assert q.value == pytest.approx(raw * scale)
    assert q.scale_stated == (scale != 1)


def test_ranges_and_inherited_scale():
    low, high = parse_quantities("$25 billion to $27 billion")
    assert (low.range_role, high.range_role) == ("low", "high")
    low, high = parse_quantities("between $25 and $27 billion")
    assert low.value == 25e9 and low.scale_stated and high.value == 27e9
    low, high = parse_quantities("4-5%")
    assert (low.kind, low.raw, high.raw) == ("percent", 4, 5)
    # Two years, or a count next to an unrelated amount, are not a range.
    assert [q.range_role for q in parse_quantities("fiscal years 2025 and 2024")] == [None, None]
    assert [q.range_role for q in parse_quantities("5 stores and 3 million shares")] == [None, None]


def test_table_row_signs_and_footnote_markers():
    row = "Greater China | 64,377 | (4)% | 66,952 | (8)% | 72,559"
    assert [q.raw for q in parse_quantities(row)] == [64377, -4, 66952, -8, 72559]
    # "(4)" after a label is a footnote marker; alone in a cell it is minus four.
    assert [q.raw for q in parse_quantities("Advertising services (4) | 46,906")] == [4, 46906]
    assert one("| (4) |").raw == -4
    # A hyphen after a letter or digit is not a minus sign.
    assert [q.raw for q in parse_quantities("2024-2025")] == [2024, 2025]
    assert one("10-K").raw == 10


def test_scale_equivalence_rounding_and_implied_scale():
    billions, millions = one("$89.3 billion"), one("89,300 million")
    cell = one("89,300")
    assert same_number(billions, millions) and same_number(billions, one("$89.30B"))
    assert not same_number(billions, cell)  # nothing says the cell is in millions
    assert same_number(billions, cell, implied=True)
    assert same_number(billions, cell, implied=True, hint=1e6)
    assert not same_number(billions, cell, implied=True, hint=1e3)
    # The rounding rule: correct rounding to the coarser side's precision only.
    exact = one("12,345")
    assert not same_number(one("$12.3 billion"), exact, implied=True)
    assert same_number(one("$12.3 billion"), exact, implied=True, rounding=True)
    assert not same_number(one("$12.4 billion"), exact, implied=True, rounding=True)
    assert not same_number(one("12,345"), one("12,346"), rounding=True)
    # A scale error is a different number when both sides state a scale ...
    assert not same_number(one("$12,345 billion"), one("$12,345 million"), rounding=True)
    # ... unless the caller only cares how the number is written.
    assert same_number(one("$12,345 billion"), one("$12,345 million"), raw_ok=True)


def test_kinds_do_not_cross():
    assert not same_number(one("4%"), one("$4"))
    assert not same_number(one("4%"), one("4 million"))
    assert same_number(one("40.5%"), one("40.5"))  # a percent column without "%"
    assert same_number(one("FY2025"), one("fiscal 2025")) and not same_number(
        one("2025"), one("2024")
    )
    assert not same_number(one("2025"), one("$2,025"))
    assert same_number(one("(4)%"), one("4%")) and not same_number(
        one("(4)%"), one("4%"), signed=True
    )


def test_helpers():
    assert format_number(12345.0) == "12345" and format_number(89.30) == "89.3"
    assert number_token("12,345") == "12345" and number_token("391.0") == "391"
    assert number_token(".50") == "0.5"
    assert scale_hint("(in millions, except per share amounts)") == 1e6
    assert scale_hint("(millions of dollars)") == 1e6 and scale_hint("$ in thousands") == 1e3
    assert scale_hint("Net sales rose.") is None
    assert find_match(one("$89.3 billion"), parse_quantities("4% and 89,300 million")).raw == 89300
    assert parse_quantities("") == [] and parse_quantities("no digits here") == []


# --- the claim filter's number check ---------------------------------------------------


def test_claim_numbers_match_the_answer_by_value():
    answer = "Apple repurchased $89,300 million of stock, 4.0% more than in fiscal 2024."
    assert numbers_not_in_answer("Apple repurchased $89.3 billion of stock.", answer) == []
    assert numbers_not_in_answer("Buybacks rose 4% from FY2024.", answer) == []
    assert numbers_not_in_answer("Apple repurchased $90 billion of stock.", answer) == ["90"]
    # As before: the same number as written passes, whatever the scale word says (the
    # filter only stops numbers the answer never mentions; labeling judges the scale).
    assert claim_problems("Apple repurchased $89,300 billion of stock.", answer, []) == []
    reasons = claim_problems("Apple repurchased $12.5 billion of stock.", answer, [])
    assert reasons == ["contains number(s) not in the answer: 12.5"]


def test_near_duplicates_compare_numbers_by_value():
    a = "Apple repurchased $89.3 billion of its common stock in fiscal 2025."
    assert near_duplicate(a, a.replace("$89.3 billion", "$89,300 million"))
    assert not near_duplicate(a, a.replace("$89.3 billion", "$89.3 million"))


# --- BM25 tokens -----------------------------------------------------------------------


def test_bm25_number_tokens_are_canonical():
    assert tokenize("$391.0 billion") == ["391", "billion"]
    assert tokenize("12,345.50") == ["12345.5"]


# --- the evidence-span suggestion --------------------------------------------------------

TABLE = (
    "[TABLE]\n | 2025 | 2024\nShare repurchases | 89,300 | 77,550\n"
    "Dividends paid | 15,234 | 14,996\n[/TABLE]\n(in millions)"
)


def test_span_scoring_matches_billions_to_a_table_in_millions():
    """The headline example: "$89.3 billion" and "89,300" (in millions) match."""
    claim = "Apple spent $89.3 billion on share repurchases in 2025."
    row = "Share repurchases | 89,300 | 77,550"
    assert matched_numbers(claim, row, 1e6) == (1, 2)  # the amount; the year is in the header
    assert overlap_score(claim, row, 1e6) > overlap_score(claim, "Dividends paid | 15,234", 1e6)
    span = best_unit_span(claim, "A-7-t0000", TABLE)
    assert span.text == row
    # The wrong scale note rules the match out.
    assert matched_numbers(claim, row, 1e3) == (0, 2)


def test_label_tool_suggests_the_row_for_a_rescaled_claim():
    chunks = [
        RetrievedChunk(rank=1, chunk_id="A-7-p0000", score=1.0, item="7", kind="prose",
                       text="The Company returned cash to shareholders through repurchases."),
        RetrievedChunk(rank=2, chunk_id="A-7-t0000", score=0.9, item="7", kind="table", text=TABLE),
    ]  # fmt: skip
    suggestion = suggest_span("Apple spent $89.3 billion on share repurchases.", chunks)
    assert suggestion.span.chunk_id == "A-7-t0000"
    assert suggestion.span.text == "Share repurchases | 89,300 | 77,550"
    rounded = suggest_span("Apple paid dividends of about $15.2 billion.", chunks)
    assert rounded.span.text == "Dividends paid | 15,234 | 14,996"


# --- query aliases -----------------------------------------------------------------------


def test_aliases_and_query_expansion():
    assert aliases(one("$89.3 billion")) == ["89300000000", "89300000", "89300"]
    assert aliases(one("12,345")) == ["12.345", "12.3"]
    assert aliases(one("402 million")) == ["402000000", "402000", "0.402"]
    assert aliases(one("4%")) == [] and aliases(one("2025")) == [] and aliases(one("40")) == []
    query = "Did Apple repurchase $89.3 billion of stock in 2025?"
    assert expand_query(query) == query + " 89300000000 89300000 89300"
    assert expand_query("What are the main risks?") == "What are the main risks?"
