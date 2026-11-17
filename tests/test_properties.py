"""Property-based tests (hypothesis) for the number normalizer and the claim filter:
round trips, scale equivalence, and no crash on garbage."""

import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from groundcheck.finnum import (  # noqa: E402
    aliases,
    expand_query,
    format_number,
    number_token,
    parse_quantities,
    same_number,
    scale_hint,
)
from groundcheck.generation.claims import (  # noqa: E402
    claim_problems,
    is_abstention,
    is_meta,
    near_duplicate,
    numbers_in,
    numbers_not_in_answer,
    validate_claims,
)
from groundcheck.retrieval.bm25 import BM25, tokenize  # noqa: E402
from groundcheck.verify.base import EvidenceChunk  # noqa: E402
from groundcheck.verify.numeric import NumericVerifier  # noqa: E402
from groundcheck.verify.spans import overlap_score  # noqa: E402

settings.register_profile("groundcheck", deadline=None, max_examples=150)
settings.load_profile("groundcheck")

# Text that looks like what the parser meets: digits, separators, signs, units, words.
ALPHABET = "0123456789.,$%()-−| \n\tFYfybnmkKMBpercentillionhousandto&'’"
garbage = st.one_of(st.text(max_size=200), st.text(alphabet=ALPHABET, max_size=120))
# Not 1900-2099: a bare four-digit integer in that range is read as a year.
amounts = st.integers(min_value=0, max_value=10**12).filter(lambda n: not 1900 <= n <= 2099)
decimals = st.integers(min_value=0, max_value=3)


def one(text):
    (quantity,) = parse_quantities(text)
    return quantity


# --- no crash on garbage -----------------------------------------------------------------


@given(garbage)
def test_parser_never_crashes_and_spans_point_into_the_text(text):
    found = parse_quantities(text)
    for q in found:
        assert 0 <= q.start < q.end <= len(text)
        assert text[q.start : q.end] == q.text
        assert q.kind in ("currency", "percent", "count", "year")
        assert q.decimals >= 0 and q.scale >= 1 and q.value == q.raw * q.scale
        assert same_number(q, q) and same_number(q, q, rounding=True, implied=True, signed=True)
    assert [q.start for q in found] == sorted(q.start for q in found)
    assert scale_hint(text) in (None, 1e3, 1e6, 1e9)
    assert expand_query(text).startswith(text)


@given(garbage, garbage)
def test_matching_is_symmetric_on_garbage(a, b):
    for x in parse_quantities(a)[:5]:
        for y in parse_quantities(b)[:5]:
            for modes in ({}, {"rounding": True}, {"implied": True, "rounding": True}):
                assert same_number(x, y, **modes) == same_number(y, x, **modes)


@given(garbage, garbage)
def test_claim_filter_and_retrieval_helpers_never_crash(claim, answer):
    assert isinstance(claim_problems(claim, answer, [answer]), list)
    assert isinstance(is_meta(claim), bool) and isinstance(is_abstention(claim), bool)
    assert near_duplicate(claim, answer) == near_duplicate(answer, claim)
    kept, rejected = validate_claims([claim, answer, claim], answer)
    assert len(kept) + len(rejected) == 3
    tokens = tokenize(claim)
    assert all(t and "," not in t for t in tokens)
    assert tokenize(" ".join(tokens)) == tokens  # tokens are already canonical
    assert 0.0 <= overlap_score(claim, answer) <= 1.0
    BM25([claim, answer]).top(expand_query(answer), 2)


@given(garbage, garbage)
def test_numeric_verifier_never_crashes(claim, passage):
    if not claim.strip():
        return
    result = NumericVerifier().verify(claim, [EvidenceChunk("X-1-p0000", passage)])
    assert result.label in ("supported", "contradicted", "unsupported")
    assert 0.0 <= result.score <= 1.0
    if result.evidence is not None:
        assert result.evidence.chunk_id == "X-1-p0000"


# --- round trips -------------------------------------------------------------------------


@given(amounts, decimals, st.booleans(), st.booleans())
def test_written_numbers_round_trip(n, places, separators, dollar):
    value = n / 10**places
    text = f"{value:,.{places}f}" if separators else f"{value:.{places}f}"
    q = one(("$" if dollar else "") + text)
    assert q.raw == pytest.approx(value) and q.decimals == places
    assert q.kind == ("currency" if dollar else "count") and not q.scale_stated
    assert q.token == format_number(value) == number_token(text)
    assert one(q.token).raw == pytest.approx(value)  # the canonical token parses back


@given(st.integers(min_value=100, max_value=10**9))
def test_accounting_negatives_round_trip(n):
    assert one(f"({n:,})").raw == -n and one(f"$({n:,})").raw == -n
    assert one(f"-{n}").raw == -n
    assert same_number(one(f"({n:,})"), one(f"{n:,}"))  # signs are ignored by default
    assert not same_number(one(f"({n:,})"), one(f"{n:,}"), signed=True)


@given(st.integers(min_value=1900, max_value=2099))
def test_fiscal_years_round_trip(year):
    for text in (str(year), f"fiscal {year}", f"fiscal year {year}", f"FY{year}", f"FY {year}"):
        q = one(text)
        assert (q.kind, q.raw) == ("year", year)
    if year >= 2000:
        assert one(f"FY{year % 100:02d}").raw == year


@given(st.text(alphabet="0123456789,.", min_size=1, max_size=20))
def test_number_token_is_idempotent(text):
    token = number_token(text)
    assert number_token(token) == token and "," not in token


# --- scale equivalence -----------------------------------------------------------------


@given(st.integers(min_value=1, max_value=10**6))
def test_the_same_amount_at_every_scale_is_the_same_number(k):
    forms = [
        f"${k} million",
        f"${k * 1000:,} thousand",
        f"${k / 1000:.3f} billion",
        f"${k * 1_000_000:,}",
        f"{k:,} million dollars",
    ]
    quantities = [one(f) for f in forms]
    for a in quantities:
        for b in quantities:
            assert same_number(a, b), (a.text, b.text)
    assert len({q.value_token for q in quantities}) == 1
    # A bare table cell matches only through an implied scale, and only the right one.
    cell = one(f"{k:,}")
    stated = one(f"${k} million")
    assert same_number(stated, cell, implied=True, hint=1e6)
    assert not same_number(stated, cell, implied=True, hint=1e3)
    # The wrong scale word is never the same number.
    assert not same_number(stated, one(f"${k} billion"))


@given(st.integers(min_value=1000, max_value=10**7))
def test_correct_rounding_to_the_claims_precision_matches(millions):
    cell = one(f"{millions:,}")
    rounded = one(f"${round(millions / 1000, 1):.1f} billion")
    modes = {"rounding": True, "implied": True, "hint": 1e6}
    assert same_number(rounded, cell, **modes)
    off = one(f"${round(millions / 1000, 1) + 0.2:.1f} billion")  # two steps away
    assert not same_number(off, cell, **modes)


@given(amounts.filter(lambda n: n > 0), st.sampled_from(["thousand", "million", "billion"]))
def test_aliases_are_the_same_amount_in_other_units(n, word):
    quantity = one(f"${n:,} {word}")
    scale = {"thousand": 1e3, "million": 1e6, "billion": 1e9}[word]
    for token in aliases(quantity):
        assert "," not in token and token != quantity.token
        assert any(
            float(token) * s == pytest.approx(n * scale, rel=1e-9) for s in (1.0, 1e3, 1e6, 1e9)
        )


# --- the claim filter ----------------------------------------------------------------------


@given(garbage)
def test_a_text_never_adds_numbers_to_itself(text):
    assert numbers_not_in_answer(text, text) == []
    assert numbers_in(text) == {q.token for q in parse_quantities(text)}


@given(st.integers(min_value=1, max_value=10**6), st.integers(min_value=1, max_value=99))
def test_rescaled_and_reformatted_claims_pass_the_number_check(k, pct):
    answer = f"Apple repurchased ${k * 1000:,} thousand of stock, up {pct}.0% in fiscal 2025."
    claim = f"Apple repurchased ${k} million of stock, up {pct}% in FY2025."
    assert numbers_not_in_answer(claim, answer) == []
    assert claim_problems(claim, answer, []) == []
    added = claim.replace(f"up {pct}%", f"up {pct + 100}%")
    assert numbers_not_in_answer(added, answer) == [str(pct + 100)]


@given(st.integers(min_value=1, max_value=99))
def test_opposite_claims_are_never_near_duplicates(pct):
    up = f"Apple's net sales in the Americas segment increased {pct}% in fiscal 2025 overall."
    assert near_duplicate(up, up)
    assert not near_duplicate(up, up.replace("increased", "decreased"))
    assert not near_duplicate(up, up.replace("increased", "did not increase by"))
