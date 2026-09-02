"""Amendment handling and unit normalisation on real-world 13F quirks."""
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sec13f.ingest import FilingRef, list_13f_filings
from sec13f.parser import normalize_value_units, select_accessions


def _filings(rows):
    return pd.DataFrame(rows, columns=["cik", "period", "accession", "submission_type", "filing_date", "amendment_type", "n_rows"])


def test_new_holdings_amendment_is_added_to_the_original():
    f = _filings([
        ("1", "2024-03-31", "A-orig", "13F-HR", "2024-05-10", "", 500),
        ("1", "2024-03-31", "A-amend", "13F-HR/A", "2024-06-01", "NEW HOLDINGS", 3),
    ])
    assert select_accessions(f) == {"A-orig", "A-amend"}


def test_restatement_replaces_the_original_and_earlier_amendments():
    f = _filings([
        ("1", "2024-03-31", "orig", "13F-HR", "2024-05-10", "", 500),
        ("1", "2024-03-31", "add1", "13F-HR/A", "2024-05-20", "NEW HOLDINGS", 2),
        ("1", "2024-03-31", "restate", "13F-HR/A", "2024-07-01", "RESTATEMENT", 480),
    ])
    assert select_accessions(f) == {"restate"}


def test_unknown_amendment_type_uses_row_count_heuristic():
    f = _filings([
        ("1", "2024-03-31", "orig", "13F-HR", "2024-05-10", "", 500),
        ("1", "2024-03-31", "small", "13F-HR/A", "2024-06-01", "", 1),      # 1 row vs 500 -> additive
        ("2", "2024-03-31", "orig2", "13F-HR", "2024-05-10", "", 100),
        ("2", "2024-03-31", "big", "13F-HR/A", "2024-06-01", "", 90),       # 90 vs 100 -> restatement
        ("3", "2024-03-31", "only-amend", "13F-HR/A", "2024-06-01", "", 7),  # no original at all -> use it
    ])
    keep = select_accessions(f)
    assert {"orig", "small", "big", "only-amend"} <= keep and "orig2" not in keep


def test_latest_original_wins_when_two_originals_exist():
    f = _filings([
        ("1", "2024-03-31", "first", "13F-HR", "2024-05-10", "", 500),
        ("1", "2024-03-31", "second", "13F-HR", "2024-05-12", "", 502),
    ])
    assert select_accessions(f) == {"second"}


def _book(cik, period, total, n=4):
    return [dict(cik=cik, period=period, cusip=f"C{i}", value_usd=total / n, accession=f"{cik}-{period}") for i in range(n)]


def test_unit_normalisation_rescales_a_quarter_still_reported_in_thousands():
    rows = _book("1", "2022-06-30", 1e11) + _book("1", "2022-09-30", 1.05e11) + _book("1", "2022-12-31", 1.1e8) + _book("1", "2023-03-31", 1.15e11) + _book("1", "2023-06-30", 1.2e11)
    h = pd.DataFrame(rows)
    fixed, factors = normalize_value_units(h)
    assert factors == {("1", "2022-12-31"): 1000.0}
    assert fixed[fixed.period == "2022-12-31"].value_usd.sum() == pytest.approx(1.1e11)
    assert fixed[fixed.period != "2022-12-31"].value_usd.sum() == pytest.approx(h[h.period != "2022-12-31"].value_usd.sum())


def test_unit_normalisation_rescales_a_quarter_reported_in_dollars_too_early():
    rows = _book("9", "2021-03-31", 5e9) + _book("9", "2021-06-30", 5.2e12) + _book("9", "2021-09-30", 5.1e9)
    fixed, factors = normalize_value_units(pd.DataFrame(rows))
    assert factors == {("9", "2021-06-30"): 0.001}
    assert fixed[fixed.period == "2021-06-30"].value_usd.sum() == pytest.approx(5.2e9)


def test_unit_normalisation_leaves_genuine_moves_alone():
    # a real 3x quarter (concentrated fund doubling down) and a first filing with no neighbours must not be touched
    rows = _book("1", "2024-03-31", 1e9) + _book("1", "2024-06-30", 3e9) + _book("1", "2024-09-30", 2.5e9) + _book("2", "2024-06-30", 7e8)
    fixed, factors = normalize_value_units(pd.DataFrame(rows))
    assert factors == {} and fixed.value_usd.sum() == pytest.approx(sum(r["value_usd"] for r in rows))


@settings(max_examples=60, deadline=None)
@given(st.lists(st.floats(min_value=1e9, max_value=4e9), min_size=3, max_size=8), st.integers(min_value=0, max_value=7))
def test_unit_normalisation_is_idempotent_and_only_uses_factors_of_1000(totals, bad):
    bad = min(bad, len(totals) - 1)
    periods = [f"20{20 + i // 4}-{['03-31', '06-30', '09-30', '12-31'][i % 4]}" for i in range(len(totals))]
    rows = []
    for i, (p, t) in enumerate(zip(periods, totals)):
        rows += _book("7", p, t / 1000 if i == bad else t)
    fixed, factors = normalize_value_units(pd.DataFrame(rows))
    assert factors == {('7', periods[bad]): 1000.0}
    again, factors2 = normalize_value_units(fixed)
    assert factors2 == {} and again.value_usd.sum() == pytest.approx(fixed.value_usd.sum())


class _Client:
    """Fake EdgarClient with one submissions block."""

    def __init__(self, block):
        self._block = block

    def submissions(self, cik):
        return {"filings": {"recent": self._block, "files": []}}


def test_list_13f_filings_keeps_originals_and_amendments_for_the_last_n_periods():
    block = {
        "accessionNumber": ["a1", "a1-amend", "a2", "a3", "other"],
        "form": ["13F-HR", "13F-HR/A", "13F-HR", "13F-HR", "10-K"],
        "filingDate": ["2024-05-10", "2024-06-01", "2024-08-10", "2024-11-10", "2024-03-01"],
        "reportDate": ["2024-03-31", "2024-03-31", "2024-06-30", "2024-09-30", "2023-12-31"],
        "primaryDocument": ["xslForm13F_X02/primary_doc.xml"] * 5,
    }
    from sec13f.managers import Manager

    refs = list_13f_filings(_Client(block), Manager(cik="1", name="M", short="M", manager_type="t"), max_filings=2)
    assert [r.accession for r in refs] == ["a3", "a2"]  # last 2 periods, newest first
    refs = list_13f_filings(_Client(block), Manager(cik="1", name="M", short="M", manager_type="t"), max_filings=3)
    assert {r.accession for r in refs} == {"a3", "a2", "a1-amend", "a1"}  # both filings of 2024Q1 come along
    assert all(isinstance(r, FilingRef) for r in refs)
