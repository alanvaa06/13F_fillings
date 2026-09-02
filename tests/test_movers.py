import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sec13f.movers import MoveMagnitude, MoverThresholds, classify_move, company_moves, magnitude_summary
from sec13f.tracker import add_weights, position_changes

IT, EN = "Information Technology", "Energy"


def _row(cik, period, cusip, value, shares, sector=IT, underlying="Equity", asset_type="Common Stock", put_call=""):
    return dict(cik=cik, period=period, cusip=cusip, issuer=cusip, display_name=cusip, ticker=cusip, put_call=put_call, value_usd=float(value),
                shares=shares, asset_type=asset_type, sector=sector, underlying_asset=underlying, title_of_class="COM", industry="")


def _h():
    q1, q2 = "2026-03-31", "2026-06-30"
    rows = [
        # A: three managers hold 100 shares each; all add 50% -> aggregate +50%, 3 buyers -> MAJOR
        _row("1", q1, "A", 100, 100), _row("2", q1, "A", 100, 100), _row("3", q1, "A", 100, 100),
        _row("1", q2, "A", 180, 150), _row("2", q2, "A", 180, 150), _row("3", q2, "A", 180, 150),
        # B: two holders, one trims 2% -> aggregate -1%, MINOR
        _row("1", q1, "B", 100, 100, EN), _row("2", q1, "B", 100, 100, EN),
        _row("1", q2, "B", 98, 98, EN), _row("2", q2, "B", 100, 100, EN),
        # C: one holder trims 0.5% of a 1000-share position -> aggregate -0.25%, below 1% -> NONE
        _row("1", q1, "C", 500, 1000), _row("2", q1, "C", 500, 1000),
        _row("1", q2, "C", 597, 995), _row("2", q2, "C", 600, 1000),
        # D: brand new for the universe (one manager opens) -> pure entry, MAJOR
        _row("3", q2, "D", 40, 4),
        # options, ETFs and preferreds must be ignored
        _row("1", q2, "A", 500, 50, IT, "Options", "Call Option", "Call"),
        _row("1", q1, "SPY", 100, 1, "ETF - Broad Equity", "Equity (ETF)", "ETF"), _row("1", q2, "SPY", 300, 3, "ETF - Broad Equity", "Equity (ETF)", "ETF"),
        _row("2", q1, "PFD", 100, 10, "Financials", "Preferred", "Preferred Stock"), _row("2", q2, "PFD", 50, 5, "Financials", "Preferred", "Preferred Stock"),
    ]
    h = pd.DataFrame(rows)
    h["manager"] = "M" + h["cik"]
    h["manager_type"] = "Hedge Fund - Value"
    return add_weights(h)


def test_company_moves_aggregates_and_classifies():
    m = company_moves(position_changes(_h()))
    m = m[m.period == "2026-06-30"].set_index("cusip")
    assert set(m.index) == {"A", "B", "C", "D"}  # no SPY, no preferred, no option row
    a = m.loc["A"]
    assert a["holders_prev"] == 3 and a["holders_cur"] == 3 and a["buyers"] == 3 and a["sellers"] == 0
    assert a["pct_shares"] == pytest.approx(0.5) and a["magnitude"] == "MAJOR"
    assert a["net_flow"] == pytest.approx(3 * 50 * 1.2)  # 50 shares each at implied price 1.2
    assert a["price_effect"] == pytest.approx(3 * 100 * 0.2)
    b = m.loc["B"]
    assert b["pct_shares"] == pytest.approx(-0.01) and b["magnitude"] == "MINOR" and b["sellers"] == 1
    c = m.loc["C"]
    assert c["magnitude"] == "NONE" and c["n_active"] == 1 and c["pct_shares"] == pytest.approx(-0.0025)
    d = m.loc["D"]
    assert np.isnan(d["pct_shares"]) and d["magnitude"] == "MAJOR" and d["new_holders"] == 1 and d["intensity"] == 1.0
    assert bool(a["eligible_intensity"]) and not bool(d["eligible_intensity"])


def test_magnitude_summary_counts_every_company_once_and_keeps_all_buckets():
    moves = company_moves(position_changes(_h()))
    s = magnitude_summary(moves, "2026-06-30").set_index("magnitude")
    assert s["companies"].sum() == 4
    assert s.loc["MAJOR", "companies"] == 2 and s.loc["MINOR", "companies"] == 1 and s.loc["NONE", "companies"] == 1
    assert list(s.index) == ["MAJOR", "MINOR", "NONE"]
    # a period where nothing moved still lists the three buckets with zeros
    only_major = moves[moves["magnitude"] == "MAJOR"]
    s2 = magnitude_summary(only_major, "2026-06-30").set_index("magnitude")
    assert list(s2.index) == ["MAJOR", "MINOR", "NONE"] and s2.loc["NONE", "companies"] == 0


def test_classify_move_rules():
    t = MoverThresholds()
    assert classify_move(0.10, 0, t) is MoveMagnitude.MAJOR
    assert classify_move(-0.05, -3, t) is MoveMagnitude.MAJOR  # breadth rule
    assert classify_move(0.05, 2, t) is MoveMagnitude.MINOR  # not enough breadth
    assert classify_move(0.01, 0, t) is MoveMagnitude.MINOR
    assert classify_move(0.009, 5, t) is MoveMagnitude.NONE  # many buyers but the aggregate barely moved
    assert classify_move(0.0, 0, t) is MoveMagnitude.NONE
    assert classify_move(float("nan"), 1, t) is MoveMagnitude.MAJOR  # pure entry


def test_thresholds_validate_ordering():
    with pytest.raises(ValueError):
        MoverThresholds(major_pct_shares=0.01, minor_pct_shares=0.05)
    with pytest.raises(ValueError):
        MoverThresholds(minor_pct_shares=0.0)
    with pytest.raises(ValueError):
        MoverThresholds(major_breadth=0)


@settings(max_examples=200, deadline=None)
@given(pct=st.floats(min_value=-5, max_value=5), net_buyers=st.integers(-20, 20))
def test_classification_is_monotone_in_intensity(pct, net_buyers):
    """A bigger move (in absolute terms) can never receive a smaller label."""
    t = MoverThresholds()
    rank = {MoveMagnitude.NONE: 0, MoveMagnitude.MINOR: 1, MoveMagnitude.MAJOR: 2}
    assert rank[classify_move(pct * 2, net_buyers, t)] >= rank[classify_move(pct, net_buyers, t)]


@settings(max_examples=100, deadline=None)
@given(pct=st.floats(min_value=-5, max_value=5), a=st.integers(-20, 20), b=st.integers(-20, 20))
def test_classification_is_monotone_in_breadth(pct, a, b):
    """More net buyers (or sellers) in the same direction can never lower the label."""
    t = MoverThresholds()
    rank = {MoveMagnitude.NONE: 0, MoveMagnitude.MINOR: 1, MoveMagnitude.MAJOR: 2}
    lo, hi = sorted((abs(a), abs(b)))
    assert rank[classify_move(pct, hi, t)] >= rank[classify_move(pct, lo, t)]
