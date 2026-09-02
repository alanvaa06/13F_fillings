import pandas as pd
import pytest

from sec13f.tracker import add_weights, consensus, equal_weight_exposure, exposure, manager_turnover, position_changes


def _h():
    rows = [
        # cik, period, cusip, issuer, put_call, value, shares, asset_type, sector, underlying
        ("1", "2026-03-31", "A", "ALPHA", "", 100.0, 10, "Common Stock", "Information Technology", "Equity"),
        ("1", "2026-03-31", "B", "BETA", "", 100.0, 20, "Common Stock", "Energy", "Equity"),
        ("1", "2026-06-30", "A", "ALPHA", "", 240.0, 20, "Common Stock", "Information Technology", "Equity"),  # doubled shares, price 10->12
        ("1", "2026-06-30", "C", "GAMMA", "", 60.0, 6, "Common Stock", "Financials", "Equity"),                # new
        ("2", "2026-03-31", "A", "ALPHA", "", 50.0, 5, "Common Stock", "Information Technology", "Equity"),
        ("2", "2026-06-30", "A", "ALPHA", "", 36.0, 3, "Common Stock", "Information Technology", "Equity"),    # trim
        ("2", "2026-06-30", "A", "ALPHA", "Put", 12.0, 1, "Put Option", "Information Technology", "Options"),
    ]
    h = pd.DataFrame(rows, columns=["cik", "period", "cusip", "issuer", "put_call", "value_usd", "shares", "asset_type", "sector", "underlying_asset"])
    h["manager"] = "M" + h["cik"]
    h["manager_type"] = "T"
    h["ticker"] = h["cusip"]
    h["title_of_class"] = "COM"
    h["industry"] = ""
    h["display_name"] = h["issuer"]
    return add_weights(h)


def test_weights_sum_to_one_per_manager_period():
    h = _h()
    s = h.groupby(["cik", "period"])["weight"].sum()
    assert (s.round(9) == 1).all()


def test_position_changes_actions_and_decomposition():
    ch = position_changes(_h())
    a = ch[(ch.cik == "1") & (ch.cusip == "A") & (ch.put_call == "")].iloc[0]
    assert a["action"] == "ADD" and a["d_shares"] == 10
    # flow = 10 new shares * implied price 12 = 120 ; price effect = 240-100-120 = 20
    assert a["flow_effect"] == pytest.approx(120) and a["price_effect"] == pytest.approx(20)
    assert ch[(ch.cik == "1") & (ch.cusip == "B")].iloc[0]["action"] == "EXIT"
    assert ch[(ch.cik == "1") & (ch.cusip == "C")].iloc[0]["action"] == "NEW"
    assert ch[(ch.cik == "2") & (ch.put_call == "Put")].iloc[0]["action"] == "NEW"
    assert ch[(ch.cik == "2") & (ch.put_call == "")].iloc[0]["action"] == "TRIM"


def test_turnover():
    ch = position_changes(_h())
    to = manager_turnover(ch)
    # cik 1: |120| + |100 exit at price 5| + |60 new| = 280 ; avg book = (200+300)/2 = 250
    assert to[("1", "2026-06-30")] == pytest.approx(280 / 250)


def test_consensus_excludes_options_and_counts_direction():
    h = _h()
    c = consensus(h, position_changes(h))
    a = c[(c.period == "2026-06-30") & (c.cusip == "A")].iloc[0]
    assert a["holders"] == 2 and a["buyers"] == 1 and a["sellers"] == 1 and a["net_buyers"] == 0
    assert a["total_value"] == 240 + 36  # put excluded


def test_exposure_shares():
    h = _h()
    e = exposure(h, "underlying_asset")
    q2 = e[e.period == "2026-06-30"].set_index("underlying_asset")["share"]
    assert q2["Equity"] == pytest.approx((240 + 60 + 36) / (240 + 60 + 36 + 12))
    ew = equal_weight_exposure(h, "underlying_asset")
    q2 = ew[ew.period == "2026-06-30"].set_index("underlying_asset")["avg_weight"]
    assert q2["Equity"] == pytest.approx((1.0 + 36 / 48) / 2)
