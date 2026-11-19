import pandas as pd
import pytest

from sec13f.cusip_links import cusip_map_for_period, detect_cusip_links, normalize_issuer
from sec13f.movers import company_moves
from sec13f.tracker import add_weights, consensus, position_changes

P0, P1 = "2026-06-30", "2026-09-30"


def _frame(rows):
    h = pd.DataFrame(rows, columns=["cik", "period", "cusip", "issuer", "put_call", "value_usd", "shares", "asset_type"])
    h["manager"] = "M" + h["cik"]
    h["manager_type"] = "T"
    h["ticker"] = ""
    h["title_of_class"] = "COM"
    h["sector"] = "Energy"
    h["industry"] = ""
    h["underlying_asset"] = "Equity"
    h["display_name"] = h["issuer"]
    return add_weights(h)


def _reorg(old="30231G102", new="30233Q108", old_name="EXXON MOBIL CORP", new_name="EXXONMOBIL HOLDINGS CORP",
           factor=1.0, asset_type="Common Stock"):
    """Five managers hold the old CUSIP, then the same five hold the new one (one of them adds)."""
    rows = []
    for i in range(1, 6):
        cik = str(i)
        rows.append((cik, P0, old, old_name, "", 1000.0, 100, asset_type))
        rows.append((cik, P0, "OTHER1", "OTHER CO", "", 500.0, 50, "Common Stock"))
        extra = 20 if i == 1 else 0
        rows.append((cik, P1, new, new_name, "", 1100.0 * (100 * factor + extra) / (100 * factor), 100 * factor + extra, asset_type))
        rows.append((cik, P1, "OTHER1", "OTHER CO", "", 500.0, 50, "Common Stock"))
    return _frame(rows)


def test_normalize_issuer_drops_legal_words():
    assert normalize_issuer("Exxon Mobil Corp") == normalize_issuer("EXXONMOBIL HOLDINGS CORP") == "EXXONMOBIL"
    assert normalize_issuer("GATES INDL CORP PLC") == normalize_issuer("GATES INDL CORP LTD")


def test_redomicile_is_linked_and_diffed_as_one_position():
    h = _reorg()
    links = detect_cusip_links(h)
    assert len(links) == 1
    r = links.iloc[0]
    assert (r.period, r.old_cusip, r.new_cusip, r.share_factor) == (P1, "30231G102", "30233Q108", 1.0)

    ch = position_changes(h, links)
    x = ch[ch.cusip == "30233Q108"]
    assert set(x.action) == {"HOLD", "ADD"}  # no NEW / EXIT
    assert ch[ch.cusip == "30231G102"].empty
    assert (x.cusip_prev == "30231G102").all()
    assert x.loc[x.cik == "1", "d_shares"].iloc[0] == 20

    moves = company_moves(ch)
    m = moves[(moves.period == P1) & (moves.cusip == "30233Q108")].iloc[0]
    assert m.new_holders == 0 and m.exits == 0 and m.buyers == 1
    c = consensus(h, ch)
    assert c[(c.period == P1) & (c.cusip == "30233Q108")].iloc[0]["net_buyers"] == 1


def test_without_links_the_reorg_is_an_exit_plus_entry():
    ch = position_changes(_reorg())
    assert set(ch[ch.cusip == "30231G102"].action) == {"EXIT"}
    assert set(ch[ch.cusip == "30233Q108"].action) == {"NEW"}


def test_reverse_split_rescales_previous_shares():
    h = _reorg(old="08862E109", new="08862E307", old_name="BEYOND MEAT INC", new_name="BEYOND MEAT INC", factor=0.1)
    links = detect_cusip_links(h)
    assert links.iloc[0].share_factor == pytest.approx(0.1)
    ch = position_changes(h, links)
    x = ch[ch.cusip == "08862E307"].set_index("cik")
    assert x.loc["2", "action"] == "HOLD" and x.loc["2", "d_shares"] == pytest.approx(0)
    assert x.loc["1", "action"] == "ADD" and x.loc["1", "d_shares"] == pytest.approx(20)


def test_etfs_and_unrelated_names_are_not_linked():
    assert detect_cusip_links(_reorg(old_name="TIDAL TRUST II", new_name="TIDAL TRUST II", asset_type="ETF")).empty
    assert detect_cusip_links(_reorg(old_name="REMAX HOLDINGS INC", new_name="REAL REMAX GROUP INC")).empty


def test_too_few_common_holders_is_not_linked():
    h = _reorg()
    h = h[~((h.cusip == "30233Q108") & h.cik.isin(["1", "2", "3"]))]  # only 2 of 5 holders carry over
    assert detect_cusip_links(h).empty


def test_option_cusips_follow_the_issue():
    links = pd.DataFrame([dict(period=P1, old_cusip="30231G102", new_cusip="30233Q108", share_factor=1.0)])
    m = cusip_map_for_period(links, P1, {"30231G102", "30231G902", "30231G952"}, {"30233Q108", "30233Q908", "30233Q958"})
    assert m == {"30231G102": ("30233Q108", 1.0), "30231G902": ("30233Q908", 1.0), "30231G952": ("30233Q958", 1.0)}
