import json
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sec13f.sectors import (
    IMPLICIT_BENCHMARK_NAME, UNIVERSE_BENCHMARK_NAME, Benchmark, SectorConfig, direct_equity_sector_weights,
    load_benchmarks, sector_positioning,
)
from sec13f.tracker import add_weights, position_changes

IT, EN, FI = "Information Technology", "Energy", "Financials"


def _row(cik, period, cusip, value, shares, sector=IT, underlying="Equity", asset_type="Common Stock", put_call=""):
    return dict(cik=cik, period=period, cusip=cusip, issuer=cusip, display_name=cusip, ticker=cusip, put_call=put_call, value_usd=float(value),
                shares=shares, asset_type=asset_type, sector=sector, underlying_asset=underlying, title_of_class="COM", industry="")


def _h(index_cik="9"):
    q1, q2 = "2026-03-31", "2026-06-30"
    rows = [
        # active manager 1: 50/50 IT/Energy in Q1, 75/25 in Q2 (bought IT)
        _row("1", q1, "A", 100, 10), _row("1", q1, "B", 100, 10, EN),
        _row("1", q2, "A", 300, 20), _row("1", q2, "B", 100, 10, EN),
        # active manager 2: 100% Energy both quarters, plus an ETF and a put that must be ignored
        _row("2", q1, "B", 50, 5, EN), _row("2", q1, "SPY", 50, 1, "ETF - Broad Equity", "Equity (ETF)", "ETF"),
        _row("2", q2, "B", 50, 5, EN), _row("2", q2, "A", 10, 1, IT, "Options", "Put Option", "Put"),
        # index manager: 60% IT / 40% Financials -> implicit benchmark
        _row(index_cik, q1, "A", 600, 60), _row(index_cik, q1, "F", 400, 40, FI),
        _row(index_cik, q2, "A", 600, 60), _row(index_cik, q2, "F", 400, 40, FI),
    ]
    h = pd.DataFrame(rows)
    h["manager"] = "M" + h["cik"]
    h["manager_type"] = h["cik"].map(lambda c: "Asset Manager - Index" if c == index_cik else "Hedge Fund - Value")
    return add_weights(h)


def test_direct_equity_weights_ignore_etf_and_options_and_sum_to_one():
    w = direct_equity_sector_weights(_h())
    assert set(w["sector"]) == {IT, EN, FI}
    assert (w.groupby(["period", "cik"])["weight"].sum().round(9) == 1).all()
    m2 = w[(w.cik == "2") & (w.period == "2026-06-30")]
    assert len(m2) == 1 and m2.iloc[0]["sector"] == EN and m2.iloc[0]["weight"] == 1.0


def test_positioning_levels_changes_and_active_weight():
    h = _h()
    sp = sector_positioning(h, position_changes(h))
    q2 = sp[sp.period == "2026-06-30"].set_index("sector")
    # EW over the two active managers only: IT = (0.75 + 0)/2, Energy = (0.25 + 1)/2, Financials = 0
    assert q2.loc[IT, "weight_ew"] == pytest.approx(0.375)
    assert q2.loc[EN, "weight_ew"] == pytest.approx(0.625)
    assert q2.loc[FI, "weight_ew"] == pytest.approx(0.0)
    assert q2.loc[IT, "d_qoq"] == pytest.approx(0.375 - 0.25)
    # implicit benchmark = index manager's book
    assert (q2["benchmark_name"] == IMPLICIT_BENCHMARK_NAME).all()
    assert q2.loc[IT, "weight_bench"] == pytest.approx(0.6) and q2.loc[FI, "weight_bench"] == pytest.approx(0.4)
    assert q2.loc[IT, "active_weight"] == pytest.approx(0.375 - 0.6)
    assert q2.loc[EN, "active_weight"] == pytest.approx(0.625)
    # breadth: manager 1 (75%) is above the 60% benchmark in IT, manager 2 (0%) is not
    assert q2.loc[IT, "overweight_breadth"] == pytest.approx(0.5)
    assert q2.loc[EN, "overweight_breadth"] == pytest.approx(1.0)
    assert q2.loc[IT, "n_active_managers"] == 2 and q2.loc[IT, "n_index_managers"] == 1
    # flows: manager 1 bought 10 A shares at implied price 15 -> +150 into IT; nothing else moved in cash equity
    assert q2.loc[IT, "net_flow"] == pytest.approx(150)
    assert q2.loc[EN, "net_flow"] == pytest.approx(0)
    assert q2.loc[IT, "n_buyers"] == 1


def test_positioning_falls_back_to_universe_without_index_managers():
    h = _h()
    h["manager_type"] = "Hedge Fund - Value"
    sp = sector_positioning(h, position_changes(h))
    assert (sp["benchmark_name"] == UNIVERSE_BENCHMARK_NAME).all()
    q2 = sp[sp.period == "2026-06-30"].set_index("sector")
    # value-weighted universe in Q2: IT 900, Energy 150, Financials 400
    assert q2.loc[IT, "weight_bench"] == pytest.approx(900 / 1450)


def test_history_columns_need_min_history():
    h = _h()
    sp = sector_positioning(h, position_changes(h), SectorConfig(min_history=4))
    assert sp["hist_percentile"].isna().all() and sp["avg_trailing"].isna().all()
    sp2 = sector_positioning(h, position_changes(h), SectorConfig(min_history=2))
    q2 = sp2[sp2.period == "2026-06-30"].set_index("sector")
    assert q2.loc[IT, "hist_percentile"] == pytest.approx(1.0)  # 0.375 is the max of its own 2-point history
    assert q2.loc[EN, "hist_percentile"] == pytest.approx(0.5)  # 0.625 < 0.75 in Q1


def test_external_benchmark_as_of_and_active_column(tmp_path: Path):
    cfg = {"benchmarks": [{"name": "S&P 500", "source": "test", "weights": {
        "2026-03-31": {IT: 30, EN: 5, FI: 65},  # not normalised on purpose
    }}]}
    p = tmp_path / "benchmarks.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    bms = load_benchmarks(p)
    assert len(bms) == 1 and bms[0].slug == "s_p_500"
    assert bms[0].as_of("2026-01-01") is None
    date, ws = bms[0].as_of("2026-06-30")
    assert date == "2026-03-31" and sum(ws.values()) == pytest.approx(1.0)
    h = _h()
    sp = sector_positioning(h, position_changes(h), benchmarks=bms)
    q2 = sp[sp.period == "2026-06-30"].set_index("sector")
    assert q2.loc[IT, "bench_s_p_500"] == pytest.approx(0.30)
    assert q2.loc[IT, "active_s_p_500"] == pytest.approx(0.375 - 0.30)
    assert q2.loc[IT, "asof_s_p_500"] == "2026-03-31"
    assert load_benchmarks(tmp_path / "missing.json") == []


def test_benchmark_rejects_bad_input():
    with pytest.raises(ValueError):
        Benchmark(name="", weights={"2026-03-31": {IT: 1.0}})
    with pytest.raises(ValueError):
        Benchmark(name="X", weights={"2026-03-31": {IT: -1.0}})
    with pytest.raises(ValueError):  # the shipped example file is all zeros on purpose: it must not load silently
        Benchmark(name="X", weights={"2026-03-31": {IT: 0.0, EN: 0.0}})


@settings(max_examples=50, deadline=None)
@given(st.lists(st.floats(min_value=0.01, max_value=1e9), min_size=1, max_size=6))
def test_benchmark_as_of_always_normalises(values):
    ws = {f"S{i}": v for i, v in enumerate(values)}
    b = Benchmark(name="B", weights={"2020-12-31": ws})
    _, norm = b.as_of("2021-03-31")
    assert sum(norm.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in norm.values())
