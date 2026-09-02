"""Sector positioning of the universe: level, change, history and active
weight against a benchmark.

Everything here is computed on the *direct equity* book of each manager
(common stock, ADRs, REITs; no ETFs, options, debt or preferreds) so the
sector weights are comparable with an equity index.

Two benchmarks are supported:

* **Implicit index** (always available): the value-weighted sector mix of the
  index managers in the universe (Vanguard, BlackRock, State Street). Their
  13F books track the US market closely, so their aggregate is a reasonable
  market-cap proxy that needs no external data.
* **External benchmarks** (optional): sector weights supplied in
  ``config/benchmarks.json`` (e.g. S&P 500 GICS weights from a data vendor).
  Snapshots are applied *as-of*: the latest snapshot dated on or before the
  reporting period is used and its date is reported next to the number.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

IMPLICIT_BENCHMARK_NAME = "Índice implícito (managers índice, ponderado por valor)"
UNIVERSE_BENCHMARK_NAME = "Universo completo (ponderado por valor)"


@dataclass(frozen=True)
class SectorConfig:
    """Parameters of the sector positioning analysis."""

    index_manager_types: frozenset[str] = frozenset({"Asset Manager - Index", "Asset Manager - Index/Active"})
    history_quarters: int = 8  # trailing window for the "vs average" gap
    min_history: int = 4  # minimum observations before history statistics are reported
    etf_sector_prefix: str = "ETF"


@dataclass(frozen=True)
class Benchmark:
    """Sector weights of an external benchmark, one snapshot per date."""

    name: str
    weights: dict[str, dict[str, float]]  # period (YYYY-MM-DD) -> sector -> weight
    source: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("benchmark needs a name")
        for period, ws in self.weights.items():
            if not ws or any(v < 0 for v in ws.values()) or sum(ws.values()) <= 0:
                raise ValueError(f"benchmark {self.name!r}: weights for {period} must be non-negative and sum to more than zero")

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")

    def as_of(self, period: str) -> Optional[tuple[str, dict[str, float]]]:
        """Latest snapshot dated on or before ``period`` (normalised to sum 1)."""
        dates = sorted(d for d in self.weights if d <= period)
        if not dates:
            return None
        ws = self.weights[dates[-1]]
        tot = float(sum(ws.values()))
        return dates[-1], {s: v / tot for s, v in ws.items()}


def load_benchmarks(path: Path) -> list[Benchmark]:
    """Read ``benchmarks.json``; an absent file means "no external benchmark"."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[Benchmark] = []
    for b in data.get("benchmarks", []):
        weights = {p: {s: float(v) for s, v in ws.items()} for p, ws in b.get("weights", {}).items()}
        if weights:
            out.append(Benchmark(name=b["name"], weights=weights, source=b.get("source", "")))
    return out


def direct_equity_sector_weights(h: pd.DataFrame, cfg: SectorConfig = SectorConfig()) -> pd.DataFrame:
    """Per (period, cik, sector): value and share of the manager's direct-equity book."""
    eq = h[(h["underlying_asset"] == "Equity") & (~h["sector"].str.startswith(cfg.etf_sector_prefix))]
    g = eq.groupby(["period", "cik", "sector"], as_index=False)["value_usd"].sum()
    g["weight"] = g["value_usd"] / g.groupby(["period", "cik"])["value_usd"].transform("sum")
    return g


def _history_percentile(values: np.ndarray) -> float:
    """Share of the observations (current included) at or below the current one."""
    return float((values <= values[-1]).mean())


def sector_positioning(h: pd.DataFrame, changes: pd.DataFrame, cfg: SectorConfig = SectorConfig(),
                       benchmarks: Optional[list[Benchmark]] = None) -> pd.DataFrame:
    """One row per (period, sector) with level, change, history, flow and
    active weight columns. See module docstring for the definitions."""
    benchmarks = benchmarks or []
    w = direct_equity_sector_weights(h, cfg)
    if w.empty:
        return pd.DataFrame()
    mtype = h.drop_duplicates("cik").set_index("cik")["manager_type"]
    w["is_index"] = w["cik"].map(mtype).isin(cfg.index_manager_types).fillna(False).astype(bool)

    piv = w.pivot_table(index=["period", "cik"], columns="sector", values="weight", aggfunc="sum", fill_value=0.0)
    idx_flag = w.drop_duplicates(["period", "cik"]).set_index(["period", "cik"])["is_index"].reindex(piv.index).fillna(False).astype(bool)
    active = piv[~idx_flag.values]
    if active.empty:  # degenerate universe made only of index managers
        active = piv
    ew = active.groupby(level="period").mean()
    n_active = active.groupby(level="period").size().rename("n_active_managers")

    bench_src = w[w["is_index"]]
    benchmark_name = IMPLICIT_BENCHMARK_NAME
    if bench_src.empty:
        bench_src, benchmark_name = w, UNIVERSE_BENCHMARK_NAME
    bv = bench_src.groupby(["period", "sector"])["value_usd"].sum()
    bench = (bv / bv.groupby(level="period").transform("sum")).unstack(fill_value=0.0).reindex(columns=ew.columns, fill_value=0.0)
    n_index = bench_src.groupby("period")["cik"].nunique().rename("n_index_managers")

    bench_aligned = bench.reindex(active.index.get_level_values("period"))
    bench_aligned.index = active.index
    ow_breadth = (active > bench_aligned.reindex(columns=active.columns, fill_value=0.0)).groupby(level="period").mean()

    out = ew.stack().rename("weight_ew").reset_index()
    out.columns = ["period", "sector", "weight_ew"]
    out = out.sort_values(["sector", "period"]).reset_index(drop=True)
    g = out.groupby("sector")["weight_ew"]
    out["weight_ew_prev"] = g.shift(1)
    out["d_qoq"] = out["weight_ew"] - out["weight_ew_prev"]
    out["weight_ew_4q"] = g.shift(4)
    out["d_yoy"] = out["weight_ew"] - out["weight_ew_4q"]
    trailing = g.transform(lambda s: s.shift(1).rolling(cfg.history_quarters, min_periods=cfg.min_history).mean())
    out["avg_trailing"] = trailing
    out["d_vs_avg"] = out["weight_ew"] - out["avg_trailing"]
    out["hist_percentile"] = g.transform(lambda s: s.expanding(min_periods=cfg.min_history).apply(_history_percentile, raw=True))
    out["hist_min"] = g.transform(lambda s: s.expanding(min_periods=cfg.min_history).min())
    out["hist_max"] = g.transform(lambda s: s.expanding(min_periods=cfg.min_history).max())

    bl = bench.stack().rename("weight_bench").reset_index()
    bl.columns = ["period", "sector", "weight_bench"]
    out = out.merge(bl, on=["period", "sector"], how="left")
    out["weight_bench"] = out["weight_bench"].fillna(0.0)
    out["active_weight"] = out["weight_ew"] - out["weight_bench"]
    out = out.sort_values(["sector", "period"]).reset_index(drop=True)
    out["active_weight_prev"] = out.groupby("sector")["active_weight"].shift(1)
    out["d_active_qoq"] = out["active_weight"] - out["active_weight_prev"]
    ob = ow_breadth.stack().rename("overweight_breadth").reset_index()
    ob.columns = ["period", "sector", "overweight_breadth"]
    out = out.merge(ob, on=["period", "sector"], how="left")
    out = out.merge(n_active.reset_index(), on="period", how="left").merge(n_index.reset_index(), on="period", how="left")
    out["n_index_managers"] = out["n_index_managers"].fillna(0).astype(int)
    out["benchmark_name"] = benchmark_name

    for b in benchmarks:
        wcol, acol, dcol = f"bench_{b.slug}", f"active_{b.slug}", f"asof_{b.slug}"
        rows = []
        for p in out["period"].unique():
            snap = b.as_of(p)
            if snap is None:
                continue
            date, ws = snap
            rows.extend({"period": p, "sector": s, wcol: v, dcol: date} for s, v in ws.items())
        if rows:
            out = out.merge(pd.DataFrame(rows), on=["period", "sector"], how="left")
            out[acol] = out["weight_ew"] - out[wcol]

    out = out.merge(_sector_flows(changes, w[~w["is_index"]][["period", "cik"]].drop_duplicates(), cfg), on=["period", "sector"], how="left")
    for c in ("net_flow", "gross_flow", "price_effect"):
        if c in out:
            out[c] = out[c].fillna(0.0)
    for c in ("n_buyers", "n_sellers"):
        if c in out:
            out[c] = out[c].fillna(0).astype(int)
    return out.sort_values(["period", "sector"]).reset_index(drop=True)


def _sector_flows(changes: pd.DataFrame, active_keys: pd.DataFrame, cfg: SectorConfig) -> pd.DataFrame:
    """Net/gross flow and price effect per (period, sector) on the active
    managers' direct-equity book."""
    if changes.empty:
        return pd.DataFrame(columns=["period", "sector", "net_flow", "gross_flow", "price_effect", "n_buyers", "n_sellers"])
    ch = changes[(changes["underlying_asset"] == "Equity") & (~changes["sector"].str.startswith(cfg.etf_sector_prefix))]
    ch = ch.merge(active_keys, on=["period", "cik"], how="inner")
    return ch.groupby(["period", "sector"], as_index=False).agg(
        net_flow=("flow_effect", "sum"), gross_flow=("abs_flow", "sum"), price_effect=("price_effect", "sum"),
        n_buyers=("action", lambda s: int(s.isin(["NEW", "ADD"]).sum())), n_sellers=("action", lambda s: int(s.isin(["EXIT", "TRIM"]).sum())),
    )
