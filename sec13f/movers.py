"""Company-level moves: how much did the universe as a whole move in each name.

Aggregates the per-manager position diff (``tracker.position_changes``) per
security and period, measures the move in a price-independent way (change in
the aggregate number of shares held by the universe) and in dollars (flow),
and classifies it as MAJOR / MINOR / NONE with explicit thresholds.

Only direct equity is considered (common stock, ADRs, REITs): options are
excluded (their notional would double count the underlying), and so are ETFs,
preferreds, debt and warrants (not "the company" in the sense analysts use).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pandas as pd


class MoveMagnitude(Enum):
    MAJOR = auto()
    MINOR = auto()
    NONE = auto()


@dataclass(frozen=True)
class MoverThresholds:
    """Rules that separate major from minor moves.

    A move is MAJOR when the universe changed its aggregate share count by at
    least ``major_pct_shares``, or when at least ``major_breadth`` more
    managers moved in one direction than in the other *and* the aggregate
    share count moved at least ``major_breadth_pct_shares``. A move is MINOR
    when the share count moved at least ``minor_pct_shares``. Below that the
    universe is considered to have held the name. A pure entry (the universe
    held nothing before) counts as a 100% move.
    """

    major_pct_shares: float = 0.10
    major_breadth: int = 3
    major_breadth_pct_shares: float = 0.03
    minor_pct_shares: float = 0.01
    min_holders_for_intensity: int = 3  # names with fewer holders are too noisy for the % ranking

    def __post_init__(self) -> None:
        if not 0 < self.minor_pct_shares <= self.major_breadth_pct_shares <= self.major_pct_shares:
            raise ValueError("thresholds must satisfy 0 < minor <= breadth <= major")
        if self.major_breadth < 1 or self.min_holders_for_intensity < 1:
            raise ValueError("breadth and holder thresholds must be >= 1")


def classify_move(pct_shares: float, net_buyers: int, t: MoverThresholds = MoverThresholds()) -> MoveMagnitude:
    """Classify one company move. ``pct_shares`` is the aggregate share-count
    change (NaN for a pure entry, which counts as a 100% move); ``net_buyers``
    is buyers minus sellers among the managers that hold the name."""
    p = 1.0 if np.isnan(pct_shares) else abs(pct_shares)
    if p >= t.major_pct_shares or (abs(net_buyers) >= t.major_breadth and p >= t.major_breadth_pct_shares):
        return MoveMagnitude.MAJOR
    if p >= t.minor_pct_shares:
        return MoveMagnitude.MINOR
    return MoveMagnitude.NONE


def company_moves(changes: pd.DataFrame, t: MoverThresholds = MoverThresholds()) -> pd.DataFrame:
    """One row per (period, cusip) with aggregate holders, direction counts,
    share-count change, flow decomposition and the MAJOR/MINOR/NONE label."""
    if changes.empty:
        return pd.DataFrame()
    ch = changes[(changes["put_call"] == "") & (changes["underlying_asset"] == "Equity") & (~changes["sector"].str.startswith("ETF"))]
    if ch.empty:
        return pd.DataFrame()
    c = ch.groupby(["period", "cusip"], as_index=False).agg(
        issuer=("display_name", "first"), ticker=("ticker", "first"), sector=("sector", "first"), asset_type=("asset_type", "first"),
        holders_prev=("shares_prev", lambda s: int((s > 0).sum())), holders_cur=("shares_cur", lambda s: int((s > 0).sum())),
        buyers=("action", lambda s: int(s.isin(["NEW", "ADD"]).sum())), sellers=("action", lambda s: int(s.isin(["EXIT", "TRIM"]).sum())),
        new_holders=("action", lambda s: int((s == "NEW").sum())), exits=("action", lambda s: int((s == "EXIT").sum())),
        value_prev=("value_usd_prev", "sum"), value_cur=("value_usd_cur", "sum"),
        shares_prev=("shares_prev", "sum"), shares_cur=("shares_cur", "sum"),
        net_flow=("flow_effect", "sum"), gross_flow=("abs_flow", "sum"), price_effect=("price_effect", "sum"),
    )
    c["net_buyers"] = c["buyers"] - c["sellers"]
    c["n_active"] = c["buyers"] + c["sellers"]
    c["d_holders"] = c["holders_cur"] - c["holders_prev"]
    c["d_shares"] = c["shares_cur"] - c["shares_prev"]
    c["pct_shares"] = c["d_shares"].div(c["shares_prev"].where(c["shares_prev"] > 0))
    c["d_value"] = c["value_cur"] - c["value_prev"]
    c["flow_pct_value"] = c["net_flow"].div(c["value_prev"].where(c["value_prev"] > 0))
    c["magnitude"] = [classify_move(p, nb, t).name for p, nb in zip(c["pct_shares"], c["net_buyers"])]
    c["intensity"] = c["pct_shares"].abs().fillna(1.0)
    c["eligible_intensity"] = np.maximum(c["holders_prev"], c["holders_cur"]) >= t.min_holders_for_intensity
    return c.sort_values(["period", "gross_flow"], ascending=[False, False]).reset_index(drop=True)


def magnitude_summary(moves: pd.DataFrame, period: str) -> pd.DataFrame:
    """Count of companies and flow per magnitude bucket for one period (every bucket present, zeros included)."""
    cols = ["magnitude", "companies", "net_flow", "gross_flow"]
    m = moves[moves["period"] == period] if not moves.empty else pd.DataFrame()
    if m.empty:
        return pd.DataFrame(columns=cols)
    s = m.groupby("magnitude").agg(companies=("cusip", "size"), net_flow=("net_flow", "sum"), gross_flow=("gross_flow", "sum"))
    s = s.reindex([k.name for k in MoveMagnitude], fill_value=0).reset_index().rename(columns={"index": "magnitude"})
    return s[cols]
