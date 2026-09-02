"""Quarter-over-quarter tracker: position changes, exposure shifts, consensus."""
from __future__ import annotations

import numpy as np
import pandas as pd

POS_KEY = ["cik", "cusip", "put_call"]


def add_weights(h: pd.DataFrame) -> pd.DataFrame:
    h = h.copy()
    tot = h.groupby(["cik", "period"])["value_usd"].transform("sum")
    h["weight"] = np.where(tot > 0, h["value_usd"] / tot, 0.0)
    h["implied_price"] = np.where(h["shares"] > 0, h["value_usd"] / h["shares"], np.nan)
    return h


def position_changes(h: pd.DataFrame) -> pd.DataFrame:
    """Diff each manager's book between consecutive reported periods.

    Value change is decomposed into a *flow* effect (shares bought/sold at
    the current implied price) and a *price* effect (old shares revalued).
    """
    periods = sorted(h["period"].unique())
    frames = []
    for prev, cur in zip(periods[:-1], periods[1:]):
        a = h[h["period"] == prev].set_index(POS_KEY)
        b = h[h["period"] == cur].set_index(POS_KEY)
        # only managers that reported in both periods
        common_ciks = set(a.index.get_level_values(0)) & set(b.index.get_level_values(0))
        a = a[a.index.get_level_values(0).isin(common_ciks)]
        b = b[b.index.get_level_values(0).isin(common_ciks)]
        label_cols = [c for c in ("manager", "manager_type", "manager_short", "issuer", "display_name", "title_of_class", "ticker", "asset_type",
                                  "underlying_asset", "sector", "industry") if c in h.columns]
        cols = label_cols + ["value_usd", "shares", "weight", "implied_price"]
        j = a[cols].join(b[cols], how="outer", lsuffix="_prev", rsuffix="_cur")
        for c in label_cols:
            j[c] = j[f"{c}_cur"].fillna(j[f"{c}_prev"])
            j.drop(columns=[f"{c}_prev", f"{c}_cur"], inplace=True)
        for c in ("value_usd", "shares", "weight"):
            j[f"{c}_prev"] = j[f"{c}_prev"].fillna(0.0)
            j[f"{c}_cur"] = j[f"{c}_cur"].fillna(0.0)
        j["period_prev"], j["period"] = prev, cur
        j["d_shares"] = j["shares_cur"] - j["shares_prev"]
        j["d_value"] = j["value_usd_cur"] - j["value_usd_prev"]
        j["d_weight"] = j["weight_cur"] - j["weight_prev"]
        j["pct_shares"] = np.where(j["shares_prev"] > 0, j["d_shares"] / j["shares_prev"], np.nan)
        px = j["implied_price_cur"].fillna(j["implied_price_prev"])
        j["flow_effect"] = j["d_shares"] * px
        j["price_effect"] = j["d_value"] - j["flow_effect"]
        conds = [
            (j["shares_prev"] == 0) & (j["shares_cur"] > 0),
            (j["shares_prev"] > 0) & (j["shares_cur"] == 0),
            j["d_shares"] > 0,
            j["d_shares"] < 0,
        ]
        j["action"] = np.select(conds, ["NEW", "EXIT", "ADD", "TRIM"], default="HOLD")
        frames.append(j.reset_index())
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["abs_flow"] = out["flow_effect"].abs()
    return out


def manager_turnover(changes: pd.DataFrame) -> pd.Series:
    """Fraction of book traded: sum |flow| / average book value, per (cik, period)."""
    if changes.empty:
        return pd.Series(dtype=float)
    g = changes.groupby(["cik", "period"])
    flow = g["abs_flow"].sum()
    avg = (g["value_usd_prev"].sum() + g["value_usd_cur"].sum()) / 2
    return (flow / avg.replace(0, np.nan)).fillna(0.0)


def manager_summary(h: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    base = (
        h.groupby(["cik", "manager", "manager_type", "period"], as_index=False)
        .agg(total_value=("value_usd", "sum"), n_positions=("cusip", "size"))
    )
    if changes.empty:
        return base
    act = changes.pivot_table(index=["cik", "period"], columns="action", values="cusip", aggfunc="count", fill_value=0).reset_index()
    flow = changes.groupby(["cik", "period"], as_index=False).agg(net_flow=("flow_effect", "sum"), gross_flow=("abs_flow", "sum"), price_effect=("price_effect", "sum"))
    out = base.merge(act, on=["cik", "period"], how="left").merge(flow, on=["cik", "period"], how="left")
    to = manager_turnover(changes).rename("turnover").reset_index()
    out = out.merge(to, on=["cik", "period"], how="left")
    out = out.sort_values(["cik", "period"])
    out["prev_value"] = out.groupby("cik")["total_value"].shift(1)
    out["value_chg_pct"] = out["total_value"] / out["prev_value"] - 1
    return out


def exposure(h: pd.DataFrame, by: str, group_col: str | None = None) -> pd.DataFrame:
    """Exposure (value and share of book) per period, by a classification
    column, optionally within a grouping (manager or manager_type)."""
    keys = ["period", by] + ([group_col] if group_col else [])
    e = h.groupby(keys, as_index=False)["value_usd"].sum()
    denom_keys = ["period"] + ([group_col] if group_col else [])
    e["share"] = e["value_usd"] / e.groupby(denom_keys)["value_usd"].transform("sum")
    e = e.sort_values(keys)
    shift_keys = [by] + ([group_col] if group_col else [])
    e["share_prev"] = e.groupby(shift_keys)["share"].shift(1)
    e["d_share"] = e["share"] - e["share_prev"]
    e["value_prev"] = e.groupby(shift_keys)["value_usd"].shift(1)
    e["d_value"] = e["value_usd"] - e["value_prev"]
    return e


def equal_weight_exposure(h: pd.DataFrame, by: str) -> pd.DataFrame:
    """Average of each manager's own allocation (so Vanguard-sized filers do
    not dominate the picture)."""
    # pivot with zeros so a manager that holds none of a category still counts in the average
    w = h.pivot_table(index=["period", "cik"], columns=by, values="weight", aggfunc="sum", fill_value=0.0)
    e = w.groupby(level="period").mean().stack().rename("avg_weight").reset_index()
    e.columns = ["period", by, "avg_weight"]
    e = e.sort_values(["period", by])
    e["avg_weight_prev"] = e.groupby(by)["avg_weight"].shift(1)
    e["d_avg_weight"] = e["avg_weight"] - e["avg_weight_prev"]
    return e


def consensus(h: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """Crowding & conviction per security and period (cash positions only; options excluded)."""
    long = h[~h["asset_type"].isin(["Put Option", "Call Option"])]
    c = long.groupby(["period", "cusip"], as_index=False).agg(
        issuer=("display_name", "first"), ticker=("ticker", "first"), sector=("sector", "first"), asset_type=("asset_type", "first"),
        holders=("cik", "nunique"), total_value=("value_usd", "sum"), avg_weight=("weight", "mean"), max_weight=("weight", "max")
    )
    if not changes.empty:
        ch = changes[changes["put_call"] == ""]
        agg = ch.groupby(["period", "cusip"], as_index=False).agg(
            buyers=("action", lambda s: int(s.isin(["NEW", "ADD"]).sum())),
            sellers=("action", lambda s: int(s.isin(["EXIT", "TRIM"]).sum())),
            new_holders=("action", lambda s: int((s == "NEW").sum())),
            exits=("action", lambda s: int((s == "EXIT").sum())),
            net_flow=("flow_effect", "sum"),
        )
        c = c.merge(agg, on=["period", "cusip"], how="left")
        for col in ("buyers", "sellers", "new_holders", "exits"):
            c[col] = c[col].fillna(0).astype(int)
        c["net_flow"] = c["net_flow"].fillna(0.0)
        c["net_buyers"] = c["buyers"] - c["sellers"]
    return c.sort_values(["period", "holders", "total_value"], ascending=[False, False, False])


def put_call_signal(h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate put vs call notional per period and per underlying."""
    o = h[h["asset_type"].isin(["Put Option", "Call Option"])]
    if o.empty:
        return pd.DataFrame(columns=["period", "issuer", "ticker", "Put Option", "Call Option", "put_call_ratio"])
    p = o.pivot_table(index=["period", "cusip"], columns="asset_type", values="value_usd", aggfunc="sum", fill_value=0).reset_index()
    names = o.groupby("cusip").agg(issuer=("display_name", "first"), ticker=("ticker", "first")).reset_index()
    p = p.merge(names, on="cusip", how="left")
    for c in ("Put Option", "Call Option"):
        if c not in p:
            p[c] = 0.0
    p["put_call_ratio"] = p["Put Option"] / (p["Call Option"].replace(0, np.nan))
    return p.sort_values(["period", "Put Option"], ascending=[False, False])


def sector_rotation(changes: pd.DataFrame, group_col: str = "manager_type") -> pd.DataFrame:
    """Net flow by sector and manager group, per period."""
    if changes.empty:
        return pd.DataFrame()
    return changes.groupby(["period", group_col, "sector"], as_index=False).agg(net_flow=("flow_effect", "sum"), gross_flow=("abs_flow", "sum"))
