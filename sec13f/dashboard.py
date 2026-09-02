"""Self-contained HTML dashboard (Plotly.js from cdnjs, data embedded as JSON)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from . import __version__

ASSET_ORDER = ["Equity", "Equity (ETF)", "Options", "Debt", "Preferred", "Fixed Income (ETF)", "Commodity (ETF)", "Warrants/Rights", "Other"]


def _records(df: pd.DataFrame, cols: list[str] | None = None) -> list[dict]:
    if df is None or df.empty:
        return []
    d = df[cols] if cols else df
    d = d.replace({np.nan: None})
    return d.to_dict(orient="records")


def build_dashboard(out_path: Path, *, h, changes, msum, fp, exp_asset_ew, exp_asset_vw, exp_sector_ew, rotation, cons, pc,
                    insights: list[dict], managers, source: str, n_filings: int, title: str = "13F Holdings Tracker") -> Path:
    periods = sorted(h["period"].unique())
    short = {m.cik: m.short for m in managers}

    # exposure by asset: merge EW + VW
    ea = exp_asset_vw[["period", "underlying_asset", "value_usd", "share", "d_share"]].merge(
        exp_asset_ew[["period", "underlying_asset", "avg_weight", "d_avg_weight"]], on=["period", "underlying_asset"], how="outer")
    asset_order = [a for a in ASSET_ORDER if a in set(ea["underlying_asset"])] + sorted(set(ea["underlying_asset"]) - set(ASSET_ORDER))

    # rotation heat: flow as % of group's book value
    rot = pd.DataFrame()
    if not rotation.empty:
        book = h.groupby(["period", "manager_type"], as_index=False)["value_usd"].sum().rename(columns={"value_usd": "book"})
        rot = rotation.merge(book, on=["period", "manager_type"], how="left")
        rot["flow_pct"] = rot["net_flow"] / rot["book"]

    # managers table
    mg = msum.merge(fp[["cik", "period", "top10_weight", "hhi", "effective_n", "options_share", "put_share", "etf_share", "credit_share",
                        "inferred_type", "inferred_conf", "inferred_reason"]], on=["cik", "period"], how="left")
    mg["short"] = mg["cik"].map(short).fillna(mg["manager"])
    for c in ("NEW", "EXIT", "ADD", "TRIM", "HOLD", "net_flow", "gross_flow", "price_effect", "turnover", "value_chg_pct"):
        if c not in mg:
            mg[c] = np.nan

    # moves: top 60 by |flow| per period per action, plus per-manager top 15 (for the detail view)
    mv = pd.DataFrame()
    if not changes.empty:
        ch = changes.copy()
        ch["short"] = ch["cik"].map(short).fillna(ch["manager"])
        top_global = ch.sort_values("abs_flow", ascending=False).groupby(["period", "action"]).head(60)
        top_mgr = ch.sort_values("abs_flow", ascending=False).groupby(["period", "cik"]).head(15)
        mv = pd.concat([top_global, top_mgr]).drop_duplicates(subset=["period", "cik", "cusip", "put_call"])
        mv = mv[mv["action"] != "HOLD"]
        mv = mv.drop(columns=["issuer"]).rename(columns={"display_name": "issuer"})
        mv = mv[["period", "cik", "short", "manager", "action", "issuer", "ticker", "put_call", "asset_type", "sector", "d_shares", "pct_shares",
                 "weight_prev", "weight_cur", "flow_effect", "price_effect", "value_usd_prev", "value_usd_cur"]]

    # holdings for the detail view: top 15 per manager-period
    hd = h.sort_values("weight", ascending=False).groupby(["period", "cik"]).head(15)
    hd = hd.drop(columns=["issuer"]).rename(columns={"display_name": "issuer"})
    hd = hd[["period", "cik", "issuer", "ticker", "put_call", "asset_type", "sector", "value_usd", "weight"]]
    ms = h.groupby(["period", "cik", "sector"], as_index=False)["weight"].sum()

    kpis = {}
    for ins in insights:
        f = ins["facts"]
        t = msum[msum["period"] == ins["period"]]
        kpis[ins["period"]] = dict(
            total_value=f.get("total_value"), total_value_chg=f.get("total_value_chg"), net_flow=f.get("net_flow"), gross_flow=f.get("gross_flow"),
            price_effect=f.get("price_effect"), n_positions=f.get("n_positions"), n_new=f.get("n_new"), n_exit=f.get("n_exit"),
            n_add=f.get("n_add"), n_trim=f.get("n_trim"), avg_turnover=(float(t["turnover"].mean()) if "turnover" in t and t["turnover"].notna().any() else None),
        )

    pcr = pd.DataFrame()
    if not pc.empty:
        pcr = pc.rename(columns={"Put Option": "put", "Call Option": "call"})[["period", "issuer", "ticker", "put", "call"]]

    data = dict(
        periods=periods, asset_order=asset_order,
        insights={i["period"]: {k: v for k, v in i.items() if k != "facts"} for i in insights},
        kpis=kpis,
        exposure_asset=_records(ea), exposure_sector=_records(exp_sector_ew), rotation=_records(rot, ["period", "manager_type", "sector", "net_flow", "flow_pct"]),
        managers=_records(mg), moves=_records(mv), consensus=_records(cons.groupby("period").head(80)), putcall=_records(pcr),
        holdings=_records(hd), mgr_sector=_records(ms),
    )
    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"), autoescape=False)
    html = env.get_template("dashboard.html.j2").render(
        title=title, source=source, n_managers=h["cik"].nunique(), n_filings=n_filings, n_rows=len(h),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"), version=__version__,
        data_json=json.dumps(data, default=_json_default).replace("</", "<\\/"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)
