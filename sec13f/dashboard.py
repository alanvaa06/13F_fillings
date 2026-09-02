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
                    insights: list[dict], managers, source: str, n_filings: int, title: str = "13F Holdings Tracker", detail_quarters: int = 12,
                    exp_sector_vw: pd.DataFrame | None = None, company_moves: pd.DataFrame | None = None, company_rows: int = 400) -> Path:
    periods = sorted(h["period"].unique())
    # position-level detail exists for EVERY quarter; it is written as one JSON file per quarter (loaded on demand by
    # the page) and the most recent `detail_quarters` are also embedded inline so the file works without a server
    detail_periods = periods
    inline_periods = periods[-max(1, detail_quarters):]
    short = {m.cik: m.short for m in managers}

    # exposure by asset: merge EW + VW
    ea = exp_asset_vw[["period", "underlying_asset", "value_usd", "share", "d_share"]].merge(
        exp_asset_ew[["period", "underlying_asset", "avg_weight", "d_avg_weight"]], on=["period", "underlying_asset"], how="outer")
    asset_order = [a for a in ASSET_ORDER if a in set(ea["underlying_asset"])] + sorted(set(ea["underlying_asset"]) - set(ASSET_ORDER))

    # exposure by sector: EW (average of manager weights) + VW (share of aggregate dollars) so the dashboard can switch between them
    es = exp_sector_ew[["period", "sector", "avg_weight", "d_avg_weight"]]
    if exp_sector_vw is not None and not exp_sector_vw.empty:
        es = es.merge(exp_sector_vw[["period", "sector", "value_usd", "share", "d_share"]], on=["period", "sector"], how="outer")

    # rotation heat: flow as % of group's book value
    rot = pd.DataFrame()
    if not rotation.empty:
        book = h.groupby(["period", "manager_type"], as_index=False)["value_usd"].sum().rename(columns={"value_usd": "book"})
        rot = rotation.merge(book, on=["period", "manager_type"], how="left")
        rot["flow_pct"] = rot["net_flow"] / rot["book"]

    # managers table
    mg = msum.merge(fp[["cik", "period", "top10_weight", "options_share", "etf_share", "inferred_type", "inferred_reason"]], on=["cik", "period"], how="left")
    mg["short"] = mg["cik"].map(short).fillna(mg["manager"])
    for c in ("NEW", "EXIT", "ADD", "TRIM", "net_flow", "gross_flow", "turnover", "value_chg_pct"):
        if c not in mg:
            mg[c] = np.nan
    mg = mg[["cik", "short", "manager", "manager_type", "period", "total_value", "n_positions", "NEW", "EXIT", "ADD", "TRIM", "net_flow", "gross_flow",
             "turnover", "value_chg_pct", "top10_weight", "options_share", "etf_share", "inferred_type", "inferred_reason"]]

    # moves: top 60 by |flow| per period per action, plus per-manager top 15 (for the detail view)
    mv = pd.DataFrame()
    if not changes.empty:
        ch = changes.copy()
        ch["short"] = ch["cik"].map(short).fillna(ch["manager"])
        ch = ch[ch["period"].isin(detail_periods)]
        top_global = ch.sort_values("abs_flow", ascending=False).groupby(["period", "action"]).head(40)
        top_mgr = ch.sort_values("abs_flow", ascending=False).groupby(["period", "cik"]).head(10)
        mv = pd.concat([top_global, top_mgr]).drop_duplicates(subset=["period", "cik", "cusip", "put_call"])
        mv = mv[mv["action"] != "HOLD"]
        mv = mv.drop(columns=["issuer"]).rename(columns={"display_name": "issuer"})
        mv = mv[["period", "cik", "short", "action", "issuer", "ticker", "put_call", "asset_type", "sector", "pct_shares",
                 "weight_prev", "weight_cur", "flow_effect"]]

    # holdings for the detail view: top 15 per manager-period
    hdet = h[h["period"].isin(detail_periods)]
    hd = hdet.sort_values("weight", ascending=False).groupby(["period", "cik"]).head(15)
    hd = hd.drop(columns=["issuer"]).rename(columns={"display_name": "issuer"})
    if not changes.empty:  # previous-quarter weight of each top position, so the detail view can chart the change
        hd = hd.merge(changes[["period", "cik", "cusip", "put_call", "weight_prev"]], on=["period", "cik", "cusip", "put_call"], how="left")
    else:
        hd["weight_prev"] = np.nan
    hd = hd[["period", "cik", "issuer", "ticker", "put_call", "asset_type", "sector", "value_usd", "weight", "weight_prev"]]
    ms = hdet.groupby(["period", "cik", "sector"], as_index=False)["weight"].sum()
    ms = ms[ms["weight"] > 0.0005]

    # long-run history, one row per period
    hist = msum.groupby("period", as_index=False).agg(total_value=("total_value", "sum"), n_managers=("cik", "nunique"), n_positions=("n_positions", "sum"))
    if "net_flow" in msum:
        hist = hist.merge(msum.groupby("period", as_index=False).agg(net_flow=("net_flow", "sum"), gross_flow=("gross_flow", "sum"), avg_turnover=("turnover", "mean")), on="period", how="left")
    hist = hist.merge(fp.groupby("period", as_index=False).agg(avg_top10=("top10_weight", "mean"), avg_options=("options_share", "mean"), avg_etf=("etf_share", "mean")), on="period", how="left")

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

    cons_det = cons[cons["period"].isin(detail_periods)]
    pcr_det = pcr[pcr["period"].isin(detail_periods)] if not pcr.empty else pcr
    # aggregated moves per company (universe-wide): buyers/sellers, aggregate share change, flow decomposition and MAJOR/MINOR/NONE label.
    # The real universe has ~7,000 issuers per quarter; embed only the rows any view can show (top by gross flow, by
    # intensity and by breadth) and ship full-universe totals separately for the KPIs.
    cm, cm_totals = pd.DataFrame(), {}
    if company_moves is not None and not company_moves.empty:
        full = company_moves[company_moves["period"].isin(detail_periods)]
        for p, grp in full.groupby("period"):
            cm_totals[p] = dict(
                total=int(len(grp)), major=int((grp["magnitude"] == "MAJOR").sum()), minor=int((grp["magnitude"] == "MINOR").sum()), none=int((grp["magnitude"] == "NONE").sum()),
                n_bought=int((grp["net_flow"] > 0).sum()), n_sold=int((grp["net_flow"] < 0).sum()),
                buy_flow=float(grp.loc[grp["net_flow"] > 0, "net_flow"].sum()), sell_flow=float(grp.loc[grp["net_flow"] < 0, "net_flow"].sum()),
                entries=int(((grp["holders_prev"] == 0) & (grp["holders_cur"] > 0)).sum()), exits=int(((grp["holders_prev"] > 0) & (grp["holders_cur"] == 0)).sum()),
            )
        keep = []
        for _, grp in full.groupby("period"):
            elig = grp[grp["eligible_intensity"] & grp["pct_shares"].notna()]
            keep += [grp.nlargest(company_rows, "gross_flow"),
                     elig.reindex(elig["pct_shares"].abs().sort_values(ascending=False).index).head(company_rows // 3),
                     grp.reindex(grp["net_buyers"].abs().sort_values(ascending=False).index).head(company_rows // 3)]
        cm = pd.concat(keep).drop_duplicates(subset=["period", "cusip"])[[
            "period", "cusip", "issuer", "ticker", "sector", "holders_prev", "holders_cur", "buyers", "sellers", "new_holders", "exits",
            "value_prev", "value_cur", "pct_shares", "net_flow", "gross_flow", "price_effect", "net_buyers", "magnitude", "eligible_intensity"]]
    # options: the dashboard charts the top 15 underlyings per quarter; keep a margin, not the whole option universe
    if not pcr_det.empty:
        pcr_det = pcr_det.assign(_tot=pcr_det["put"] + pcr_det["call"]).sort_values("_tot", ascending=False).groupby("period").head(60).drop(columns="_tot")
    # per-quarter detail payloads
    detail: dict[str, dict] = {}
    frames = dict(moves=mv, consensus=cons_det.groupby("period").head(60) if not cons_det.empty else cons_det, putcall=pcr_det, holdings=hd, mgr_sector=ms, companies=cm)
    for p in periods:
        payload = {k: _records(df[df["period"] == p]) if (df is not None and not df.empty) else [] for k, df in frames.items()}
        payload["companies_totals"] = cm_totals.get(p)
        detail[p] = _round(payload)
    data_dir = out_path.parent / "dashboard_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for stale in data_dir.glob("*.json"):
        if stale.stem not in detail:
            stale.unlink()
    for p, payload in detail.items():
        (data_dir / f"{p}.json").write_text(json.dumps(payload, default=_json_default, separators=(",", ":")), encoding="utf-8")
    data = dict(
        periods=periods, detail_periods=detail_periods, inline_periods=inline_periods, detail_url="dashboard_data/{period}.json",
        asset_order=asset_order, history=_records(hist),
        insights={i["period"]: {k: v for k, v in i.items() if k != "facts"} for i in insights},
        kpis=kpis,
        exposure_asset=_records(ea), exposure_sector=_records(es), rotation=_records(rot, ["period", "manager_type", "sector", "net_flow", "flow_pct"]),
        managers=_records(mg), detail={p: detail[p] for p in inline_periods},
    )
    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"), autoescape=False)
    html = env.get_template("dashboard.html.j2").render(
        title=title, source=source, n_managers=h["cik"].nunique(), n_filings=n_filings, n_rows=len(h),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"), version=__version__,
        data_json=json.dumps(_round(data), default=_json_default, separators=(",", ":")).replace("</", "<\\/"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _round(o):
    if isinstance(o, float):
        return round(o, 6)
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v) for v in o]
    return o


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)
