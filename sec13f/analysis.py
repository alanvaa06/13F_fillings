"""Turn tracker tables into a structured, human-readable quarterly read."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _usd(x: float) -> str:
    """Compact dollar amount with the sign in front of the currency symbol (-$1.2B)."""
    a, sign = abs(x), ("-" if x < 0 else "")
    if a >= 1e12:
        return f"{sign}${a / 1e12:,.2f}T"
    if a >= 1e9:
        return f"{sign}${a / 1e9:,.1f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:,.0f}M"
    return f"{sign}${a:,.0f}"


def _n(count: int, singular: str, plural: str) -> str:
    """'1 comprador' / '3 compradores'."""
    return f"{count} {singular if count == 1 else plural}"


_ACRONYMS = {"ETF", "ADR", "ADS", "MSCI", "EAFE", "S&P", "SPDR", "REIT", "PFD", "NV", "SA", "AG", "PLC", "LLC", "LP", "USA", "US", "FTSE", "QQQ", "SPD", "AT&T", "IBM", "HCA", "CVS", "GE", "RTX", "KKR", "PNC", "TJX", "UPS", "MGM", "PDD", "JD", "ASML", "SAP", "TSM", "NVR"}


def pretty(name: str) -> str:
    """Title-case an EDGAR issuer name, keeping acronyms and the (TICKER) suffix upper-case."""
    out = []
    for w in (name or "").split():
        if w.startswith("(") and w.endswith(")"):
            out.append(w.upper())
        elif w.upper() in _ACRONYMS:
            out.append(w.upper())
        else:
            out.append(w[:1].upper() + w[1:].lower())
    return " ".join(out)


def _pp(x: float) -> str:
    v = round(x * 100, 1)
    if v == 0:
        return "0.0 pp"
    return f"{v:+.1f} pp"


def quarter_label(period: str) -> str:
    y, m = period[:4], int(period[5:7])
    return f"{y}Q{(m - 1) // 3 + 1}"


def build_insights(h: pd.DataFrame, changes: pd.DataFrame, msum: pd.DataFrame, fp: pd.DataFrame,
                   exp_asset_ew: pd.DataFrame, exp_sector_ew: pd.DataFrame, cons: pd.DataFrame,
                   pc: pd.DataFrame, period: str, *, sector_pos: Optional[pd.DataFrame] = None,
                   moves: Optional[pd.DataFrame] = None) -> dict:
    """Return {'period', 'headline', 'bullets': [...], 'facts': {...}} for one period."""
    sector_pos = sector_pos if sector_pos is not None else pd.DataFrame()
    moves = moves if moves is not None else pd.DataFrame()
    prev_periods = sorted(p for p in h["period"].unique() if p < period)
    prev = prev_periods[-1] if prev_periods else None
    cur = h[h["period"] == period]
    bullets: list[dict] = []
    facts: dict = {}

    # --- universe
    n_mgr = cur["cik"].nunique()
    tot = cur["value_usd"].sum()
    facts.update(n_managers=n_mgr, total_value=tot, n_positions=len(cur))
    if prev:
        ptot = h.loc[h["period"] == prev, "value_usd"].sum()
        facts["total_value_chg"] = tot / ptot - 1 if ptot else np.nan

    # --- flows
    ch = changes[changes["period"] == period] if not changes.empty else pd.DataFrame()
    if not ch.empty:
        net = ch["flow_effect"].sum()
        gross = ch["abs_flow"].sum()
        px = ch["price_effect"].sum()
        facts.update(net_flow=net, gross_flow=gross, price_effect=px,
                     n_new=int((ch["action"] == "NEW").sum()), n_exit=int((ch["action"] == "EXIT").sum()),
                     n_add=int((ch["action"] == "ADD").sum()), n_trim=int((ch["action"] == "TRIM").sum()))
        bullets.append(dict(
            kind="flows",
            text=(f"El valor agregado del universo cambió {_usd(net + px)} en el trimestre: {_usd(px)} por precio y "
                  f"{_usd(net)} por flujo neto (compras menos ventas). Actividad bruta de {_usd(gross)} con "
                  f"{facts['n_new']} posiciones nuevas y {facts['n_exit']} salidas."),
        ))

    # --- asset mix (equal-weighted)
    ea = exp_asset_ew[exp_asset_ew["period"] == period].dropna(subset=["d_avg_weight"])
    if not ea.empty:
        up = ea.loc[ea["d_avg_weight"].idxmax()]
        dn = ea.loc[ea["d_avg_weight"].idxmin()]
        bullets.append(dict(
            kind="asset_mix",
            text=(f"Mezcla de activos (promedio simple entre managers): sube {up['underlying_asset']} ({_pp(up['d_avg_weight'])}) "
                  f"y baja {dn['underlying_asset']} ({_pp(dn['d_avg_weight'])})."),
        ))

    # --- sector rotation (equal-weighted)
    es = exp_sector_ew[(exp_sector_ew["period"] == period)].dropna(subset=["d_avg_weight"])
    es = es[~es["sector"].str.startswith("ETF")]
    if not es.empty:
        top = es.nlargest(2, "d_avg_weight")
        bot = es.nsmallest(2, "d_avg_weight")
        bullets.append(dict(
            kind="sector",
            text=("Rotación sectorial: mayor incremento de peso en "
                  + " y ".join(f"{r.sector} ({_pp(r.d_avg_weight)})" for r in top.itertuples())
                  + "; mayor reducción en "
                  + " y ".join(f"{r.sector} ({_pp(r.d_avg_weight)})" for r in bot.itertuples()) + "."),
        ))

    # --- sector positioning vs benchmark (direct equity, active managers)
    sp = sector_pos[sector_pos["period"] == period] if not sector_pos.empty else pd.DataFrame()
    if not sp.empty:
        ow = sp.loc[sp["active_weight"].idxmax()]
        uw = sp.loc[sp["active_weight"].idxmin()]
        text = (f"Posicionamiento sectorial de los managers activos en acciones directas, frente al benchmark {sp['benchmark_name'].iloc[0]}: "
                f"mayor sobreponderación en {ow['sector']} ({_pp(ow['active_weight'])}; {ow['overweight_breadth']:.0%} de los managers por encima del benchmark) "
                f"y mayor infraponderación en {uw['sector']} ({_pp(uw['active_weight'])}; {uw['overweight_breadth']:.0%} por encima).")
        da = sp.dropna(subset=["d_active_qoq"])
        if not da.empty:
            mv = da.loc[da["d_active_qoq"].abs().idxmax()]
            text += f" Mayor cambio de peso activo en el trimestre: {mv['sector']} ({_pp(mv['d_active_qoq'])})."
        bullets.append(dict(kind="sector_positioning", text=text))
        facts.update(sector_top_overweight=dict(sector=ow["sector"], active_weight=float(ow["active_weight"]), breadth=float(ow["overweight_breadth"])),
                     sector_top_underweight=dict(sector=uw["sector"], active_weight=float(uw["active_weight"]), breadth=float(uw["overweight_breadth"])),
                     sector_benchmark=str(sp["benchmark_name"].iloc[0]))
        dq = sp.dropna(subset=["d_qoq"])
        if not dq.empty and "net_flow" in sp:
            top = dq.loc[dq["d_qoq"].idxmax()]
            hist = ""
            if not pd.isna(top.get("hist_percentile", np.nan)):
                hist = f"; su peso actual está en el percentil {top['hist_percentile'] * 100:.0f} de su historia"
            bullets.append(dict(kind="sector_driver", text=(
                f"El sector que más peso ganó, {top['sector']} ({_pp(top['d_qoq'])} QoQ"
                + (f", {_pp(top['d_yoy'])} YoY" if not pd.isna(top.get("d_yoy", np.nan)) else "")
                + f"), recibió flujo neto de {_usd(top['net_flow'])} frente a un efecto precio de {_usd(top['price_effect'])}"
                + f" ({_n(int(top['n_buyers']), 'compra', 'compras')} / {_n(int(top['n_sellers']), 'venta', 'ventas')}){hist}.")))

        # historical extremes: sectors at the top / bottom of their own history
        hp = sp.dropna(subset=["hist_percentile"])
        if not hp.empty:
            highs = hp[hp["hist_percentile"] >= 0.95].sort_values("weight_ew", ascending=False)
            lows = hp[hp["hist_percentile"] <= 0.05].sort_values("weight_ew", ascending=False)
            parts = []
            if not highs.empty:
                parts.append("en máximos de su historia: " + ", ".join(f"{r.sector} ({r.weight_ew:.1%})" for r in highs.itertuples()))
            if not lows.empty:
                parts.append("en mínimos: " + ", ".join(f"{r.sector} ({r.weight_ew:.1%})" for r in lows.itertuples()))
            if parts:
                bullets.append(dict(kind="sector_extremes", text="Extremos históricos del peso sectorial (percentil dentro de la historia disponible): " + "; ".join(parts) + "."))
                facts["sector_extremes"] = dict(highs=highs["sector"].tolist(), lows=lows["sector"].tolist())

    # --- consensus buys / sells
    c = cons[cons["period"] == period] if not cons.empty else pd.DataFrame()
    if not c.empty and "net_buyers" in c:
        buys = c[c["net_buyers"] > 0].sort_values(["net_buyers", "net_flow"], ascending=False).head(3)
        sells = c[c["net_buyers"] < 0].sort_values(["net_buyers", "net_flow"], ascending=[True, True]).head(3)
        if not buys.empty:
            bullets.append(dict(kind="consensus_buy", text="Compras de consenso (más compradores netos): "
                                + ", ".join(f"{pretty(r.issuer)} ({_n(r.buyers, 'comprador', 'compradores')} / {_n(r.sellers, 'vendedor', 'vendedores')}, flujo {_usd(r.net_flow)})" for r in buys.itertuples()) + "."))
        if not sells.empty:
            bullets.append(dict(kind="consensus_sell", text="Ventas de consenso: "
                                + ", ".join(f"{pretty(r.issuer)} ({_n(r.sellers, 'vendedor', 'vendedores')} / {_n(r.buyers, 'comprador', 'compradores')}, flujo {_usd(r.net_flow)})" for r in sells.itertuples()) + "."))
        crowded = c.nlargest(3, "holders")
        facts["most_held"] = [dict(issuer=r.issuer, holders=int(r.holders), value=float(r.total_value)) for r in crowded.itertuples()]

    # --- biggest single moves
    if not ch.empty:
        big = ch.reindex(ch["abs_flow"].sort_values(ascending=False).index).head(4)
        parts = []
        for r in big.itertuples():
            verb = {"NEW": "abre", "EXIT": "liquida", "ADD": "aumenta", "TRIM": "reduce"}.get(r.action, "mantiene")
            parts.append(f"{r.manager} {verb} {pretty(r.display_name)} ({_usd(r.flow_effect)})")
        bullets.append(dict(kind="moves", text="Movimientos individuales más grandes: " + "; ".join(parts) + "."))

    # --- company-level moves (universe aggregate)
    mv = moves[moves["period"] == period] if not moves.empty else pd.DataFrame()
    if not mv.empty:
        counts = mv["magnitude"].value_counts()
        top = mv.reindex(mv["gross_flow"].sort_values(ascending=False).index).head(3)
        parts = []
        for r in top.itertuples():
            size = "entrada nueva al universo" if pd.isna(r.pct_shares) else f"títulos agregados {r.pct_shares:+.0%}"
            parts.append(f"{pretty(r.issuer)} ({size}, {_n(r.buyers, 'comprador', 'compradores')} / {_n(r.sellers, 'vendedor', 'vendedores')}, flujo neto {_usd(r.net_flow)})")
        n_major, n_minor, n_none = int(counts.get("MAJOR", 0)), int(counts.get("MINOR", 0)), int(counts.get("NONE", 0))
        bullets.append(dict(kind="company_moves", text=(
            "Empresas con mayor movimiento agregado: " + "; ".join(parts)
            + f". De {len(mv)} empresas en cartera, {n_major} tuvieron un cambio mayor, {n_minor} menor y {n_none} ninguno.")))
        facts.update(companies_total=int(len(mv)), companies_major=n_major, companies_minor=n_minor, companies_none=n_none,
                     top_company_moves=[dict(issuer=r.issuer, ticker=r.ticker, pct_shares=(None if pd.isna(r.pct_shares) else float(r.pct_shares)),
                                             net_flow=float(r.net_flow), buyers=int(r.buyers), sellers=int(r.sellers), magnitude=r.magnitude) for r in top.itertuples()])

    # --- managers: turnover and type mismatch
    m = msum[msum["period"] == period]
    if not m.empty and "turnover" in m and m["turnover"].notna().any():
        hi = m.nlargest(3, "turnover")
        bullets.append(dict(kind="turnover", text="Mayor rotación de cartera: "
                            + ", ".join(f"{r.manager} ({r.turnover:.0%})" for r in hi.itertuples()) + "."))
    f = fp[fp["period"] == period]
    if not f.empty and "manager_type" in f:
        mism = f[[not _types_agree(a, b) for a, b in zip(f["manager_type"], f["inferred_type"])]]
        if not mism.empty:
            bullets.append(dict(kind="type_mismatch", text="Perfil de cartera distinto al tipo declarado: "
                                + "; ".join(f"{r.manager} (declarado {r.manager_type}, cartera parece {r.inferred_type}: {r.inferred_reason})" for r in mism.head(4).itertuples()) + "."))

    # --- options
    p = pc[pc["period"] == period] if not pc.empty else pd.DataFrame()
    if not p.empty:
        puts, calls = p["Put Option"].sum(), p["Call Option"].sum()
        top_put = p.nlargest(3, "Put Option")
        bullets.append(dict(kind="options", text=(f"Opciones reportadas: puts {_usd(puts)} vs calls {_usd(calls)} "
                            f"(ratio {puts / calls if calls else float('nan'):.2f}). Mayor cobertura con puts en "
                            + ", ".join(pretty(r.issuer) for r in top_put.itertuples()) + ".")))
        facts.update(put_notional=puts, call_notional=calls)

    headline = (f"{quarter_label(period)}: {n_mgr} managers, {_usd(tot)} en posiciones reportadas"
                + (f", {facts['total_value_chg']:+.1%} vs trimestre anterior" if prev else ""))
    return dict(period=period, label=quarter_label(period), headline=headline, bullets=bullets, facts=facts)


_TYPE_MAP = {
    "index": {"Index / Broad Asset Manager"},
    "quant": {"Quant / Systematic", "Index / Broad Asset Manager"},
    "multi": {"Multi-strategy / Options-heavy", "Quant / Systematic"},
    "activist": {"Concentrated / Activist", "Concentrated Value"},
    "macro": {"Macro / Allocator (ETF-heavy)", "Concentrated Value", "Long-only Stock Picker"},
    "value": {"Concentrated Value", "Long-only Stock Picker", "Credit / Special Situations"},
    "opportunistic": {"Concentrated Value", "Long-only Stock Picker", "Credit / Special Situations"},
    "growth": {"Concentrated Value", "Long-only Stock Picker"},
}


def _types_agree(declared: str, inferred: str) -> bool:
    d = (declared or "").lower()
    for key, ok in _TYPE_MAP.items():
        if key in d and inferred in ok:
            return True
    if "conglomerate" in d and inferred in _TYPE_MAP["value"]:
        return True
    if "family office" in d:
        return True  # family offices are heterogeneous by nature
    return False
