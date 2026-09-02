"""Turn tracker tables into a structured, human-readable quarterly read."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _usd(x: float) -> str:
    a = abs(x)
    if a >= 1e12:
        return f"${x / 1e12:,.2f}T"
    if a >= 1e9:
        return f"${x / 1e9:,.1f}B"
    if a >= 1e6:
        return f"${x / 1e6:,.0f}M"
    return f"${x:,.0f}"


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
    return f"{x * 100:+.1f} pp"


def quarter_label(period: str) -> str:
    y, m = period[:4], int(period[5:7])
    return f"{y}Q{(m - 1) // 3 + 1}"


def build_insights(h: pd.DataFrame, changes: pd.DataFrame, msum: pd.DataFrame, fp: pd.DataFrame,
                   exp_asset_ew: pd.DataFrame, exp_sector_ew: pd.DataFrame, cons: pd.DataFrame,
                   pc: pd.DataFrame, period: str) -> dict:
    """Return {'period', 'headline', 'bullets': [...], 'facts': {...}} for one period."""
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

    # --- consensus buys / sells
    c = cons[cons["period"] == period] if not cons.empty else pd.DataFrame()
    if not c.empty and "net_buyers" in c:
        buys = c[c["net_buyers"] > 0].sort_values(["net_buyers", "net_flow"], ascending=False).head(3)
        sells = c[c["net_buyers"] < 0].sort_values(["net_buyers", "net_flow"], ascending=[True, True]).head(3)
        if not buys.empty:
            bullets.append(dict(kind="consensus_buy", text="Compras de consenso (más compradores netos): "
                                + ", ".join(f"{pretty(r.issuer)} ({r.buyers} compradores / {r.sellers} vendedores, flujo {_usd(r.net_flow)})" for r in buys.itertuples()) + "."))
        if not sells.empty:
            bullets.append(dict(kind="consensus_sell", text="Ventas de consenso: "
                                + ", ".join(f"{pretty(r.issuer)} ({r.sellers} vendedores / {r.buyers} compradores, flujo {_usd(r.net_flow)})" for r in sells.itertuples()) + "."))
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
