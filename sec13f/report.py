"""Markdown report (Spanish) generated from the tracker tables."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from .analysis import _pp, _usd, quarter_label


def _md_table(df: pd.DataFrame, cols: dict[str, str], fmts: dict[str, callable] | None = None) -> str:
    fmts = fmts or {}
    head = "| " + " | ".join(cols.values()) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for r in df.itertuples(index=False):
        d = r._asdict()
        cells = []
        for c in cols:
            v = d.get(c, "")
            f = fmts.get(c)
            cells.append(f(v) if f else ("" if pd.isna(v) else str(v)))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def build_report(insights: list[dict], msum: pd.DataFrame, fp: pd.DataFrame, exp_asset_ew: pd.DataFrame,
                 exp_sector_ew: pd.DataFrame, cons: pd.DataFrame, changes: pd.DataFrame, source: str) -> str:
    latest = insights[-1]
    period = latest["period"]
    out = [f"# 13F Tracker — Reporte trimestral {latest['label']}\n",
           f"_Generado {datetime.now():%Y-%m-%d %H:%M} · fuente de datos: **{source}** · periodo de reporte {period}_\n",
           "> " + latest["headline"] + "\n",
           "## Lectura del trimestre\n"]
    out += [f"- {b['text']}" for b in latest["bullets"]]

    out.append("\n## Managers en el universo\n")
    m = msum[msum["period"] == period].merge(
        fp[fp["period"] == period][["cik", "top10_weight", "options_share", "etf_share", "inferred_type"]], on="cik", how="left"
    ).sort_values("total_value", ascending=False)
    out.append(_md_table(m, {
        "manager": "Manager", "manager_type": "Tipo declarado", "inferred_type": "Tipo inferido", "total_value": "Valor 13F",
        "n_positions": "Posiciones", "top10_weight": "Top-10", "turnover": "Rotación", "net_flow": "Flujo neto", "value_chg_pct": "Δ valor",
    }, {"total_value": _usd, "top10_weight": lambda v: f"{v:.0%}", "turnover": lambda v: "" if pd.isna(v) else f"{v:.0%}",
        "net_flow": lambda v: "" if pd.isna(v) else _usd(v), "value_chg_pct": lambda v: "" if pd.isna(v) else f"{v:+.1%}"}))

    out.append("\n## Exposición por tipo de activo (promedio simple entre managers)\n")
    ea = exp_asset_ew[exp_asset_ew["period"] == period].sort_values("avg_weight", ascending=False)
    out.append(_md_table(ea, {"underlying_asset": "Tipo de activo", "avg_weight": "Peso promedio", "d_avg_weight": "Δ vs. trimestre previo"},
                         {"avg_weight": lambda v: f"{v:.1%}", "d_avg_weight": lambda v: "" if pd.isna(v) else _pp(v)}))

    out.append("\n## Exposición sectorial (promedio simple entre managers)\n")
    es = exp_sector_ew[exp_sector_ew["period"] == period].sort_values("avg_weight", ascending=False)
    out.append(_md_table(es, {"sector": "Sector", "avg_weight": "Peso promedio", "d_avg_weight": "Δ vs. trimestre previo"},
                         {"avg_weight": lambda v: f"{v:.1%}", "d_avg_weight": lambda v: "" if pd.isna(v) else _pp(v)}))

    c = cons[cons["period"] == period]
    if not c.empty:
        out.append("\n## Consenso: posiciones más compartidas\n")
        out.append(_md_table(c.nlargest(10, "holders"), {"issuer": "Emisor", "ticker": "Ticker", "sector": "Sector", "holders": "Tenedores", "total_value": "Valor agregado", "net_buyers": "Compradores netos"},
                             {"total_value": _usd}))
        if "net_buyers" in c:
            out.append("\n## Consenso: compras y ventas netas\n")
            out.append("**Más comprados (compradores − vendedores):**\n")
            out.append(_md_table(c.sort_values(["net_buyers", "net_flow"], ascending=False).head(8),
                                 {"issuer": "Emisor", "buyers": "Compradores", "sellers": "Vendedores", "new_holders": "Nuevos", "net_flow": "Flujo neto"}, {"net_flow": _usd}))
            out.append("\n**Más vendidos:**\n")
            out.append(_md_table(c.sort_values(["net_buyers", "net_flow"], ascending=[True, True]).head(8),
                                 {"issuer": "Emisor", "buyers": "Compradores", "sellers": "Vendedores", "exits": "Salidas", "net_flow": "Flujo neto"}, {"net_flow": _usd}))

    ch = changes[changes["period"] == period] if not changes.empty else pd.DataFrame()
    if not ch.empty:
        out.append("\n## Movimientos individuales más grandes (por flujo estimado)\n")
        big = ch.reindex(ch["abs_flow"].sort_values(ascending=False).index).head(20)
        out.append(_md_table(big, {"manager": "Manager", "action": "Acción", "display_name": "Emisor", "asset_type": "Tipo", "pct_shares": "Δ títulos", "flow_effect": "Flujo", "weight_prev": "Peso previo", "weight_cur": "Peso actual"},
                             {"pct_shares": lambda v: "" if pd.isna(v) else f"{v:+.0%}", "flow_effect": _usd, "weight_prev": lambda v: f"{v:.1%}", "weight_cur": lambda v: f"{v:.1%}"}))

    out.append("\n## Metodología\n")
    out.append("- **Fuente:** formularios 13F-HR (y enmiendas 13F-HR/A) descargados de SEC EDGAR vía la API de submissions y los archivos XML del information table. "
               "Los valores se normalizan a dólares (los filings anteriores al 3-ene-2023 reportan en miles).\n"
               "- **Clasificación de activos:** reglas sobre `titleOfClass`, `putCall` y `sshPrnamtType`, más un maestro de valores por CUSIP; la clasificación sectorial usa el maestro, palabras clave del emisor y un modelo Naive Bayes sobre el nombre del emisor cuando no hay coincidencia. Cada etiqueta lleva un score de confianza.\n"
               "- **Cambios:** el Δ de valor de cada posición se descompone en *efecto flujo* (Δ títulos × precio implícito del trimestre actual) y *efecto precio* (resto).\n"
               "- **Exposición equal-weight:** promedio simple de los pesos de cada manager, para que los filers muy grandes (índices) no dominen la lectura. La versión ponderada por valor también se reporta en el dashboard.\n"
               "- **Tipo de manager inferido:** huella de cartera (número de posiciones, concentración top-10/HHI, share de opciones/ETF/crédito, rotación).\n"
               "- **Limitaciones del 13F:** solo posiciones largas en valores de la sección 13(f) de EE.UU. (no cortos, no bonos soberanos, no derivados OTC, no posiciones internacionales sin ADR); rezago de hasta 45 días; las opciones se reportan por valor nocional del subyacente.\n")
    return "\n".join(out)
