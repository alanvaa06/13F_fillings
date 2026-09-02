"""Markdown report (Spanish) generated from the tracker tables."""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

import pandas as pd

from .analysis import _pp, _usd, pretty, quarter_label
from .movers import MoverThresholds, magnitude_summary
from .sectors import Benchmark, SectorConfig

_MAGNITUDE_ES = {"MAJOR": "Mayor", "MINOR": "Menor", "NONE": "Sin cambio"}


def _pct(v: float, digits: int = 1) -> str:
    return "" if pd.isna(v) else f"{v:.{digits}%}"


def _pp_or_blank(v: float) -> str:
    return "" if pd.isna(v) else _pp(v)


def _usd_or_blank(v: float) -> str:
    return "" if pd.isna(v) else _usd(v)


def _percentile(v: float) -> str:
    """0-100 integer percentile (blank when history is too short)."""
    return "" if pd.isna(v) else f"{v * 100:.0f}"


def _sector_sections(sp: pd.DataFrame, period: str, cfg: SectorConfig, benchmarks: list[Benchmark]) -> list[str]:
    s = sp[sp["period"] == period]
    if s.empty:
        return []
    bname = s["benchmark_name"].iloc[0]
    n_act, n_idx = int(s["n_active_managers"].iloc[0]), int(s["n_index_managers"].iloc[0])
    out = ["\n## Exposición sectorial (renta variable directa)\n",
           (f"_Pesos sobre el libro de **acciones directas** (sin ETFs, opciones, deuda ni preferentes). **EW** = promedio simple de los "
            f"{n_act} managers activos (se excluyen los managers índice). Benchmark: **{bname}**"
            + (f", construido con los {n_idx} managers índice del universo" if n_idx else "")
            + ". Peso activo = EW − benchmark. Percentil histórico = posición del peso actual dentro de toda su historia disponible "
            f"(mínimo {cfg.min_history} trimestres); vs prom. = diferencia contra el promedio de los {cfg.history_quarters} trimestres previos._\n")]
    s = s.sort_values("weight_ew", ascending=False)
    out.append("### Nivel, cambio e historia\n")
    out.append(_md_table(s, {"sector": "Sector", "weight_ew": "Peso EW", "d_qoq": "Δ QoQ", "d_yoy": "Δ YoY", "d_vs_avg": f"vs prom. {cfg.history_quarters}T",
                             "hist_percentile": "Percentil hist. (0-100)", "hist_min": "Mín. hist.", "hist_max": "Máx. hist."},
                         {"weight_ew": _pct, "d_qoq": _pp_or_blank, "d_yoy": _pp_or_blank, "d_vs_avg": _pp_or_blank,
                          "hist_percentile": _percentile, "hist_min": _pct, "hist_max": _pct}))
    if "net_flow" in s:
        out.append("\n### Flujo vs. precio en el trimestre (managers activos)\n")
        out.append("_Descomposición en dólares del Δ valor de cada sector: flujo neto (compras − ventas a precio del trimestre) y efecto precio (revalorización de lo que ya se tenía)._\n")
        out.append(_md_table(s.sort_values("net_flow", ascending=False), {"sector": "Sector", "net_flow": "Flujo neto", "price_effect": "Efecto precio", "gross_flow": "Flujo bruto",
                                                                          "n_buyers": "Compras", "n_sellers": "Ventas", "d_qoq": "Δ peso EW"},
                             {"net_flow": _usd, "price_effect": _usd, "gross_flow": _usd, "d_qoq": _pp_or_blank}))
    out.append("\n### Posicionamiento relativo al benchmark\n")
    out.append(f"_Benchmark = {bname}. Peso activo = EW − benchmark; % managers OW = proporción de managers activos con un peso mayor al del benchmark en ese sector._\n")
    cols = {"sector": "Sector", "weight_ew": "Peso EW", "weight_bench": "Benchmark", "active_weight": "Peso activo", "d_active_qoq": "Δ activo QoQ", "overweight_breadth": "% managers OW"}
    fmts: dict[str, Callable] = {"weight_ew": _pct, "weight_bench": _pct, "active_weight": _pp_or_blank, "d_active_qoq": _pp_or_blank, "overweight_breadth": lambda v: _pct(v, 0)}
    notes = []
    for b in benchmarks:
        wcol, acol, dcol = f"bench_{b.slug}", f"active_{b.slug}", f"asof_{b.slug}"
        if wcol in s and s[wcol].notna().any():
            cols[wcol], cols[acol] = b.name, f"Activo vs {b.name}"
            fmts[wcol], fmts[acol] = _pct, _pp_or_blank
            asof = s[dcol].dropna().iloc[0]
            notes.append(f"{b.name}: pesos al {asof}" + (f" ({b.source})" if b.source else ""))
    out.append(_md_table(s.sort_values("active_weight", ascending=False), cols, fmts))
    if notes:
        out.append("_Benchmarks externos — " + "; ".join(notes) + "._\n")
    elif benchmarks:
        out.append("_Hay benchmarks externos configurados pero ninguno tiene un corte en o antes de este periodo._\n")
    hist_periods = sorted(sp["period"].unique())[-cfg.history_quarters:]
    if len(hist_periods) >= 3:
        piv = sp[sp["period"].isin(hist_periods)].pivot_table(index="sector", columns="period", values="weight_ew").reindex(s["sector"])
        piv.columns = [quarter_label(p) for p in piv.columns]
        piv = piv.reset_index()
        out.append(f"\n### Trayectoria del peso EW (últimos {len(hist_periods)} trimestres)\n")
        out.append(_md_table(piv, {"sector": "Sector", **{c: c for c in piv.columns if c != "sector"}}, {c: _pct for c in piv.columns if c != "sector"}))
    return out


def _mover_sections(moves: pd.DataFrame, period: str, t: MoverThresholds) -> list[str]:
    m = moves[moves["period"] == period]
    if m.empty:
        return []
    out = ["\n## Empresas que más se movieron (agregado del universo)\n",
           (f"_Solo acciones directas (común, ADR, REIT; sin opciones, ETFs, preferentes ni deuda), agregando a todos los managers que reportaron ambos trimestres. "
            f"**Δ títulos** = variación % de los títulos agregados en manos del universo (independiente del precio). "
            f"Clasificación: **mayor** si |Δ títulos| ≥ {t.major_pct_shares:.0%}, o si hay al menos {t.major_breadth} compradores (o vendedores) netos y |Δ títulos| ≥ {t.major_breadth_pct_shares:.0%}; "
            f"**menor** si |Δ títulos| ≥ {t.minor_pct_shares:.0%}; **sin cambio** por debajo de ese umbral. "
            f"Las entradas nuevas al universo cuentan como cambio mayor._\n")]
    summ = magnitude_summary(moves, period)
    summ["magnitude"] = summ["magnitude"].map(_MAGNITUDE_ES)
    summ["share"] = summ["companies"] / summ["companies"].sum()
    out.append("### Resumen por magnitud\n")
    out.append(_md_table(summ, {"magnitude": "Magnitud", "companies": "Empresas", "share": "% del universo", "net_flow": "Flujo neto", "gross_flow": "Flujo bruto"},
                         {"share": lambda v: _pct(v, 0), "net_flow": _usd, "gross_flow": _usd}))
    m = m.assign(issuer_pretty=m["issuer"].map(pretty), holders=m["holders_prev"].astype(str) + " → " + m["holders_cur"].astype(str),
                 buy_sell=m["buyers"].astype(str) + " / " + m["sellers"].astype(str), magnitude_es=m["magnitude"].map(_MAGNITUDE_ES))
    cols = {"issuer_pretty": "Emisor", "sector": "Sector", "magnitude_es": "Cambio neto", "holders": "Tenedores (prev → act.)", "buy_sell": "Compr. / Vend.",
            "pct_shares": "Δ títulos", "net_flow": "Flujo neto", "price_effect": "Efecto precio", "value_cur": "Valor agregado"}
    fmts = {"pct_shares": lambda v: "nueva" if pd.isna(v) else f"{v:+.1%}", "net_flow": _usd, "price_effect": _usd, "value_cur": _usd}
    out.append("\n### Mayores movimientos por monto (flujo bruto)\n")
    out.append("_Ordenado por flujo bruto (compras + ventas en dólares). Un nombre puede tener mucho flujo bruto y aun así quedar 'sin cambio' neto cuando unos managers compran lo que otros venden: es rotación entre manos, no un cambio de posición del universo._\n")
    out.append(_md_table(m.reindex(m["gross_flow"].sort_values(ascending=False).index).head(15), cols, fmts))
    elig = m[m["eligible_intensity"] & (m["magnitude"] != "NONE")]
    buys = elig[elig["net_flow"] > 0].sort_values("intensity", ascending=False).head(10)
    sells = elig[elig["net_flow"] < 0].sort_values("intensity", ascending=False).head(10)
    icols = {k: v for k, v in cols.items() if k not in ("price_effect",)}
    if not buys.empty:
        out.append(f"\n### Mayor intensidad de compra (Δ títulos agregados, ≥{t.min_holders_for_intensity} tenedores)\n")
        out.append(_md_table(buys, icols, fmts))
    if not sells.empty:
        out.append(f"\n### Mayor intensidad de venta (Δ títulos agregados, ≥{t.min_holders_for_intensity} tenedores)\n")
        out.append(_md_table(sells, icols, fmts))
    return out


def _md_table(df: pd.DataFrame, cols: dict[str, str], fmts: dict[str, callable] | None = None) -> str:
    fmts = fmts or {}
    head = "| " + " | ".join(cols.values()) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for d in df.to_dict("records"):  # not itertuples: it renames columns that are not identifiers (e.g. "2024Q1")
        cells = []
        for c in cols:
            v = d.get(c, "")
            f = fmts.get(c)
            cells.append(f(v) if f else ("" if pd.isna(v) else str(v)))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def build_report(insights: list[dict], msum: pd.DataFrame, fp: pd.DataFrame, exp_asset_ew: pd.DataFrame,
                 exp_sector_ew: pd.DataFrame, cons: pd.DataFrame, changes: pd.DataFrame, source: str, *,
                 sector_pos: Optional[pd.DataFrame] = None, moves: Optional[pd.DataFrame] = None,
                 benchmarks: Optional[list[Benchmark]] = None, sector_cfg: SectorConfig = SectorConfig(),
                 thresholds: MoverThresholds = MoverThresholds()) -> str:
    sector_pos = sector_pos if sector_pos is not None else pd.DataFrame()
    moves = moves if moves is not None else pd.DataFrame()
    benchmarks = benchmarks or []
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

    if not changes.empty:
        out.append("\n## Actividad por tipo de manager\n")
        ch = changes[changes["period"] == period]
        bt = ch.groupby("manager_type", as_index=False).agg(managers=("cik", "nunique"), net_flow=("flow_effect", "sum"), gross_flow=("abs_flow", "sum"),
                                                              nuevas=("action", lambda x: int((x == "NEW").sum())), salidas=("action", lambda x: int((x == "EXIT").sum())))
        book = msum[msum["period"] == period].groupby("manager_type", as_index=False)["total_value"].sum()
        bt = bt.merge(book, on="manager_type", how="left")
        bt["flow_pct"] = bt["net_flow"] / bt["total_value"]
        out.append(_md_table(bt.sort_values("gross_flow", ascending=False), {"manager_type": "Tipo de manager", "managers": "Managers", "total_value": "Valor 13F", "net_flow": "Flujo neto", "flow_pct": "Flujo neto / valor", "gross_flow": "Flujo bruto", "nuevas": "Nuevas", "salidas": "Salidas"},
                             {"total_value": _usd, "net_flow": _usd, "gross_flow": _usd, "flow_pct": lambda v: f"{v:+.1%}"}))

    out.append("\n## Exposición por tipo de activo (promedio simple entre managers)\n")
    ea = exp_asset_ew[exp_asset_ew["period"] == period].sort_values("avg_weight", ascending=False)
    out.append(_md_table(ea, {"underlying_asset": "Tipo de activo", "avg_weight": "Peso promedio", "d_avg_weight": "Δ vs. trimestre previo"},
                         {"avg_weight": lambda v: f"{v:.1%}", "d_avg_weight": lambda v: "" if pd.isna(v) else _pp(v)}))

    sector_out = _sector_sections(sector_pos, period, sector_cfg, benchmarks)
    if sector_out:
        out += sector_out
    else:  # no direct-equity book at all: fall back to the total-book equal-weight table
        out.append("\n## Exposición sectorial (promedio simple entre managers, libro total)\n")
        es = exp_sector_ew[exp_sector_ew["period"] == period].sort_values("avg_weight", ascending=False)
        out.append(_md_table(es, {"sector": "Sector", "avg_weight": "Peso promedio", "d_avg_weight": "Δ vs. trimestre previo"},
                             {"avg_weight": _pct, "d_avg_weight": _pp_or_blank}))

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

    out += _mover_sections(moves, period, thresholds)

    ch = changes[changes["period"] == period] if not changes.empty else pd.DataFrame()
    if not ch.empty:
        out.append("\n## Movimientos individuales más grandes (por manager, flujo estimado)\n")
        big = ch.reindex(ch["abs_flow"].sort_values(ascending=False).index).head(20)
        out.append(_md_table(big, {"manager": "Manager", "action": "Acción", "display_name": "Emisor", "asset_type": "Tipo", "pct_shares": "Δ títulos", "flow_effect": "Flujo", "weight_prev": "Peso previo", "weight_cur": "Peso actual"},
                             {"pct_shares": lambda v: "" if pd.isna(v) else f"{v:+.0%}", "flow_effect": _usd, "weight_prev": lambda v: f"{v:.1%}", "weight_cur": lambda v: f"{v:.1%}"}))

    # ---- long-run history, one row per year (Q4, or the latest quarter of the final year)
    periods = sorted(msum["period"].unique())
    if len(periods) > 4:
        out.append("\n## Serie histórica (un corte por año)\n")
        yearly = {}
        for p in periods:
            yearly[p[:4]] = p  # keeps the last quarter available in each year
        rows = []
        for y, p in sorted(yearly.items()):
            m = msum[msum["period"] == p]
            ea = exp_asset_ew[(exp_asset_ew["period"] == p) & (exp_asset_ew["underlying_asset"] == "Equity")]
            es = exp_sector_ew[(exp_sector_ew["period"] == p) & (~exp_sector_ew["sector"].str.startswith("ETF"))]
            top = es.nlargest(1, "avg_weight")
            rows.append(dict(period=quarter_label(p), managers=int(m["cik"].nunique()), total_value=float(m["total_value"].sum()),
                             net_flow=(float(m["net_flow"].sum()) if "net_flow" in m and m["net_flow"].notna().any() else float("nan")),
                             equity_w=(float(ea["avg_weight"].iloc[0]) if not ea.empty else float("nan")),
                             top_sector=(f"{top['sector'].iloc[0]} ({top['avg_weight'].iloc[0]:.1%})" if not top.empty else "")))
        out.append(_md_table(pd.DataFrame(rows), {"period": "Trimestre", "managers": "Managers", "total_value": "Valor 13F", "net_flow": "Flujo neto (trim.)", "equity_w": "Equity directo (EW)", "top_sector": "Sector líder (EW)"},
                             {"total_value": _usd, "net_flow": lambda v: "" if pd.isna(v) else _usd(v), "equity_w": lambda v: "" if pd.isna(v) else f"{v:.1%}"}))

    out.append("\n## Metodología\n")
    out.append("- **Fuente:** formularios 13F-HR (y enmiendas 13F-HR/A) descargados de SEC EDGAR vía la API de submissions y los archivos XML del information table. "
               "Los valores se normalizan a dólares (los filings anteriores al 3-ene-2023 reportan en miles).\n"
               "- **Clasificación de activos:** reglas sobre `titleOfClass`, `putCall` y `sshPrnamtType`, más un maestro de valores por CUSIP; la clasificación sectorial usa el maestro, palabras clave del emisor y un modelo Naive Bayes sobre el nombre del emisor cuando no hay coincidencia. Cada etiqueta lleva un score de confianza.\n"
               "- **Cambios:** el Δ de valor de cada posición se descompone en *efecto flujo* (Δ títulos × precio implícito del trimestre actual) y *efecto precio* (resto).\n"
               "- **Exposición equal-weight:** promedio simple de los pesos de cada manager, para que los filers muy grandes (índices) no dominen la lectura. La versión ponderada por valor también se reporta en el dashboard.\n"
               "- **Posicionamiento sectorial:** se calcula sobre el libro de acciones directas de cada manager (sin ETFs, opciones, deuda ni preferentes) para que sea comparable con un índice de renta variable. "
               "El benchmark implícito es la mezcla sectorial ponderada por valor de los managers índice del universo (sus 13F replican de cerca el mercado estadounidense); si se configura `config/benchmarks.json` "
               "con pesos de un índice externo (p. ej. S&P 500), se aplican *as-of* (último corte disponible en o antes del periodo) y se reportan junto a la fecha del corte. "
               "El peso activo es EW − benchmark; el % de managers OW mide la amplitud de la sobreponderación. La descomposición flujo/precio por sector suma los efectos de las posiciones de los managers activos.\n"
               "- **Movimientos por empresa:** agregación de los diffs de todos los managers por CUSIP (solo acciones en efectivo). La intensidad se mide como variación % de los títulos agregados (independiente del precio) y la materialidad por flujo bruto en dólares; "
               "los umbrales de la clasificación mayor/menor/sin cambio se indican en la propia sección.\n"
               "- **Tipo de manager inferido:** huella de cartera (número de posiciones, concentración top-10/HHI, share de opciones/ETF/crédito, rotación).\n"
               "- **Limitaciones del 13F:** solo posiciones largas en valores de la sección 13(f) de EE.UU. (no cortos, no bonos soberanos, no derivados OTC, no posiciones internacionales sin ADR); rezago de hasta 45 días; las opciones se reportan por valor nocional del subyacente.\n")
    return "\n".join(out)
