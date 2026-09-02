"""Offline sample generator.

Writes synthetic-but-schema-exact 13F-HR filings (primary_doc.xml + infotable.xml
+ meta.json) into data/raw so the full pipeline can run without network access.
Holdings are *not* real; identifiers for listed common stock, ADRs and ETFs are.

The engine simulates N quarters ending 2026-06-30 with a market factor that
follows approximate S&P 500 quarterly returns (2011Q4-2025Q1 from memory,
later quarters invented), sector betas, idiosyncratic noise, IPO dates for
securities, first-filing dates for managers and a handful of scripted events.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from .config import Settings
from .ingest import FilingRef
from .managers import Manager

LAST_PERIOD = date(2026, 6, 30)

# Approximate S&P 500 total-return by quarter, 2011Q4 .. 2026Q2 (percent).
MARKET_RETURNS = {
    "2011-12-31": 11.8, "2012-03-31": 12.6, "2012-06-30": -2.8, "2012-09-30": 6.4, "2012-12-31": -0.4,
    "2013-03-31": 10.6, "2013-06-30": 2.9, "2013-09-30": 5.2, "2013-12-31": 10.5,
    "2014-03-31": 1.8, "2014-06-30": 5.2, "2014-09-30": 1.1, "2014-12-31": 4.9,
    "2015-03-31": 0.9, "2015-06-30": 0.3, "2015-09-30": -6.4, "2015-12-31": 7.0,
    "2016-03-31": 1.3, "2016-06-30": 2.5, "2016-09-30": 3.9, "2016-12-31": 3.8,
    "2017-03-31": 6.1, "2017-06-30": 3.1, "2017-09-30": 4.5, "2017-12-31": 6.6,
    "2018-03-31": -0.8, "2018-06-30": 3.4, "2018-09-30": 7.7, "2018-12-31": -13.5,
    "2019-03-31": 13.6, "2019-06-30": 4.3, "2019-09-30": 1.7, "2019-12-31": 9.1,
    "2020-03-31": -19.6, "2020-06-30": 20.5, "2020-09-30": 8.9, "2020-12-31": 12.1,
    "2021-03-31": 6.2, "2021-06-30": 8.5, "2021-09-30": 0.6, "2021-12-31": 11.0,
    "2022-03-31": -4.6, "2022-06-30": -16.1, "2022-09-30": -4.9, "2022-12-31": 7.6,
    "2023-03-31": 7.5, "2023-06-30": 8.7, "2023-09-30": -3.3, "2023-12-31": 11.7,
    "2024-03-31": 10.6, "2024-06-30": 4.3, "2024-09-30": 5.9, "2024-12-31": 2.4,
    "2025-03-31": -4.3, "2025-06-30": 10.9, "2025-09-30": 8.1, "2025-12-31": 2.7,
    "2026-03-31": 3.0, "2026-06-30": 4.5,
}
REF_PRICE_PERIOD = "2025-09-30"  # issuers.json ref_price is quoted as of this quarter

SECTOR_BETA = {
    "Information Technology": 1.25, "Communication Services": 1.05, "Consumer Discretionary": 1.15, "Consumer Staples": 0.6,
    "Health Care": 0.8, "Financials": 1.1, "Energy": 0.9, "Industrials": 1.05, "Materials": 1.0, "Utilities": 0.45, "Real Estate": 0.8,
    "ETF - Broad Equity": 1.0, "ETF - International Equity": 0.9, "ETF - Sector": 1.0, "ETF - Thematic": 1.5,
    "ETF - Fixed Income": 0.05, "ETF - Commodity": 0.1,
}
SECTOR_ALPHA = {  # annualised excess drift, so long histories look plausible
    "Information Technology": 0.08, "Communication Services": 0.02, "Consumer Discretionary": 0.02, "Energy": -0.04,
    "Utilities": -0.03, "Consumer Staples": -0.03, "Health Care": -0.01, "ETF - Fixed Income": 0.02, "ETF - Commodity": 0.04,
}

# style -> (n_positions, concentration exponent, option share, turnover, sector tilts)
PROFILES = {
    "value": dict(n=40, conc=1.6, opt=0.0, turnover=0.10, tilt={"Financials": 2.0, "Consumer Staples": 1.8, "Energy": 1.6, "Health Care": 1.2, "ETF": 0.0}),
    "macro": dict(n=90, conc=0.9, opt=0.10, turnover=0.35, tilt={"ETF - Broad Equity": 12.0, "ETF - International Equity": 9.0, "ETF - Commodity": 9.0, "ETF - Fixed Income": 8.0, "ETF - Sector": 6.0, "Financials": 1.0}),
    "quant": dict(n=150, conc=0.5, opt=0.05, turnover=0.55, tilt={}),
    "multistrat": dict(n=150, conc=0.6, opt=0.35, turnover=0.50, tilt={}),
    "activist": dict(n=12, conc=1.3, opt=0.15, turnover=0.20, tilt={"Industrials": 1.5, "Consumer Discretionary": 1.5, "Real Estate": 1.5, "ETF": 0.0}),
    "growth": dict(n=40, conc=1.3, opt=0.05, turnover=0.30, tilt={"Information Technology": 2.5, "Communication Services": 2.0, "Consumer Discretionary": 1.8, "ETF": 0.0, "Utilities": 0.1, "Energy": 0.2}),
    "index": dict(n=200, conc=0.55, opt=0.0, turnover=0.03, tilt={"ETF": 0.0}),
}
# Sample long value (USD) as of the last quarter; earlier quarters are deflated by the market factor.
AUM = {
    "1067983": 285e9, "1350694": 22e9, "1037389": 70e9, "1423053": 160e9, "1336528": 12e9, "1791786": 15e9, "1167483": 25e9,
    "1061768": 6e9, "1029160": 7e9, "1649339": 0.3e9, "1536411": 3.5e9, "102909": 4.8e12, "1364742": 4.2e12, "1179392": 45e9,
    "1656456": 6e9, "1061165": 14e9, "1103804": 28e9, "1135730": 30e9, "1009207": 110e9, "1273087": 200e9, "1167557": 55e9,
    "1603466": 40e9, "1079114": 2e9, "1040273": 7e9, "921669": 10e9, "1418814": 6e9, "1517137": 5e9, "1345471": 6e9,
    "1159159": 2e9, "934639": 5e9, "1138995": 3e9, "1035674": 2e9, "923093": 8e9, "1165408": 50e9, "1318757": 30e9,
    "1218710": 25e9, "1595888": 60e9, "1446194": 300e9, "909661": 20e9, "1541617": 6e9, "1166559": 45e9, "1374": 450e9,
    "919079": 130e9, "315066": 1.4e12, "80255": 800e9, "93751": 2.3e12, "902219": 500e9, "29440": 180e9, "813917": 70e9, "1088875": 120e9,
}
DEFAULT_AUM = 5e9

# Scripted events keyed by (cik, period). Values: ticker -> share multiplier (0 = exit), "new": [tickers or TICKER:Put/Call]. All synthetic.
EVENTS = {
    ("1067983", "2016-03-31"): {"new": ["AAPL"]},
    ("1067983", "2016-12-31"): {"AAPL": 3.0, "new": ["DAL"]},
    ("1067983", "2017-06-30"): {"AAPL": 1.6},
    ("1067983", "2018-06-30"): {"AAPL": 1.3},
    ("1067983", "2020-06-30"): {"DAL": 0.0},
    ("1067983", "2020-09-30"): {"AAPL": 0.96, "WFC": 0.5},
    ("1067983", "2022-03-31"): {"new": ["OXY"], "CVX": 3.0},
    ("1067983", "2022-06-30"): {"OXY": 1.5},
    ("1067983", "2024-03-31"): {"AAPL": 0.87},
    ("1067983", "2024-06-30"): {"AAPL": 0.5, "BAC": 0.9},
    ("1067983", "2024-09-30"): {"AAPL": 0.75, "BAC": 0.8},
    ("1067983", "2025-03-31"): {"new": ["STZ", "DPZ"], "BAC": 0.9},
    ("1067983", "2025-12-31"): {"AAPL": 0.87, "OXY": 1.15},
    ("1067983", "2026-03-31"): {"AAPL": 0.80, "CVX": 1.10, "SIRI": 1.25},
    ("1067983", "2026-06-30"): {"AAPL": 0.75, "OXY": 1.10, "BAC": 0.85, "new": ["HEI", "POOL"]},
    ("1167483", "2020-12-31"): {"new": ["SNOW", "PLTR", "ABNB"]},
    ("1167483", "2021-06-30"): {"SNOW": 1.5, "new": ["COIN", "RBLX"]},
    ("1167483", "2022-06-30"): {"SNOW": 0.5, "COIN": 0.0, "RBLX": 0.0, "META": 0.6},
    ("1167483", "2025-12-31"): {"new": ["NVDA", "AVGO", "ARM"], "META": 1.3},
    ("1167483", "2026-03-31"): {"NVDA": 1.4, "SNOW": 0.5, "new": ["SMCI"]},
    ("1167483", "2026-06-30"): {"SMCI": 0.0, "AMD": 1.5},
    ("1649339", "2020-12-31"): {"new": ["GOOGL:Call", "PFE"]},
    ("1649339", "2021-06-30"): {"new": ["TSLA:Put", "ARKK:Put"]},
    ("1649339", "2021-09-30"): {"TSLA:Put": 0.0, "ARKK:Put": 0.0},
    ("1649339", "2023-09-30"): {"new": ["SPY:Put", "QQQ:Put"]},
    ("1649339", "2023-12-31"): {"SPY:Put": 0.0, "QQQ:Put": 0.0},
    ("1649339", "2024-06-30"): {"new": ["BABA", "JD"]},
    ("1649339", "2026-03-31"): {"new": ["NVDA:Put", "SMCI:Put"], "JD": 1.3},
    ("1649339", "2026-06-30"): {"NVDA:Put": 0.0, "new": ["EL"]},
    ("1350694", "2020-03-31"): {"new": ["SPY:Put"]},
    ("1350694", "2020-06-30"): {"SPY:Put": 0.0, "GLD": 1.8},
    ("1350694", "2025-12-31"): {"EEM": 0.7, "GLD": 1.4},
    ("1350694", "2026-03-31"): {"EEM": 0.6, "GLD": 1.3, "IVV": 1.15},
    ("1350694", "2026-06-30"): {"SPY": 0.85, "TLT": 1.3, "new": ["XLE"]},
    ("1336528", "2014-03-31"): {"new": ["CP"]},
    ("1336528", "2018-12-31"): {"new": ["HLT", "CMG"]},
    ("1336528", "2020-06-30"): {"new": ["LOW"], "HLT": 1.4},
    ("1336528", "2026-03-31"): {"new": ["NKE"], "CMG": 0.8},
    ("1336528", "2026-06-30"): {"new": ["UBER"]},
    ("1791786", "2025-12-31"): {"new": ["SWK", "HON"]},
    ("1791786", "2026-06-30"): {"new": ["PSX"], "SWK": 1.25},
    ("1029160", "2026-06-30"): {"new": ["SPY:Put", "QQQ:Put"], "AMZN": 1.2},
    ("1656456", "2025-12-31"): {"new": ["MU", "AMZN"], "META": 1.2},
    ("1656456", "2026-06-30"): {"new": ["NVDA"], "INTC": 0.0},
    ("921669", "2013-09-30"): {"new": ["AAPL"]},
    ("921669", "2016-06-30"): {"AAPL": 0.0},
    ("1345471", "2017-09-30"): {"new": ["PG"]},
    ("1345471", "2021-03-31"): {"PG": 0.0},
}

XML_HEAD = '<?xml version="1.0" encoding="UTF-8"?>\n'


def quarter_ends(n: int, last: date = LAST_PERIOD) -> list[date]:
    out, d = [], last
    for _ in range(n):
        out.append(d)
        d = (d.replace(day=1) - timedelta(days=1))          # last day of previous month
        d = (d.replace(day=1) - timedelta(days=1))
        d = (d.replace(day=1) - timedelta(days=1))
    return sorted(out)


def filing_date_for(period: date) -> date:
    """13F is due 45 days after quarter end; sample files on the deadline (or the next weekday)."""
    d = period + timedelta(days=45)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _load_issuers(settings: Settings) -> list[dict]:
    # enriched master entries (from SEC/OpenFIGI) carry no reference price; the sample generator needs one
    return [i for i in json.loads(settings.issuers_file.read_text(encoding="utf-8"))["issuers"] if i.get("ref_price")]


def _market_path(periods: list[date], rng: np.random.Generator) -> np.ndarray:
    """Cumulative market factor, 1.0 at the first period."""
    f = [1.0]
    for p in periods[1:]:
        r = MARKET_RETURNS.get(p.isoformat())
        if r is None:
            r = rng.normal(2.2, 7.0)
        f.append(f[-1] * (1 + r / 100))
    return np.array(f)


def _price_paths(issuers: list[dict], periods: list[date], rng: np.random.Generator) -> dict[str, np.ndarray]:
    mkt = _market_path(periods, rng)
    mkt_ret = np.diff(np.log(mkt))
    ref_idx = [p.isoformat() for p in periods].index(REF_PRICE_PERIOD) if REF_PRICE_PERIOD in [p.isoformat() for p in periods] else len(periods) - 1
    paths = {}
    for it in issuers:
        sec = it["sector"]
        beta = SECTOR_BETA.get(sec, 1.0)
        alpha = SECTOR_ALPHA.get(sec, 0.0) / 4
        idio = 0.03 if it["asset_type"] == "ETF" else (0.11 if sec == "Information Technology" else 0.08)
        if it["asset_type"] in ("Convertible Debt", "Corporate Debt", "Preferred Stock"):
            beta, idio = 0.15, 0.02
        rets = beta * mkt_ret + alpha + rng.normal(0, idio, size=len(mkt_ret)) - 0.5 * idio ** 2
        lvl = np.concatenate([[0.0], np.cumsum(rets)])
        lvl -= lvl[ref_idx]
        paths[it["ticker"]] = np.round(float(it["ref_price"]) * np.exp(lvl), 2)
    return paths, mkt


def _tilt_weights(m: Manager, issuers: list[dict]) -> np.ndarray:
    prof = PROFILES.get(m.style, PROFILES["value"])
    weights = []
    for it in issuers:
        w = 1.0
        sec = it["sector"]
        for k, v in prof["tilt"].items():
            if sec.startswith(k):
                w *= v
        if it["asset_type"] in ("Preferred Stock", "Convertible Debt", "Corporate Debt", "Warrant", "Unit"):
            w *= 0.35 if m.style in ("value", "activist", "macro") else 0.05
        if m.style == "index" and it["asset_type"] == "ETF":
            w = 0.0
        weights.append(w)
    weights = np.array(weights, dtype=float)
    if weights.sum() == 0:
        weights = np.ones(len(issuers))
    return weights


def _pick_universe(m: Manager, issuers: list[dict], rng: np.random.Generator) -> list[dict]:
    prof = PROFILES.get(m.style, PROFILES["value"])
    weights = _tilt_weights(m, issuers)
    n = min(prof["n"], int((weights > 0).sum()))
    idx = rng.choice(len(issuers), size=n, replace=False, p=weights / weights.sum())
    chosen = [issuers[i] for i in idx]
    ranks = np.arange(1, n + 1)
    sizes = 1.0 / ranks ** prof["conc"]
    rng.shuffle(sizes)
    boost = 4.0 if m.style == "macro" else 1.0
    return [dict(c, size=float(s) * (boost if c["asset_type"] == "ETF" else 1.0)) for c, s in zip(chosen, sizes)]


def _write_filing(dest: Path, m: Manager, ref: FilingRef, rows: list[dict], in_thousands: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    period_us = f"{ref.report_period[5:7]}-{ref.report_period[8:10]}-{ref.report_period[:4]}"
    div = 1000 if in_thousands else 1
    vals = [int(round(r["value"] / div)) for r in rows]
    total = sum(vals)
    primary = f"""{XML_HEAD}<edgarSubmission xmlns="http://www.sec.gov/edgar/thirteenffiler" xmlns:com="http://www.sec.gov/edgar/common">
  <headerData>
    <submissionType>13F-HR</submissionType>
    <filerInfo>
      <liveTestFlag>LIVE</liveTestFlag>
      <filer><credentials><cik>{m.cik10}</cik></credentials></filer>
      <periodOfReport>{period_us}</periodOfReport>
    </filerInfo>
  </headerData>
  <formData>
    <coverPage>
      <reportCalendarOrQuarter>{period_us}</reportCalendarOrQuarter>
      <isAmendment>false</isAmendment>
      <filingManager>
        <name>{escape(m.name)}</name>
        <address><com:street1>SAMPLE</com:street1><com:city>SAMPLE</com:city><com:stateOrCountry>NY</com:stateOrCountry><com:zipCode>00000</com:zipCode></address>
      </filingManager>
      <reportType>13F HOLDINGS REPORT</reportType>
      <form13FFileNumber>028-00000</form13FFileNumber>
      <provideInfoForInstruction5>N</provideInfoForInstruction5>
    </coverPage>
    <signatureBlock><name>Sample Signer</name><title>CFO</title><phone>000-000-0000</phone><signature>/s/ Sample</signature><city>SAMPLE</city><stateOrCountry>NY</stateOrCountry><signatureDate>{ref.filing_date[5:7]}-{ref.filing_date[8:10]}-{ref.filing_date[:4]}</signatureDate></signatureBlock>
    <summaryPage>
      <otherIncludedManagersCount>0</otherIncludedManagersCount>
      <tableEntryTotal>{len(rows)}</tableEntryTotal>
      <tableValueTotal>{total}</tableValueTotal>
      <isConfidentialOmitted>false</isConfidentialOmitted>
    </summaryPage>
  </formData>
</edgarSubmission>
"""
    parts = [XML_HEAD, '<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">\n']
    for r, v in zip(rows, vals):
        pc = f"\n    <putCall>{r['put_call']}</putCall>" if r["put_call"] else ""
        parts.append(
            f"""  <infoTable>
    <nameOfIssuer>{escape(r['issuer'])}</nameOfIssuer>
    <titleOfClass>{escape(r['title_of_class'])}</titleOfClass>
    <cusip>{r['cusip']}</cusip>
    <value>{v}</value>
    <shrsOrPrnAmt>
      <sshPrnamt>{int(r['shares'])}</sshPrnamt>
      <sshPrnamtType>{r['sh_prn']}</sshPrnamtType>
    </shrsOrPrnAmt>{pc}
    <investmentDiscretion>SOLE</investmentDiscretion>
    <votingAuthority>
      <Sole>{int(r['shares'])}</Sole>
      <Shared>0</Shared>
      <None>0</None>
    </votingAuthority>
  </infoTable>
"""
        )
    parts.append("</informationTable>\n")
    (dest / "primary_doc.xml").write_text(primary)
    (dest / "infotable.xml").write_text("".join(parts))
    (dest / "meta.json").write_text(json.dumps(asdict(ref), indent=2))


def generate_sample(managers: list[Manager], settings: Settings, seed: int = 13, quarters: int = 4) -> list[Path]:
    settings.ensure_dirs()
    issuers = _load_issuers(settings)
    by_ticker = {i["ticker"]: i for i in issuers}
    rng = np.random.default_rng(seed)
    periods = quarter_ends(quarters)
    piso = [p.isoformat() for p in periods]
    prices, mkt = _price_paths(issuers, periods, rng)
    out: list[Path] = []

    def live(it: dict, p: str) -> bool:
        return it.get("ipo", "") <= p

    for m in managers:
        prof = PROFILES.get(m.style, PROFILES["value"])
        aum_last = AUM.get(m.cik, DEFAULT_AUM)
        first_idx = next((i for i, p in enumerate(piso) if p >= (m.since or "0000")), None)
        if first_idx is None:
            continue
        # sub-generator per manager so adding a manager does not reshuffle the others
        mrng = np.random.default_rng([seed, int(m.cik)])
        book: dict[str, dict] = {}
        for qi in range(first_idx, len(periods)):
            p = piso[qi]
            aum = aum_last * mkt[qi] / mkt[-1] * (0.85 + 0.3 * mrng.random())
            if not book:
                universe = [u for u in _pick_universe(m, issuers, mrng) if live(u, p)]
                tot = sum(u["size"] for u in universe) or 1.0
                for u in universe:
                    key, put_call = u["ticker"], ""
                    if prof["opt"] > 0 and mrng.random() < prof["opt"]:
                        put_call = "Put" if mrng.random() < 0.5 else "Call"
                        key = f"{u['ticker']}:{put_call}"
                    tv = aum * u["size"] / tot * (0.2 if put_call else 1.0)
                    book[key] = {"it": u, "shares": max(100, int(tv / prices[u["ticker"]][qi])), "put_call": put_call}
            else:
                # organic turnover: big positions get trimmed rather than liquidated outright
                bv = {k: v["shares"] * float(prices[v["it"]["ticker"]][qi - 1]) for k, v in book.items()}
                tot_bv = sum(bv.values()) or 1.0
                for key in list(book):
                    r = mrng.random()
                    if r < prof["turnover"] * 0.25 and bv[key] / tot_bv < 0.03:
                        del book[key]
                    elif r < prof["turnover"] * 0.6:
                        book[key]["shares"] = int(book[key]["shares"] * mrng.uniform(0.4, 0.9))
                    elif r < prof["turnover"]:
                        book[key]["shares"] = int(book[key]["shares"] * mrng.uniform(1.1, 1.8))
                held = {v["it"]["ticker"] for v in book.values()}
                # replace what left plus some organic additions, so the book keeps its target breadth
                n_new = max(0, prof["n"] - len(book)) + max(1, int(round(prof["n"] * prof["turnover"] * 0.15)))
                candidates = [i for i in issuers if i["ticker"] not in held and live(i, p)]
                cw = _tilt_weights(m, candidates) if candidates else np.array([])
                if candidates and cw.sum() > 0:
                    picks = mrng.choice(len(candidates), size=min(n_new, int((cw > 0).sum())), replace=False, p=cw / cw.sum())
                    for ci in picks:
                        c = candidates[ci]
                        put_call = ""
                        if prof["opt"] > 0 and mrng.random() < prof["opt"]:
                            put_call = "Put" if mrng.random() < 0.5 else "Call"
                        tv = aum * mrng.uniform(0.002, 0.02) * (4.0 if (m.style == "macro" and c["asset_type"] == "ETF") else 1.0)
                        key = f"{c['ticker']}:{put_call}" if put_call else c["ticker"]
                        book[key] = {"it": c, "shares": max(100, int(tv / prices[c["ticker"]][qi])), "put_call": put_call}
                for k, v in EVENTS.get((m.cik, p), {}).items():
                    if k == "new":
                        for tk in v:
                            base, _, pc = tk.partition(":")
                            if base not in by_ticker or not live(by_ticker[base], p):
                                continue
                            tv = aum * mrng.uniform(0.01, 0.06) * (0.2 if pc else 1.0)
                            book[tk] = {"it": by_ticker[base], "shares": max(100, int(tv / prices[base][qi])), "put_call": pc}
                    elif k in book:
                        if v == 0.0:
                            del book[k]
                        else:
                            book[k]["shares"] = int(book[k]["shares"] * v)
            rows = []
            for key, pos in book.items():
                it = pos["it"]
                rows.append({
                    "issuer": it["issuer"], "title_of_class": it["title_of_class"], "cusip": it["cusip"],
                    "value": pos["shares"] * float(prices[it["ticker"]][qi]), "shares": pos["shares"],
                    "sh_prn": "PRN" if it["asset_type"] in ("Convertible Debt", "Corporate Debt") else "SH",
                    "put_call": pos["put_call"],
                })
            rows.sort(key=lambda r: r["issuer"])
            period, fdate = periods[qi], filing_date_for(periods[qi])
            acc = f"{m.cik10}-{str(period.year)[2:]}-{qi:03d}{int(m.cik) % 1000:03d}"
            ref = FilingRef(m.cik, m.name, acc, "13F-HR", fdate.isoformat(), period.isoformat(), "primary_doc.xml", source="sample")
            dest = settings.raw_dir / m.cik / ref.folder_name
            _write_filing(dest, m, ref, rows, in_thousands=fdate.isoformat() < "2023-01-03")
            out.append(dest)
    return out
