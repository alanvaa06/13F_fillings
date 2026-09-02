"""Offline sample generator.

Writes synthetic-but-schema-exact 13F-HR filings (primary_doc.xml + infotable.xml
+ meta.json) into data/raw so the full pipeline can run without network access.
Holdings are *not* real; identifiers for listed common stock, ADRs and ETFs are.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from .config import Settings
from .ingest import FilingRef
from .managers import Manager

QUARTERS = [
    (date(2025, 9, 30), date(2025, 11, 14)),
    (date(2025, 12, 31), date(2026, 2, 17)),
    (date(2026, 3, 31), date(2026, 5, 15)),
    (date(2026, 6, 30), date(2026, 8, 14)),
]

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
AUM = {  # sample total long value in USD
    "1067983": 285e9, "1350694": 22e9, "1037389": 70e9, "1423053": 160e9, "1336528": 12e9,
    "1791786": 15e9, "1167483": 25e9, "1061768": 6e9, "1029160": 7e9, "1649339": 0.3e9,
    "1536411": 3.5e9, "102909": 4.8e12, "1364742": 4.2e12, "1179392": 45e9, "1656456": 6e9,
}

# Scripted events so the sample tells a story (all synthetic).
EVENTS = {
    ("1067983", 1): {"AAPL": 0.87, "OXY": 1.15},
    ("1067983", 2): {"AAPL": 0.80, "CVX": 1.10, "SIRI": 1.25},
    ("1067983", 3): {"AAPL": 0.75, "OXY": 1.10, "BAC": 0.85, "new": ["HEI", "POOL"]},
    ("1167483", 1): {"new": ["NVDA", "AVGO", "ARM"], "META": 1.3},
    ("1167483", 2): {"NVDA": 1.4, "SNOW": 0.5, "new": ["SMCI"]},
    ("1167483", 3): {"SMCI": 0.0, "AMD": 1.5},
    ("1649339", 2): {"new": ["NVDA:Put", "SMCI:Put", "BABA"], "JD": 1.3},
    ("1649339", 3): {"NVDA:Put": 0.0, "new": ["EL", "MOH"]},
    ("1350694", 1): {"EEM": 0.7, "GLD": 1.4},
    ("1350694", 2): {"EEM": 0.6, "GLD": 1.3, "IVV": 1.15},
    ("1350694", 3): {"SPY": 0.85, "TLT": 1.3, "new": ["XLE"]},
    ("1336528", 2): {"new": ["NKE"], "CMG": 0.8},
    ("1336528", 3): {"new": ["UBER"], "HHH": 1.0},
    ("1791786", 1): {"new": ["SWK", "HON"]},
    ("1791786", 3): {"new": ["PSX", "BP"], "SWK": 1.25},
    ("1029160", 3): {"new": ["SPY:Put", "QQQ:Put"], "AMZN": 1.2},
    ("1656456", 1): {"new": ["MU", "AMZN"], "META": 1.2},
    ("1656456", 3): {"new": ["NVDA"], "INTC": 0.0},
}

XML_HEAD = '<?xml version="1.0" encoding="UTF-8"?>\n'


def _load_issuers(settings: Settings) -> list[dict]:
    return json.loads(settings.issuers_file.read_text())["issuers"]


def _price_paths(issuers: list[dict], rng: np.random.Generator) -> dict[str, list[float]]:
    sector_drift = {
        "Information Technology": 0.06, "Communication Services": 0.04, "Consumer Discretionary": 0.02,
        "Consumer Staples": 0.01, "Health Care": -0.01, "Financials": 0.03, "Energy": 0.05,
        "Industrials": 0.03, "Materials": 0.02, "Utilities": 0.01, "Real Estate": 0.00,
    }
    paths = {}
    for it in issuers:
        drift = sector_drift.get(it["sector"], 0.02)
        vol = 0.04 if it["asset_type"] == "ETF" else (0.10 if it["sector"] == "Information Technology" else 0.08)
        p = float(it["ref_price"])
        seq = [p]
        for _ in range(len(QUARTERS) - 1):
            p *= float(np.exp(rng.normal(drift, vol)))
            seq.append(round(p, 2))
        paths[it["ticker"]] = seq
    return paths


def _pick_universe(m: Manager, issuers: list[dict], rng: np.random.Generator) -> list[dict]:
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
    weights = np.array(weights)
    if weights.sum() == 0:
        weights = np.ones(len(issuers))
    n = min(prof["n"], (weights > 0).sum())
    idx = rng.choice(len(issuers), size=n, replace=False, p=weights / weights.sum())
    chosen = [issuers[i] for i in idx]
    # position sizes: power-law by rank
    ranks = np.arange(1, n + 1)
    sizes = 1.0 / ranks ** prof["conc"]
    rng.shuffle(sizes)
    for it, s in zip(chosen, sizes):
        it = dict(it)
    boost = 4.0 if m.style == "macro" else 1.0
    return [dict(c, size=float(s) * (boost if c["asset_type"] == "ETF" else 1.0)) for c, s in zip(chosen, sizes)]


def _write_filing(dest: Path, m: Manager, ref: FilingRef, rows: list[dict]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    period_us = f"{ref.report_period[5:7]}-{ref.report_period[8:10]}-{ref.report_period[:4]}"
    total = int(round(sum(r["value"] for r in rows)))
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
    for r in rows:
        pc = f"\n    <putCall>{r['put_call']}</putCall>" if r["put_call"] else ""
        parts.append(
            f"""  <infoTable>
    <nameOfIssuer>{escape(r['issuer'])}</nameOfIssuer>
    <titleOfClass>{escape(r['title_of_class'])}</titleOfClass>
    <cusip>{r['cusip']}</cusip>
    <value>{int(round(r['value']))}</value>
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


def generate_sample(managers: list[Manager], settings: Settings, seed: int = 13) -> list[Path]:
    settings.ensure_dirs()
    issuers = _load_issuers(settings)
    by_ticker = {i["ticker"]: i for i in issuers}
    rng = np.random.default_rng(seed)
    prices = _price_paths(issuers, rng)
    out: list[Path] = []

    for m in managers:
        prof = PROFILES.get(m.style, PROFILES["value"])
        aum = AUM.get(m.cik, 5e9)
        book = {}  # key -> {issuer dict, shares, put_call}
        universe = _pick_universe(m, issuers, rng)
        tot = sum(u["size"] for u in universe)
        for u in universe:
            key = u["ticker"]
            put_call = ""
            if prof["opt"] > 0 and rng.random() < prof["opt"]:
                put_call = "Put" if rng.random() < 0.5 else "Call"
                key = f"{u['ticker']}:{put_call}"
            target_value = aum * u["size"] / tot
            if put_call:
                target_value *= 0.2  # options sleeves are a fraction of a cash position
            shares = max(100, int(target_value / prices[u["ticker"]][0]))
            book[key] = {"it": u, "shares": shares, "put_call": put_call}

        for qi, (period, fdate) in enumerate(QUARTERS):
            if qi > 0:
                # organic turnover
                for key in list(book):
                    r = rng.random()
                    if r < prof["turnover"] * 0.25:
                        del book[key]
                    elif r < prof["turnover"] * 0.6:
                        book[key]["shares"] = int(book[key]["shares"] * rng.uniform(0.4, 0.9))
                    elif r < prof["turnover"]:
                        book[key]["shares"] = int(book[key]["shares"] * rng.uniform(1.1, 1.8))
                n_new = int(prof["n"] * prof["turnover"] * 0.25)
                candidates = [i for i in issuers if i["ticker"] not in {v["it"]["ticker"] for v in book.values()}]
                for c in rng.choice(candidates, size=min(n_new, len(candidates)), replace=False):
                    tv = aum * rng.uniform(0.002, 0.02)
                    book[c["ticker"]] = {"it": c, "shares": max(100, int(tv / prices[c["ticker"]][qi])), "put_call": ""}
                # scripted events
                for k, v in EVENTS.get((m.cik, qi), {}).items():
                    if k == "new":
                        for tk in v:
                            base, _, pc = tk.partition(":")
                            if base not in by_ticker:
                                continue
                            tv = aum * rng.uniform(0.01, 0.06)
                            book[tk] = {"it": by_ticker[base], "shares": max(100, int(tv / prices[base][qi])), "put_call": pc}
                    elif k in book:
                        if v == 0.0:
                            del book[k]
                        else:
                            book[k]["shares"] = int(book[k]["shares"] * v)
            rows = []
            for key, pos in book.items():
                it = pos["it"]
                px = prices[it["ticker"]][qi]
                rows.append(
                    {
                        "issuer": it["issuer"], "title_of_class": it["title_of_class"], "cusip": it["cusip"],
                        "value": pos["shares"] * px, "shares": pos["shares"],
                        "sh_prn": "PRN" if it["asset_type"] in ("Convertible Debt", "Corporate Debt") else "SH",
                        "put_call": pos["put_call"],
                    }
                )
            rows.sort(key=lambda r: r["issuer"])
            acc = f"0000000000-{period.year % 100:02d}-{int(m.cik) % 100000:05d}{qi}"
            acc = f"{m.cik10}-{str(period.year)[2:]}-{qi:02d}{int(m.cik) % 1000:03d}"
            ref = FilingRef(m.cik, m.name, acc, "13F-HR", fdate.isoformat(), period.isoformat(), "primary_doc.xml", source="sample")
            dest = settings.raw_dir / m.cik / ref.folder_name
            _write_filing(dest, m, ref, rows)
            out.append(dest)
    return out
