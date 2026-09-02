"""Intelligent classification of 13F holdings and managers.

Three layers, applied in order, each with a confidence score:
  1. Security master lookup (config/issuers.json) by CUSIP  -> confidence 1.0
  2. Deterministic rules on title-of-class / put-call / issuer keywords -> 0.6-0.9
  3. Naive-Bayes text model over issuer-name tokens, trained on layers 1-2 -> 0.3-0.7
Manager types come from the config (declared) and from a portfolio-fingerprint
model (inferred), so the two can be compared.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SECTORS = [
    "Information Technology", "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Health Care", "Financials", "Energy", "Industrials", "Materials", "Utilities", "Real Estate",
]

# ---------------------------------------------------------------- asset type
_ETF_ISSUER = re.compile(r"\b(ISHARES|SPDR|VANGUARD (INDEX|INTL|WORLD|BD|SCOTTSDALE)|INVESCO (QQQ|EXCHANGE)|SELECT SECTOR|PROSHARES|DIREXION|ARK ETF|WISDOMTREE|SCHWAB STRATEGIC|GLOBAL X|VANECK|FIRST TR EXCHANGE|ETF|INDEX FDS?|INDEX FUND)\b")
_ETF_CLASS = re.compile(r"\b(ETF|TR UNIT|UNIT SER|INDEX|SHS BEN INT|GOLD SHS|SILVER|UNITS?)\b")
_ADR = re.compile(r"\b(ADR|ADS|SPON ADR|SPONSORED ADR|SPONSORED ADS|SPON ADS|NY REG|AMERICAN DEP)\b")
_PFD = re.compile(r"\b(PFD|PREF|PREFERRED|DEP SHS|DEPOSITARY)\b")
_DEBT = re.compile(r"\b(NOTE|NOTES|DEBENTURE|DEB|BOND|BD|SR NT|SUB NT|\d+\.\d{2,3}%)\b")
_CONV = re.compile(r"\b(CONV|CONVERTIBLE|CV)\b")
_WARRANT = re.compile(r"\b(WT|WTS|WARRANT|WARRANTS)\b")
_RIGHT = re.compile(r"\b(RT|RTS|RIGHT|RIGHTS)\b")
_UNIT = re.compile(r"\bUNIT\b")
_REIT = re.compile(r"\b(REIT|REALTY|PPTY|PROPERTIES|PROPERTY TR|REAL ESTATE|APARTMENT|STORAGE|TOWER|INDUSTRIAL TR)\b")
_COMMON = re.compile(r"\b(COM|COMMON|ORD|SHS|SHARES|CL [A-C]|CLASS [A-C]|CAP STK|STK)\b")
_SPAC = re.compile(r"\b(ACQUISITION|ACQ CORP|SPAC)\b")


def classify_asset_type(title_of_class: str, issuer: str, put_call: str, sh_prn: str) -> tuple[str, float]:
    t = (title_of_class or "").upper()
    i = (issuer or "").upper()
    if put_call in ("Put", "Call"):
        return f"{put_call} Option", 1.0
    if _WARRANT.search(t):
        return "Warrant", 0.9
    if _RIGHT.search(t):
        return "Right", 0.85
    if _PFD.search(t):
        return "Preferred Stock", 0.9
    if _DEBT.search(t) or sh_prn == "PRN":
        return ("Convertible Debt", 0.9) if _CONV.search(t) else ("Corporate Debt", 0.85)
    if _ETF_ISSUER.search(i) or (_ETF_CLASS.search(t) and "TR" in i.split()):
        return "ETF", 0.9
    if _ADR.search(t):
        return "ADR", 0.95
    if _REIT.search(i):
        return "REIT", 0.75
    if _UNIT.search(t) and _SPAC.search(i):
        return "SPAC Unit", 0.8
    if _UNIT.search(t):
        return "Unit", 0.7
    if _COMMON.search(t):
        return "Common Stock", 0.85
    return "Common Stock", 0.5


# ------------------------------------------------------------------- sector
_SECTOR_KEYWORDS: list[tuple[str, re.Pattern]] = [
    ("Financials", re.compile(r"\b(BANK|BANCORP|BANCSHARES|FINL|FINANCIAL|CAPITAL|CAP|ACQUISITION|INSURANCE|ASSURANCE|TRUST CO|HLDGS? BANC|BROKERAGE|SECURITIES|ASSET MGMT|INVESTMENT|CREDIT|MORTGAGE|PAYMENT|EXCHANGE)\b")),
    ("Health Care", re.compile(r"\b(PHARMA|PHARMACEUTICAL|THERAPEUTICS|BIO|BIOSCIENCE|BIOTECH|MEDICAL|HEALTH|HEALTHCARE|SURGICAL|GENOMIC|DIAGNOSTIC|LABS?|LABORATORIES|CLINIC|DENTAL|DEVICES?)\b")),
    ("Energy", re.compile(r"\b(OIL|GAS|PETE|PETROLEUM|ENERGY|RESOURCES|DRILLING|OFFSHORE|PIPELINE|MIDSTREAM|REFIN|COAL|EXPLORATION)\b")),
    ("Information Technology", re.compile(r"\b(SOFTWARE|SEMICONDUCTOR|SEMICOND|MICRO|DEVICES|TECHNOLOG|TECH|SYSTEMS|COMPUTER|DATA|CLOUD|DIGITAL|NETWORKS?|CYBER|ELECTRONICS?|INSTRUMENTS)\b")),
    ("Communication Services", re.compile(r"\b(COMMUNICATIONS?|TELECOM|WIRELESS|MEDIA|BROADCAST|ENTERTAINMENT|INTERACTIVE|GAMES?|GAMING|PUBLISHING|CABLE|STREAMING)\b")),
    ("Consumer Discretionary", re.compile(r"\b(MOTORS?|AUTO|AUTOMOTIVE|RETAIL|STORES?|RESTAURANTS?|APPAREL|BRANDS|HOTELS?|RESORTS?|CRUISE|LEISURE|HOME|FURNITURE|TOYS?|LUXURY|TRAVEL|BOOKING)\b")),
    ("Consumer Staples", re.compile(r"\b(FOODS?|BEVERAGE|BREWING|TOBACCO|GROCERY|HOUSEHOLD|PRODUCTS|CONSUMER|WHOLESALE|WHSL|COLA|SNACK|DAIRY|NUTRITION)\b")),
    ("Industrials", re.compile(r"\b(INDUSTRIES|INDL|AEROSPACE|DEFENSE|AIRLINES?|AIR LINES|RAILROAD|RAIL|MACHINERY|MACHINES?|ENGINEERING|CONSTRUCTION|LOGISTICS|FREIGHT|PARCEL|TRANSPORT|ELECTRIC EQUIP|TOOLS)\b")),
    ("Materials", re.compile(r"\b(CHEMICAL|CHEM|MINING|MINERALS?|METALS?|STEEL|GOLD|COPPER|LITHIUM|PAPER|PACKAGING|CEMENT|MATERIALS?|GASES)\b")),
    ("Utilities", re.compile(r"\b(UTILITIES|UTIL|ELECTRIC|POWER|WATER|ENERGY CORP|GAS & ELEC|EDISON)\b")),
    ("Real Estate", re.compile(r"\b(REIT|REALTY|PPTY|PROPERTIES|PROPERTY|REAL ESTATE|APARTMENT|STORAGE|TOWER|LAND)\b")),
]

_TOKEN = re.compile(r"[A-Z0-9&]+")
_STOP = {"INC", "CORP", "CO", "LTD", "PLC", "LLC", "LP", "SA", "NV", "AG", "SE", "DEL", "NEW", "HLDGS", "HOLDINGS", "GROUP", "THE", "&", "COS", "COM"}


def _tokens(name: str) -> list[str]:
    return [t for t in _TOKEN.findall((name or "").upper()) if t not in _STOP and len(t) > 1]


class NaiveBayesSector:
    """Multinomial NB over issuer-name tokens (pure numpy; no sklearn dependency)."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.classes: list[str] = []
        self.class_log_prior: dict[str, float] = {}
        self.token_counts: dict[str, Counter] = {}
        self.class_totals: dict[str, int] = {}
        self.vocab: set[str] = set()

    def fit(self, names: list[str], labels: list[str]) -> "NaiveBayesSector":
        cc = Counter(labels)
        self.classes = sorted(cc)
        n = len(labels)
        self.class_log_prior = {c: math.log(cc[c] / n) for c in self.classes}
        self.token_counts = {c: Counter() for c in self.classes}
        for name, lab in zip(names, labels):
            self.token_counts[lab].update(_tokens(name))
        self.vocab = set().union(*(set(c) for c in self.token_counts.values()))
        self.class_totals = {c: sum(self.token_counts[c].values()) for c in self.classes}
        return self

    def predict(self, name: str) -> tuple[str, float]:
        toks = _tokens(name)
        if not toks or not self.classes:
            return "Unknown", 0.0
        V = len(self.vocab)
        scores = {}
        for c in self.classes:
            s = self.class_log_prior[c]
            tot = self.class_totals[c] + self.alpha * V
            for t in toks:
                s += math.log((self.token_counts[c][t] + self.alpha) / tot)
            scores[c] = s
        m = max(scores.values())
        probs = {c: math.exp(v - m) for c, v in scores.items()}
        z = sum(probs.values())
        best = max(probs, key=probs.get)
        return best, probs[best] / z


class SecurityClassifier:
    def __init__(self, issuers_file: Path):
        data = json.loads(Path(issuers_file).read_text())["issuers"]
        self.master = {i["cusip"]: i for i in data}
        self.master_by_cusip6 = defaultdict(list)
        for i in data:
            self.master_by_cusip6[i["cusip"][:6]].append(i)
        self.nb = NaiveBayesSector().fit([i["issuer"] for i in data], [i["sector"] for i in data])

    # -- single security ---------------------------------------------------
    def classify(self, cusip: str, issuer: str, title_of_class: str, put_call: str, sh_prn: str) -> dict:
        asset_type, at_conf = classify_asset_type(title_of_class, issuer, put_call, sh_prn)
        ref = self.master.get(cusip)
        if ref is None and cusip[:6] in self.master_by_cusip6:
            ref = self.master_by_cusip6[cusip[:6]][0]  # same issuer, other class
        if ref is not None:
            out = dict(
                ticker=ref["ticker"].split(".")[0], sector=ref["sector"], industry=ref["industry"],
                country=ref["country"], sector_conf=1.0, sector_method="master",
            )
            if put_call not in ("Put", "Call") and at_conf < 0.9:
                asset_type, at_conf = ref["asset_type"], 1.0
        else:
            sector, conf, method = self._sector_from_text(issuer, asset_type, title_of_class)
            out = dict(ticker="", sector=sector, industry="", country="", sector_conf=conf, sector_method=method)
        if asset_type == "ETF" and not out["sector"].startswith("ETF"):
            out["sector"] = self._etf_bucket(issuer, title_of_class)
            out["sector_conf"], out["sector_method"] = 0.8, "rule"
        out.update(asset_type=asset_type, asset_type_conf=at_conf, underlying_asset=self._underlying(asset_type, out["sector"]))
        return out

    @staticmethod
    def _etf_bucket(issuer: str, toc: str) -> str:
        s = f"{issuer} {toc}".upper()
        if re.search(r"\b(GOLD|SILVER|OIL|COMMODITY|GSCI|METALS?)\b", s):
            return "ETF - Commodity"
        if re.search(r"\b(BD|BOND|TREAS|TRS|CORP|CREDIT|HI YD|HIGH YIELD|TIPS|AGG|MUNI)\b", s):
            return "ETF - Fixed Income"
        if re.search(r"\b(EMERG|EAFE|INTL|INTERNATIONAL|CHINA|JAPAN|EUROPE|EX-US|EX US|WORLD|FTSE)\b", s):
            return "ETF - International Equity"
        if re.search(r"\b(SELECT SECTOR|SBI INT|FINL|ENERGY|TECH|HEALTH|UTIL|INDUS|MATER|STAPLE|DISCRET|REAL ESTATE|SEMICOND)\b", s):
            return "ETF - Sector"
        if re.search(r"\b(ARK|INNOVATION|THEMATIC|ROBOTICS|CLEAN|CLOUD|AI )\b", s):
            return "ETF - Thematic"
        return "ETF - Broad Equity"

    def _sector_from_text(self, issuer: str, asset_type: str, toc: str = "") -> tuple[str, float, str]:
        if asset_type == "ETF":
            return self._etf_bucket(issuer, toc), 0.8, "rule"
        if asset_type == "REIT":
            return "Real Estate", 0.9, "rule"
        name = (issuer or "").upper()
        hits = [(sec, len(pat.findall(name))) for sec, pat in _SECTOR_KEYWORDS]
        hits = [h for h in hits if h[1] > 0]
        nb_sec, nb_conf = self.nb.predict(name)
        if hits:
            hits.sort(key=lambda h: -h[1])
            sec = hits[0][0]
            conf = 0.7 if len(hits) == 1 else 0.6
            if nb_sec == sec:
                conf = min(0.9, conf + 0.2)
            return sec, conf, "keyword"
        if nb_conf >= 0.45:
            return nb_sec, round(0.3 + 0.4 * nb_conf, 2), "naive_bayes"
        return "Unclassified", 0.0, "none"

    @staticmethod
    def _underlying(asset_type: str, sector: str) -> str:
        """Coarse exposure bucket used for the asset-mix views."""
        if asset_type in ("Common Stock", "ADR", "REIT", "SPAC Unit", "Unit"):
            return "Equity"
        if asset_type == "ETF":
            if sector == "ETF - Fixed Income":
                return "Fixed Income (ETF)"
            if sector == "ETF - Commodity":
                return "Commodity (ETF)"
            return "Equity (ETF)"
        if asset_type in ("Put Option", "Call Option"):
            return "Options"
        if asset_type in ("Convertible Debt", "Corporate Debt"):
            return "Debt"
        if asset_type == "Preferred Stock":
            return "Preferred"
        if asset_type in ("Warrant", "Right"):
            return "Warrants/Rights"
        return "Other"

    # -- dataframe ----------------------------------------------------------
    def classify_frame(self, holdings: pd.DataFrame) -> pd.DataFrame:
        keys = holdings[["cusip", "issuer", "title_of_class", "put_call", "sh_prn"]].drop_duplicates()
        recs = []
        for r in keys.itertuples(index=False):
            d = self.classify(r.cusip, r.issuer, r.title_of_class, r.put_call, r.sh_prn)
            d.update(cusip=r.cusip, issuer=r.issuer, title_of_class=r.title_of_class, put_call=r.put_call, sh_prn=r.sh_prn)
            recs.append(d)
        lut = pd.DataFrame(recs)
        lut["display_name"] = [
            display_name(r.issuer, r.title_of_class, r.asset_type, r.ticker) for r in lut.itertuples()
        ]
        return holdings.merge(lut, on=["cusip", "issuer", "title_of_class", "put_call", "sh_prn"], how="left")


def display_name(issuer: str, title_of_class: str, asset_type: str, ticker: str = "") -> str:
    """Human-readable security name. ETFs and non-common instruments need the class
    to be distinguishable (every iShares fund is filed as 'ISHARES TR')."""
    issuer = (issuer or "").strip()
    toc = (title_of_class or "").strip()
    if asset_type == "ETF" or asset_type in ("Preferred Stock", "Convertible Debt", "Corporate Debt", "Warrant", "Right", "Unit", "SPAC Unit"):
        name = f"{issuer} {toc}".strip()
    else:
        name = issuer
    return f"{name} ({ticker})" if ticker else name


# ------------------------------------------------------------ manager type
def manager_fingerprint(h: pd.DataFrame) -> pd.DataFrame:
    """Per manager-period portfolio statistics."""
    g = h.groupby(["cik", "manager", "period"])
    rows = []
    for (cik, mgr, per), d in g:
        tot = d["value_usd"].sum()
        w = d["value_usd"] / tot if tot else d["value_usd"] * 0
        opt = d.loc[d["asset_type"].isin(["Put Option", "Call Option"]), "value_usd"].sum() / tot if tot else 0
        put = d.loc[d["asset_type"] == "Put Option", "value_usd"].sum() / tot if tot else 0
        etf = d.loc[d["asset_type"] == "ETF", "value_usd"].sum() / tot if tot else 0
        debt = d.loc[d["underlying_asset"].isin(["Debt", "Preferred", "Warrants/Rights"]), "value_usd"].sum() / tot if tot else 0
        top10 = w.nlargest(10).sum()
        hhi = float((w ** 2).sum())
        rows.append(dict(cik=cik, manager=mgr, period=per, total_value=tot, n_positions=len(d), top10_weight=top10,
                         hhi=hhi, effective_n=(1 / hhi if hhi else 0), options_share=opt, put_share=put,
                         etf_share=etf, credit_share=debt))
    return pd.DataFrame(rows)


def infer_manager_type(fp: pd.DataFrame, turnover: pd.Series | None = None) -> pd.DataFrame:
    """Rule-based fingerprint -> inferred manager archetype with a confidence."""
    out = []
    for r in fp.itertuples(index=False):
        to = float(turnover.get((r.cik, r.period), np.nan)) if turnover is not None else np.nan
        label, conf, why = "Long-only Stock Picker", 0.5, []
        if r.etf_share > 0.25:
            label, conf = "Macro / Allocator (ETF-heavy)", 0.8
            why.append(f"ETF {r.etf_share:.0%}")
        elif r.options_share > 0.2:
            label, conf = "Multi-strategy / Options-heavy", 0.85
            why.append(f"options {r.options_share:.0%}")
        elif r.n_positions >= 120 and r.top10_weight < 0.35:
            label, conf = ("Index / Broad Asset Manager", 0.8) if (np.isnan(to) or to < 0.15) else ("Quant / Systematic", 0.8)
            why.append(f"{r.n_positions} positions, top-10 {r.top10_weight:.0%}")
        elif r.n_positions <= 20 and r.top10_weight > 0.7:
            label, conf = "Concentrated / Activist", 0.85
            why.append(f"{r.n_positions} positions, top-10 {r.top10_weight:.0%}")
        elif r.credit_share > 0.15:
            label, conf = "Credit / Special Situations", 0.7
            why.append(f"credit {r.credit_share:.0%}")
        elif r.top10_weight > 0.5:
            label, conf = "Concentrated Value", 0.7
            why.append(f"top-10 {r.top10_weight:.0%}")
        if not np.isnan(to):
            why.append(f"turnover {to:.0%}")
        out.append(dict(cik=r.cik, period=r.period, inferred_type=label, inferred_conf=conf, inferred_reason="; ".join(why)))
    return fp.merge(pd.DataFrame(out), on=["cik", "period"])
