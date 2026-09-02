"""Grow the security master (config/issuers.json) from the securities the universe actually holds.

Pipeline per unknown CUSIP (ranked by dollar value held):
  1. CUSIP -> ticker / name / security type      OpenFIGI mapping API (exact identifier match)
  2. ticker -> CIK                               SEC company_tickers_exchange.json
     (fallback: normalised issuer name -> CIK)   same file, token matching
  3. CIK -> SIC code + description               SEC submissions API
  4. SIC -> GICS-style sector                    range table below, with description overrides

Entries written this way carry ``source`` and ``sic`` so they can be audited or replaced by vendor data
later; the classifier treats them like any other master row (sector_method = "master").
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd
import requests

from .classify import SecurityClassifier, classify_asset_type

log = logging.getLogger(__name__)

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
SOURCE_TAG = "openfigi+sec_sic"


@dataclass(frozen=True)
class EnrichConfig:
    top_n: int = 2500                 # unknown CUSIPs to resolve, by value held
    openfigi_batch: int = 10          # jobs per request (limit without an API key)
    openfigi_sleep: float = 2.6       # seconds between OpenFIGI requests (25 req/min without key)
    sec_sleep: float = 0.13           # stay under the SEC's 10 req/s
    user_agent: str = "13F-Tracker research contact@example.com"
    openfigi_api_key: str = ""

    def __post_init__(self) -> None:
        if self.top_n < 1 or not 1 <= self.openfigi_batch <= 100:
            raise ValueError("top_n must be >= 1 and openfigi_batch within 1..100")


# ---------------------------------------------------------------- SIC -> sector
# (lo, hi, sector) ranges on the 4-digit SIC code; first match wins, more specific ranges first.
_SIC_RANGES: list[tuple[int, int, str]] = [
    (1311, 1389, "Energy"), (2911, 2999, "Energy"), (1000, 1099, "Materials"), (1200, 1299, "Energy"), (1400, 1499, "Materials"),
    (100, 999, "Consumer Staples"), (1500, 1799, "Industrials"),
    (2000, 2199, "Consumer Staples"), (2200, 2399, "Consumer Discretionary"), (2400, 2499, "Materials"), (2500, 2599, "Consumer Discretionary"),
    (2600, 2699, "Materials"), (2700, 2799, "Communication Services"),
    (2833, 2836, "Health Care"), (2840, 2844, "Consumer Staples"), (2800, 2899, "Materials"),
    (3011, 3011, "Consumer Discretionary"), (3000, 3099, "Materials"), (3100, 3199, "Consumer Discretionary"), (3200, 3299, "Materials"),
    (3300, 3399, "Materials"), (3400, 3499, "Industrials"),
    (3570, 3579, "Information Technology"), (3500, 3599, "Industrials"),
    (3630, 3639, "Consumer Discretionary"), (3651, 3652, "Consumer Discretionary"), (3660, 3699, "Information Technology"), (3600, 3629, "Industrials"),
    (3559, 3559, "Industrials"),
    (3711, 3716, "Consumer Discretionary"), (3751, 3751, "Consumer Discretionary"), (3700, 3799, "Industrials"),
    (3841, 3851, "Health Care"), (3826, 3826, "Health Care"), (3825, 3825, "Information Technology"), (3827, 3829, "Information Technology"),
    (3861, 3861, "Information Technology"), (3873, 3873, "Consumer Discretionary"), (3800, 3899, "Industrials"),
    (3942, 3949, "Consumer Discretionary"), (3900, 3999, "Industrials"),
    (4812, 4899, "Communication Services"), (4000, 4799, "Industrials"), (4950, 4959, "Industrials"), (4900, 4999, "Utilities"),
    (5122, 5122, "Health Care"), (5140, 5149, "Consumer Staples"), (5180, 5182, "Consumer Staples"), (5000, 5199, "Industrials"),
    (5411, 5411, "Consumer Staples"), (5912, 5912, "Consumer Staples"), (5200, 5999, "Consumer Discretionary"),
    (6500, 6553, "Real Estate"), (6798, 6798, "Real Estate"), (6000, 6799, "Financials"),
    (7011, 7011, "Consumer Discretionary"), (7200, 7299, "Consumer Discretionary"), (7311, 7319, "Communication Services"),
    (7370, 7379, "Information Technology"), (7300, 7399, "Industrials"), (7500, 7699, "Consumer Discretionary"),
    (7812, 7841, "Communication Services"), (7900, 7999, "Communication Services"),
    (8000, 8099, "Health Care"), (8731, 8731, "Health Care"), (8100, 8999, "Industrials"),
]
_DESC_OVERRIDES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"REAL ESTATE INVESTMENT TRUST", re.I), "Real Estate"),
    (re.compile(r"PHARMACEUTICAL|BIOLOGICAL|MEDICAL|HEALTH|HOSPITAL|SURGICAL|DIAGNOSTIC", re.I), "Health Care"),
    (re.compile(r"SEMICONDUCTOR|PREPACKAGED SOFTWARE|COMPUTER PROGRAMMING|COMPUTER STORAGE|COMPUTER COMMUNICATIONS|COMPUTER PERIPHERAL", re.I), "Information Technology"),
    (re.compile(r"CRUDE PETROLEUM|NATURAL GAS|OIL & GAS|PETROLEUM REFINING", re.I), "Energy"),
    (re.compile(r"GOLD|SILVER|METAL MINING", re.I), "Materials"),
    (re.compile(r"TELEPHONE|RADIOTELEPHONE|CABLE|TELEVISION|BROADCASTING|MOTION PICTURE", re.I), "Communication Services"),
    (re.compile(r"ELECTRIC SERVICES|GAS TRANSMISSION|WATER SUPPLY|GAS DISTRIBUTION", re.I), "Utilities"),
]


# Large names whose SIC-derived sector contradicts their GICS sector (SIC predates today's industry structure).
TICKER_SECTOR_OVERRIDES: dict[str, str] = {
    "LRCX": "Information Technology", "KLAC": "Information Technology", "TER": "Information Technology", "COHR": "Information Technology",
    "CIEN": "Information Technology", "GLW": "Information Technology", "ENTG": "Information Technology", "AMKR": "Information Technology",
    "ONTO": "Information Technology", "MKSI": "Information Technology", "KEYS": "Information Technology", "TDY": "Information Technology",
    "GEV": "Industrials", "HWM": "Industrials", "ETN": "Industrials", "EMR": "Industrials", "ROK": "Industrials", "AME": "Industrials",
    "SHW": "Materials", "PPG": "Materials", "ECL": "Materials", "IFF": "Materials", "CE": "Materials",
    "GOOGL": "Communication Services", "GOOG": "Communication Services", "META": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "TMUS": "Communication Services", "EA": "Communication Services", "TTWO": "Communication Services",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "BKNG": "Consumer Discretionary", "ABNB": "Consumer Discretionary",
    "V": "Financials", "MA": "Financials", "PYPL": "Financials", "FI": "Financials", "FIS": "Financials", "GPN": "Financials",
    "MSCI": "Financials", "SPGI": "Financials", "MCO": "Financials", "ICE": "Financials", "CME": "Financials", "NDAQ": "Financials",
    "CTAS": "Industrials", "PAYX": "Industrials", "ADP": "Industrials", "VRSK": "Industrials", "BR": "Industrials", "UBER": "Industrials",
    "GEHC": "Health Care", "IQV": "Health Care", "LH": "Health Care", "DGX": "Health Care",
}


def sic_to_sector(sic: Optional[int], description: str = "") -> str:
    """GICS-style sector for a SIC code; the description resolves ambiguous ranges. Unclassified when unknown."""
    if description:
        for pat, sector in _DESC_OVERRIDES:
            if pat.search(description):
                return sector
    if sic is None or sic <= 0:
        return "Unclassified"
    for lo, hi, sector in _SIC_RANGES:
        if lo <= sic <= hi:
            return sector
    return "Unclassified"


# ---------------------------------------------------------------- name matching
_ABBREV = {
    "MATLS": "MATERIALS", "HLDGS": "HOLDINGS", "HLDNGS": "HOLDINGS", "HLDG": "HOLDINGS", "INTL": "INTERNATIONAL", "FINL": "FINANCIAL",
    "SVCS": "SERVICES", "SVC": "SERVICES", "MFG": "MANUFACTURING", "PWR": "POWER", "GRP": "GROUP", "SYS": "SYSTEMS", "RES": "RESOURCES",
    "ENTMT": "ENTERTAINMENT", "PHARMA": "PHARMACEUTICALS", "PHARMACEUTICAL": "PHARMACEUTICALS", "CMNTY": "COMMUNITY", "BANCORP": "BANCORP",
    "TECH": "TECHNOLOGY", "TECHNOLOGIES": "TECHNOLOGY", "COMMUNICATIONS": "COMMUNICATION", "LABS": "LABORATORIES", "PPTYS": "PROPERTIES",
    "PPTY": "PROPERTY", "INDS": "INDUSTRIES", "IND": "INDUSTRIES", "ELEC": "ELECTRIC", "MTRS": "MOTORS", "PRODS": "PRODUCTS", "PROD": "PRODUCTS",
    "NATL": "NATIONAL", "AMERN": "AMERICAN", "GENL": "GENERAL", "CORPORATION": "CORP", "INCORPORATED": "INC", "COMPANY": "CO", "LIMITED": "LTD",
    "TR": "TRUST", "ENERGY": "ENERGY", "PARTNERS": "PARTNERS", "&": "AND",
}
_DROP = {"INC", "CORP", "CO", "LTD", "PLC", "PL", "LT", "LLC", "LP", "L.P.", "NV", "SA", "AG", "SE", "AB", "AS", "OYJ", "SPA", "CL", "CLASS", "NEW", "DEL",
         "COM", "ORD", "SHS", "ADR", "ADS", "SPONSORED", "SPON", "THE", "OF", "HOLDINGS", "GROUP", "TRUST", "A", "B", "C", "I", "II", "III",
         "SH", "SHARE", "SHARES", "USD", "US", "N", "V", "DE", "MD", "MN", "PA", "OH", "CT", "NJ", "NY", "TX", "GA", "FL", "CA", "WA", "MA", "IL"}


def normalize_name(name: str) -> tuple[str, ...]:
    """Uppercase tokens with abbreviations expanded and legal/class noise removed. Empty tuple for empty input."""
    s = re.sub(r"/[A-Z]{2,3}/?$", " ", (name or "").upper())  # trailing state markers: /DE/, /MN
    s = re.sub(r"[^A-Z0-9& ]+", " ", s)
    toks = [_ABBREV.get(t, t) for t in s.split()]
    toks = [t for t in toks if t not in _DROP]
    return tuple(toks)


class SecCompanyIndex:
    """SEC listed-company file: ticker -> (cik, name) and normalised name -> candidates."""

    def __init__(self, rows: Iterable[tuple[int, str, str, str]]):
        self.by_ticker: dict[str, tuple[int, str]] = {}
        self.by_name: dict[tuple[str, ...], list[tuple[int, str, str]]] = defaultdict(list)
        self.by_first: dict[str, list[tuple[tuple[str, ...], int, str, str]]] = defaultdict(list)
        for cik, name, ticker, _exchange in rows:
            t = (ticker or "").upper()
            if t and t not in self.by_ticker:
                self.by_ticker[t] = (int(cik), name)
            key = normalize_name(name)
            if key:
                self.by_name[key].append((int(cik), t, name))
                self.by_first[key[0]].append((key, int(cik), t, name))

    @classmethod
    def fetch(cls, user_agent: str) -> "SecCompanyIndex":
        data = requests.get(SEC_TICKERS_URL, headers={"User-Agent": user_agent}, timeout=60).json()
        return cls(tuple(r) for r in data["data"])

    def match_ticker(self, ticker: str) -> Optional[tuple[int, str]]:
        return self.by_ticker.get((ticker or "").upper().replace("/", "-"))

    def match_name(self, issuer: str, min_score: float = 0.6) -> Optional[tuple[int, str, str]]:
        """(cik, ticker, name) by normalised token match: exact key first, then best Jaccard among names sharing the first token."""
        key = normalize_name(issuer)
        if not key:
            return None
        if key in self.by_name:
            return self.by_name[key][0]
        best, best_score, runner = None, 0.0, 0.0
        ks = set(key)
        for ck, cik, t, name in self.by_first.get(key[0], []):
            cs = set(ck)
            score = len(ks & cs) / len(ks | cs)
            if score > best_score:
                best, runner, best_score = (cik, t, name), best_score, score
            elif score > runner:
                runner = score
        if best and best_score >= min_score and best_score > runner:
            return best
        return None


# ---------------------------------------------------------------- external lookups
def openfigi_lookup(cusips: list[str], cfg: EnrichConfig, post: Callable = requests.post) -> dict[str, dict]:
    """CUSIP -> {ticker, name, securityType, securityType2, marketSector, exchCode}; missing keys were not found."""
    out: dict[str, dict] = {}
    headers = {"Content-Type": "application/json"}
    if cfg.openfigi_api_key:
        headers["X-OPENFIGI-APIKEY"] = cfg.openfigi_api_key
    for i in range(0, len(cusips), cfg.openfigi_batch):
        batch = cusips[i:i + cfg.openfigi_batch]
        jobs = [{"idType": "ID_CUSIP", "idValue": c} for c in batch]
        for attempt in range(4):
            r = post(OPENFIGI_URL, json=jobs, headers=headers, timeout=60)
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        else:
            log.warning("OpenFIGI kept rate-limiting; stopping after %d CUSIPs", i)
            break
        for c, res in zip(batch, r.json()):
            data = res.get("data") or []
            if data:
                d = data[0]
                out[c] = {k: d.get(k) for k in ("ticker", "name", "securityType", "securityType2", "marketSector", "exchCode")}
        if i + cfg.openfigi_batch < len(cusips):
            time.sleep(cfg.openfigi_sleep)
    return out


def sec_sic(cik: int, cfg: EnrichConfig, get: Callable = requests.get, retries: int = 4) -> tuple[Optional[int], str]:
    """SIC code and description for a CIK. Transient network errors and 429/5xx are retried with backoff;
    a CIK that still cannot be read yields (None, "") instead of aborting the whole run."""
    r = None
    for attempt in range(retries):
        try:
            r = get(SEC_SUBMISSIONS_URL.format(cik=cik), headers={"User-Agent": cfg.user_agent}, timeout=60)
        except requests.RequestException as exc:
            log.warning("SEC submissions %s: %s (attempt %d)", cik, exc, attempt + 1)
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(5 * (attempt + 1))
            continue
        break
    if r is None or r.status_code != 200:
        return None, ""
    d = r.json()
    sic = d.get("sic")
    try:
        sic = int(sic) if sic not in (None, "") else None
    except ValueError:
        sic = None
    return sic, d.get("sicDescription") or ""


# ---------------------------------------------------------------- orchestration
_FIGI_ASSET = {"ETP": "ETF", "REIT": "REIT", "ADR": "ADR", "Depositary Receipt": "ADR", "Preferred": "Preferred Stock", "Closed-End Fund": "ETF", "Mutual Fund": "ETF"}


def select_targets(holdings: pd.DataFrame, master_cusips: set[str], top_n: int) -> pd.DataFrame:
    """Unknown CUSIPs ranked by total value held across all periods, with the most common issuer name / class."""
    h = holdings[~holdings["cusip"].isin(master_cusips) & (holdings["cusip"].str.len() == 9)]
    h = h[~h["cusip"].str[:6].isin({c[:6] for c in master_cusips})]
    if h.empty:
        return pd.DataFrame(columns=["cusip", "issuer", "title_of_class", "value_usd"])
    val = h.groupby("cusip")["value_usd"].sum()
    names = h.groupby("cusip").agg(issuer=("issuer", lambda s: s.mode().iat[0]), title_of_class=("title_of_class", lambda s: s.mode().iat[0]))
    t = names.join(val).reset_index().sort_values("value_usd", ascending=False)
    return t.head(top_n)


def build_entries(targets: pd.DataFrame, figi: dict[str, dict], sec_index: SecCompanyIndex, sic_of: Callable[[int], tuple[Optional[int], str]],
                  sic_cache: dict[str, dict]) -> tuple[list[dict], dict]:
    """Resolve each target to a master entry. ``sic_cache`` (cik -> {sic, desc}) is updated in place."""
    entries: list[dict] = []
    stats = dict(targets=int(len(targets)), figi_hits=0, ticker_matches=0, name_matches=0, sic_found=0, written=0, no_sector=0)
    for r in targets.itertuples(index=False):
        f = figi.get(r.cusip)
        cik = ticker = ""
        sec_name = ""
        if f:
            stats["figi_hits"] += 1
            hit = sec_index.match_ticker(f.get("ticker") or "")
            if hit:
                cik, sec_name = hit
                ticker = (f.get("ticker") or "").upper()
                stats["ticker_matches"] += 1
        if not cik:
            hit = sec_index.match_name(r.issuer)
            if hit:
                cik, ticker, sec_name = hit
                stats["name_matches"] += 1
        if not cik:
            continue
        ck = str(cik)
        if ck not in sic_cache:
            sic, desc = sic_of(int(cik))
            sic_cache[ck] = {"sic": sic, "desc": desc}
        sic, desc = sic_cache[ck]["sic"], sic_cache[ck]["desc"]
        if sic:
            stats["sic_found"] += 1
        asset_type, _ = classify_asset_type(r.title_of_class, r.issuer, "", "SH")
        if f and f.get("securityType") in _FIGI_ASSET:
            asset_type = _FIGI_ASSET[f["securityType"]]
        if asset_type == "ETF":
            sector = SecurityClassifier._etf_bucket(r.issuer, r.title_of_class)
        elif asset_type == "REIT":
            sector = "Real Estate"
        else:
            sector = sic_to_sector(sic, desc)
        sector = TICKER_SECTOR_OVERRIDES.get((ticker or "").upper(), sector) if asset_type not in ("ETF", "REIT") else sector
        if sector == "Unclassified":
            stats["no_sector"] += 1
            continue
        entries.append(dict(
            ticker=ticker or (f.get("ticker") if f else "") or "", cusip=r.cusip, issuer=r.issuer, title_of_class=r.title_of_class, asset_type=asset_type,
            sector=sector, industry=(desc or "").title(), country=("US" if (f or {}).get("exchCode") == "US" else ""),
            source=SOURCE_TAG, sic=sic, cik=int(cik), sec_name=sec_name,
        ))
        stats["written"] += 1
    return entries, stats


def load_master(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_master(path: Path, master: dict, new_entries: list[dict]) -> int:
    """Append entries whose CUSIP is not already present; returns the number added."""
    have = {i["cusip"] for i in master["issuers"]}
    added = [e for e in new_entries if e["cusip"] not in have]
    master["issuers"].extend(added)
    Path(path).write_text(json.dumps(master, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return len(added)


def enrich_master(holdings: pd.DataFrame, master_path: Path, cache_path: Path, cfg: EnrichConfig,
                  sec_index: Optional[SecCompanyIndex] = None) -> dict:
    """End-to-end: pick targets, resolve them, append to the master. Returns the stats dict."""
    master = load_master(master_path)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {"figi": {}, "sic": {}}
    targets = select_targets(holdings, {i["cusip"] for i in master["issuers"]}, cfg.top_n)
    log.info("Resolving %d unknown CUSIPs (%.1f%% of the value held by unknown securities)", len(targets),
             100 * targets["value_usd"].sum() / max(holdings.loc[~holdings["cusip"].isin({i['cusip'] for i in master['issuers']}), "value_usd"].sum(), 1))
    todo = [c for c in targets["cusip"] if c not in cache["figi"]]
    if todo:
        found = openfigi_lookup(todo, cfg)
        for c in todo:
            cache["figi"][c] = found.get(c)  # None = looked up, not found
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    figi = {c: v for c, v in cache["figi"].items() if v}
    sec_index = sec_index or SecCompanyIndex.fetch(cfg.user_agent)

    calls = 0

    def sic_of(cik: int) -> tuple[Optional[int], str]:
        nonlocal calls
        time.sleep(cfg.sec_sleep)
        calls += 1
        if calls % 25 == 0:  # checkpoint so an aborted run keeps its SIC lookups
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        return sec_sic(cik, cfg)

    entries, stats = build_entries(targets, figi, sec_index, sic_of, cache["sic"])
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    stats["added"] = save_master(master_path, master, entries)
    stats["master_size"] = len(master["issuers"])
    return stats


def reclassify_enriched(master_path: Path) -> int:
    """Recompute the sector of every enriched entry from its stored SIC/description with the current mapping. Returns changes."""
    master = load_master(master_path)
    changed = 0
    for e in master["issuers"]:
        if e.get("source") != SOURCE_TAG or e.get("asset_type") in ("ETF", "REIT"):
            continue
        sector = sic_to_sector(e.get("sic"), e.get("industry", ""))
        sector = TICKER_SECTOR_OVERRIDES.get((e.get("ticker") or "").upper(), sector)
        if sector != "Unclassified" and sector != e["sector"]:
            e["sector"] = sector
            changed += 1
    Path(master_path).write_text(json.dumps(master, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return changed
