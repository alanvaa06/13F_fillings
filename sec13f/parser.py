"""Parse 13F-HR XML filings (cover page + information table) into DataFrames.

Namespace-agnostic: EDGAR has changed the informationTable namespace over the
years, so every lookup uses local-name() matching.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree

from .config import VALUE_IN_DOLLARS_FROM

log = logging.getLogger(__name__)

NUMERIC_COLUMNS = ["value_usd", "shares", "vote_sole", "vote_shared", "vote_none"]
HOLDING_COLUMNS = [
    "cik", "manager", "accession", "form", "filing_date", "period",
    "issuer", "title_of_class", "cusip", "figi", "value_usd", "shares",
    "sh_prn", "put_call", "discretion", "other_managers",
    "vote_sole", "vote_shared", "vote_none",
]


def _ln(tag: str) -> str:
    return f"*[local-name()='{tag}']"


def _txt(node, tag: str, default: str = "") -> str:
    r = node.xpath(f"./{_ln(tag)}/text()")
    return r[0].strip() if r else default


def _deep_txt(root, tag: str, default: str = "") -> str:
    r = root.xpath(f".//{_ln(tag)}/text()")
    return r[0].strip() if r else default


def _to_date(s: str) -> str:
    """Normalise MM-DD-YYYY / YYYY-MM-DD / MM/DD/YYYY to ISO."""
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%m-%d-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return s


def parse_cover(primary_xml: bytes) -> dict:
    root = etree.fromstring(primary_xml)
    return {
        "submission_type": _deep_txt(root, "submissionType"),
        "cik": _deep_txt(root, "cik").lstrip("0"),
        "period": _to_date(_deep_txt(root, "periodOfReport")),
        "manager": _deep_txt(root, "name"),
        "is_amendment": _deep_txt(root, "isAmendment", "false").lower() == "true",
        "amendment_type": _deep_txt(root, "amendmentType"),
        "report_type": _deep_txt(root, "reportType"),
        "file_number": _deep_txt(root, "form13FFileNumber"),
        "other_managers_count": int(_deep_txt(root, "otherIncludedManagersCount", "0") or 0),
        "table_entry_total": int(_deep_txt(root, "tableEntryTotal", "0") or 0),
        "table_value_total": float(_deep_txt(root, "tableValueTotal", "0") or 0),
        "confidential_omitted": _deep_txt(root, "isConfidentialOmitted", "false").lower() == "true",
    }


def value_multiplier(filing_date: str) -> int:
    """Values are in thousands before 2023-01-03 and in dollars afterwards."""
    return 1 if _to_date(filing_date) >= VALUE_IN_DOLLARS_FROM else 1000


def parse_infotable(xml: bytes, meta: dict) -> pd.DataFrame:
    root = etree.fromstring(xml)
    mult = value_multiplier(meta.get("filing_date", "2030-01-01"))
    rows = []
    for it in root.xpath(f"//{_ln('infoTable')}"):
        sp = it.xpath(f"./{_ln('shrsOrPrnAmt')}")
        va = it.xpath(f"./{_ln('votingAuthority')}")
        shares = _txt(sp[0], "sshPrnamt", "0") if sp else "0"
        rows.append(
            {
                "cik": meta["cik"],
                "manager": meta["manager"],
                "accession": meta["accession"],
                "form": meta["form"],
                "filing_date": _to_date(meta["filing_date"]),
                "period": _to_date(meta["report_period"]),
                "issuer": _txt(it, "nameOfIssuer").upper(),
                "title_of_class": _txt(it, "titleOfClass").upper(),
                "cusip": _txt(it, "cusip").upper().strip(),
                "figi": _txt(it, "figi"),
                "value_usd": float(_txt(it, "value", "0").replace(",", "") or 0) * mult,
                "shares": float(shares.replace(",", "") or 0),
                "sh_prn": (_txt(sp[0], "sshPrnamtType", "SH") if sp else "SH").upper(),
                "put_call": _txt(it, "putCall").title(),
                "discretion": _txt(it, "investmentDiscretion").upper(),
                "other_managers": _txt(it, "otherManager"),
                "vote_sole": float(_txt(va[0], "Sole", "0") or 0) if va else 0.0,
                "vote_shared": float(_txt(va[0], "Shared", "0") or 0) if va else 0.0,
                "vote_none": float(_txt(va[0], "None", "0") or 0) if va else 0.0,
            }
        )
    df = pd.DataFrame(rows, columns=HOLDING_COLUMNS)
    # Placeholder tables: some filers (e.g. Norges Bank) submit a single "NA" / 000000000 / $0 row when every
    # position is under confidential treatment and file the real table later as 13F-HR/A. Treat as no data.
    placeholder = (df["cusip"].isin({"000000000", "", "NA"}) | df["issuer"].isin({"NA", "N/A", "NONE"})) & (df["value_usd"] == 0) & (df["shares"] == 0)
    if placeholder.any():
        log.info("Dropped %d placeholder row(s) from %s (%s)", int(placeholder.sum()), meta.get("accession"), meta.get("report_period"))
        df = df[~placeholder].reset_index(drop=True)
    return df


_CUSIP_RE = re.compile(r"(?<![A-Z0-9])([0-9A-Z]{6}[0-9A-Z]{2}[0-9])(?![A-Z0-9])")
_NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")
_CLASS_HEADS = {"COM", "CL", "CLASS", "SHS", "ORD", "SPONSORED", "SPON", "PFD", "NOTE", "NOTES", "DEB", "DEBENTURE", "BD", "WT", "WTS",
                "UNIT", "UNITS", "ADR", "ADS", "RT", "RTS", "CAP", "STK", "SH", "SHARE", "SHARES", "ETF", "TR"}


def parse_text_table(text: str, meta: dict) -> pd.DataFrame:
    """Best-effort parser for pre-2013 ASCII 13F tables.

    Each holdings line contains: issuer, class, CUSIP, value (x$1000), shares,
    SH/PRN, [PUT/CALL], discretion, [managers], voting sole/shared/none. Column
    positions vary by filer, so we anchor on the 9-character CUSIP and read the
    numeric fields to its right.
    """
    rows = []
    mult = value_multiplier(meta.get("filing_date", "2000-01-01"))
    in_table = False
    for raw in text.splitlines():
        line = raw.rstrip()
        up = line.upper()
        if "<TABLE>" in up or "FORM 13F INFORMATION TABLE" in up:
            in_table = True
            continue
        if "</TABLE>" in up:
            in_table = False
            continue
        if not in_table or not line.strip():
            continue
        m = _CUSIP_RE.search(up)
        if not m or m.start() < 8:
            continue
        left, right = line[: m.start()].strip(), up[m.end():]
        nums = [n.replace(",", "") for n in _NUM_RE.findall(right)]
        if len(nums) < 2:
            continue
        toks = right.split()
        put_call = "Put" if "PUT" in toks else ("Call" if "CALL" in toks else "")
        sh_prn = "PRN" if "PRN" in toks else "SH"
        disc = next((t for t in toks if t in ("SOLE", "SHARED", "DEFINED", "DFND", "OTR", "OTHER")), "")
        # issuer / class split: the class starts at the last "class head" token (COM, CL, SHS, PFD, NOTE...)
        lt = left.split()
        heads = [i for i, t in enumerate(lt) if i >= 1 and t.upper() in _CLASS_HEADS]
        cut = heads[-1] if heads else max(1, len(lt) - 1)
        issuer, cls = " ".join(lt[:cut]), " ".join(lt[cut:])
        try:
            value, shares = float(nums[0]), float(nums[1])
        except ValueError:
            continue
        # voting authority = last three numbers; anything in between is the "other managers" field
        tail = nums[-3:] if len(nums) >= 5 else nums[2:5]
        vote = [float(x) for x in tail] + [0.0] * 3
        other = " ".join(nums[2:-3]) if len(nums) > 5 else ""
        rows.append({
            "cik": meta["cik"], "manager": meta["manager"], "accession": meta["accession"], "form": meta["form"],
            "filing_date": _to_date(meta["filing_date"]), "period": _to_date(meta["report_period"]),
            "issuer": issuer.upper(), "title_of_class": cls.upper(), "cusip": m.group(1), "figi": "",
            "value_usd": value * mult, "shares": shares, "sh_prn": sh_prn, "put_call": put_call, "discretion": disc,
            "other_managers": other, "vote_sole": vote[0], "vote_shared": vote[1], "vote_none": vote[2],
        })
    return pd.DataFrame(rows, columns=HOLDING_COLUMNS)


PARSER_VERSION = 3  # bump when parse_infotable / parse_cover / parse_text_table change, so cached results are rebuilt


def _cache_is_fresh(folder: Path) -> bool:
    cache, ccover = folder / "parsed.parquet", folder / "parsed_cover.json"
    if not (cache.exists() and ccover.exists()):
        return False
    try:
        if json.loads(ccover.read_text(encoding="utf-8")).get("_parser_version") != PARSER_VERSION:
            return False
    except (OSError, ValueError):
        return False
    stamp = cache.stat().st_mtime
    return all(stamp >= p.stat().st_mtime for p in (folder / "infotable.xml", folder / "infotable.txt", folder / "primary_doc.xml", folder / "meta.json") if p.exists())


def parse_filing_folder(folder: Path, use_cache: bool = True) -> tuple[dict, pd.DataFrame]:
    """Parse one filing folder. Results are cached next to the raw files (parsed.parquet + parsed_cover.json):
    parsing 4,000 XML tables takes about an hour, reading the cache takes a minute. The cache is invalidated by
    PARSER_VERSION and by the raw files' modification times."""
    if use_cache and _cache_is_fresh(folder):
        cover = json.loads((folder / "parsed_cover.json").read_text(encoding="utf-8"))
        cover.pop("_parser_version", None)
        return cover, pd.read_parquet(folder / "parsed.parquet")
    cover, holdings = _parse_filing_folder(folder)
    if use_cache:
        try:
            holdings.to_parquet(folder / "parsed.parquet", index=False)
            (folder / "parsed_cover.json").write_text(json.dumps({**cover, "_parser_version": PARSER_VERSION}, default=str), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - a cache miss is never worth failing the build
            log.debug("Could not cache %s: %s", folder, exc)
    return cover, holdings


def _parse_filing_folder(folder: Path) -> tuple[dict, pd.DataFrame]:
    meta = json.loads((folder / "meta.json").read_text())
    # Identity comes from meta.json (written from the submissions API); the cover page only enriches it.
    # A missing or unreadable cover must never drop the filing from the universe.
    cover = {"cik": str(meta.get("cik", "")).lstrip("0"), "period": _to_date(meta.get("report_period", "")), "manager": meta.get("manager", ""),
             "submission_type": meta.get("form", ""), "cover_parsed": False}
    pdoc = folder / "primary_doc.xml"
    if pdoc.exists():
        raw = pdoc.read_bytes()
        if raw.lstrip()[:15].lower().startswith((b"<!doctype html", b"<html")):
            log.warning("Cover page in %s is an HTML rendering, not XML; using meta.json (re-run fetch to repair)", folder)
        else:
            try:
                parsed = parse_cover(raw)
                cover.update({k: v for k, v in parsed.items() if v not in ("", None)})
                cover["cover_parsed"] = True
            except etree.XMLSyntaxError as exc:
                log.warning("Bad cover page in %s: %s", folder, exc)
    if (folder / "infotable.xml").exists():
        holdings = parse_infotable((folder / "infotable.xml").read_bytes(), meta)
    else:
        holdings = parse_text_table((folder / "infotable.txt").read_text(errors="replace"), meta)
        cover["text_format"] = True
    # Reconcile: cover-page total vs sum of table (in the same units)
    mult = value_multiplier(meta.get("filing_date", "2030-01-01"))
    if cover.get("table_value_total"):
        declared = cover["table_value_total"] * mult
        actual = holdings["value_usd"].sum()
        cover["reconciliation_gap_pct"] = round(100 * (actual - declared) / declared, 3) if declared else None
    cover.update({"accession": meta["accession"], "n_rows": len(holdings), "source": meta.get("source", "sec")})
    return cover, holdings


def aggregate_positions(holdings: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate rows (same manager/period/cusip/put_call) which happen
    when several sub-managers report the same security."""
    keys = ["cik", "manager", "period", "cusip", "put_call"]
    agg = (
        holdings.groupby(keys, as_index=False, dropna=False)
        .agg(
            accession=("accession", "first"),
            form=("form", "first"),
            filing_date=("filing_date", "max"),
            issuer=("issuer", "first"),
            title_of_class=("title_of_class", "first"),
            sh_prn=("sh_prn", "first"),
            value_usd=("value_usd", "sum"),
            shares=("shares", "sum"),
            vote_sole=("vote_sole", "sum"),
            vote_shared=("vote_shared", "sum"),
            vote_none=("vote_none", "sum"),
            n_rows=("cusip", "size"),
        )
    )
    return agg


def parse_all(folders: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    covers, frames = [], []
    for f in folders:
        try:
            c, h = parse_filing_folder(f)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to parse %s: %s", f, exc)
            continue
        covers.append(c)
        frames.append(h)
    holdings = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=HOLDING_COLUMNS)
    # an empty filing frame carries object dtypes and would poison the concat
    for c in NUMERIC_COLUMNS:
        holdings[c] = pd.to_numeric(holdings[c], errors="coerce").astype(float).fillna(0.0)
    filings = pd.DataFrame(covers)
    if not holdings.empty:
        keep = select_accessions(filings)
        holdings = holdings[holdings["accession"].isin(keep)].copy()
        filings["used"] = filings["accession"].isin(keep)
        holdings, factors = normalize_value_units(holdings)
        filings["unit_factor"] = [factors.get((c, p), 1.0) for c, p in zip(filings["cik"], filings["period"])]
    return filings, holdings


def _is_amendment(form: str) -> bool:
    return str(form or "").upper().endswith("/A")


def select_accessions(filings: pd.DataFrame) -> set[str]:
    """Which accessions make up each (cik, period) book.

    Base = the latest original 13F-HR. Amendments are applied in filing order: a RESTATEMENT replaces the book,
    a NEW HOLDINGS amendment adds the positions omitted from the original. When the cover page did not say which,
    an amendment with at least half as many rows as the current book is treated as a restatement.
    """
    if filings.empty:
        return set()
    f = filings.copy()
    f["filing_date"] = f["filing_date"].fillna("") if "filing_date" in f else ""
    f["form"] = f["submission_type"] if "submission_type" in f else ""
    f["amendment_type"] = f["amendment_type"].fillna("") if "amendment_type" in f else ""
    keep: set[str] = set()
    for (_cik, _period), grp in f.groupby(["cik", "period"], sort=False):
        grp = grp.sort_values("filing_date")
        originals = grp[~grp["form"].map(_is_amendment)]
        book: list[str] = [originals["accession"].iloc[-1]] if not originals.empty else []
        book_rows = int(originals["n_rows"].iloc[-1]) if not originals.empty else 0
        for a in grp[grp["form"].map(_is_amendment)].itertuples():
            atype = str(a.amendment_type or "").upper()
            restates = atype.startswith("RESTAT") or (not atype and (book_rows == 0 or a.n_rows >= 0.5 * book_rows))
            if restates:
                book, book_rows = [a.accession], int(a.n_rows)
            else:
                book.append(a.accession)
                book_rows += int(a.n_rows)
        keep.update(book)
    return keep


def normalize_value_units(holdings: pd.DataFrame, threshold: float = 200.0, max_fixes_per_manager: int = 80) -> tuple[pd.DataFrame, dict]:
    """Detect filings reported in the wrong unit and rescale them by 1000.

    The SEC switched 13F values from thousands to dollars for filings made from 2023-01-03, but filers moved at their
    own pace (many 2022Q4 and some 2023 tables are still in thousands, a few earlier ones already in dollars).

    Per manager, the quarter whose book total deviates most from the median of its neighbouring quarters (up to two on
    each side) is rescaled by x1000 or /1000 when the deviation exceeds `threshold` AND the rescaled total lands within
    5x of the neighbours; then references are recomputed and the search repeats. Fixing the worst quarter first matters:
    a corrupt quarter also distorts its neighbours' references. Returns the factor applied per (cik, period).
    """
    factors: dict[tuple[str, str], float] = {}
    if holdings.empty:
        return holdings, factors
    tot = holdings.groupby(["cik", "period"])["value_usd"].sum()
    log_thr = np.log10(threshold)
    for cik, s in tot.groupby(level=0):
        s = s.droplevel(0).sort_index().astype(float).copy()
        if len(s) < 2:
            continue
        for _ in range(max_fixes_per_manager):
            vals = s.to_numpy()
            worst, worst_dev = None, 0.0
            for i, (period, total) in enumerate(s.items()):
                lo, hi = max(0, i - 2), min(len(vals), i + 3)
                neigh = [v for j, v in enumerate(vals[lo:hi], start=lo) if j != i and v > 0]
                if not neigh or total <= 0:
                    continue
                ref = float(np.median(neigh))
                dev = abs(np.log10(total / ref))
                if dev > worst_dev:
                    worst_dev, worst = dev, (period, total, ref)
            if worst is None or worst_dev < log_thr:
                break
            period, total, ref = worst
            f = 1000.0 if total < ref else 0.001
            if not (ref / 5 <= total * f <= ref * 5):
                break  # not a unit error: rescaling would not bring it back in line
            s[period] = total * f
            factors[(cik, period)] = factors.get((cik, period), 1.0) * f
            log.info("Unit fix: cik %s period %s rescaled x%g (book total %.3g vs neighbours %.3g)", cik, period, f, total, ref)
    if factors:
        key = list(zip(holdings["cik"], holdings["period"]))
        factor = pd.Series([factors.get(k, 1.0) for k in key], index=holdings.index)
        holdings = holdings.assign(value_usd=holdings["value_usd"] * factor)
    return holdings, factors
