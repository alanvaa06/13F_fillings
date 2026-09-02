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


def parse_filing_folder(folder: Path) -> tuple[dict, pd.DataFrame]:
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
    # Keep only the latest filing per (cik, period); amendments that restate replace originals.
    if not holdings.empty:
        latest = holdings.groupby(["cik", "period"])["filing_date"].transform("max")
        holdings = holdings[holdings["filing_date"] == latest].copy()
    return filings, holdings
