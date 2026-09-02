"""Parse 13F-HR XML filings (cover page + information table) into DataFrames.

Namespace-agnostic: EDGAR has changed the informationTable namespace over the
years, so every lookup uses local-name() matching.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from lxml import etree

from .config import VALUE_IN_DOLLARS_FROM

log = logging.getLogger(__name__)

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
    return df


def parse_filing_folder(folder: Path) -> tuple[dict, pd.DataFrame]:
    meta = json.loads((folder / "meta.json").read_text())
    cover = {}
    pdoc = folder / "primary_doc.xml"
    if pdoc.exists():
        try:
            cover = parse_cover(pdoc.read_bytes())
        except etree.XMLSyntaxError as exc:
            log.warning("Bad cover page in %s: %s", folder, exc)
    holdings = parse_infotable((folder / "infotable.xml").read_bytes(), meta)
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
    filings = pd.DataFrame(covers)
    # Keep only the latest filing per (cik, period); amendments that restate replace originals.
    if not holdings.empty:
        latest = holdings.groupby(["cik", "period"])["filing_date"].transform("max")
        holdings = holdings[holdings["filing_date"] == latest].copy()
    return filings, holdings
