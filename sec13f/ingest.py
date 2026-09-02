"""Discover and download 13F-HR filings for a manager universe.

Each filing is stored under  data/raw/<cik>/<accession-nodash>/  with:
  - primary_doc.xml       cover page / summary page (EDGAR schema)
  - infotable.xml         holdings (informationTable schema)
  - meta.json             accession, form, filing date, report period, source
This is exactly the layout the sample generator writes, so parsing is
identical for live SEC data and offline samples.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from .config import SEC_SUBMISSIONS, Settings
from .edgar_client import EdgarClient
from .managers import Manager

log = logging.getLogger(__name__)


@dataclass
class FilingRef:
    cik: str
    manager: str
    accession: str
    form: str
    filing_date: str
    report_period: str
    primary_document: str = ""
    source: str = "sec"
    filer_cik: str = ""  # CIK the filing was actually submitted under (may be a previous CIK)

    @property
    def folder_name(self) -> str:
        return self.accession.replace("-", "")


def _iter_submission_blocks(client: EdgarClient, cik: str):
    """Yield the `recent` block plus every paginated `filings.files` block, so
    long histories (the submissions API caps `recent` at ~1000 filings) are covered."""
    sub = client.submissions(cik)
    yield sub.get("filings", {}).get("recent", {})
    for extra in sub.get("filings", {}).get("files", []):
        name = extra.get("name")
        if name:
            try:
                yield client.get_json(f"{SEC_SUBMISSIONS}/{name}")
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not load %s: %s", name, exc)


def list_13f_filings(client: EdgarClient, manager: Manager, form_types=("13F-HR", "13F-HR/A"), max_filings: int = 8,
                     since: str = "") -> list[FilingRef]:
    refs: list[FilingRef] = []
    for cik in (manager.cik, *manager.previous_ciks):
        for block in _iter_submission_blocks(client, cik):
            for acc, form, fdate, rdate, pdoc in zip(
                block.get("accessionNumber", []),
                block.get("form", []),
                block.get("filingDate", []),
                block.get("reportDate", []),
                block.get("primaryDocument", []),
            ):
                if form in form_types and (not since or (rdate or fdate) >= since):
                    refs.append(FilingRef(manager.cik, manager.name, acc, form, fdate, rdate, pdoc, filer_cik=cik))
    # Keep EVERY filing (original and amendments) for the latest `max_filings` report periods. An amendment can be a
    # full restatement or only the holdings omitted from the original ("NEW HOLDINGS"); the parser combines them.
    refs.sort(key=lambda r: (r.report_period, r.filing_date), reverse=True)
    periods: list[str] = []
    for r in refs:
        if r.report_period not in periods:
            periods.append(r.report_period)
    keep = set(periods[:max_filings])
    seen_acc: set[str] = set()
    out: list[FilingRef] = []
    for r in refs:
        if r.report_period in keep and r.accession not in seen_acc:
            seen_acc.add(r.accession)
            out.append(r)
    return out


_INFOTABLE_PAT = re.compile(r"(infotable|information_table|form13fInfoTable|13f.*table)", re.I)


def _pick_infotable(xml_items: list[dict], primary: str) -> str:
    """Name of the information-table XML among the filing's index items.

    The table has no fixed name (infotable.xml, form13fInfoTable.xml, <filer>13Fq22026_holding.xml, ...)
    but it is always by far the largest XML in the folder, so size decides; the filename pattern is only
    the tie-breaker when the index carries no sizes. The cover document (``primary``) is never eligible."""
    candidates = [it for it in xml_items if it.get("name") and it["name"] != primary]
    sized = sorted(candidates, key=lambda it: int(it.get("size") or 0), reverse=True)
    if sized and int(sized[0].get("size") or 0) > 0:
        return sized[0]["name"]
    return next((it["name"] for it in candidates if _INFOTABLE_PAT.search(it["name"])), next((it["name"] for it in candidates), ""))


_XSL_PREFIX = re.compile(r"^xsl[^/]*/", re.I)


def _raw_primary_name(primary_document: str) -> str:
    """The submissions API points at the XSL-rendered HTML view (``xslForm13F_X02/primary_doc.xml``);
    the raw XML is the same file name without that folder prefix."""
    return _XSL_PREFIX.sub("", primary_document or "")


def download_filing(client: EdgarClient, ref: FilingRef, settings: Settings) -> Path:
    dest = settings.raw_dir / ref.cik / ref.folder_name
    filer = ref.filer_cik or ref.cik
    if ((dest / "infotable.xml").exists() or (dest / "infotable.txt").exists()) and (dest / "meta.json").exists():
        # Complete folder. Repair only a missing cover page (e.g. after an HTML rendering was deleted).
        want = _raw_primary_name(ref.primary_document)
        if (dest / "infotable.xml").exists() and want.lower().endswith(".xml") and not (dest / "primary_doc.xml").exists():
            (dest / "primary_doc.xml").write_bytes(client.filing_file(filer, ref.accession, want))
            log.info("Repaired cover page for %s %s (%s)", ref.manager, ref.form, ref.report_period)
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    idx = client.filing_index(filer, ref.accession)
    items = idx.get("directory", {}).get("item", [])
    xml_items = [it for it in items if it["name"].lower().endswith(".xml")]
    xml_files = [it["name"] for it in xml_items]
    want = _raw_primary_name(ref.primary_document)
    primary = want if want.lower().endswith(".xml") else next((n for n in xml_files if "primary" in n.lower()), "")
    infotable = _pick_infotable(xml_items, primary)
    if infotable:
        (dest / "infotable.xml").write_bytes(client.filing_file(filer, ref.accession, infotable))
        if primary:
            (dest / "primary_doc.xml").write_bytes(client.filing_file(filer, ref.accession, primary))
    else:
        # Pre-2013 filings: no XML, the holdings table lives in the complete submission text file.
        txt = f"{ref.accession}.txt"
        (dest / "infotable.txt").write_bytes(client.filing_file(filer, ref.accession, txt))
        log.info("No XML in %s; stored text submission for best-effort parsing", ref.accession)
    (dest / "meta.json").write_text(json.dumps(asdict(ref), indent=2))
    log.info("Downloaded %s %s (%s)", ref.manager, ref.form, ref.report_period)
    return dest


def fetch_universe(managers: list[Manager], settings: Settings, quarters: int = 4) -> list[Path]:
    client = EdgarClient(settings)
    paths: list[Path] = []
    for m in managers:
        try:
            refs = list_13f_filings(client, m, settings.form_types, max_filings=quarters, since=m.since)
        except Exception as exc:  # noqa: BLE001
            log.error("Could not list filings for %s (%s): %s", m.name, m.cik, exc)
            continue
        for ref in refs:
            try:
                paths.append(download_filing(client, ref, settings))
            except Exception as exc:  # noqa: BLE001
                log.error("Download failed %s %s: %s", m.name, ref.accession, exc)
    return paths


def iter_local_filings(settings: Settings) -> list[Path]:
    """All filing folders on disk (from SEC or sample)."""
    return sorted(p.parent for p in settings.raw_dir.glob("*/*/meta.json"))


def quarter_end(d: date) -> date:
    """Calendar quarter end for a date."""
    q = (d.month - 1) // 3
    m = 3 * (q + 1)
    last = {3: 31, 6: 30, 9: 30, 12: 31}[m]
    return date(d.year, m, last)
