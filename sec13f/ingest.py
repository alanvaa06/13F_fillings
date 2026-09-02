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

from .config import Settings
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

    @property
    def folder_name(self) -> str:
        return self.accession.replace("-", "")


def list_13f_filings(client: EdgarClient, manager: Manager, form_types=("13F-HR", "13F-HR/A"), max_filings: int = 8) -> list[FilingRef]:
    sub = client.submissions(manager.cik)
    recent = sub.get("filings", {}).get("recent", {})
    refs: list[FilingRef] = []
    for acc, form, fdate, rdate, pdoc in zip(
        recent.get("accessionNumber", []),
        recent.get("form", []),
        recent.get("filingDate", []),
        recent.get("reportDate", []),
        recent.get("primaryDocument", []),
    ):
        if form in form_types:
            refs.append(FilingRef(manager.cik, manager.name, acc, form, fdate, rdate, pdoc))
    # keep the latest filing per report period (amendments supersede originals
    # only when they are restatements; we keep both and let the parser decide)
    refs.sort(key=lambda r: (r.report_period, r.filing_date), reverse=True)
    seen: set[str] = set()
    out: list[FilingRef] = []
    for r in refs:
        if r.report_period in seen:
            continue
        seen.add(r.report_period)
        out.append(r)
        if len(out) >= max_filings:
            break
    return out


_INFOTABLE_PAT = re.compile(r"(infotable|information_table|form13fInfoTable|13f.*table)", re.I)


def download_filing(client: EdgarClient, ref: FilingRef, settings: Settings) -> Path:
    dest = settings.raw_dir / ref.cik / ref.folder_name
    if (dest / "infotable.xml").exists() and (dest / "meta.json").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    idx = client.filing_index(ref.cik, ref.accession)
    items = idx.get("directory", {}).get("item", [])
    xml_files = [it["name"] for it in items if it["name"].lower().endswith(".xml")]
    primary = ref.primary_document or next((n for n in xml_files if "primary" in n.lower()), "")
    infotable = next((n for n in xml_files if _INFOTABLE_PAT.search(n) and n != primary), "")
    if not infotable:
        # fall back: any xml that is not the primary doc
        infotable = next((n for n in xml_files if n != primary), "")
    if not infotable:
        raise RuntimeError(f"No information table XML found in {ref.accession}: {xml_files}")
    (dest / "infotable.xml").write_bytes(client.filing_file(ref.cik, ref.accession, infotable))
    if primary:
        (dest / "primary_doc.xml").write_bytes(client.filing_file(ref.cik, ref.accession, primary))
    (dest / "meta.json").write_text(json.dumps(asdict(ref), indent=2))
    log.info("Downloaded %s %s (%s)", ref.manager, ref.form, ref.report_period)
    return dest


def fetch_universe(managers: list[Manager], settings: Settings, quarters: int = 4) -> list[Path]:
    client = EdgarClient(settings)
    paths: list[Path] = []
    for m in managers:
        try:
            refs = list_13f_filings(client, m, settings.form_types, max_filings=quarters)
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
