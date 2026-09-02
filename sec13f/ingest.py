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
    if ((dest / "infotable.xml").exists() or (dest / "infotable.txt").exists()) and (dest / "meta.json").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    filer = ref.filer_cik or ref.cik
    idx = client.filing_index(filer, ref.accession)
    items = idx.get("directory", {}).get("item", [])
    xml_files = [it["name"] for it in items if it["name"].lower().endswith(".xml")]
    primary = ref.primary_document if ref.primary_document.lower().endswith(".xml") else next((n for n in xml_files if "primary" in n.lower()), "")
    infotable = next((n for n in xml_files if _INFOTABLE_PAT.search(n) and n != primary), "")
    if not infotable:
        infotable = next((n for n in xml_files if n != primary), "")
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
