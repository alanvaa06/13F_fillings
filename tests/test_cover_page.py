"""Real EDGAR quirks around the 13F cover page: the submissions API points at the XSL-rendered HTML
view, and a filing must survive a missing or non-XML cover."""
import json
from pathlib import Path

from sec13f.ingest import _pick_infotable, _raw_primary_name
from sec13f.parser import parse_all, parse_filing_folder


def test_pick_infotable_prefers_largest_non_cover_xml():
    items = [{"name": "primary_doc.xml", "size": "2020"}, {"name": "renaissance13Fq22026_holding.xml", "size": "1795507"},
             {"name": "0001037389-26-000059-index.html", "size": ""}]
    assert _pick_infotable(items, "primary_doc.xml") == "renaissance13Fq22026_holding.xml"
    # the cover is never chosen even when it is the only other file
    assert _pick_infotable([{"name": "primary_doc.xml", "size": "2020"}], "primary_doc.xml") == ""
    # without sizes fall back to the name pattern, then to anything that is not the cover
    nosize = [{"name": "primary_doc.xml"}, {"name": "misc.xml"}, {"name": "form13fInfoTable.xml"}]
    assert _pick_infotable(nosize, "primary_doc.xml") == "form13fInfoTable.xml"
    assert _pick_infotable([{"name": "primary_doc.xml"}, {"name": "misc.xml"}], "primary_doc.xml") == "misc.xml"

INFOTABLE = b"""<?xml version="1.0"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable><nameOfIssuer>APPLE INC</nameOfIssuer><titleOfClass>COM</titleOfClass><cusip>037833100</cusip><value>1000</value>
    <shrsOrPrnAmt><sshPrnamt>10</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion><votingAuthority><Sole>10</Sole><Shared>0</Shared><None>0</None></votingAuthority></infoTable>
</informationTable>"""

META = {"cik": "1067983", "manager": "Berkshire Hathaway Inc", "accession": "0000950123-24-000001", "form": "13F-HR",
        "filing_date": "2024-05-15", "report_period": "2024-03-31", "primary_document": "xslForm13F_X02/primary_doc.xml", "source": "sec", "filer_cik": "1067983"}


def _folder(tmp_path: Path, cover: bytes | None) -> Path:
    d = tmp_path / "1067983" / "000095012324000001"
    d.mkdir(parents=True)
    (d / "infotable.xml").write_bytes(INFOTABLE)
    (d / "meta.json").write_text(json.dumps(META))
    if cover is not None:
        (d / "primary_doc.xml").write_bytes(cover)
    return d


def test_xsl_prefix_is_stripped_from_primary_document():
    assert _raw_primary_name("xslForm13F_X02/primary_doc.xml") == "primary_doc.xml"
    assert _raw_primary_name("xslF345X05/form.xml") == "form.xml"
    assert _raw_primary_name("primary_doc.xml") == "primary_doc.xml"
    assert _raw_primary_name("") == ""


def test_html_cover_falls_back_to_meta(tmp_path: Path):
    d = _folder(tmp_path, b"<!DOCTYPE html PUBLIC ...>\n<html><head><meta charset=utf-8></head><body>rendered</body></html>")
    cover, holdings = parse_filing_folder(d)
    assert cover["cik"] == "1067983" and cover["period"] == "2024-03-31" and cover["manager"] == "Berkshire Hathaway Inc"
    assert cover["cover_parsed"] is False and "reconciliation_gap_pct" not in cover
    assert len(holdings) == 1 and holdings.iloc[0]["cusip"] == "037833100"


def test_missing_cover_still_yields_cik_and_period_columns(tmp_path: Path):
    d = _folder(tmp_path, None)
    filings, holdings = parse_all([d])
    assert list(filings["cik"]) == ["1067983"] and list(filings["period"]) == ["2024-03-31"]
    assert len(holdings) == 1


def test_placeholder_confidential_table_yields_no_rows(tmp_path: Path):
    placeholder = b"""<?xml version="1.0"?><informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
      <infoTable><nameOfIssuer>NA</nameOfIssuer><titleOfClass>NA</titleOfClass><cusip>000000000</cusip><value>0</value>
      <shrsOrPrnAmt><sshPrnamt>0</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt><investmentDiscretion>SOLE</investmentDiscretion>
      <votingAuthority><Sole>0</Sole><Shared>0</Shared><None>0</None></votingAuthority></infoTable></informationTable>"""
    d = _folder(tmp_path, None)
    (d / "infotable.xml").write_bytes(placeholder)
    cover, holdings = parse_filing_folder(d, use_cache=False)  # the file is rewritten below within the same second
    assert holdings.empty and cover["n_rows"] == 0
    # a real row next to a placeholder row keeps the real one
    (d / "infotable.xml").write_bytes(INFOTABLE.replace(b"</informationTable>", placeholder.split(b"informationtable\">")[1]))
    _, holdings = parse_filing_folder(d, use_cache=False)
    assert len(holdings) == 1 and holdings.iloc[0]["cusip"] == "037833100"


def test_parse_cache_roundtrip_and_invalidation(tmp_path: Path):
    import time

    from sec13f import parser

    d = _folder(tmp_path, None)
    cover1, h1 = parse_filing_folder(d)
    assert (d / "parsed.parquet").exists() and (d / "parsed_cover.json").exists()
    cover2, h2 = parse_filing_folder(d)  # served from cache
    assert cover2 == cover1 and h2.equals(h1) and "_parser_version" not in cover2
    # a newer raw file invalidates the cache
    time.sleep(0.05)
    (d / "infotable.xml").write_bytes(INFOTABLE.replace(b"<value>1000</value>", b"<value>2000</value>"))
    import os
    os.utime(d / "infotable.xml", None)
    _, h3 = parse_filing_folder(d)
    assert h3.iloc[0]["value_usd"] == 2000
    # a parser version bump invalidates it too
    (d / "parsed_cover.json").write_text('{"_parser_version": -1}', encoding="utf-8")
    assert parser._cache_is_fresh(d) is False
    # and use_cache=False bypasses it without touching the files
    cover4, _ = parse_filing_folder(d, use_cache=False)
    assert cover4["cik"] == "1067983"


def test_xml_cover_enriches_meta(tmp_path: Path):
    cover_xml = b"""<?xml version="1.0"?><edgarSubmission xmlns="http://www.sec.gov/edgar/thirteenffiler">
      <headerData><submissionType>13F-HR</submissionType><filerInfo><filer><credentials><cik>0001067983</cik></credentials></filer>
      <periodOfReport>03-31-2024</periodOfReport></filerInfo></headerData>
      <formData><summaryPage><otherIncludedManagersCount>0</otherIncludedManagersCount><tableEntryTotal>1</tableEntryTotal><tableValueTotal>1000</tableValueTotal></summaryPage></formData>
    </edgarSubmission>"""
    d = _folder(tmp_path, cover_xml)
    cover, _ = parse_filing_folder(d)
    assert cover["cover_parsed"] is True and cover["table_entry_total"] == 1 and cover["reconciliation_gap_pct"] == 0
