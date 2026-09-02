import json
from pathlib import Path

import pandas as pd

from sec13f.parser import aggregate_positions, parse_cover, parse_infotable, value_multiplier

INFOTABLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer><titleOfClass>COM</titleOfClass><cusip>037833100</cusip>
    <value>1000</value>
    <shrsOrPrnAmt><sshPrnamt>10</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
    <votingAuthority><Sole>10</Sole><Shared>0</Shared><None>0</None></votingAuthority>
  </infoTable>
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer><titleOfClass>COM</titleOfClass><cusip>037833100</cusip>
    <value>500</value>
    <shrsOrPrnAmt><sshPrnamt>5</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
    <putCall>Put</putCall>
    <investmentDiscretion>DFND</investmentDiscretion>
    <otherManager>2</otherManager>
    <votingAuthority><Sole>0</Sole><Shared>5</Shared><None>0</None></votingAuthority>
  </infoTable>
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer><titleOfClass>COM</titleOfClass><cusip>037833100</cusip>
    <value>200</value>
    <shrsOrPrnAmt><sshPrnamt>2</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
    <votingAuthority><Sole>2</Sole><Shared>0</Shared><None>0</None></votingAuthority>
  </infoTable>
</informationTable>"""

META_NEW = dict(cik="1", manager="Test", accession="0000000001-26-000001", form="13F-HR", filing_date="2026-08-14", report_period="2026-06-30")
META_OLD = dict(META_NEW, filing_date="2022-11-14", report_period="2022-09-30")


def test_value_units_switch():
    assert value_multiplier("2022-12-30") == 1000
    assert value_multiplier("2023-01-03") == 1
    assert value_multiplier("2026-08-14") == 1


def test_parse_infotable_new_units():
    df = parse_infotable(INFOTABLE, META_NEW)
    assert len(df) == 3
    assert df["value_usd"].tolist() == [1000.0, 500.0, 200.0]
    assert df["put_call"].tolist() == ["", "Put", ""]
    assert df["discretion"].iloc[1] == "DFND"
    assert df["other_managers"].iloc[1] == "2"
    assert df["period"].iloc[0] == "2026-06-30"


def test_parse_infotable_old_units_in_thousands():
    df = parse_infotable(INFOTABLE, META_OLD)
    assert df["value_usd"].iloc[0] == 1_000_000.0


def test_aggregate_positions_merges_submanager_rows_but_keeps_puts_separate():
    df = parse_infotable(INFOTABLE, META_NEW)
    agg = aggregate_positions(df)
    assert len(agg) == 2
    cash = agg[agg["put_call"] == ""].iloc[0]
    assert cash["shares"] == 12 and cash["value_usd"] == 1200 and cash["n_rows"] == 2


def test_parse_cover_handles_us_dates():
    xml = b"""<?xml version="1.0"?><edgarSubmission xmlns="http://www.sec.gov/edgar/thirteenffiler">
    <headerData><submissionType>13F-HR</submissionType><filerInfo><filer><credentials><cik>0001067983</cik></credentials></filer>
    <periodOfReport>06-30-2026</periodOfReport></filerInfo></headerData>
    <formData><coverPage><isAmendment>false</isAmendment><filingManager><name>Berkshire Hathaway Inc</name></filingManager></coverPage>
    <summaryPage><tableEntryTotal>40</tableEntryTotal><tableValueTotal>123456</tableValueTotal></summaryPage></formData></edgarSubmission>"""
    c = parse_cover(xml)
    assert c["cik"] == "1067983" and c["period"] == "2026-06-30" and c["table_entry_total"] == 40
    assert c["manager"] == "Berkshire Hathaway Inc" and c["is_amendment"] is False


TEXT_13F = """<TABLE>
                                                    VALUE   SHARES/  SH/ PUT/ INVSTMT  OTHER   VOTING AUTHORITY
NAME OF ISSUER              TITLE OF CLASS  CUSIP   (x$1000) PRN AMT  PRN CALL DSCRETN MANAGERS SOLE  SHARED NONE
APPLE INC                   COM             037833100  152,340   400,000  SH       SOLE            400,000  0  0
BANK OF AMERICA CORP        COM             060505104    5,000 1,000,000  SH  PUT  SOLE          1,000,000  0  0
INTL BUSINESS MACHS CORP    COM             459200101   64,500   350,000  SH       DEFINED   4     350,000  0  0
BERKSHIRE HATHAWAY INC DEL  CL B NEW        084670702   10,000    50,000  SH       SOLE             50,000  0  0
</TABLE>"""


def test_parse_text_table_pre_2013():
    from sec13f.parser import parse_text_table

    meta = dict(cik="1", manager="X", accession="a", form="13F-HR", filing_date="2012-02-14", report_period="2011-12-31")
    df = parse_text_table(TEXT_13F, meta)
    assert len(df) == 4
    aapl = df.iloc[0]
    assert aapl["cusip"] == "037833100" and aapl["value_usd"] == 152_340_000 and aapl["shares"] == 400_000
    assert df.iloc[1]["put_call"] == "Put"
    ibm = df.iloc[2]
    assert ibm["discretion"] == "DEFINED" and ibm["other_managers"] == "4" and ibm["vote_sole"] == 350_000
    brk = df.iloc[3]
    assert brk["issuer"] == "BERKSHIRE HATHAWAY INC DEL" and brk["title_of_class"] == "CL B NEW"
