from sec13f.classify import SecurityClassifier, classify_asset_type, display_name
from sec13f.config import Settings

clf = SecurityClassifier(Settings().issuers_file)


def test_asset_type_rules():
    assert classify_asset_type("COM", "APPLE INC", "", "SH")[0] == "Common Stock"
    assert classify_asset_type("COM", "APPLE INC", "Put", "SH")[0] == "Put Option"
    assert classify_asset_type("COM", "APPLE INC", "Call", "SH")[0] == "Call Option"
    assert classify_asset_type("SPONSORED ADR", "TAIWAN SEMICONDUCTOR", "", "SH")[0] == "ADR"
    assert classify_asset_type("PFD SER L", "BANK OF AMERICA", "", "SH")[0] == "Preferred Stock"
    assert classify_asset_type("NOTE 0.750% 12/1", "MICROSTRATEGY", "", "PRN")[0] == "Corporate Debt"
    assert classify_asset_type("NOTE CONV 1.25%", "SNOWFLAKE", "", "PRN")[0] == "Convertible Debt"
    assert classify_asset_type("WT EXP 08/03/2027", "OCCIDENTAL PETE", "", "SH")[0] == "Warrant"
    assert classify_asset_type("MSCI EMERG MKT ETF", "ISHARES TR", "", "SH")[0] == "ETF"
    assert classify_asset_type("TR UNIT", "SPDR S&P 500 ETF TR", "", "SH")[0] == "ETF"
    assert classify_asset_type("UNIT", "FOO ACQUISITION CORP", "", "SH")[0] == "SPAC Unit"
    assert classify_asset_type("COM", "SIMON PPTY GROUP INC NEW", "", "SH")[0] == "REIT"


def test_master_lookup_wins():
    d = clf.classify("037833100", "APPLE INC", "COM", "", "SH")
    assert d["sector"] == "Information Technology" and d["sector_method"] == "master" and d["ticker"] == "AAPL"
    assert d["underlying_asset"] == "Equity"


def test_same_issuer_other_class_uses_cusip6():
    d = clf.classify("037833999", "APPLE INC", "COM", "", "SH")
    assert d["sector"] == "Information Technology"


def test_keyword_and_bayes_fallback():
    assert clf.classify("999999991", "FIRST REPUBLIC BANCORP", "COM", "", "SH")["sector"] == "Financials"
    assert clf.classify("999999992", "ACME THERAPEUTICS INC", "COM", "", "SH")["sector"] == "Health Care"
    assert clf.classify("999999993", "VERTEX ENERGY INC", "COM", "", "SH")["sector"] == "Energy"
    d = clf.classify("999999994", "ISHARES TR", "CORE MSCI EAFE ETF", "", "SH")
    assert d["asset_type"] == "ETF" and d["sector"] == "ETF - International Equity"


def test_option_underlying_bucket():
    d = clf.classify("037833100", "APPLE INC", "COM", "Put", "SH")
    assert d["asset_type"] == "Put Option" and d["underlying_asset"] == "Options"


def test_display_name_distinguishes_etfs():
    assert display_name("ISHARES TR", "MSCI EMERG MKT ETF", "ETF", "EEM") == "ISHARES TR MSCI EMERG MKT ETF (EEM)"
    assert display_name("APPLE INC", "COM", "Common Stock", "AAPL") == "APPLE INC (AAPL)"
    assert display_name("FOO CORP", "COM", "Common Stock", "") == "FOO CORP"
