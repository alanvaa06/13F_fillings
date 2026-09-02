"""Issuer-master enrichment: SIC mapping, name normalisation, matching and entry building (no network)."""
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sec13f.enrich import EnrichConfig, SecCompanyIndex, build_entries, normalize_name, select_targets, sic_to_sector

SECTORS = {"Energy", "Materials", "Industrials", "Consumer Discretionary", "Consumer Staples", "Health Care", "Financials",
           "Information Technology", "Communication Services", "Utilities", "Real Estate", "Unclassified"}


@pytest.mark.parametrize("sic,desc,expected", [
    (3674, "SEMICONDUCTORS & RELATED DEVICES", "Information Technology"),
    (7372, "SERVICES-PREPACKAGED SOFTWARE", "Information Technology"),
    (2834, "PHARMACEUTICAL PREPARATIONS", "Health Care"),
    (6798, "REAL ESTATE INVESTMENT TRUSTS", "Real Estate"),
    (6022, "STATE COMMERCIAL BANKS", "Financials"),
    (1311, "CRUDE PETROLEUM & NATURAL GAS", "Energy"),
    (4911, "ELECTRIC SERVICES", "Utilities"),
    (5411, "RETAIL-GROCERY STORES", "Consumer Staples"),
    (5731, "RETAIL-RADIO TV & CONSUMER ELECTRONICS STORES", "Consumer Discretionary"),
    (3711, "MOTOR VEHICLES & PASSENGER CAR BODIES", "Consumer Discretionary"),
    (4813, "TELEPHONE COMMUNICATIONS (NO RADIOTELEPHONE)", "Communication Services"),
    (3576, "COMPUTER COMMUNICATIONS EQUIPMENT", "Information Technology"),
    (2860, "INDUSTRIAL ORGANIC CHEMICALS", "Materials"),
    (3312, "STEEL WORKS, BLAST FURNACES & ROLLING MILLS", "Materials"),
    (8731, "SERVICES-COMMERCIAL PHYSICAL & BIOLOGICAL RESEARCH", "Health Care"),
    (6770, "BLANK CHECKS", "Financials"),
    (None, "", "Unclassified"),
    (9999, "", "Unclassified"),
])
def test_sic_to_sector(sic, desc, expected):
    assert sic_to_sector(sic, desc) == expected


@settings(max_examples=200, deadline=None)
@given(st.integers(min_value=0, max_value=9999))
def test_sic_to_sector_always_returns_a_known_label(sic):
    assert sic_to_sector(sic) in SECTORS


def test_normalize_name_expands_abbreviations_and_drops_noise():
    assert normalize_name("APPLIED MATLS INC") == ("APPLIED", "MATERIALS")
    assert normalize_name("Applied Materials Inc /DE") == ("APPLIED", "MATERIALS")
    assert normalize_name("SEAGATE TECHNOLOGY HLDNGS PL") == ("SEAGATE", "TECHNOLOGY")  # truncated "PLC" is noise too
    assert normalize_name("Alphabet Inc. Class A") == ("ALPHABET",)
    assert normalize_name("") == ()


def _index():
    return SecCompanyIndex([
        (6951, "APPLIED MATERIALS INC /DE", "AMAT", "Nasdaq"),
        (1652044, "Alphabet Inc.", "GOOGL", "Nasdaq"),
        (1652044, "Alphabet Inc.", "GOOG", "Nasdaq"),
        (707549, "LAM RESEARCH CORP", "LRCX", "Nasdaq"),
        (1137789, "SEAGATE TECHNOLOGY HOLDINGS PLC", "STX", "Nasdaq"),
    ])


def test_index_matches_ticker_and_name():
    ix = _index()
    assert ix.match_ticker("lrcx") == (707549, "LAM RESEARCH CORP")
    assert ix.match_name("APPLIED MATLS INC")[0] == 6951
    assert ix.match_name("ALPHABET INC CL A")[0] == 1652044
    assert ix.match_name("SEAGATE TECHNOLOGY HLDNGS PL")[0] == 1137789  # partial-token match above threshold
    assert ix.match_name("TOTALLY UNKNOWN WIDGETS") is None


def test_select_targets_ranks_unknown_cusips_by_value():
    h = pd.DataFrame({
        "cusip": ["512807306", "512807306", "037833100", "038222105", "BADCUSIP"],
        "issuer": ["LAM RESEARCH CORP", "LAM RESEARCH CORP", "APPLE INC", "APPLIED MATLS INC", "X"],
        "title_of_class": ["COM", "COM", "COM", "COM", "COM"],
        "value_usd": [100.0, 50.0, 1e9, 120.0, 5.0],
    })
    t = select_targets(h, master_cusips={"037833100"}, top_n=10)
    assert list(t["cusip"]) == ["512807306", "038222105"]  # Apple known, malformed CUSIP skipped, Lam first (150 > 120)


def test_build_entries_uses_figi_then_name_fallback_and_skips_unresolved():
    targets = pd.DataFrame({
        "cusip": ["512807306", "038222105", "999999999"],
        "issuer": ["LAM RESEARCH CORP", "APPLIED MATLS INC", "MYSTERY CO"],
        "title_of_class": ["COM", "COM", "COM"], "value_usd": [3.0, 2.0, 1.0],
    })
    figi = {"512807306": {"ticker": "LRCX", "name": "LAM RESEARCH CORP", "securityType": "Common Stock", "exchCode": "US"}}
    sics = {707549: (3559, "SPECIAL INDUSTRY MACHINERY, NEC"), 6951: (3674, "SEMICONDUCTORS & RELATED DEVICES")}
    cache: dict = {}
    entries, stats = build_entries(targets, figi, _index(), lambda cik: sics[cik], cache)
    by = {e["cusip"]: e for e in entries}
    assert set(by) == {"512807306", "038222105"}
    assert by["512807306"]["ticker"] == "LRCX" and by["512807306"]["sector"] == "Information Technology" and by["512807306"]["country"] == "US"  # ticker override beats SIC 3559
    assert by["038222105"]["ticker"] == "AMAT" and by["038222105"]["sector"] == "Information Technology" and by["038222105"]["source"] == "openfigi+sec_sic"
    assert stats["figi_hits"] == 1 and stats["ticker_matches"] == 1 and stats["name_matches"] == 1 and stats["written"] == 2
    assert set(cache) == {"707549", "6951"}  # SIC lookups cached by CIK


def test_sec_sic_retries_transient_errors_and_gives_up_cleanly(monkeypatch):
    import requests as rq
    from sec13f import enrich

    monkeypatch.setattr(enrich.time, "sleep", lambda s: None)

    class Resp:
        def __init__(self, status, body=None):
            self.status_code, self._body = status, body or {}

        def json(self):
            return self._body

    calls = {"n": 0}

    def flaky_get(url, headers, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise rq.ConnectionError("remote end closed connection")
        if calls["n"] == 2:
            return Resp(503)
        return Resp(200, {"sic": "3674", "sicDescription": "SEMICONDUCTORS & RELATED DEVICES"})

    assert enrich.sec_sic(6951, EnrichConfig(), get=flaky_get) == (3674, "SEMICONDUCTORS & RELATED DEVICES")
    assert calls["n"] == 3
    always_down = lambda url, headers, timeout: (_ for _ in ()).throw(rq.ConnectionError("down"))  # noqa: E731
    assert enrich.sec_sic(6951, EnrichConfig(), get=always_down, retries=2) == (None, "")


def test_sic_mapping_fixes_for_gics_disagreements():
    # "(No Computer Equip)" must not be read as a computer business
    assert sic_to_sector(3600, "ELECTRONIC & OTHER ELECTRICAL EQUIPMENT (NO COMPUTER EQUIP)") == "Industrials"
    # equipment makers stay in IT as in GICS
    assert sic_to_sector(3827, "OPTICAL INSTRUMENTS & LENSES") == "Information Technology"
    assert sic_to_sector(3825, "INSTRUMENTS FOR MEAS & TESTING OF ELECTRICITY & ELEC SIGNALS") == "Information Technology"
    assert sic_to_sector(3559, "SPECIAL INDUSTRY MACHINERY, NEC") == "Industrials"  # generic machinery; LRCX handled by ticker override


def test_ticker_override_and_offline_reclassification(tmp_path):
    import json
    from sec13f.enrich import SOURCE_TAG, TICKER_SECTOR_OVERRIDES, reclassify_enriched

    targets = pd.DataFrame({"cusip": ["512807306"], "issuer": ["LAM RESEARCH CORP"], "title_of_class": ["COM"], "value_usd": [1.0]})
    figi = {"512807306": {"ticker": "LRCX", "securityType": "Common Stock", "exchCode": "US"}}
    entries, _ = build_entries(targets, figi, _index(), lambda cik: (3559, "SPECIAL INDUSTRY MACHINERY, NEC"), {})
    assert entries[0]["sector"] == TICKER_SECTOR_OVERRIDES["LRCX"] == "Information Technology"
    # an older master with the pre-fix sector gets corrected offline from its stored SIC / description
    master = {"issuers": [
        {"ticker": "GEV", "cusip": "36828A101", "issuer": "GE VERNOVA INC", "asset_type": "Common Stock", "sector": "Information Technology",
         "industry": "Electronic & Other Electrical Equipment (No Computer Equip)", "sic": 3600, "source": SOURCE_TAG},
        {"ticker": "AAPL", "cusip": "037833100", "issuer": "APPLE INC", "asset_type": "Common Stock", "sector": "Information Technology", "industry": "x", "ref_price": 1},
    ]}
    p = tmp_path / "issuers.json"
    p.write_text(json.dumps(master), encoding="utf-8")
    assert reclassify_enriched(p) == 1
    out = json.loads(p.read_text(encoding="utf-8"))["issuers"]
    assert out[0]["sector"] == "Industrials" and out[1]["sector"] == "Information Technology"  # curated rows untouched


def test_config_validation():
    with pytest.raises(ValueError):
        EnrichConfig(top_n=0)
    with pytest.raises(ValueError):
        EnrichConfig(openfigi_batch=500)
