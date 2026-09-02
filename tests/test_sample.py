from datetime import date

from sec13f.managers import Manager, cik_alias_map, load_managers
from sec13f.config import Settings
from sec13f.sample import filing_date_for, quarter_ends


def test_quarter_ends_60_covers_15_years():
    q = quarter_ends(60)
    assert q[0] == date(2011, 9, 30) and q[-1] == date(2026, 6, 30) and len(q) == 60
    assert all(d.month in (3, 6, 9, 12) for d in q)


def test_filing_date_is_45_days_after_quarter_on_a_weekday():
    d = filing_date_for(date(2026, 6, 30))
    assert d >= date(2026, 8, 14) and d.weekday() < 5


def test_alias_map_from_config():
    ms = load_managers(Settings().managers_file)
    aliases = cik_alias_map(ms)
    assert aliases["1048445"] == "1791786"  # Elliott's previous CIK


def test_since_and_ipo_respected(tmp_path):
    from sec13f.ingest import iter_local_filings
    from sec13f.parser import parse_all
    from sec13f.sample import generate_sample

    s = Settings(raw_dir=tmp_path / "raw", cache_dir=tmp_path / "c", processed_dir=tmp_path / "p", output_dir=tmp_path / "o")
    scion = [m for m in load_managers(s.managers_file) if m.cik == "1649339"]
    generate_sample(scion, s, quarters=60)
    filings, raw = parse_all(iter_local_filings(s))
    assert filings["period"].min() >= "2016-12-31"
    # no Snowflake before its 2020 IPO
    snow = raw[raw["cusip"] == "833445109"]
    assert snow.empty or snow["period"].min() >= "2020-09-30"
