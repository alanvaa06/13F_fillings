"""End-to-end: generate sample filings into a temp dir and build every artifact."""
from pathlib import Path

from sec13f.classify import SecurityClassifier
from sec13f.config import Settings
from sec13f.ingest import iter_local_filings
from sec13f.managers import load_managers
from sec13f.parser import aggregate_positions, parse_all
from sec13f.sample import generate_sample
from sec13f.tracker import add_weights, position_changes


def test_sample_roundtrip(tmp_path: Path):
    s = Settings(raw_dir=tmp_path / "raw", cache_dir=tmp_path / "cache", processed_dir=tmp_path / "proc", output_dir=tmp_path / "out")
    managers = load_managers(s.managers_file, ["1067983", "1649339"])
    paths = generate_sample(managers, s, seed=1)
    assert len(paths) == 8
    filings, raw = parse_all(iter_local_filings(s))
    assert len(filings) == 8 and (filings["reconciliation_gap_pct"].abs() < 0.01).all()
    h = add_weights(SecurityClassifier(s.issuers_file).classify_frame(aggregate_positions(raw)))
    assert h["sector"].ne("Unclassified").all()
    ch = position_changes(h)
    assert set(ch["action"]) >= {"NEW", "EXIT", "ADD", "TRIM", "HOLD"}


def test_cli_build_end_to_end(tmp_path: Path, monkeypatch):
    import sec13f.cli as cli

    s = Settings(raw_dir=tmp_path / "raw", cache_dir=tmp_path / "cache", processed_dir=tmp_path / "proc", output_dir=tmp_path / "out")
    monkeypatch.setattr(cli, "Settings", lambda: s)
    cli.main(["all", "--source", "sample", "--cik", "1067983", "1336528", "1029160"])
    out = s.output_dir
    assert (out / "dashboard.html").stat().st_size > 100_000
    assert "Lectura del trimestre" in (out / "report.md").read_text()
    assert (out / "holdings.csv").exists() and (out / "changes.csv").exists() and (out / "consensus.csv").exists()
