"""Command-line entry point.

  python -m sec13f.cli sample                 # write offline sample filings
  python -m sec13f.cli fetch  --quarters 4    # download real filings from SEC EDGAR
  python -m sec13f.cli build                  # parse -> classify -> track -> analyze -> dashboard + report
  python -m sec13f.cli all --source sec|sample
  python -m sec13f.cli verify                 # check manager CIKs against EDGAR
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys

import pandas as pd

from .analysis import build_insights
from .classify import SecurityClassifier, infer_manager_type, manager_fingerprint
from .config import Settings
from .dashboard import build_dashboard
from .ingest import fetch_universe, iter_local_filings
from .managers import cik_alias_map, load_managers
from .movers import MoverThresholds, company_moves
from .parser import aggregate_positions, parse_all
from .report import build_report
from .sectors import SectorConfig, load_benchmarks, sector_positioning
from .tracker import (add_weights, consensus, equal_weight_exposure, exposure, manager_summary, manager_turnover,
                      position_changes, put_call_signal, sector_rotation)

log = logging.getLogger("sec13f")


def cmd_sample(args, settings: Settings):
    from .sample import generate_sample

    if args.clean and settings.raw_dir.exists():
        shutil.rmtree(settings.raw_dir)
    managers = load_managers(settings.managers_file, args.cik)
    paths = generate_sample(managers, settings, seed=args.seed, quarters=args.quarters)
    log.info("Wrote %d sample filings for %d managers under %s", len(paths), len(managers), settings.raw_dir)


def cmd_fetch(args, settings: Settings):
    if args.clean and settings.raw_dir.exists():
        shutil.rmtree(settings.raw_dir)
    managers = load_managers(settings.managers_file, args.cik)
    paths = fetch_universe(managers, settings, quarters=args.quarters)
    log.info("Downloaded/cached %d filings", len(paths))


def cmd_verify(args, settings: Settings):
    from .edgar_client import EdgarClient

    client = EdgarClient(settings)
    for m in load_managers(settings.managers_file, args.cik):
        try:
            sub = client.submissions(m.cik)
            forms = sub.get("filings", {}).get("recent", {}).get("form", [])
            n13 = sum(1 for f in forms if f.startswith("13F"))
            print(f"OK   {m.cik:>8}  config='{m.name}'  edgar='{sub.get('name')}'  13F filings (recent): {n13}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {m.cik:>8}  {m.name}: {exc}")


def cmd_enrich(args, settings: Settings):
    """Grow config/issuers.json from the securities held (OpenFIGI + SEC SIC codes). Needs a prior build."""
    from .enrich import EnrichConfig, enrich_master

    pq = settings.processed_dir / "holdings.parquet"
    if pq.exists():
        holdings = pd.read_parquet(pq, columns=["cusip", "issuer", "title_of_class", "value_usd"])
    elif (settings.output_dir / "holdings.csv").exists():
        holdings = pd.read_csv(settings.output_dir / "holdings.csv", usecols=["cusip", "issuer", "title_of_class", "value_usd"], dtype={"cusip": str})
    else:
        log.error("No holdings found. Run `build` first.")
        sys.exit(2)
    cfg = EnrichConfig(top_n=args.top, user_agent=settings.user_agent, openfigi_api_key=os.environ.get("OPENFIGI_API_KEY", ""), sec_sleep=args.sec_sleep)
    stats = enrich_master(holdings, settings.issuers_file, settings.cache_dir / "enrich_cache.json", cfg)
    print("Issuer master enrichment")
    for k, v in stats.items():
        print(f"  {k:16s} {v}")
    print("Re-run `build` to apply the new classifications.")


def cmd_build(args, settings: Settings):
    settings.ensure_dirs()
    folders = iter_local_filings(settings)
    if not folders:
        log.error("No filings under %s. Run `sample` or `fetch` first.", settings.raw_dir)
        sys.exit(2)
    managers = load_managers(settings.managers_file)
    mtype = {m.cik: m.manager_type for m in managers}
    mshort = {m.cik: m.short for m in managers}

    filings, raw = parse_all(folders)
    aliases = cik_alias_map(managers)
    if aliases:  # merge filings submitted under previous CIKs into the canonical manager
        raw["cik"] = raw["cik"].replace(aliases)
        filings["cik"] = filings["cik"].replace(aliases)
    source = "sample" if (filings["source"] == "sample").all() else ("sec" if (filings["source"] == "sec").all() else "mixed")
    log.info("Parsed %d filings, %d holding rows (%s)", len(filings), len(raw), source)
    # `fetch --quarters N` takes the last N filings *per manager*; a filer that stopped reporting drags in old
    # quarters where it is alone. Cap the history to the last N calendar quarters present so the series stay comparable.
    hq = getattr(args, "history_quarters", None) or (getattr(args, "quarters", None) if args.cmd == "all" else None)
    if hq:
        keep = sorted(raw["period"].unique())[-hq:]
        raw, filings = raw[raw["period"].isin(keep)], filings[filings["period"].isin(keep)]
        log.info("History capped to the last %d quarters (%s .. %s)", len(keep), keep[0], keep[-1])
    h = aggregate_positions(raw)
    h = SecurityClassifier(settings.issuers_file).classify_frame(h)
    h["manager_type"] = h["cik"].map(mtype).fillna("Unknown")
    h["manager_short"] = h["cik"].map(mshort).fillna(h["manager"])
    h = add_weights(h)

    changes = position_changes(h)
    turnover = manager_turnover(changes)
    fp = infer_manager_type(manager_fingerprint(h), turnover)
    fp["manager_type"] = fp["cik"].map(mtype).fillna("Unknown")
    msum = manager_summary(h, changes)
    exp_asset_vw = exposure(h, "underlying_asset")
    exp_asset_ew = equal_weight_exposure(h, "underlying_asset")
    exp_sector_ew = equal_weight_exposure(h, "sector")
    exp_sector_vw = exposure(h, "sector")
    exp_sector_type = exposure(h, "sector", "manager_type")
    rot = sector_rotation(changes, "manager_type")
    cons = consensus(h, changes)
    pc = put_call_signal(h)
    benchmarks = load_benchmarks(settings.benchmarks_file)
    if benchmarks:
        log.info("External benchmarks: %s", ", ".join(b.name for b in benchmarks))
    sector_cfg = SectorConfig()
    sector_pos = sector_positioning(h, changes, sector_cfg, benchmarks)
    thresholds = MoverThresholds()
    moves = company_moves(changes, thresholds)

    periods = sorted(h["period"].unique())
    insights = [build_insights(h, changes, msum, fp, exp_asset_ew, exp_sector_ew, cons, pc, p, sector_pos=sector_pos, moves=moves) for p in periods]

    # persist tables
    pdir, odir = settings.processed_dir, settings.output_dir
    h.to_parquet(pdir / "holdings.parquet", index=False)
    if not changes.empty:
        changes.to_parquet(pdir / "changes.parquet", index=False)
    position_tables = {"holdings": h, "changes": changes} if getattr(args, "position_csv", False) else {}
    for name, df in {
        "filings": filings, **position_tables, "manager_summary": msum.merge(fp.drop(columns=["manager", "manager_type"]), on=["cik", "period"], how="left"),
        "exposure_asset_value_weighted": exp_asset_vw, "exposure_asset_equal_weighted": exp_asset_ew,
        "exposure_sector_equal_weighted": exp_sector_ew, "exposure_sector_value_weighted": exp_sector_vw, "exposure_sector_by_manager_type": exp_sector_type,
        "sector_rotation": rot, "consensus": cons, "put_call": pc,
        "sector_positioning": sector_pos, "company_moves": moves,
    }.items():
        if df is not None and not df.empty:
            df.to_csv(odir / f"{name}.csv", index=False)
    (odir / "insights.json").write_text(json.dumps(insights, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    report = build_report(insights, msum, fp, exp_asset_ew, exp_sector_ew, cons, changes, source,
                          sector_pos=sector_pos, moves=moves, benchmarks=benchmarks, sector_cfg=sector_cfg, thresholds=thresholds)
    (odir / "report.md").write_text(report, encoding="utf-8")
    dash = build_dashboard(odir / "dashboard.html", h=h, changes=changes, msum=msum, fp=fp, exp_asset_ew=exp_asset_ew, exp_asset_vw=exp_asset_vw,
                           exp_sector_ew=exp_sector_ew, exp_sector_vw=exp_sector_vw, rotation=rot, cons=cons, pc=pc, insights=insights, managers=managers, company_moves=moves,
                           source=source, n_filings=len(filings), title=args.title, detail_quarters=args.detail_quarters)
    log.info("Report: %s", odir / "report.md")
    log.info("Dashboard: %s", dash)
    print(insights[-1]["headline"])
    for b in insights[-1]["bullets"]:
        print(" -", b["text"])


def main(argv=None):
    # Windows consoles default to cp1252; never let a stray glyph in a printed bullet kill a 20-minute run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(prog="sec13f", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="generate offline sample filings")
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--quarters", type=int, default=60)
    s.add_argument("--cik", nargs="*")
    s.add_argument("--clean", action="store_true", help="delete data/raw first")
    s.set_defaults(fn=cmd_sample)

    f = sub.add_parser("fetch", help="download 13F filings from SEC EDGAR")
    f.add_argument("--quarters", type=int, default=60)
    f.add_argument("--cik", nargs="*")
    f.add_argument("--clean", action="store_true")
    f.set_defaults(fn=cmd_fetch)

    e = sub.add_parser("enrich-issuers", help="grow config/issuers.json from held securities via OpenFIGI + SEC SIC codes")
    e.add_argument("--top", type=int, default=2500, help="unknown CUSIPs to resolve, ranked by value held")
    e.add_argument("--sec-sleep", type=float, default=0.13, help="seconds between SEC submissions requests (raise it while a fetch is running)")
    e.set_defaults(fn=cmd_enrich)

    v = sub.add_parser("verify", help="check configured CIKs against EDGAR")
    v.add_argument("--cik", nargs="*")
    v.set_defaults(fn=cmd_verify)

    b = sub.add_parser("build", help="parse, classify, track, analyze, render")
    b.add_argument("--title", default="13F Holdings Tracker")
    b.add_argument("--detail-quarters", type=int, default=4, help="recent quarters whose position-level detail is embedded inline; every quarter also gets output/dashboard_data/<period>.json, loaded on demand")
    b.add_argument("--position-csv", action="store_true", help="also write holdings.csv and changes.csv (multi-GB with real data; parquet is always written)")
    b.add_argument("--history-quarters", type=int, default=None, help="keep only the last N calendar quarters on disk (default: everything)")
    b.set_defaults(fn=cmd_build)

    a = sub.add_parser("all", help="sample|fetch then build")
    a.add_argument("--source", choices=["sec", "sample"], default="sample")
    a.add_argument("--quarters", type=int, default=60)
    a.add_argument("--detail-quarters", type=int, default=4)
    a.add_argument("--position-csv", action="store_true", help="also write holdings.csv and changes.csv (multi-GB with real data; parquet is always written)")
    a.add_argument("--seed", type=int, default=13)
    a.add_argument("--cik", nargs="*")
    a.add_argument("--clean", action="store_true")
    a.add_argument("--title", default="13F Holdings Tracker")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings()
    if args.cmd == "all":
        (cmd_fetch if args.source == "sec" else cmd_sample)(args, settings)
        cmd_build(args, settings)
    else:
        args.fn(args, settings)


if __name__ == "__main__":
    main()
