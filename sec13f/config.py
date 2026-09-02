"""Central configuration for the 13F pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "output"
CONFIG_DIR = ROOT / "config"

# SEC fair-access policy: identify yourself and stay under 10 requests/second.
# https://www.sec.gov/os/accessing-edgar-data
DEFAULT_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "13F-Tracker research contact@example.com"
)
SEC_MAX_REQUESTS_PER_SECOND = float(os.environ.get("SEC_RPS", "8"))

SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions"
SEC_BROWSE = "https://www.sec.gov/cgi-bin/browse-edgar"
SEC_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"

# Filings submitted on/after this date report values in whole dollars.
# Earlier filings report values in thousands of dollars.
VALUE_IN_DOLLARS_FROM = "2023-01-03"

FORM_TYPES = ("13F-HR", "13F-HR/A")


@dataclass
class Settings:
    user_agent: str = DEFAULT_USER_AGENT
    raw_dir: Path = RAW_DIR
    cache_dir: Path = CACHE_DIR
    processed_dir: Path = PROCESSED_DIR
    output_dir: Path = OUTPUT_DIR
    managers_file: Path = CONFIG_DIR / "managers.json"
    issuers_file: Path = CONFIG_DIR / "issuers.json"
    benchmarks_file: Path = CONFIG_DIR / "benchmarks.json"  # optional; see benchmarks.example.json
    quarters: int = 4
    form_types: tuple = field(default_factory=lambda: FORM_TYPES)

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.cache_dir, self.processed_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)
