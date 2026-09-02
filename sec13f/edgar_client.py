"""Thin, polite HTTP client for SEC EDGAR.

Implements the SEC fair-access requirements: a descriptive User-Agent, a
request-rate ceiling (default 8 req/s, SEC limit is 10), retries with
exponential back-off, and an on-disk cache so re-runs never re-download.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import requests

from .config import (
    SEC_ARCHIVES,
    SEC_BROWSE,
    SEC_MAX_REQUESTS_PER_SECOND,
    SEC_SUBMISSIONS,
    Settings,
)

log = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, rps: float):
        self.min_interval = 1.0 / rps if rps > 0 else 0.0
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


class EdgarClient:
    def __init__(self, settings: Settings | None = None, rps: float = SEC_MAX_REQUESTS_PER_SECOND):
        self.settings = settings or Settings()
        self.settings.ensure_dirs()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.settings.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "*/*",
            }
        )
        self.limiter = RateLimiter(rps)

    # ------------------------------------------------------------------ core
    def _cache_path(self, url: str) -> Path:
        h = hashlib.sha1(url.encode()).hexdigest()
        return self.settings.cache_dir / f"{h}.bin"

    def get(self, url: str, *, use_cache: bool = True, retries: int = 4, timeout: int = 30) -> bytes:
        cp = self._cache_path(url)
        if use_cache and cp.exists():
            return cp.read_bytes()
        backoff = 1.0
        last_exc: Exception | None = None
        for attempt in range(retries):
            self.limiter.wait()
            try:
                resp = self.session.get(url, timeout=timeout)
                if resp.status_code == 200:
                    cp.write_bytes(resp.content)
                    return resp.content
                if resp.status_code in (403, 429, 500, 502, 503, 504):
                    log.warning("EDGAR %s -> HTTP %s (attempt %d)", url, resp.status_code, attempt + 1)
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
            except requests.RequestException as exc:  # network-level failure
                last_exc = exc
                log.warning("EDGAR %s -> %s (attempt %d)", url, exc, attempt + 1)
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError(f"Failed to fetch {url}: {last_exc}")

    def get_json(self, url: str, **kw) -> Any:
        return json.loads(self.get(url, **kw))

    # --------------------------------------------------------------- helpers
    def submissions(self, cik: str) -> dict:
        """Filing history for a CIK (data.sec.gov submissions API)."""
        return self.get_json(f"{SEC_SUBMISSIONS}/CIK{cik.zfill(10)}.json", use_cache=False)

    def filing_index(self, cik: str, accession: str) -> dict:
        """JSON directory listing for one accession folder."""
        acc_nodash = accession.replace("-", "")
        return self.get_json(f"{SEC_ARCHIVES}/{int(cik)}/{acc_nodash}/index.json")

    def filing_file(self, cik: str, accession: str, filename: str) -> bytes:
        acc_nodash = accession.replace("-", "")
        return self.get(f"{SEC_ARCHIVES}/{int(cik)}/{acc_nodash}/{filename}")

    def lookup_cik(self, company: str) -> list[dict]:
        """Resolve a company name to CIK candidates via the EDGAR company browser."""
        url = f"{SEC_BROWSE}?company={requests.utils.quote(company)}&type=13F-HR&output=atom"
        raw = self.get(url, use_cache=False).decode("utf-8", "replace")
        from lxml import etree

        root = etree.fromstring(raw.encode())
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out = []
        for e in root.findall(".//a:entry", ns):
            title = e.findtext("a:title", default="", namespaces=ns)
            link = e.find("a:link", ns)
            href = link.get("href") if link is not None else ""
            cik = ""
            if "CIK=" in href:
                cik = href.split("CIK=")[1].split("&")[0].lstrip("0")
            out.append({"title": title, "cik": cik, "href": href})
        return out
