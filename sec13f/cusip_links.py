"""Link a security's old and new CUSIP when a corporate action changes the identifier.

The tracker diffs books by CUSIP, so a redomicile, holding-company reorganisation, ADR-to-ordinary conversion or
reverse split (new CUSIP, same company) otherwise shows up as every holder exiting the old CUSIP and opening the
new one: fake consensus buys/sells and the largest "moves" of the quarter.

A link (old -> new, effective at period ``cur``) is accepted when, between two consecutive periods:
  * the old CUSIP disappears from the whole universe and the new one first appears in it;
  * both are cash equity (common stock, ADR, REIT); ETFs are excluded because many report the *trust* as the
    issuer (``ISHARES TR``), so different funds of one trust would look like the same company;
  * the normalised issuer names match (legal-form words such as INC/PLC/LTD/HOLDINGS dropped);
  * mostly the same managers hold both (Jaccard overlap of holder sets >= ``min_overlap``).
Each CUSIP is linked at most once per period, best overlap first.

``share_factor`` converts old shares into new ones (reverse splits, exchange ratios). It is the median per-holder
share ratio among the managers holding both; inside ``[1/split_band, split_band]`` it is taken as 1:1 so that
ordinary trading by the holders is not mistaken for an exchange ratio.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

LINK_COLUMNS = ["period", "old_cusip", "new_cusip", "old_issuer", "new_issuer", "holders_prev", "holders_cur", "overlap",
                "name_ratio", "share_factor"]
LINKABLE_ASSET_TYPES = ("Common Stock", "ADR", "REIT")

_LEGAL_WORDS = {
    "A", "ADR", "ADS", "AG", "B", "CL", "CLASS", "CO", "COM", "COMPANY", "CORP", "CORPORATION", "DEL", "GROUP", "GRP",
    "HLDG", "HLDGS", "HOLDING", "HOLDINGS", "INC", "INCORPORATED", "LIMITED", "LLC", "LP", "LTD", "NEW", "NV", "ORD",
    "PLC", "SA", "SE", "SHS", "SPON", "SPONSORED", "THE", "US",
}


def normalize_issuer(name: str) -> str:
    """Issuer name without punctuation, spaces or legal-form words: 'Exxon Mobil Corp' and 'ExxonMobil Holdings
    Corp' both become 'EXXONMOBIL'."""
    tokens = re.sub(r"[^A-Z0-9 ]", " ", str(name or "").upper()).split()
    return "".join(t for t in tokens if t not in _LEGAL_WORDS)


def _name_ratio(a: str, b: str) -> float:
    if not a or not b or a[:3] != b[:3]:
        return 0.0
    return 1.0 if a == b else SequenceMatcher(None, a, b).ratio()


def detect_cusip_links(h: pd.DataFrame, min_holders: int = 3, min_overlap: float = 0.6, min_name_ratio: float = 0.85,
                       split_band: float = 1.25) -> pd.DataFrame:
    """Old -> new CUSIP links per period (see module docstring). ``h`` needs period, cik, cusip, issuer, put_call,
    asset_type and shares."""
    if h.empty:
        return pd.DataFrame(columns=LINK_COLUMNS)
    cash = h[(h["put_call"] == "") & h["asset_type"].isin(LINKABLE_ASSET_TYPES)][["period", "cik", "cusip", "issuer", "shares"]]
    every = {p: set(g) for p, g in h.groupby("period")["cusip"]}  # any asset type / option row counts as "present"
    by_period = dict(tuple(cash.groupby("period")))
    periods = sorted(every)
    out: list[dict] = []
    for prev, cur in zip(periods[:-1], periods[1:]):
        a, b = by_period.get(prev), by_period.get(cur)
        if a is None or b is None:
            continue
        common = set(a["cik"]) & set(b["cik"])
        a, b = a[a["cik"].isin(common)], b[b["cik"].isin(common)]
        holders_a = a.groupby("cusip")["cik"].agg(set)
        holders_b = b.groupby("cusip")["cik"].agg(set)
        olds = [c for c in set(holders_a.index) - every[cur] if len(holders_a[c]) >= min_holders]
        news = [c for c in set(holders_b.index) - every[prev] if len(holders_b[c]) >= min_holders]
        if not olds or not news:
            continue
        name_a = a[a["cusip"].isin(olds)].groupby("cusip")["issuer"].agg(lambda s: s.mode().iat[0])
        name_b = b[b["cusip"].isin(news)].groupby("cusip")["issuer"].agg(lambda s: s.mode().iat[0])
        norm_b = {c: normalize_issuer(n) for c, n in name_b.items()}
        cands = []
        for o in olds:
            no = normalize_issuer(name_a[o])
            for n in news:
                r = _name_ratio(no, norm_b[n])
                if r < min_name_ratio:
                    continue
                ov = len(holders_a[o] & holders_b[n]) / len(holders_a[o] | holders_b[n])
                if ov >= min_overlap:
                    cands.append((ov, r, o, n))
        used: set[str] = set()
        for ov, r, o, n in sorted(cands, reverse=True):
            if o in used or n in used:
                continue
            used |= {o, n}
            sa = a[a["cusip"] == o].groupby("cik")["shares"].sum()
            sb = b[b["cusip"] == n].groupby("cik")["shares"].sum()
            both = sa.index.intersection(sb.index)
            ratios = (sb[both] / sa[both].where(sa[both] > 0)).dropna()
            factor = float(np.median(ratios)) if len(ratios) else 1.0
            if 1 / split_band <= factor <= split_band or not np.isfinite(factor) or factor <= 0:
                factor = 1.0
            out.append(dict(period=cur, old_cusip=o, new_cusip=n, old_issuer=name_a[o], new_issuer=name_b[n],
                            holders_prev=len(holders_a[o]), holders_cur=len(holders_b[n]), overlap=round(ov, 3),
                            name_ratio=round(r, 3), share_factor=factor))
    return pd.DataFrame(out, columns=LINK_COLUMNS)


def cusip_map_for_period(links: pd.DataFrame, period: str, prev_cusips: set[str], cur_cusips: set[str]) -> dict[str, tuple[str, float]]:
    """{old cusip: (new cusip, share factor)} to apply to the previous book before diffing into ``period``.

    Besides the linked equity CUSIPs, option-specific CUSIPs of the same issue (old issuer prefix plus the same
    issue number under the new prefix, e.g. 30231G902 -> 30233Q908) are carried over when the old one is gone and
    exactly one matching new one appears this period."""
    if links is None or links.empty:
        return {}
    sel = links[links["period"] == period]
    m: dict[str, tuple[str, float]] = {}
    by_issue: dict[str, list[str]] = {}
    for c in cur_cusips - prev_cusips:
        by_issue.setdefault(c[:8], []).append(c)
    for r in sel.itertuples():
        m[r.old_cusip] = (r.new_cusip, float(r.share_factor))
        old6, new6 = r.old_cusip[:6], r.new_cusip[:6]
        if old6 == new6:
            continue  # same issuer prefix (reverse split etc.): option CUSIPs are not re-issued under a new prefix
        for c in prev_cusips - cur_cusips:
            if c[:6] == old6 and c != r.old_cusip and c not in m:
                targets = by_issue.get(new6 + c[6:8], [])
                if len(targets) == 1:
                    m[c] = (targets[0], float(r.share_factor))
    return m
