"""Manager universe loading."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Manager:
    cik: str
    name: str
    short: str
    manager_type: str
    style: str = ""
    since: str = ""  # first report period (ISO) the filer is expected to have
    previous_ciks: tuple = ()

    @property
    def cik10(self) -> str:
        return self.cik.zfill(10)


def load_managers(path: Path, ciks: Iterable[str] | None = None) -> list[Manager]:
    data = json.loads(Path(path).read_text())
    managers = [
        Manager(
            cik=str(m["cik"]).lstrip("0"),
            name=m["name"],
            short=m.get("short", m["name"]),
            manager_type=m.get("manager_type", "Unknown"),
            style=m.get("style", ""),
            since=m.get("since", ""),
            previous_ciks=tuple(str(c).lstrip("0") for c in m.get("previous_ciks", [])),
        )
        for m in data["managers"]
    ]
    if ciks:
        wanted = {str(c).lstrip("0") for c in ciks}
        managers = [m for m in managers if m.cik in wanted]
    return managers


def cik_alias_map(managers: Iterable[Manager]) -> dict[str, str]:
    """previous CIK -> canonical CIK."""
    return {old: m.cik for m in managers for old in m.previous_ciks}
