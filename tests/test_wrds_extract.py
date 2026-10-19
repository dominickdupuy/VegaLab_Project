"""Dominick Dupuy: the WRDS pull's command line, without WRDS.

The script used to read its arguments and import wrds at module import, so neither
``--help`` nor an import worked anywhere but on the machine that ran the pull. The
connection and the query are stood in for; what is tested is the argument parsing,
the account, and the resume rule.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from scripts import wrds_extract


def test_help_needs_neither_wrds_nor_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_:
        wrds_extract.main(["--help"])
    assert exit_.value.code == 0
    assert "out first_year last_year [secids]" in capsys.readouterr().out


def test_the_account_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("wrds")
    fake.Connection = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wrds", fake)
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    assert wrds_extract.connect()["wrds_username"] == "dominickdupuy"
    monkeypatch.setenv("WRDS_USERNAME", "someone_else")
    assert wrds_extract.connect()["wrds_username"] == "someone_else"


def test_the_vendored_ticker_list_maps_tickers_to_secids() -> None:
    secids = json.loads(wrds_extract.SECIDS.read_text())
    assert secids["SPX"] == [108105]
    assert all(isinstance(i, int) for ids in secids.values() for i in ids)


class FakeDB:
    def close(self) -> None:
        pass


def test_a_pull_writes_per_ticker_year_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queries: list[str] = []

    def fetch(db: Any, sql: str, dest: Path, schema: Any) -> int:
        queries.append(sql)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"parquet")
        return 3

    monkeypatch.setattr(wrds_extract, "connect", FakeDB)
    monkeypatch.setattr(wrds_extract, "fetch", fetch)
    monkeypatch.setattr(wrds_extract, "schema_for", lambda db, table: None)
    tickers = tmp_path / "secids.json"
    tickers.write_text(json.dumps({"SPX": [108105], "XSP": [125063, 189691]}))
    out = tmp_path / "out"
    assert wrds_extract.main([str(out), "2024", "2025", str(tickers)]) == 0
    assert len(queries) == 8  # 2 years x 2 tickers x (opprcd, secprd)
    assert (out / "opprcd" / "year=2025" / "XSP.parquet").exists()
    assert "secid in (125063,189691) and date <= '2025-08-29'" in queries[2]
    assert "DONE opprcd rows 12" in (out / "download.log").read_text()

    queries.clear()
    assert wrds_extract.main([str(out), "2024", "2025", str(tickers)]) == 0
    assert queries == []
