"""Build the real SPX data in the shared folder, one resumable stage at a time.

Lane: Hemdutt Rao (data adapter).

    python scripts/build_spx_panel.py inventory     --databento DIR [--max-mb 15]
    python scripts/build_spx_panel.py extract-opra  --databento DIR
    python scripts/build_spx_panel.py panel         [--years 2010 2026]
    python scripts/build_spx_panel.py fred

Every stage reads from and writes to ``VEGALAB_DATA`` (the team's shared folder) and
skips work whose output already exists, so a rerun after a dropped connection picks
up where it stopped. The tars are read where they sit; a cloud-only file is fetched
on demand, one at a time.

``inventory`` records every Databento job's query. Jobs above ``--max-mb`` are listed
by size and marked uninspected rather than downloaded: the daily closing snapshots
are small, and the large jobs (definitions and statistics for whole symbol groups)
are not needed to build the panel.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

from optspread.realdata import OPTIONMETRICS_END, data_root
from optspread.realdata.opra import extract_spx_close, read_batch_info
from optspread.realdata.optionmetrics import read_opprcd, read_secprd
from optspread.realdata.panel import (
    atm_term,
    coverage,
    daily_table,
    finalise,
    parity_spot,
    seam_frame,
    surface_grid,
    trading_calendar,
)
from optspread.realdata.surface import price_surface

DEFAULT_DATABENTO = Path.home() / "Library/CloudStorage/OneDrive-UniversityofFlorida/databento"
INVENTORY_FIELDS = (
    "tar",
    "job_id",
    "dataset",
    "schema",
    "start",
    "end",
    "n_symbols",
    "symbols",
    "dbn_files",
    "size_bytes",
    "inspected",
)


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def inventory(databento: Path, out: Path, max_mb: float) -> list[dict[str, str]]:
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, str]] = {}
    if out.exists():
        with out.open(newline="") as fh:
            done = {r["tar"]: r for r in csv.DictReader(fh) if r["inspected"] == "yes"}
    rows: list[dict[str, str]] = []
    tars = sorted(databento.glob("*.tar"))
    for i, tar in enumerate(tars, 1):
        if tar.name in done:
            rows.append(done[tar.name])
            continue
        size = tar.stat().st_size
        if size > max_mb * 1e6:
            rows.append({"tar": tar.name, "size_bytes": str(size), "inspected": "no"})
            continue
        info = read_batch_info(tar)
        rows.append({k: str(v) for k, v in info.as_row().items()} | {"inspected": "yes"})
        if i % 25 == 0:
            _log(f"inventory {i}/{len(tars)}")
            _write_inventory(out, rows)
    _write_inventory(out, rows)
    _log(f"inventory: {len(rows)} tars, {sum(r['inspected'] == 'yes' for r in rows)} inspected")
    return rows


def _write_inventory(out: Path, rows: list[dict[str, str]]) -> None:
    tmp = out.with_suffix(".csv.part")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INVENTORY_FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out)


def extract_opra(databento: Path, inventory_csv: Path, out_dir: Path) -> int:
    with inventory_csv.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["inspected"] == "yes"]
    jobs = [
        r
        for r in rows
        if r["dataset"] == "OPRA.PILLAR"
        and r["schema"] == "cbbo-1m"
        and any(s.split(".")[0] in ("SPX", "SPXW") for s in r["symbols"].split(","))
    ]
    _log(f"extract-opra: {len(jobs)} closing-snapshot jobs")
    written = 0
    for i, r in enumerate(sorted(jobs, key=lambda r: r["start"]), 1):
        day = r["start"]  # each closing job covers one trading day
        dest = out_dir / f"year={day[:4]}" / f"{day}.parquet"
        if dest.exists():
            continue
        got, quotes = extract_spx_close(databento / r["tar"])
        if got.isoformat() != day:
            raise ValueError(f"{r['tar']}: query day {day} but quotes for {got}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(".parquet.part")
        quotes.to_parquet(part, index=False, compression="zstd")
        part.replace(dest)
        written += 1
        _log(f"extract-opra {i}/{len(jobs)} {day}: {len(quotes):,} SPX/SPXW quotes")
    return written


OM_END = date.fromisoformat(OPTIONMETRICS_END)


def _priced_by_month(quotes: pd.DataFrame) -> pd.DataFrame:
    """Price month by month: parity groups never cross a date, and memory stays bounded."""
    if quotes.empty:
        return quotes
    month = pd.to_datetime(quotes["date"]).dt.to_period("M")
    return pd.concat([price_surface(g) for _, g in quotes.groupby(month)], ignore_index=True)


def _databento_year(root: Path, year: int) -> pd.DataFrame:
    files = sorted((root / "databento" / "spx_close" / f"year={year}").glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    q = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    q["am_settled"] = q["root"] == "SPX"  # the SPX root is AM-settled, SPXW is PM
    return q


def build_year(year: int) -> dict[str, pd.DataFrame]:
    """One year of the panel, written to its partition; returns the small summaries."""
    root = data_root()
    dest = root / "panel" / "spx" / f"year={year}" / "part.parquet"
    cache = root / "panel" / "_summaries" / f"year={year}"
    names = ("term", "counts", "spot_parity", "seam", "surface")
    if dest.exists() and all((cache / f"{n}.parquet").exists() for n in names):
        _log(f"panel {year}: cached")
        return {n: pd.read_parquet(cache / f"{n}.parquet") for n in names}
    frames: list[pd.DataFrame] = []
    dropped: dict[tuple[date, str], int] = {}
    om_overlap = pd.DataFrame()
    db_overlap = pd.DataFrame()

    if year <= OM_END.year:
        om = read_opprcd(root, year)
        priced = _priced_by_month(om[om["date"] <= OM_END])
        panel, _ = finalise(priced, "optionmetrics")
        frames.append(panel)
        om_overlap = panel[pd.to_datetime(panel["date"]) >= pd.Timestamp("2025-07-01")]
    db = _databento_year(root, year)
    if not db.empty:
        priced = _priced_by_month(db)
        panel, _ = finalise(priced, "databento")
        after = panel[panel["date"] > OM_END]
        frames.append(after)
        db_overlap = panel[panel["date"] <= OM_END]

    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not full.empty:
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(".parquet.part")
        full.to_parquet(part, index=False, compression="zstd")
        part.replace(dest)
    counts = full.groupby(["date", "source"]).size().rename("n_contracts").reset_index()
    seam = seam_frame(om_overlap, db_overlap) if not db_overlap.empty else pd.DataFrame()
    _log(f"panel {year}: {len(full):,} rows, {full['date'].nunique() if len(full) else 0} days")
    del dropped
    summary = {
        "term": atm_term(full) if not full.empty else pd.DataFrame(),
        "counts": counts,
        "spot_parity": parity_spot(full).rename("spot_parity").reset_index()
        if not full.empty
        else pd.DataFrame(),
        "seam": seam,
        "surface": surface_grid(full) if not full.empty else pd.DataFrame(),
    }
    cache.mkdir(parents=True, exist_ok=True)
    for name, frame in summary.items():
        frame.to_parquet(cache / f"{name}.parquet", index=False)
    return summary


def _cat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    kept = [f for f in frames if not f.empty]
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()


def build_panel(years: range, workers: int) -> None:
    root = data_root()
    out = root / "panel"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(build_year, years))
    term = _cat([r["term"] for r in results])
    counts = _cat([r["counts"] for r in results])
    sp = _cat([r["spot_parity"] for r in results])
    seam = _cat([r["seam"] for r in results])

    closes = pd.concat(
        [read_secprd(root, y) for y in years if y <= OM_END.year], ignore_index=True
    ).set_index("date")["close"]
    per_day = counts.groupby("date").agg(source=("source", "first"), n=("n_contracts", "sum"))
    daily = daily_table(
        term, closes, sp.set_index("date")["spot_parity"], per_day["source"], per_day["n"]
    )
    daily.to_parquet(out / "spx_daily.parquet", index=False)
    calendar = trading_calendar(pd.Index(closes.index), max(counts["date"]))
    cov = coverage(
        calendar,
        counts.groupby("date", as_index=False).agg(
            source=("source", "first"), n_contracts=("n_contracts", "sum")
        ),
    )
    cov.to_parquet(out / "coverage.parquet", index=False)
    seam.to_parquet(out / "seam_overlap.parquet", index=False)
    term.to_parquet(out / "atm_term.parquet", index=False)
    _cat([r["surface"] for r in results]).to_parquet(out / "surface_grid.parquet", index=False)
    missing = cov[cov["missing"]]
    _log(
        f"panel: {len(cov)} trading days {cov['date'].min()}..{cov['date'].max()}, "
        f"{len(missing)} missing: {[d.isoformat() for d in missing['date']][:20]}"
    )


FRED_SERIES = ("DGS3MO", "SOFR", "SP500")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def fred_check() -> pd.DataFrame:
    """Public FRED series against what the quotes implied: a sanity check, not an input.

    The 3-month bill and SOFR bracket the parity-implied 30-day rate (option boxes
    fund near SOFR, a little above bills); the S&P 500 close checks the spot, which
    after the OptionMetrics ceiling is the prepaid forward of the shortest expiry.
    Only public series are fetched; nothing of ours is sent.
    """
    import io
    import urllib.request

    root = data_root()
    ref = root / "reference" / "fred"
    ref.mkdir(parents=True, exist_ok=True)
    frames = []
    for sid in FRED_SERIES:
        with urllib.request.urlopen(FRED_URL.format(sid), timeout=60) as resp:
            raw = resp.read()
        (ref / f"{sid}.csv").write_bytes(raw)
        f = pd.read_csv(io.BytesIO(raw), na_values=".")
        f.columns = ["date", sid]
        f["date"] = pd.to_datetime(f["date"]).dt.date
        frames.append(f.set_index("date"))
    fred = pd.concat(frames, axis=1)
    daily = pd.read_parquet(root / "panel" / "spx_daily.parquet").set_index("date")
    j = daily[["spot", "rate30", "source"]].join(fred, how="left")
    j["year"] = [d.year for d in j.index]
    j["rate_minus_bill_bp"] = (j["rate30"] - j["DGS3MO"] / 100) * 1e4
    j["rate_minus_sofr_bp"] = (j["rate30"] - j["SOFR"] / 100) * 1e4
    j["spot_vs_sp500_bp"] = (j["spot"] / j["SP500"] - 1) * 1e4
    summary = j.groupby(["year", "source"])[
        ["rate_minus_bill_bp", "rate_minus_sofr_bp", "spot_vs_sp500_bp"]
    ].median()
    summary.reset_index().to_parquet(root / "panel" / "fred_check.parquet", index=False)
    _log("fred check (medians, basis points):\n" + summary.round(1).to_string())
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--databento", type=Path, default=DEFAULT_DATABENTO)
    inv.add_argument("--max-mb", type=float, default=15.0)
    ext = sub.add_parser("extract-opra")
    ext.add_argument("--databento", type=Path, default=DEFAULT_DATABENTO)
    pan = sub.add_parser("panel")
    pan.add_argument("--years", type=int, nargs=2, default=(2010, 2026))
    pan.add_argument("--workers", type=int, default=3)
    sub.add_parser("fred")
    args = ap.parse_args(argv)

    root = data_root()
    inventory_csv = root / "databento" / "inventory.csv"
    if args.stage == "inventory":
        inventory(args.databento, inventory_csv, args.max_mb)
    elif args.stage == "extract-opra":
        n = extract_opra(args.databento, inventory_csv, root / "databento" / "spx_close")
        _log(f"extract-opra: {n} new days written")
    elif args.stage == "panel":
        build_panel(range(args.years[0], args.years[1] + 1), args.workers)
    elif args.stage == "fred":
        fred_check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
