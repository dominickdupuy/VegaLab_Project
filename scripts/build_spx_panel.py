"""Build the real SPX data in the shared folder, one resumable stage at a time.

Lane: Hemdutt Rao (data adapter).

    python scripts/build_spx_panel.py inventory     --databento DIR [--max-mb 15]
    python scripts/build_spx_panel.py extract-opra  --databento DIR
    python scripts/build_spx_panel.py panel         [--years 2010 2026]
    python scripts/build_spx_panel.py fred
    python scripts/build_spx_panel.py manifest      [--hash-inputs]
    python scripts/build_spx_panel.py check         [--panel DIR]

Every stage reads from and writes to ``VEGALAB_DATA`` (the team's shared folder) and
skips work whose output already exists, so a rerun after a dropped connection picks
up where it stopped. The tars are read where they sit; a cloud-only file is fetched
on demand, one at a time.

``inventory`` records every Databento job's query. Jobs above ``--max-mb`` are listed
by size and marked uninspected rather than downloaded: the daily closing snapshots
are small, and the large jobs (definitions and statistics for whole symbol groups)
are not needed to build the panel.

``manifest`` records the sha256 and size of every file the build wrote, and the size
of every raw ``opprcd``/``secprd`` input it read; ``--hash-inputs`` adds the inputs'
sha256, which reads every byte of them.

``check`` builds the real provider's chain for every day of both splits from a panel
(``--panel``, default the shared folder's) and runs the leak suite's chain checks on
each, calendar arbitrage among them (#29); it exits 1 if any day fails. Point it at a
local copy of ``panel/`` to keep it off the sync client.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from optspread.config import MarketConfig
    from optspread.realdata.provider import SurfaceStore
    from optspread.types import Chain

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


def extract_opra(
    databento: Path, inventory_csv: Path, out_dir: Path, shard: tuple[int, int] = (0, 1)
) -> int:
    with inventory_csv.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["inspected"] == "yes"]
    jobs = [
        r
        for r in rows
        if r["dataset"] == "OPRA.PILLAR"
        and r["schema"] == "cbbo-1m"
        and any(s.split(".")[0] in ("SPX", "SPXW") for s in r["symbols"].split(","))
    ]
    jobs = sorted(jobs, key=lambda r: r["start"])
    k, n = shard  # shard k of n takes every n-th day, so n processes split the work
    jobs = jobs[k::n]
    _log(f"extract-opra: {len(jobs)} closing-snapshot jobs (shard {k} of {n})")
    written = 0
    for i, r in enumerate(jobs, 1):
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
# Bump when a change to optspread.realdata alters what a year builds to: the cached
# summaries of the old version are then ignored and every year is rebuilt. The cache
# is keyed on this number and nothing else, so a change to the normaliser, the pricer,
# ``finalise``, ``atm_term`` or ``surface_grid`` without a bump is silently not built
# (a change to the daily table alone needs none: ``build_panel`` recomputes it from
# the summaries every run). 3: an AM expiry before a closure settles the trading day
# before, and every vendor root is SPX or SPXW (#31). 4: the root is the settlement
# flag on every row (the 2010 PM weeklies under SPX are SPXW), an expiry dated on an
# unscheduled closure settles the trading day before, an expiry whose forward is off
# the day's curve is dropped as stale, and the daily carry is against the parity spot
# (``carry30_parity``) (#31).
PANEL_VERSION = 4


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
    q = pd.concat([_read(f) for f in files], ignore_index=True)
    q["am_settled"] = q["root"] == "SPX"  # the SPX root is AM-settled, SPXW is PM
    return q


def build_year(year: int) -> dict[str, pd.DataFrame]:
    """One year of the panel, written to its partition; returns the small summaries."""
    root = data_root()
    dest = root / "panel" / "spx" / f"year={year}" / "part.parquet"
    cache = root / "panel" / "_summaries" / f"v{PANEL_VERSION}" / f"year={year}"
    names = ("term", "counts", "spot_parity", "seam", "surface")
    if dest.exists() and all((cache / f"{n}.parquet").exists() for n in names):
        _log(f"panel {year}: cached")
        return {n: _read(cache / f"{n}.parquet") for n in names}
    frames: list[pd.DataFrame] = []
    dropped: list[pd.Series] = []
    om_overlap = pd.DataFrame()
    db_overlap = pd.DataFrame()

    if year <= OM_END.year:
        om = read_opprcd(root, year)
        priced = _priced_by_month(om[om["date"] <= OM_END])
        panel, lost = finalise(priced, "optionmetrics")
        frames.append(panel)
        dropped.append(lost)
        om_overlap = panel[pd.to_datetime(panel["date"]) >= pd.Timestamp("2025-07-01")]
    db = _databento_year(root, year)
    if not db.empty:
        priced = _priced_by_month(db)
        panel, lost = finalise(priced, "databento")
        dropped.append(lost[[d > OM_END for d in lost.index]])
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
    if dropped:
        lost_by_day = pd.concat(dropped).groupby(level=0).sum().rename("n_dropped")
        counts = counts.merge(lost_by_day, left_on="date", right_index=True, how="left")
        counts["n_dropped"] = counts["n_dropped"].fillna(0).astype("int64")
    seam = seam_frame(om_overlap, db_overlap) if not db_overlap.empty else pd.DataFrame()
    _log(f"panel {year}: {len(full):,} rows, {full['date'].nunique() if len(full) else 0} days")
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


def _read(path: Path, attempts: int = 6) -> pd.DataFrame:
    """Read a parquet from the shared folder, riding out the sync client's timeouts."""
    for attempt in range(1, attempts + 1):
        try:
            return pd.read_parquet(path)
        except (TimeoutError, OSError) as exc:
            if attempt == attempts:
                raise
            _log(f"read {path.name} attempt {attempt}: {exc!r}")
            time.sleep(15 * attempt)
    raise AssertionError("unreachable")


def _write(frame: pd.DataFrame, dest: Path, attempts: int = 5) -> None:
    """Write beside ``dest`` under a fresh name, then rename over it.

    Opening a synced OneDrive file to overwrite it can time out while the sync
    client holds it; a new file never does, and the rename is atomic. Retried with
    a growing pause before giving up loudly.
    """
    for attempt in range(1, attempts + 1):
        part = dest.with_name(f"{dest.name}.{os.getpid()}.{attempt}.part")
        try:
            frame.to_parquet(part, index=False)
            part.replace(dest)
            return
        except (TimeoutError, OSError) as exc:
            _log(f"write {dest.name} attempt {attempt}: {exc!r}")
            time.sleep(10 * attempt)
    raise TimeoutError(f"could not write {dest} after {attempts} attempts")


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
    _write(daily, out / "spx_daily.parquet")
    calendar = trading_calendar(pd.Index(closes.index), max(counts["date"]))
    cov = coverage(
        calendar,
        counts.groupby("date", as_index=False).agg(
            source=("source", "first"),
            n_contracts=("n_contracts", "sum"),
            n_dropped=("n_dropped", "sum"),
        ),
    )
    _write(cov, out / "coverage.parquet")
    _write(seam, out / "seam_overlap.parquet")
    _write(term, out / "atm_term.parquet")
    _write(_cat([r["surface"] for r in results]), out / "surface_grid.parquet")
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
    daily = _read(root / "panel" / "spx_daily.parquet").set_index("date")
    j = daily[["spot", "rate30", "source"]].join(fred, how="left")
    j["year"] = [d.year for d in j.index]
    j["rate_minus_bill_bp"] = (j["rate30"] - j["DGS3MO"] / 100) * 1e4
    j["rate_minus_sofr_bp"] = (j["rate30"] - j["SOFR"] / 100) * 1e4
    j["spot_vs_sp500_bp"] = (j["spot"] / j["SP500"] - 1) * 1e4
    summary = j.groupby(["year", "source"])[
        ["rate_minus_bill_bp", "rate_minus_sofr_bp", "spot_vs_sp500_bp"]
    ].median()
    _write(summary.reset_index(), root / "panel" / "fred_check.parquet")
    _log("fred check (medians, basis points):\n" + summary.round(1).to_string())
    return summary


def _sha256(path: Path, attempts: int = 6) -> str:
    """The file's sha256, retrying the sync client's read timeouts from the start."""
    import hashlib

    for attempt in range(1, attempts + 1):
        h = hashlib.sha256()
        try:
            with path.open("rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            return h.hexdigest()
        except (TimeoutError, OSError) as exc:
            if attempt == attempts:
                raise
            _log(f"hash {path.name} attempt {attempt}: {exc!r}")
            time.sleep(15 * attempt)
    raise AssertionError("unreachable")


# The raw inputs the panel stage reads, as the WRDS pull (or the vegalab import) left
# them. They are recorded beside what the build wrote, so a teammate can tell whether
# a different panel came from different code or from different inputs.
RAW_INPUTS = ("opprcd", "secprd")


def raw_inputs(root: Path) -> list[Path]:
    """Every ``opprcd``/``secprd`` Parquet under ``root``, in path order."""
    return sorted(f for kind in RAW_INPUTS for f in (root / kind).rglob("*.parquet"))


def data_manifest(repo_copy: Path, hash_inputs: bool = False) -> dict[str, dict[str, object]]:
    """sha256 and size of every built file in the shared folder: what the repo can verify.

    Covers the inventory, the extracted closing quotes and the panel (not the
    per-year cache) under ``files``, and the raw ``opprcd``/``secprd`` inputs under
    ``inputs``. An input always has its size; its sha256 only with ``hash_inputs``,
    because that reads every byte of several gigabytes through the sync client, and
    without it the entry says ``null`` rather than leaving the file out. The same
    JSON is written to the shared folder and to ``repo_copy``, so a teammate can
    check that their copy of the data is the one the code was built and tested
    against.
    """
    import json

    root = data_root()
    files = sorted(
        [root / "databento" / "inventory.csv"]
        + list((root / "databento" / "spx_close").rglob("*.parquet"))
        + [f for f in (root / "panel").rglob("*.parquet") if "_summaries" not in f.parts]
    )
    out: dict[str, dict[str, object]] = {}
    total = 0
    for f in files:
        h = _sha256(f)
        size = f.stat().st_size
        total += size
        out[f.relative_to(root).as_posix()] = {"sha256": h, "bytes": size}
    inputs: dict[str, dict[str, object]] = {}
    raw_total = 0
    for f in raw_inputs(root):
        size = f.stat().st_size
        raw_total += size
        inputs[f.relative_to(root).as_posix()] = {
            "sha256": _sha256(f) if hash_inputs else None,
            "bytes": size,
        }
    blob = {
        "panel_version": PANEL_VERSION,
        "files": out,
        "inputs": inputs,
        "inputs_hashed": hash_inputs,
    }
    text = json.dumps(blob, indent=1, sort_keys=True)
    (root / "MANIFEST.json").write_text(text + "\n")
    repo_copy.write_text(text + "\n")
    _log(f"manifest: {len(out)} files, {total / 1e9:.2f} GB")
    if hash_inputs:
        _log(f"manifest: {len(inputs)} raw inputs, {raw_total / 1e9:.2f} GB, hashed")
    else:
        _log(
            f"manifest: {len(inputs)} raw inputs, {raw_total / 1e9:.2f} GB, recorded by size "
            "only; pass --hash-inputs to add their sha256 (reads every byte, slow on the "
            "synced folder)"
        )
    return out


def calendar_check(panel_dir: Path) -> int:
    """The leak suite's chain checks on the real provider's chain for every day of both
    splits of the panel under ``panel_dir``. Prints a line per split and the first
    failing days; returns how many days failed."""
    from optspread.config import MarketConfig
    from optspread.leaks import chain_violations
    from optspread.realdata.provider import HistoricalChainProvider, SurfaceStore

    grid = _read(panel_dir / "surface_grid.parquet")
    daily = _read(panel_dir / "spx_daily.parquet")
    cfg = MarketConfig()
    failed = 0
    for split in ("train", "eval"):
        store = SurfaceStore.from_frames(grid, daily, split)
        bad = [
            (day, v)
            for day, chain in zip(store.dates, day_chains(cfg, store), strict=True)
            if (v := chain_violations(chain))
        ]
        failed += len(bad)
        starts = HistoricalChainProvider(cfg, store).n_starts
        _log(f"check {split}: {store.n} days, {starts} window starts, {len(bad)} days failing")
        for day, v in bad[:10]:
            _log(f"  {day}: {v[0]}")
    return failed


def day_chains(config: MarketConfig, store: SurfaceStore) -> list[Chain]:
    """The chain the provider serves for each day of ``store``, at the start of a window."""
    from optspread.realdata.provider import HistoricalChainProvider

    provider = HistoricalChainProvider(
        config.model_copy(update={"n_days": min(config.n_days, store.n - 1)}), store
    )
    out: list[Chain] = []
    for i in range(store.n):
        provider._start, provider._day = i, 0
        out.append(provider._chain())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--databento", type=Path, default=DEFAULT_DATABENTO)
    inv.add_argument("--max-mb", type=float, default=15.0)
    ext = sub.add_parser("extract-opra")
    ext.add_argument("--databento", type=Path, default=DEFAULT_DATABENTO)
    ext.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("K", "N"))
    pan = sub.add_parser("panel")
    pan.add_argument("--years", type=int, nargs=2, default=(2010, 2026))
    pan.add_argument("--workers", type=int, default=3)
    sub.add_parser("fred")
    man = sub.add_parser("manifest")
    man.add_argument("--repo-copy", type=Path, default=Path("docs/DATA_MANIFEST.json"))
    chk = sub.add_parser("check")
    chk.add_argument("--panel", type=Path, default=None, help="a panel/ folder (default: shared)")
    man.add_argument(
        "--hash-inputs",
        action="store_true",
        help="also sha256 the raw opprcd/secprd inputs (reads several GB; sizes are always kept)",
    )
    args = ap.parse_args(argv)

    root = data_root()
    inventory_csv = root / "databento" / "inventory.csv"
    if args.stage == "inventory":
        inventory(args.databento, inventory_csv, args.max_mb)
    elif args.stage == "extract-opra":
        n = extract_opra(
            args.databento, inventory_csv, root / "databento" / "spx_close", tuple(args.shard)
        )
        _log(f"extract-opra: {n} new days written")
    elif args.stage == "panel":
        build_panel(range(args.years[0], args.years[1] + 1), args.workers)
    elif args.stage == "fred":
        fred_check()
    elif args.stage == "manifest":
        data_manifest(args.repo_copy, hash_inputs=bool(args.hash_inputs))
    elif args.stage == "check":
        return 1 if calendar_check(args.panel or root / "panel") else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
