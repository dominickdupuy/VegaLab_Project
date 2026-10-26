"""Stage the vegalab repo's OptionMetrics extract as the per-year files the panel reads.

Lane: Hemdutt Rao (data adapter).

    python -m scripts.import_vegalab_extract --source ../vegalab/data [--years 2010 2022]
    python scripts/build_spx_panel.py panel --years 2010 2022 --workers 1

The vegalab extraction (``vegalab/data/EXTRACTION.md``) holds the same WRDS ``opprcd``
rows as ``scripts/wrds_extract.py`` writes, as one gzipped CSV over every development
date (``optionmetrics_spx_quotes_devall.csv.gz``, through 2022-08-24; the holdout
years are not in it), and the ``secprd`` closes as ``optionmetrics_spx_open.csv``.
This writes them where ``optspread.realdata.optionmetrics`` looks, one zstd Parquet
per year under ``VEGALAB_DATA``, streaming the CSV so memory stays bounded. Nothing is
transformed: the columns are the vendor's, and normalising them stays the reader's job.
Years whose files exist are skipped, so a killed import resumes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from optspread.realdata import data_root
from optspread.realdata.optionmetrics import OPPRCD_COLUMNS, opprcd_path, secprd_path

QUOTES = "optionmetrics_spx_quotes_devall.csv.gz"
CLOSES = "optionmetrics_spx_open.csv"
CHUNK_ROWS = 2_000_000
# Fixed per column, so every chunk has the schema of the first (a chunk whose forwards
# are all empty would otherwise read them as another type).
DTYPES = {c: "float64" for c in OPPRCD_COLUMNS if c not in ("date", "exdate")} | {
    "symbol": "string",
    "cp_flag": "string",
}


def import_closes(source: Path, root: Path, years: range) -> None:
    closes = pd.read_csv(source / CLOSES, usecols=["date", "close"], parse_dates=["date"])
    for year, g in closes.groupby(closes["date"].dt.year):
        dest = secprd_path(root, int(year))
        if int(year) in years and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            g.to_parquet(dest, index=False, compression="zstd")


def import_quotes(source: Path, root: Path, years: range) -> dict[int, int]:
    todo = {y for y in years if not opprcd_path(root, y).exists()}
    writers: dict[int, pq.ParquetWriter] = {}
    rows: dict[int, int] = {}
    reader = pd.read_csv(
        source / QUOTES,
        usecols=OPPRCD_COLUMNS,
        dtype=DTYPES,
        parse_dates=["date", "exdate"],
        chunksize=CHUNK_ROWS,
    )
    try:
        for chunk in reader:
            for year, g in chunk.groupby(chunk["date"].dt.year):
                y = int(year)
                if y not in todo:
                    continue
                table = pa.Table.from_pandas(g[OPPRCD_COLUMNS], preserve_index=False)
                if y not in writers:
                    part = opprcd_path(root, y).with_suffix(".parquet.part")
                    part.parent.mkdir(parents=True, exist_ok=True)
                    writers[y] = pq.ParquetWriter(part, table.schema, compression="zstd")
                writers[y].write_table(table)
                rows[y] = rows.get(y, 0) + len(g)
            print(f"  read through {chunk['date'].max():%Y-%m-%d}", flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    for y in writers:  # renamed only once the whole CSV is read, so a part is never final
        part = opprcd_path(root, y).with_suffix(".parquet.part")
        part.replace(opprcd_path(root, y))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--source", type=Path, required=True, help="vegalab/data")
    parser.add_argument("--years", type=int, nargs=2, default=(2010, 2022))
    args = parser.parse_args(argv)
    years = range(args.years[0], args.years[1] + 1)
    root = data_root()
    import_closes(args.source, root, years)
    rows = import_quotes(args.source, root, years)
    for y in sorted(rows):
        print(f"opprcd {y}: {rows[y]:,} rows -> {opprcd_path(root, y)}")
    print(f"secprd and opprcd for {years.start}..{years.stop - 1} under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
