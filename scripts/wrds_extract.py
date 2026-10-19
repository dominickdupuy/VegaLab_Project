"""Download OptionMetrics opprcd (every contract, daily) for a ticker list to zstd Parquet.

Lane: Dominick Dupuy (the WRDS pull).

    pip install -e ".[dev,data,wrds]"
    WRDS_USERNAME=<you> python -m scripts.wrds_extract "$VEGALAB_DATA" 2010 2025

One file per (ticker, year): OUT/opprcd/year=YYYY/<TICKER>.parquet, written to .part and
renamed on success, so a rerun resumes by skipping finished files. Underlying daily
prices go to OUT/secprd/year=YYYY/<TICKER>.parquet the same way. The ticker list maps
each ticker to its OptionMetrics secids (default: scripts/wrds_secids.json). The
account is ``$WRDS_USERNAME``; the wrds client asks for its password or reads
~/.pgpass. docs/DATA.md, Extraction, has the rest.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:  # the wrds extra; imported where it is used, so --help needs none of it
    import wrds

SECIDS = Path(__file__).resolve().parent / "wrds_secids.json"
USERNAME_ENV = "WRDS_USERNAME"
DEFAULT_USERNAME = "dominickdupuy"
END_DATE = "2025-08-29"  # vendor ceiling at extraction, recorded in docs/DATA.md
CHUNK = 250_000
ZSTD_LEVEL = 19
LOG_FILENAME = "download.log"


def log(out: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with (out / LOG_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def connect() -> wrds.Connection:
    import wrds

    username = os.environ.get(USERNAME_ENV) or DEFAULT_USERNAME
    return wrds.Connection(wrds_username=username, autoconnect=True, verbose=False)


PG_TO_ARROW = {
    "double precision": pa.float64(),
    "date": pa.date32(),
    "integer": pa.int32(),
    "bigint": pa.int64(),
    "smallint": pa.int16(),
    "numeric": pa.float64(),
    "real": pa.float32(),
    "character varying": pa.string(),
    "text": pa.string(),
    "character": pa.string(),
}
_SCHEMAS: dict[str, pa.Schema] = {}


def schema_for(db: wrds.Connection, table: str) -> pa.Schema:
    if table not in _SCHEMAS:
        df = db.raw_sql(
            "select column_name, data_type from information_schema.columns "
            f"where table_schema='optionm' and table_name='{table}' "
            "order by ordinal_position"
        )
        pairs = zip(df.column_name, df.data_type, strict=True)
        _SCHEMAS[table] = pa.schema([(c, PG_TO_ARROW[t]) for c, t in pairs])
    return _SCHEMAS[table]


def fetch(db: wrds.Connection, sql: str, dest: Path, schema: pa.Schema) -> int:
    from sqlalchemy import text

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".parquet.part")
    rows = 0
    writer: pq.ParquetWriter | None = None
    # wrds' engine runs AUTOCOMMIT; a server-side (streaming) cursor needs a transaction.
    with (
        db.engine.connect().execution_options(
            isolation_level="READ COMMITTED", stream_results=True
        ) as conn,
        conn.begin(),
    ):
        result = conn.execute(text(sql))
        cols = list(result.keys())
        while batch := result.fetchmany(CHUNK):
            frame = pd.DataFrame.from_records(batch, columns=cols)
            table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    part,
                    schema,
                    compression="zstd",
                    compression_level=ZSTD_LEVEL,
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.write_table(table, row_group_size=CHUNK)
            rows += len(batch)
    if writer is None:  # no rows: write nothing, but leave a marker so reruns skip it
        dest.with_suffix(".empty").touch()
        return 0
    writer.close()
    os.replace(part, dest)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("out", type=Path, help="output folder, e.g. $VEGALAB_DATA")
    parser.add_argument("first_year", type=int, help="first opprcd/secprd year to pull")
    parser.add_argument("last_year", type=int, help="last year to pull, inclusive")
    parser.add_argument(
        "secids",
        type=Path,
        nargs="?",
        default=SECIDS,
        help=f"JSON of ticker -> OptionMetrics secids (default: {SECIDS.name})",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    tickers: dict[str, list[int]] = json.loads(Path(args.secids).read_text())

    out.mkdir(parents=True, exist_ok=True)
    db = connect()
    total = 0
    t0 = time.time()
    for year in range(int(args.first_year), int(args.last_year) + 1):
        for ticker, ids in tickers.items():
            idlist = ",".join(str(i) for i in ids)
            jobs = [
                (
                    f"opprcd{year}",
                    f"select * from optionm.opprcd{year} where secid in ({idlist}) "
                    f"and date <= '{END_DATE}' "
                    "order by secid, date, exdate, cp_flag, strike_price",
                ),
                (
                    f"secprd{year}",
                    f"select * from optionm.secprd{year} where secid in ({idlist}) "
                    f"and date <= '{END_DATE}' order by secid, date",
                ),
            ]
            for table, sql in jobs:
                kind = table[:6]
                dest = out / kind / f"year={year}" / f"{ticker}.parquet"
                if dest.exists() or dest.with_suffix(".empty").exists():
                    continue
                for attempt in range(1, 6):
                    try:
                        t = time.time()
                        n = fetch(db, sql, dest, schema_for(db, table))
                        total += n if kind == "opprcd" else 0
                        size = dest.stat().st_size / 1e6 if dest.exists() else 0.0
                        log(
                            out,
                            f"{kind} {year} {ticker}: {n:,} rows, {size:,.1f} MB, "
                            f"{time.time() - t:,.0f}s (opprcd total {total:,}, "
                            f"elapsed {(time.time() - t0) / 3600:.2f}h)",
                        )
                        break
                    except Exception as exc:  # network drops on multi-hour pulls
                        log(out, f"ERROR {kind} {year} {ticker} attempt {attempt}: {exc!r}")
                        time.sleep(30 * attempt)
                        with contextlib.suppress(Exception):
                            db.close()
                        db = connect()
                else:
                    log(out, f"GAVE UP {kind} {year} {ticker}")
    log(out, f"DONE opprcd rows {total:,} in {(time.time() - t0) / 3600:.2f}h")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
