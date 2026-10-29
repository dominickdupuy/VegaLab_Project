"""Databento OPRA batches: inventory every job, pull SPX and SPXW closing quotes.

Lane: Hemdutt Rao (data adapter).

A Databento batch download is one tar per job: ``metadata.json`` (the query that
produced it), ``manifest.json``, ``condition.json`` and one or more ``.dbn.zst``
files. The jobs in the shared folder mix datasets, schemas and symbol groups, so
nothing is assumed from a file name: every tar is read for its query first.

The closing snapshot is the ``cbbo-1m`` bar for the last minute of regular trading,
the consolidated best bid and offer that OptionMetrics' ``best_bid`` and
``best_offer`` describe. Each job asks for the minute before the close and the
minute after it (15:59 to 16:01 New York, or 12:59 to 13:01 on an early close), so
the earlier of a file's two bars is the close. ``cbbo-1m`` writes a bar only for an
instrument whose book changed in that minute, so an illiquid contract can be absent
from the closing bar and present in the next one; each instrument therefore keeps
its earliest record in the closing window (``CLOSE_WINDOW``), and later records are
dropped.

Each tar is decoded in memory and only its SPX and SPXW rows are written out, so
the 20 GB of OPRA batches never have to sit on local disk at once.
"""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from optspread.realdata import ROOTS

NY = "America/New_York"


@dataclass(frozen=True, slots=True)
class BatchInfo:
    """One Databento job, as its own ``metadata.json`` describes it."""

    tar: str
    job_id: str
    dataset: str
    schema: str
    start: str  # ISO date of the query start (UTC)
    end: str
    n_symbols: int
    symbols: tuple[str, ...]
    dbn_files: tuple[str, ...]
    size_bytes: int

    @property
    def has_spx(self) -> bool:
        return any(s.split(".")[0] in ROOTS for s in self.symbols)

    @property
    def is_spx_close(self) -> bool:
        return self.dataset == "OPRA.PILLAR" and self.schema == "cbbo-1m" and self.has_spx

    def as_row(self) -> dict[str, Any]:
        return {
            "tar": self.tar,
            "job_id": self.job_id,
            "dataset": self.dataset,
            "schema": self.schema,
            "start": self.start,
            "end": self.end,
            "n_symbols": self.n_symbols,
            "symbols": ",".join(self.symbols),
            "dbn_files": ",".join(self.dbn_files),
            "size_bytes": self.size_bytes,
        }


def _iso(ns_or_iso: Any) -> str:
    if isinstance(ns_or_iso, int | float):
        return pd.Timestamp(int(ns_or_iso), unit="ns", tz="UTC").date().isoformat()
    return pd.Timestamp(str(ns_or_iso)).date().isoformat()


def parse_metadata(
    meta: dict[str, Any], tar: str, dbn_files: tuple[str, ...], size: int
) -> BatchInfo:
    q = meta.get("query", meta)
    syms = q.get("symbols")
    symbols = tuple(syms) if isinstance(syms, list) else (str(syms),)
    return BatchInfo(
        tar=tar,
        job_id=str(meta.get("job_id", Path(tar).stem)),
        dataset=str(q.get("dataset")),
        schema=str(q.get("schema")),
        start=_iso(q.get("start")),
        end=_iso(q.get("end")),
        n_symbols=len(symbols),
        symbols=symbols,
        dbn_files=dbn_files,
        size_bytes=size,
    )


def read_batch_info(path: Path) -> BatchInfo:
    """The job's query, read from inside its tar without extracting anything to disk."""
    with tarfile.open(path) as tf:
        members = tf.getmembers()
        meta_member = next(m for m in members if m.name.endswith("metadata.json"))
        handle = tf.extractfile(meta_member)
        if handle is None:
            raise ValueError(f"{path.name}: metadata.json is not a regular file")
        meta = json.loads(handle.read())
        dbn = tuple(Path(m.name).name for m in members if m.name.endswith(".dbn.zst"))
    return parse_metadata(meta, path.name, dbn, path.stat().st_size)


# -- OSI symbols ------------------------------------------------------------- #


def parse_osi(symbols: pd.Series) -> pd.DataFrame:
    """Split OSI option symbols (``"SPXW  250709C05500000"``) into their four fields.

    Root is the first six characters, right-padded; then expiry ``YYMMDD``, the right,
    and the strike in thousandths of a point. A symbol that does not parse is NaN.
    """
    s = symbols.astype("string")
    parts = s.str.extract(r"^(?P<root>.{1,6}?)\s*(?P<ymd>\d{6})(?P<cp>[CP])(?P<k>\d{8})$")
    return pd.DataFrame(
        {
            "root": parts["root"].str.strip(),
            "expiry": pd.to_datetime(parts["ymd"], format="%y%m%d", errors="coerce").dt.date,
            "cp": parts["cp"],
            "strike": pd.to_numeric(parts["k"], errors="coerce") / 1000.0,
        },
        index=symbols.index,
    )


# -- the closing snapshot ---------------------------------------------------- #

QUOTE_COLUMNS = ("date", "root", "expiry", "cp", "strike", "bid", "ask", "bid_size", "ask_size")
CONTRACT = ["root", "expiry", "cp", "strike"]

# The closing window: from the file's first bar (the close, 15:59 or 12:59 New York)
# through the bar one minute later, the last the job asks for. A contract with no
# update in the closing minute takes its quote from that next bar; nothing later
# than it is read, whatever else a file holds.
CLOSE_WINDOW = pd.Timedelta(minutes=1)


def closing_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    """SPX and SPXW closing quotes, one row per contract: its earliest bar in the window.

    ``frame`` is a decoded cbbo-1m DBN frame indexed by ``ts_recv`` (UTC) with the
    ``symbol`` column mapped. The window runs from the file's first bar, which is the
    close, through ``CLOSE_WINDOW`` after it, so a contract quoted in the closing bar
    keeps that quote and one missing from it keeps its first later quote rather than
    being lost for the day. Rows with no two-sided market keep NaN prices; the panel
    decides what is usable, not the extractor.
    """
    if frame.empty:
        return pd.DataFrame(columns=list(QUOTE_COLUMNS))
    first = frame.index.min()
    window = frame[(frame.index >= first) & (frame.index <= first + CLOSE_WINDOW)]
    osi = parse_osi(window["symbol"])
    keep = osi["root"].isin(ROOTS).to_numpy()
    rows, osi = window[keep], osi[keep]
    bid = rows["bid_px_00"].to_numpy(dtype=np.float64)
    ask = rows["ask_px_00"].to_numpy(dtype=np.float64)
    out = pd.DataFrame(
        {
            "date": rows.index.tz_convert(NY).date,
            "root": osi["root"].to_numpy(),
            "expiry": osi["expiry"].to_numpy(),
            "cp": osi["cp"].to_numpy(),
            "strike": osi["strike"].to_numpy(dtype=np.float64),
            "bid": np.where(bid > 0.0, bid, np.nan),
            "ask": np.where(ask > 0.0, ask, np.nan),
            "bid_size": rows["bid_sz_00"].to_numpy(dtype=np.float64),
            "ask_size": rows["ask_sz_00"].to_numpy(dtype=np.float64),
        }
    )
    # Each contract's earliest bar; within that bar the last record, as before.
    ts = pd.Series(rows.index.asi8, index=out.index)
    earliest = ts == ts.groupby([out[c] for c in CONTRACT]).transform("min")
    out = out[earliest.to_numpy()]
    return out.drop_duplicates(CONTRACT, keep="last").reset_index(drop=True)


def extract_spx_close(path: Path) -> tuple[date, pd.DataFrame]:  # pragma: no cover - vendor IO
    """Decode one daily cbbo-1m tar in memory and return its SPX/SPXW closing quotes."""
    import databento as db

    with tarfile.open(path) as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith(".dbn.zst"))
        handle = tf.extractfile(member)
        if handle is None:
            raise ValueError(f"{path.name}: {member.name} is not a regular file")
        store = db.DBNStore.from_bytes(io.BytesIO(handle.read()))
    quotes = closing_quotes(store.to_df())
    if quotes.empty:
        raise ValueError(f"{path.name}: no SPX/SPXW rows in the closing window")
    days = set(quotes["date"])
    if len(days) != 1:
        raise ValueError(f"{path.name}: expected one trading day, found {sorted(days)}")
    return days.pop(), quotes
