#!/usr/bin/env python3
"""
Resample a Magnavis magnetic export CSV to **1 Hz** by flooring ``timestamp`` to whole seconds
and taking the **arithmetic mean** of all numeric samples in that second **per sensor_id**.

Input columns (typical): id, sensor_id, timestamp, b_x, b_y, b_z, lat, lon, alt, theta_x, theta_y, theta_z

Output columns: id, sensor_id, timestamp, <numeric means>, n_samples
  - ``timestamp`` is the floored second (no sub-second fraction).
  - ``n_samples`` counts raw rows averaged for that row.
  - ``id`` is a fresh sequential integer (original ``id`` is dropped).

Example::

  python tools/resample_magnetic_csv_to_1hz.py \\
    --input magnetic_data_20260426_060000_to_20260427_090000.csv \\
    --output magnetic_data_20260426_060000_to_20260427_090000_1hz.csv \\
    --chunksize 400000
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


NUMERIC_COLS = ["b_x", "b_y", "b_z", "lat", "lon", "alt", "theta_x", "theta_y", "theta_z"]


def _read_usecols(path: Path) -> List[str]:
    head = pd.read_csv(path, nrows=0)
    cols = list(head.columns)
    need = ["sensor_id", "timestamp"] + [c for c in NUMERIC_COLS if c in cols]
    missing = {"sensor_id", "timestamp"} - set(cols)
    if missing:
        raise SystemExit(f"CSV missing required columns: {sorted(missing)}")
    for c in NUMERIC_COLS:
        if c not in cols:
            raise SystemExit(f"CSV missing expected numeric column {c!r}; have {cols}")
    return need


def resample_to_1hz(
    input_path: Path,
    output_path: Path,
    *,
    chunksize: int,
    time_start: str | None,
    time_end: str | None,
) -> None:
    usecols = _read_usecols(input_path)
    t0 = pd.to_datetime(time_start, errors="coerce") if (time_start or "").strip() else None
    t1 = pd.to_datetime(time_end, errors="coerce") if (time_end or "").strip() else None
    if time_start and pd.isna(t0):
        raise SystemExit(f"Invalid --start: {time_start!r}")
    if time_end and pd.isna(t1):
        raise SystemExit(f"Invalid --end: {time_end!r}")

    sum_df: pd.DataFrame | None = None
    cnt_df: pd.DataFrame | None = None
    n_raw = 0

    for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=int(chunksize)):
        n_raw += len(chunk)
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
        chunk = chunk.dropna(subset=["timestamp"])
        chunk["tsec"] = chunk["timestamp"].dt.floor("s")
        if t0 is not None:
            chunk = chunk.loc[chunk["tsec"] >= t0]
        if t1 is not None:
            chunk = chunk.loc[chunk["tsec"] <= t1]
        if chunk.empty:
            continue
        for c in NUMERIC_COLS:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

        gsum = chunk.groupby(["sensor_id", "tsec"], sort=False)[NUMERIC_COLS].sum()
        gcnt = chunk.groupby(["sensor_id", "tsec"], sort=False)[NUMERIC_COLS].count()
        if sum_df is None:
            sum_df = gsum
            cnt_df = gcnt
        else:
            sum_df = sum_df.add(gsum, fill_value=0.0)
            cnt_df = cnt_df.add(gcnt, fill_value=0.0)

    if sum_df is None or cnt_df is None or sum_df.empty:
        raise SystemExit("No rows after filtering; check --input / --start / --end.")

    mean_df = sum_df.div(cnt_df.replace(0, pd.NA)).astype("float64")
    out = mean_df.reset_index().rename(columns={"tsec": "timestamp"})
    ns = (
        cnt_df[[NUMERIC_COLS[0]]]
        .rename(columns={NUMERIC_COLS[0]: "n_samples"})
        .reset_index()
        .rename(columns={"tsec": "timestamp"})
    )
    out = out.merge(ns, on=["sensor_id", "timestamp"], how="left")
    out["n_samples"] = out["n_samples"].fillna(0).astype("int64")

    out = out.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    out.insert(0, "id", range(1, len(out) + 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(
        f"Wrote {output_path} ({len(out)} rows, 1 Hz per sensor_id; averaged from {n_raw} raw rows read)."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Resample magnetic CSV to 1 Hz (mean within each second per sensor).")
    p.add_argument("--input", "-i", type=Path, required=True, help="Source magnetic_data_*.csv")
    p.add_argument("--output", "-o", type=Path, required=True, help="Destination CSV path")
    p.add_argument("--chunksize", type=int, default=400_000, help="Rows per read_csv chunk (default 400000)")
    p.add_argument(
        "--start",
        default="",
        help="Optional inclusive lower bound on **floored** second (e.g. 2026-04-27 06:15:00).",
    )
    p.add_argument(
        "--end",
        default="",
        help="Optional inclusive upper bound on **floored** second (e.g. 2026-04-27 06:35:00).",
    )
    args = p.parse_args()
    resample_to_1hz(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        chunksize=args.chunksize,
        time_start=str(args.start or "").strip() or None,
        time_end=str(args.end or "").strip() or None,
    )


if __name__ == "__main__":
    main()
