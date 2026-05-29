#!/usr/bin/env python3
"""
Build synthetic app.log lines for simple forecast baselines (EWMA / median / Savitzky–Golay)
using the same AnomalyDetector.calculate_differences path as the GUI app.

Invoked by ``run_suite.py`` in this directory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# .../src/benchmark_feb13_2026/this_file.py → src/
SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Anomaly_detector import AnomalyDetector  # noqa: E402


def _parse_csv_end_arg(raw: str) -> pd.Timestamp | None:
    raw = str(raw or "").strip()
    if not raw:
        return None
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _mag_series_from_export(
    csv_path: Path,
    sensor_tag: str,
    end_cap: pd.Timestamp | None = None,
) -> tuple[list[pd.Timestamp], list[float]]:
    usecols = ["sensor_id", "timestamp", "b_x", "b_y", "b_z"]
    df = pd.read_csv(csv_path, usecols=usecols)
    df = df[df["sensor_id"].astype(str).str.contains(sensor_tag, regex=False)].copy()
    if df.empty:
        raise SystemExit(f"No rows for sensor containing {sensor_tag!r} in {csv_path}")
    df["t"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["t"])
    bx = pd.to_numeric(df["b_x"], errors="coerce")
    by = pd.to_numeric(df["b_y"], errors="coerce")
    bz = pd.to_numeric(df["b_z"], errors="coerce")
    df["mag"] = np.sqrt(bx * bx + by * by + bz * bz)
    df = df.dropna(subset=["mag"]).sort_values("t")
    if end_cap is not None:
        df = df[df["t"] <= end_cap]
    if df.empty:
        raise SystemExit(f"No rows for sensor {sensor_tag!r} in {csv_path} after --csv-end truncation")
    times = [pd.Timestamp(x).to_pydatetime() for x in df["t"]]
    vals = [float(x) for x in df["mag"]]
    return times, vals


def _predict_ewma_shifted(y: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    s = pd.Series(y, dtype=float)
    lvl = s.ewm(alpha=alpha, adjust=False).mean()
    pred = lvl.shift(1)
    pred = pred.bfill()
    return pred.to_numpy(dtype=float)


def _predict_median_shifted(y: np.ndarray, window: int = 31) -> np.ndarray:
    s = pd.Series(y, dtype=float)
    pred = s.rolling(window=max(3, int(window)), min_periods=1).median().shift(1)
    pred = pred.bfill()
    return pred.to_numpy(dtype=float)


def _predict_savgol_shifted(y: np.ndarray, window_length: int = 31, polyorder: int = 3) -> np.ndarray:
    from scipy.signal import savgol_filter

    n = len(y)
    wl = max(polyorder + 2, min(int(window_length) | 1, n if (n % 2 == 1) else n - 1))  # odd
    if wl > n:
        wl = n if (n % 2 == 1) else n - 1
    if wl <= polyorder:
        return y.copy()
    sm = savgol_filter(y, window_length=wl, polyorder=polyorder, mode="interp")
    pred = np.roll(sm, 1)
    pred[0] = sm[0]
    return pred


def _one_sensor_anomaly_lines(
    mode: str,
    csv_path: Path,
    sensor_display: str,
    sensor_match: str,
    k: float,
    end_cap: pd.Timestamp | None,
) -> list[tuple[pd.Timestamp, str]]:
    times, y_list = _mag_series_from_export(csv_path, sensor_match, end_cap)
    y = np.asarray(y_list, dtype=float)
    if mode == "ewma":
        pred = _predict_ewma_shifted(y)
    elif mode == "median":
        pred = _predict_median_shifted(y)
    elif mode == "savgol":
        pred = _predict_savgol_shifted(y)
    else:
        raise ValueError(mode)

    det = AnomalyDetector(
        threshold_multiplier=float(k),
        min_samples_for_threshold=20,
        std_relative_floor=0.02,
    )
    diff = det.calculate_differences(times, y_list, times, pred.tolist())
    out: list[tuple[pd.Timestamp, str]] = []
    for _, row in diff[diff["is_anomaly"]].iterrows():
        t = pd.Timestamp(row["time"]).floor("s")
        ts = pd.Timestamp(row["time"]).strftime("%Y-%m-%d %H:%M:%S")
        mag = float(row["actual"])
        line = f"[{sensor_display}] Anomaly detected | time={ts} | magnitude={mag:.1f} nT"
        out.append((t, line))
    return out


def run_baseline(
    mode: str,
    csv_path: Path,
    sensor_display: str,
    sensor_match: str,
    k: float,
    out_log: Path,
    end_cap: pd.Timestamp | None,
) -> None:
    events = _one_sensor_anomaly_lines(mode, csv_path, sensor_display, sensor_match, k, end_cap)
    events.sort(key=lambda x: x[0])
    lines = [f"Magnavis offline baseline ({mode}) k={k} sensor={sensor_display}"]
    lines.extend(e[1] for e in events)
    out_log.parent.mkdir(parents=True, exist_ok=True)
    out_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_log} ({len(lines) - 1} anomaly lines)")


def run_baseline_multi_sensor(
    mode: str,
    csv_path: Path,
    sensor_tags: list[str],
    k: float,
    out_log: Path,
    end_cap: pd.Timestamp | None,
) -> None:
    """One log file: anomalies from each tag (e.g. OBS2_1..3), merged in time order."""
    events: list[tuple[pd.Timestamp, str]] = []
    for tag in sensor_tags:
        events.extend(_one_sensor_anomaly_lines(mode, csv_path, tag, tag, k, end_cap))
    events.sort(key=lambda x: x[0])
    header = f'Magnavis offline baseline ({mode}) k={k} sensors={",".join(sensor_tags)}'
    lines = [header] + [e[1] for e in events]
    out_log.parent.mkdir(parents=True, exist_ok=True)
    out_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_log} ({len(lines) - 1} anomaly lines across {len(sensor_tags)} sensors)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--mode", choices=("ewma", "median", "savgol"), required=True)
    p.add_argument("--sensor-display", default="OBS2_1", help="Bracket label (single-sensor mode)")
    p.add_argument("--sensor-match", default="OBS2_1", help="CSV sensor_id substring (single-sensor mode)")
    p.add_argument(
        "--sensors",
        default="",
        help="Comma-separated tags (e.g. OBS2_1,OBS2_2,OBS2_3). If set, merges anomalies from each sensor into one log.",
    )
    p.add_argument("--k", type=float, default=2.5)
    p.add_argument(
        "--csv-end",
        default="",
        help="Truncate magnetic rows at this timestamp inclusive (e.g. 2026-02-13 16:18:30). Empty = use full file.",
    )
    p.add_argument("--out-log", type=Path, required=True)
    args = p.parse_args()
    end_cap = _parse_csv_end_arg(str(args.csv_end))
    raw = str(args.sensors).strip()
    if raw:
        tags = [x.strip() for x in raw.split(",") if x.strip()]
        if not tags:
            raise SystemExit("--sensors was empty after parsing")
        run_baseline_multi_sensor(args.mode, args.csv, tags, args.k, args.out_log, end_cap)
    else:
        run_baseline(args.mode, args.csv, args.sensor_display, args.sensor_match, args.k, args.out_log, end_cap)


if __name__ == "__main__":
    main()
