#!/usr/bin/env python3
"""
Build synthetic app.log lines for simple forecast baselines (EWMA / median / Savitzky–Golay)
using the same AnomalyDetector.calculate_differences path as the GUI app.

Magnetic exports may contain **multiple samples per wall-clock second**. ``app.py`` CSV
ingestion (``_csv_raw_to_timeseries_df``) downsamples to **1 Hz** by flooring timestamps to
seconds and **averaging** ``|B|`` within each second. This module applies the same rule before
forecasting so baselines match headless CSV replay and the evaluator's per-second grid.

Supports ``--skip-initial-minutes``: drop all samples before (first_timestamp + N minutes)
so baselines align with "after historic load" replay (default 62 for improved benchmark).

Invoked by ``run_suite.py`` in this directory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# .../src/benchmark_feb13_2026_improved/this_file.py → src/
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


def _parse_csv_start_arg(raw: str) -> pd.Timestamp | None:
    raw = str(raw or "").strip()
    if not raw:
        return None
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _downsample_to_1hz_mean(
    times: list[pd.Timestamp], mags: list[float]
) -> tuple[list[pd.Timestamp], list[float]]:
    """
    Align with ``ApplicationTemp._csv_raw_to_timeseries_df`` (``app.py``): one sample per
    second, ``mag = mean(|B|)`` over all rows whose timestamp falls in that second.
    """
    if not times:
        return [], []
    df = pd.DataFrame({"t": pd.to_datetime(times), "m": mags})
    df["t"] = df["t"].dt.floor("s")
    g = df.groupby("t", as_index=False)["m"].mean().sort_values("t").reset_index(drop=True)
    out_t = [pd.Timestamp(x).to_pydatetime() for x in g["t"]]
    out_m = [float(x) for x in g["m"]]
    return out_t, out_m


def _load_multi_sensor_mag_series_chunked(
    csv_path: Path,
    sensor_tags: list[str],
    end_cap: pd.Timestamp | None,
    start_cap: pd.Timestamp | None,
    *,
    chunksize: int = 400_000,
) -> dict[str, tuple[list[pd.Timestamp], list[float]]]:
    """
    One pass over the export (chunked) for all ``sensor_tags`` — avoids re-reading huge CSVs
    once per sensor in ``run_baseline_multi_sensor``.
    """
    buckets: dict[str, list[tuple[pd.Timestamp, float]]] = {t: [] for t in sensor_tags}
    usecols = ["sensor_id", "timestamp", "b_x", "b_y", "b_z"]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=int(chunksize)):
        chunk["t"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
        chunk = chunk.dropna(subset=["t"])
        # Match magnetic eval grid: trim by **floor second**, not raw timestamp (see app.py CSV load).
        tsec = chunk["t"].dt.floor("s")
        if start_cap is not None:
            chunk = chunk.loc[tsec >= start_cap]
        if end_cap is not None:
            chunk = chunk.loc[tsec <= end_cap]
        if chunk.empty:
            continue
        sid = chunk["sensor_id"].astype(str)
        bx = pd.to_numeric(chunk["b_x"], errors="coerce")
        by = pd.to_numeric(chunk["b_y"], errors="coerce")
        bz = pd.to_numeric(chunk["b_z"], errors="coerce")
        mag = np.sqrt(bx * bx + by * by + bz * bz)
        chunk = chunk.assign(_mag=mag).dropna(subset=["_mag"])
        for tag in sensor_tags:
            sub = chunk[sid.str.contains(tag, regex=False)]
            if sub.empty:
                continue
            for t, m in zip(sub["t"], sub["_mag"]):
                buckets[tag].append((pd.Timestamp(t), float(m)))
    out: dict[str, tuple[list[pd.Timestamp], list[float]]] = {}
    for tag in sensor_tags:
        pairs = sorted(buckets[tag], key=lambda x: x[0])
        if not pairs:
            raise SystemExit(
                f"No rows for sensor {tag!r} in {csv_path} after --csv-start/--csv-end / chunk filter"
            )
        t_raw = [a for a, _ in pairs]
        m_raw = [b for _, b in pairs]
        out[tag] = _downsample_to_1hz_mean(t_raw, m_raw)
    return out


def _mag_series_from_export(
    csv_path: Path,
    sensor_tag: str,
    end_cap: pd.Timestamp | None = None,
    start_cap: pd.Timestamp | None = None,
) -> tuple[list[pd.Timestamp], list[float]]:
    return _load_multi_sensor_mag_series_chunked(csv_path, [sensor_tag], end_cap, start_cap)[sensor_tag]


def _trim_initial_minutes(
    times: list[pd.Timestamp],
    vals: list[float],
    skip_minutes: float,
    sensor_tag: str,
) -> tuple[list[pd.Timestamp], list[float]]:
    """Keep only rows with t >= times[0] + skip_minutes (per-sensor timeline)."""
    if skip_minutes <= 0:
        return times, vals
    if not times:
        return times, vals
    t0 = pd.Timestamp(times[0])
    cutoff = t0 + pd.Timedelta(minutes=float(skip_minutes))
    out_t: list[pd.Timestamp] = []
    out_v: list[float] = []
    for t, v in zip(times, vals):
        if pd.Timestamp(t) >= cutoff:
            out_t.append(t)
            out_v.append(v)
    if len(out_v) < 30:
        raise SystemExit(
            f"After --skip-initial-minutes={skip_minutes}, sensor {sensor_tag!r} has only {len(out_v)} points."
        )
    return out_t, out_v


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


def _anomaly_lines_from_mag_series(
    mode: str,
    sensor_display: str,
    k: float,
    times: list[pd.Timestamp],
    y_list: list[float],
    skip_initial_minutes: float,
    trim_label: str,
    *,
    ewma_alpha: float = 0.35,
    median_window: int = 31,
    savgol_window: int = 31,
    savgol_polyorder: int = 3,
    detector_alpha: float = 0.995,
) -> list[tuple[pd.Timestamp, str]]:
    times, y_list = _trim_initial_minutes(times, y_list, skip_initial_minutes, trim_label)
    y = np.asarray(y_list, dtype=float)
    if mode == "ewma":
        pred = _predict_ewma_shifted(y, alpha=float(ewma_alpha))
    elif mode == "median":
        pred = _predict_median_shifted(y, window=int(median_window))
    elif mode == "savgol":
        pred = _predict_savgol_shifted(
            y, window_length=int(savgol_window), polyorder=int(savgol_polyorder)
        )
    else:
        raise ValueError(mode)

    det = AnomalyDetector(
        threshold_multiplier=float(k),
        min_samples_for_threshold=20,
        error_smoothing_alpha=float(detector_alpha),
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


def _one_sensor_anomaly_lines(
    mode: str,
    csv_path: Path,
    sensor_display: str,
    sensor_match: str,
    k: float,
    end_cap: pd.Timestamp | None,
    start_cap: pd.Timestamp | None,
    skip_initial_minutes: float,
    *,
    ewma_alpha: float = 0.35,
    median_window: int = 31,
    savgol_window: int = 31,
    savgol_polyorder: int = 3,
    detector_alpha: float = 0.995,
) -> list[tuple[pd.Timestamp, str]]:
    times, y_list = _mag_series_from_export(csv_path, sensor_match, end_cap, start_cap)
    return _anomaly_lines_from_mag_series(
        mode,
        sensor_display,
        k,
        times,
        y_list,
        skip_initial_minutes,
        sensor_match,
        ewma_alpha=ewma_alpha,
        median_window=median_window,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
        detector_alpha=detector_alpha,
    )


def _baseline_header(
    mode: str,
    k: float,
    sensors: str,
    skip_initial_minutes: float,
    *,
    ewma_alpha: float,
    median_window: int,
    savgol_window: int,
    savgol_polyorder: int,
    detector_alpha: float,
) -> str:
    parts = [
        f"Magnavis offline baseline ({mode}) k={k} sensors={sensors}",
        f"ewma_alpha={ewma_alpha}",
        f"median_window={median_window}",
        f"savgol_window={savgol_window}",
        f"savgol_polyorder={savgol_polyorder}",
        f"detector_alpha={detector_alpha}",
    ]
    if skip_initial_minutes > 0:
        parts.append(f"skip_initial_minutes={skip_initial_minutes}")
    return " ".join(parts)


def run_baseline(
    mode: str,
    csv_path: Path,
    sensor_display: str,
    sensor_match: str,
    k: float,
    out_log: Path,
    end_cap: pd.Timestamp | None,
    start_cap: pd.Timestamp | None,
    skip_initial_minutes: float,
    *,
    ewma_alpha: float = 0.35,
    median_window: int = 31,
    savgol_window: int = 31,
    savgol_polyorder: int = 3,
    detector_alpha: float = 0.995,
) -> None:
    events = _one_sensor_anomaly_lines(
        mode,
        csv_path,
        sensor_display,
        sensor_match,
        k,
        end_cap,
        start_cap,
        skip_initial_minutes,
        ewma_alpha=ewma_alpha,
        median_window=median_window,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
        detector_alpha=detector_alpha,
    )
    events.sort(key=lambda x: x[0])
    lines = [
        _baseline_header(
            mode,
            k,
            sensor_display,
            skip_initial_minutes,
            ewma_alpha=ewma_alpha,
            median_window=median_window,
            savgol_window=savgol_window,
            savgol_polyorder=savgol_polyorder,
            detector_alpha=detector_alpha,
        )
    ]
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
    start_cap: pd.Timestamp | None,
    skip_initial_minutes: float,
    *,
    ewma_alpha: float = 0.35,
    median_window: int = 31,
    savgol_window: int = 31,
    savgol_polyorder: int = 3,
    detector_alpha: float = 0.995,
) -> None:
    """One log file: anomalies from each tag (e.g. OBS2_1..3), merged in time order."""
    series = _load_multi_sensor_mag_series_chunked(csv_path, sensor_tags, end_cap, start_cap)
    events: list[tuple[pd.Timestamp, str]] = []
    for tag in sensor_tags:
        times, y_list = series[tag]
        events.extend(
            _anomaly_lines_from_mag_series(
                mode,
                tag,
                k,
                times,
                y_list,
                skip_initial_minutes,
                tag,
                ewma_alpha=ewma_alpha,
                median_window=median_window,
                savgol_window=savgol_window,
                savgol_polyorder=savgol_polyorder,
                detector_alpha=detector_alpha,
            )
        )
    events.sort(key=lambda x: x[0])
    header = _baseline_header(
        mode,
        k,
        ",".join(sensor_tags),
        skip_initial_minutes,
        ewma_alpha=ewma_alpha,
        median_window=median_window,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
        detector_alpha=detector_alpha,
    )
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
        "--ewma-alpha",
        type=float,
        default=0.35,
        help="Forecast EWMA smoothing α (ewma mode only; lower = slower tracking).",
    )
    p.add_argument(
        "--median-window",
        type=int,
        default=31,
        help="Rolling median window in seconds (median mode).",
    )
    p.add_argument(
        "--savgol-window",
        type=int,
        default=31,
        help="Savitzky–Golay window length in samples (odd, savgol mode).",
    )
    p.add_argument(
        "--savgol-polyorder",
        type=int,
        default=3,
        help="Savitzky–Golay polynomial order (savgol mode).",
    )
    p.add_argument(
        "--detector-alpha",
        type=float,
        default=0.995,
        help="AnomalyDetector EWMA α on |actual−pred| (threshold memory).",
    )
    p.add_argument(
        "--csv-end",
        default="",
        help="Truncate magnetic rows at this timestamp inclusive (e.g. 2026-02-13 16:18:30). Empty = use full file.",
    )
    p.add_argument(
        "--csv-start",
        default="",
        help="Keep magnetic rows with timestamp >= this value (inclusive). Empty = no lower bound.",
    )
    p.add_argument(
        "--skip-initial-minutes",
        type=float,
        default=0.0,
        help="Drop samples before first_timestamp + N minutes per sensor (aligns with historic load). 0 = off.",
    )
    p.add_argument("--out-log", type=Path, required=True)
    args = p.parse_args()
    end_cap = _parse_csv_end_arg(str(args.csv_end))
    start_cap = _parse_csv_start_arg(str(args.csv_start))
    skip = float(args.skip_initial_minutes or 0.0)
    kw = dict(
        ewma_alpha=float(args.ewma_alpha),
        median_window=int(args.median_window),
        savgol_window=int(args.savgol_window),
        savgol_polyorder=int(args.savgol_polyorder),
        detector_alpha=float(args.detector_alpha),
    )
    raw = str(args.sensors).strip()
    if raw:
        tags = [x.strip() for x in raw.split(",") if x.strip()]
        if not tags:
            raise SystemExit("--sensors was empty after parsing")
        run_baseline_multi_sensor(
            args.mode, args.csv, tags, args.k, args.out_log, end_cap, start_cap, skip, **kw
        )
    else:
        run_baseline(
            args.mode,
            args.csv,
            args.sensor_display,
            args.sensor_match,
            args.k,
            args.out_log,
            end_cap,
            start_cap,
            skip,
            **kw,
        )


if __name__ == "__main__":
    main()
