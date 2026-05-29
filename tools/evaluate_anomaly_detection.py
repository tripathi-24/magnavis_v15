#!/usr/bin/env python3
"""
Evaluate anomaly detection quality using app logs and ground-truth anomaly windows.

Inputs
------
1) app.log (contains lines like: "[OBS2_2] Anomaly detected | time=2026-02-16 15:18:07 | ...")
2) Ground truth, either:
   - Experiment_Data.csv (HHMM start/end), or
   - ``--gt-mode manual_app`` + ``--manual-csv-basename``: same magnet/trimmer intervals as
     ``app.py`` ``_MANUAL_EXPERIMENT_CSV_GT`` (brown / introduced-anomaly UI bands).

Optional
--------
- ``--session-dir``: session folder (used when you pass
  ``--restrict-point-seconds-to-session-predict-pairs``).
- ``--restrict-point-seconds-to-session-predict-pairs``: point-level confusion uses **only**
  seconds where ``predict_input.csv`` and ``predict_out.csv`` inner-join on ``x`` (union or
  ``--predict-pairs-mode intersection`` across sensors). **Warning:** GT windows often extend
  **outside** that set (e.g. magnet events before dense paired predictions), so many GT
  anomaly seconds become **omitted** from the matrix (not counted as FN). Default is **off**:
  the timeline includes all GT and log times so ~all GT anomaly seconds appear as positives.
- ``--magnetic-csv`` (optional): further restrict point-level seconds to timestamps present
  in that export (after optional ``--magnetic-csv-end`` and ``--magnetic-csv-skip-initial-minutes``,
  mirroring offline baselines / headless CSV cap).
  With ``--restrict-point-seconds-to-session-predict-pairs`` and ``--session-dir``, the
  allowed set is the **intersection** of both.

Outputs
-------
- models/anomaly_eval/<prefix>_summary.json
- models/anomaly_eval/<prefix>_point_metrics.csv
- models/anomaly_eval/<prefix>_event_metrics.csv
- models/anomaly_eval/<prefix>_metrics_plot.png
- models/anomaly_eval/<prefix>_confusion_matrix.png

Notes
-----
- Point-level metrics are computed per second over the evaluation timeline (optionally clipped).
- Event-level metrics are computed by comparing predicted anomaly episodes against GT intervals.
- By default (--prediction-sensor-mode union_all), a second counts as **predicted anomalous**
  if **any** sensor logs ``Anomaly detected`` for that time (OR across sensors). Use
  ``--prediction-sensor-mode filter`` to restrict predictions to ``--sensor`` only.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "models" / "anomaly_eval"
ANOMALY_RE = re.compile(
    r"\[(?P<sensor>[^\]]+)\]\s+Anomaly detected\s+\|\s+time=(?P<ts>[^|]+)\|",
    re.IGNORECASE,
)

# Must stay in sync with ``src/app.py`` ``_MANUAL_EXPERIMENT_CSV_GT`` (UI brown / introduced bands).
# Tuple layout: (base_day, magnet_ranges, trimmer_ranges). Apr 2026 exports: five magnet windows only
# (same wall-clock bands for OBS1-only or OBS2-only; see app.py comments for resolved times).
_MANUAL_EXPERIMENT_CSV_GT: Dict[str, Tuple[datetime, List[Tuple[str, str]], List[Tuple[str, str]]]] = {
    "magnetic_data_20260206_151500_to_20260206_161500.csv": (
        datetime(2026, 2, 6),
        [
            ("1536", "1537"),
            ("1543", "1543.5"),
            ("1545", "1546"),
            ("1555", "1556"),
            ("1601", "1603"),
        ],
        [("1546", "1550"), ("1552", "1554"), ("1557", "1559")],
    ),
    "magnetic_data_20260210_110000_to_20260210_124500.csv": (
        datetime(2026, 2, 10),
        [
            ("1226", "1227"),
            ("1227", "1228"),
            ("1229", "1231"),
            ("1232", "1233"),
            ("1234", "1235"),
            ("1236", "1237"),
        ],
        [],
    ),
    "magnetic_data_20260213_150000_to_20260213_163000.csv": (
        datetime(2026, 2, 13),
        [
            ("160623", "160747"),
            ("160910", "161010"),
            ("161315", "161415"),
        ],
        [],
    ),
    "magnetic_data_20260213_150000_to_20260213_180000_synthetic_no_anomaly.csv": (
        datetime(2026, 2, 13),
        [
            ("1515", "1545"),
            ("1610", "1640"),
            ("1700", "1730"),
        ],
        [],
    ),
    # Apr 2026 magnet GT on 2026-04-27 (OBS1 and OBS2; same HHMM/HHMMSS cells as app.py).
    # Resolved: 06:25:20–06:54:00, 07:25:00–07:55:03, 08:26:04–08:56:00, 06:55:00–07:24:00, 07:55:51–08:24:46.
    "magnetic_data_20260426_060000_to_20260427_090000.csv": (
        datetime(2026, 4, 27),
        [
            ("062520", "065400"),
            ("072500", "075503"),
            ("082604", "085600"),
            ("065500", "072400"),
            ("075551", "082446"),
        ],
        [],
    ),
    "magnetic_data_20260426_060000_to_20260427_090000_1hz.csv": (
        datetime(2026, 4, 27),
        [
            ("062520", "065400"),
            ("072500", "075503"),
            ("082604", "085600"),
            ("065500", "072400"),
            ("075551", "082446"),
        ],
        [],
    ),
}


@dataclass
class GtInterval:
    start: datetime
    end: datetime


@dataclass
class Event:
    start: datetime
    end: datetime

    @property
    def duration_s(self) -> int:
        return int((self.end - self.start).total_seconds()) + 1


def _clock_cell_to_datetime(base_date: datetime, cell) -> datetime:
    """
    Parse a schedule cell as either:
    - 4-digit HHMM (e.g. 1226 -> 12:26:00)
    - 6-digit HHMMSS (e.g. 160635 -> 16:06:35)
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        raise ValueError("empty cell")
    s = str(cell).strip()
    if "." in s and s.replace(".", "", 1).isdigit():
        try:
            s = str(int(float(s)))
        except ValueError:
            pass
    if not s:
        raise ValueError("empty string")
    if s.isdigit() and len(s) == 6:
        hour = int(s[0:2])
        minute = int(s[2:4])
        second = int(s[4:6])
    elif s.isdigit() and len(s) == 4:
        hour = int(s[0:2])
        minute = int(s[2:4])
        second = 0
    else:
        raise ValueError(f"Unsupported clock token (use HHMM or HHMMSS): {cell!r}")
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise ValueError(f"Invalid time parts: {cell!r}")
    return datetime(base_date.year, base_date.month, base_date.day, hour, minute, second)


def _naive_ts_floor(ts: object) -> pd.Timestamp:
    """UTC-naive second floor so CSV / log timestamps match ``date_range`` indices for ``isin``."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.floor("s")


def _load_gt_intervals(experiment_file: Path, base_date: datetime) -> List[GtInterval]:
    raw = pd.read_csv(experiment_file, header=None)
    df = pd.DataFrame(
        {
            "start_cell": raw.iloc[:, 1],
            "end_cell": raw.iloc[:, 2],
        }
    ).dropna()

    intervals: List[GtInterval] = []
    for row in df.itertuples(index=False):
        try:
            start = _clock_cell_to_datetime(base_date, row.start_cell)
            end = _clock_cell_to_datetime(base_date, row.end_cell)
        except ValueError:
            continue
        # Handle cross-midnight interval if present.
        if end <= start:
            end = end + timedelta(days=1)
        intervals.append(GtInterval(start=start, end=end))

    intervals.sort(key=lambda x: x.start)
    return intervals


def _load_gt_intervals_manual_app(csv_basename: str) -> List[GtInterval]:
    """Same intervals as ``app.py`` ``_apply_manual_ground_truth_for_known_csv_experiment`` for known CSVs."""
    cfg = _MANUAL_EXPERIMENT_CSV_GT.get(csv_basename)
    if cfg is None:
        keys = ", ".join(sorted(_MANUAL_EXPERIMENT_CSV_GT))
        raise KeyError(f"Unknown manual CSV basename {csv_basename!r}. Known keys: {keys}")
    base_day, magnet_ranges, trimmer_ranges = cfg
    intervals: List[GtInterval] = []
    for start_cell, end_cell in list(magnet_ranges) + list(trimmer_ranges):
        try:
            start = _clock_cell_to_datetime(base_day, start_cell)
            end = _clock_cell_to_datetime(base_day, end_cell)
        except ValueError:
            continue
        if end <= start:
            end = end + timedelta(days=1)
        intervals.append(GtInterval(start=start, end=end))
    intervals.sort(key=lambda x: x.start)
    return intervals


def _parse_optional_timestamp_arg(raw: str) -> Optional[pd.Timestamp]:
    raw = str(raw or "").strip()
    if not raw:
        return None
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp string: {raw!r}")
    return pd.Timestamp(ts)


def _allowed_seconds_from_magnetic_csv(
    path: Path,
    require_all_obs2: bool,
    require_all_obs1: bool = False,
    start_cap: Optional[pd.Timestamp] = None,
    end_cap: Optional[pd.Timestamp] = None,
    skip_initial_minutes: float = 0.0,
) -> Set[pd.Timestamp]:
    """
    Floor to 1 Hz timestamps present in a magnetic export CSV (timestamp, sensor_id).

    Optional ``start_cap`` / ``end_cap``: keep rows with start_cap <= t <= end_cap (benchmark CSV trim).

    Optional ``skip_initial_minutes``: drop rows with t < cutoff. If ``require_all_obs2`` /
    ``require_all_obs1``, cutoff = max over the three OBS* tags of (first_timestamp(sensor) + skip);
    else cutoff = min(t) + skip on the filtered frame (all rows).
    """
    df = pd.read_csv(path, usecols=["timestamp", "sensor_id"])
    df["t"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce").dt.floor("s")
    df = df.dropna(subset=["t"])
    start_cap_ts = pd.Timestamp(start_cap) if start_cap is not None else None
    end_cap_ts = pd.Timestamp(end_cap) if end_cap is not None else None
    if start_cap_ts is not None:
        df = df[df["t"] >= start_cap_ts]
    if end_cap_ts is not None:
        df = df[df["t"] <= end_cap_ts]
    if df.empty:
        return set()

    want_sufs: Tuple[str, str, str]
    if require_all_obs1 and not require_all_obs2:
        want_sufs = ("OBS1_1", "OBS1_2", "OBS1_3")
    elif require_all_obs2 and not require_all_obs1:
        want_sufs = ("OBS2_1", "OBS2_2", "OBS2_3")
    elif require_all_obs1 and require_all_obs2:
        raise ValueError("require_all_obs1 and require_all_obs2 cannot both be true")
    else:
        want_sufs = ()

    def _obs_tag(sid: object) -> str:
        s = str(sid)
        for suf in ("OBS2_1", "OBS2_2", "OBS2_3", "OBS1_1", "OBS1_2", "OBS1_3"):
            if suf in s:
                return suf
        return s

    df["tag"] = df["sensor_id"].map(_obs_tag)

    skip_m = float(skip_initial_minutes or 0.0)
    if skip_m > 0:
        if want_sufs:
            cutoffs: List[pd.Timestamp] = []
            for suf in want_sufs:
                sub = df.loc[df["tag"] == suf, "t"]
                if sub.empty:
                    raise ValueError(
                        f"--magnetic-csv-skip-initial-minutes requires rows for {suf} in {path} "
                        f"(after --magnetic-csv-end filter)."
                    )
                cutoffs.append(sub.min() + pd.Timedelta(minutes=skip_m))
            lo = max(cutoffs)
            df = df[df["t"] >= lo]
        else:
            lo = df["t"].min() + pd.Timedelta(minutes=skip_m)
            df = df[df["t"] >= lo]

    if df.empty:
        return set()

    if want_sufs:
        want = set(want_sufs)
        by_t = df.groupby("t")["tag"].agg(set)
        return {_naive_ts_floor(t) for t, tags in by_t.items() if want.issubset(tags)}
    return {_naive_ts_floor(x) for x in df["t"].unique()}


def _allowed_seconds_from_session_predict_pairs(
    session_dir: Path, mode: str
) -> Set[pd.Timestamp]:
    """
    Seconds where both actual (predict_input ``y``) and forecast (predict_out ``y``) exist
    for the same timestamp ``x`` (inner join), aggregated across per-sensor session folders.
    ``mode`` is ``union`` (default): second kept if any sensor has a pair at that ``x``;
    ``intersection``: second kept only if every sensor subfolder that has both CSVs has a row at ``x``.
    """
    mode_l = str(mode or "union").strip().lower()
    per_sensor: List[Set[pd.Timestamp]] = []
    for sub in sorted(session_dir.iterdir()):
        if not sub.is_dir():
            continue
        pin, pout = sub / "predict_input.csv", sub / "predict_out.csv"
        if not pin.is_file() or not pout.is_file():
            continue
        inp = pd.read_csv(pin, parse_dates=["x"])
        out = pd.read_csv(pout, parse_dates=["x"])
        inner = inp.merge(out, on="x", how="inner", suffixes=("_a", "_b"))
        if inner.empty:
            continue
        per_sensor.append({_naive_ts_floor(x) for x in inner["x"]})
    if not per_sensor:
        return set()
    if mode_l == "intersection":
        return set.intersection(*per_sensor)
    return set.union(*per_sensor)


def _parse_anomaly_times_from_log(log_file: Path, sensor_filter: str) -> List[datetime]:
    """
    Parse ``[Sensor] Anomaly detected | time=...`` lines. When sensor_filter is ALL/*,
    every sensor contributes; timestamps are floored to 1 Hz and de-duplicated (OR across
    sensors at the same second).
    """
    times: List[datetime] = []
    sf = sensor_filter.strip().upper()
    use_all = sf in {"ALL", "*"}

    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = ANOMALY_RE.search(line)
        if not m:
            continue
        sensor = str(m.group("sensor")).strip()
        if (not use_all) and (sf not in sensor.upper()):
            continue
        ts_str = str(m.group("ts")).strip()
        try:
            ts = pd.to_datetime(ts_str).to_pydatetime()
            times.append(ts)
        except Exception:
            continue

    # unique + sorted at second resolution
    sec = sorted({pd.Timestamp(t).floor("s").to_pydatetime() for t in times})
    return sec


def _merge_points_into_events(points: Sequence[datetime], max_gap_seconds: int) -> List[Event]:
    if not points:
        return []
    pts = sorted(points)
    events: List[Event] = []
    cur_start = pts[0]
    cur_end = pts[0]
    for t in pts[1:]:
        if (t - cur_end).total_seconds() <= max_gap_seconds:
            cur_end = t
        else:
            events.append(Event(start=cur_start, end=cur_end))
            cur_start = t
            cur_end = t
    events.append(Event(start=cur_start, end=cur_end))
    return events


def _overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return not (a_end < b_start or b_end < a_start)


def _event_metrics(
    gt_intervals: Sequence[GtInterval],
    pred_events: Sequence[Event],
    tolerance_seconds: int,
) -> Tuple[dict, pd.DataFrame]:
    tol = timedelta(seconds=tolerance_seconds)
    gt_detected = []
    for gt in gt_intervals:
        gt_s = gt.start - tol
        gt_e = gt.end + tol
        hit = any(_overlaps(gt_s, gt_e, pe.start, pe.end) for pe in pred_events)
        gt_detected.append(hit)

    pred_true = []
    for pe in pred_events:
        pe_s = pe.start - tol
        pe_e = pe.end + tol
        hit = any(_overlaps(pe_s, pe_e, gt.start, gt.end) for gt in gt_intervals)
        pred_true.append(hit)

    tp_gt = int(sum(gt_detected))
    n_gt = int(len(gt_intervals))
    tp_pred = int(sum(pred_true))
    n_pred = int(len(pred_events))

    recall = (tp_gt / n_gt) if n_gt else 0.0
    precision = (tp_pred / n_pred) if n_pred else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    details = pd.DataFrame(
        {
            "type": ["gt_intervals", "pred_events"],
            "count": [n_gt, n_pred],
            "matched": [tp_gt, tp_pred],
            "unmatched": [n_gt - tp_gt, n_pred - tp_pred],
        }
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "n_gt_events": n_gt,
        "n_pred_events": n_pred,
        "n_gt_detected": tp_gt,
        "n_pred_true": tp_pred,
    }, details


def _point_metrics(
    gt_intervals: Sequence[GtInterval],
    pred_points: Sequence[datetime],
    tolerance_seconds: int,
    allowed_seconds: Optional[Set[pd.Timestamp]] = None,
    point_restriction: str = "none",
) -> Tuple[dict, pd.DataFrame]:
    if not gt_intervals and not pred_points:
        raise ValueError("No GT intervals and no predicted anomalies found.")

    start_candidates = []
    end_candidates = []
    if gt_intervals:
        start_candidates.append(min(x.start for x in gt_intervals))
        end_candidates.append(max(x.end for x in gt_intervals))
    if pred_points:
        start_candidates.append(min(pred_points))
        end_candidates.append(max(pred_points))

    t0 = min(start_candidates).replace(microsecond=0)
    t1 = max(end_candidates).replace(microsecond=0)
    if t1 < t0:
        t1 = t0

    # When restricting to e.g. predict_in∩predict_out seconds, that set can extend **beyond**
    # the min/max of *logged* anomaly times alone. Otherwise most allowed seconds are missing
    # from ``date_range`` and the confusion matrix silently shrinks.
    allowed_norm: Optional[Set[pd.Timestamp]] = None
    if allowed_seconds is not None:
        allowed_norm = {_naive_ts_floor(x) for x in allowed_seconds}
        lo_a = min(allowed_norm)
        hi_a = max(allowed_norm)
        t0 = min(_naive_ts_floor(t0), lo_a).to_pydatetime()
        t1 = max(_naive_ts_floor(t1), hi_a).to_pydatetime()

    timeline = pd.date_range(t0, t1, freq="1s")
    y_true = pd.Series(False, index=timeline)
    y_pred = pd.Series(False, index=timeline)

    for gt in gt_intervals:
        lo = pd.Timestamp(gt.start).floor("s")
        hi = pd.Timestamp(gt.end).floor("s")
        y_true.loc[(y_true.index >= lo) & (y_true.index <= hi)] = True

    tol = int(max(0, tolerance_seconds))
    for p in pred_points:
        center = pd.Timestamp(p).floor("s")
        lo = center - pd.Timedelta(seconds=tol)
        hi = center + pd.Timedelta(seconds=tol)
        y_pred.loc[(y_pred.index >= lo) & (y_pred.index <= hi)] = True

    if allowed_seconds is not None:
        assert allowed_norm is not None
        idx_floor = pd.DatetimeIndex([_naive_ts_floor(x) for x in y_true.index])
        allowed_index = y_true.index[idx_floor.isin(allowed_norm)]
        if len(allowed_index) == 0:
            raise ValueError(
                "No overlap between per-second timeline and the allowed-timestamp set "
                f"({point_restriction}). Check --session-dir / --magnetic-csv and date range."
            )
        yt = y_true.loc[allowed_index].astype(int).to_numpy()
        yp = y_pred.loc[allowed_index].astype(int).to_numpy()
    else:
        yt = y_true.astype(int).to_numpy()
        yp = y_pred.astype(int).to_numpy()

    n_gt_positive_in_timeline = int(y_true.sum())
    n_gt_positive_evaluated = int(yt.sum())

    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    acc = float(accuracy_score(yt, yp))
    prec = float(precision_score(yt, yp, zero_division=0))
    rec = float(recall_score(yt, yp, zero_division=0))
    f1 = float(f1_score(yt, yp, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": specificity,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "timeline_points": int(len(timeline)),
        "evaluated_points": int(len(yt)),
        "n_allowed_seconds": int(len(allowed_seconds)) if allowed_seconds is not None else None,
        "point_restriction": str(point_restriction),
        "n_gt_positive_seconds_in_timeline": int(n_gt_positive_in_timeline),
        "n_gt_positive_seconds_evaluated": int(n_gt_positive_evaluated),
    }

    detail = pd.DataFrame(
        [
            {"metric": "accuracy", "value": acc},
            {"metric": "precision", "value": prec},
            {"metric": "recall", "value": rec},
            {"metric": "f1_score", "value": f1},
            {"metric": "specificity", "value": specificity},
        ]
    )
    return metrics, detail


def _plot_metrics(point: dict, event: dict, out_png: Path, title: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Point-level
    labels_a = ["Accuracy", "Precision", "Recall", "F1", "Specificity"]
    vals_a = [
        point["accuracy"],
        point["precision"],
        point["recall"],
        point["f1_score"],
        point["specificity"],
    ]
    colors_a = ["#2563EB", "#059669", "#DC2626", "#7C3AED", "#0EA5E9"]
    bars_a = axes[0].bar(labels_a, vals_a, color=colors_a, edgecolor="black", linewidth=0.6)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Point-level (per second)")
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=20)
    for b, v in zip(bars_a, vals_a):
        axes[0].annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", xytext=(0, 5), textcoords="offset points", fontsize=9)

    # Event-level
    labels_b = ["Precision", "Recall", "F1"]
    vals_b = [event["precision"], event["recall"], event["f1_score"]]
    colors_b = ["#16A34A", "#EA580C", "#9333EA"]
    bars_b = axes[1].bar(labels_b, vals_b, color=colors_b, edgecolor="black", linewidth=0.6)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Event-level (episodes)")
    axes[1].tick_params(axis="x", rotation=10)
    for b, v in zip(bars_b, vals_b):
        axes[1].annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", xytext=(0, 5), textcoords="offset points", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix_png(cm: np.ndarray, metrics: dict, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 6.0), dpi=300, facecolor="white")
    total = float(np.sum(cm)) or 1.0
    pct = cm / total
    im = ax.imshow(pct, cmap="Blues", vmin=0.0, vmax=max(0.35, float(pct.max())))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction")
    labels = ["Normal (0)", "Anomaly (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted (log detections)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ground truth", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    thr = float(pct.max()) * 0.55 if pct.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            c, p = int(cm[i, j]), float(pct[i, j])
            ax.text(
                j,
                i,
                f"n={c}\n{p * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if p > thr else "#111",
            )
    footer = (
        f"Acc {metrics['accuracy']:.3f}  Prec {metrics['precision']:.3f}  "
        f"Rec {metrics['recall']:.3f}  F1 {metrics['f1_score']:.3f}  Spec {metrics['specificity']:.3f}"
    )
    fig.text(0.5, 0.02, footer, ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate anomaly detection metrics from app.log + GT intervals "
        "(Experiment_Data.csv or the same manual schedule as app.py for known magnetic CSVs)."
    )
    p.add_argument("--log-file", required=True, help="Path to app.log")
    p.add_argument(
        "--gt-mode",
        choices=("experiment", "manual_app"),
        default="experiment",
        help="experiment: read --experiment-file. manual_app: GT from app.py schedule (--manual-csv-basename).",
    )
    p.add_argument(
        "--manual-csv-basename",
        default="magnetic_data_20260213_150000_to_20260213_163000.csv",
        help="When --gt-mode manual_app: basename must match a key in the evaluator's manual GT table (synced with app.py).",
    )
    p.add_argument(
        "--experiment-file",
        default=str(PROJECT_ROOT / "Experiment_Data.csv"),
        help="When --gt-mode experiment: path to Experiment_Data.csv",
    )
    p.add_argument(
        "--session-dir",
        default=None,
        help="App session folder (UUID under src/sessions). Required with "
        "--restrict-point-seconds-to-session-predict-pairs.",
    )
    p.add_argument(
        "--restrict-point-seconds-to-session-predict-pairs",
        action="store_true",
        help="Point-level rows = only seconds with predict_input∩predict_out (see --session-dir). "
        "Omitting this flag uses the full GT∪log timeline so all GT anomaly seconds are counted.",
    )
    p.add_argument(
        "--predict-pairs-mode",
        choices=("union", "intersection"),
        default="union",
        help="How to combine inner-join timestamp sets across sensor subfolders (default: union).",
    )
    p.add_argument(
        "--magnetic-csv",
        default=None,
        help="Optional magnetic export CSV; if used alone, restricts to seconds present in that file only "
        "(does not prove a model prediction existed). Prefer --session-dir for actual+predicted pairs; "
        "if both are set, the allowed set is the intersection. Session restriction applies only with "
        "--restrict-point-seconds-to-session-predict-pairs.",
    )
    p.add_argument(
        "--magnetic-csv-require-all-obs2",
        action="store_true",
        help="With --magnetic-csv: only seconds that have at least one row for OBS2_1, OBS2_2, and OBS2_3.",
    )
    p.add_argument(
        "--magnetic-csv-require-all-obs1",
        action="store_true",
        help="With --magnetic-csv: only seconds that have at least one row for OBS1_1, OBS1_2, and OBS1_3.",
    )
    p.add_argument(
        "--magnetic-csv-end",
        default="",
        help="With --magnetic-csv: only timestamps <= this value (inclusive), e.g. 2026-02-13 16:18:30. "
        "Empty = use full file span.",
    )
    p.add_argument(
        "--magnetic-csv-start",
        default="",
        help="With --magnetic-csv: only timestamps >= this value (inclusive). Empty = no lower bound.",
    )
    p.add_argument(
        "--magnetic-csv-skip-initial-minutes",
        type=float,
        default=0.0,
        help="With --magnetic-csv: drop seconds before warmup (per-sensor first t + N minutes when "
        "--magnetic-csv-require-all-obs1/obs2; else global min(t)+N). Matches offline_statistical_baselines.",
    )
    p.add_argument(
        "--sensor",
        default="ALL",
        help="Output naming / legacy tag. Prediction pool is controlled by --prediction-sensor-mode.",
    )
    p.add_argument(
        "--prediction-sensor-mode",
        choices=("union_all", "filter"),
        default="union_all",
        help="union_all (default): predicted-positive at a second if ANY sensor logged an anomaly "
        "then (OR). filter: only log lines whose [sensor] matches --sensor count as predictions.",
    )
    p.add_argument("--base-date", default=None, help="Base date for HHMM GT intervals: YYYY-MM-DD (default: infer from first anomaly)")
    p.add_argument("--point-tolerance-sec", type=int, default=0, help="Tolerance around each predicted anomaly second for point-level labeling")
    p.add_argument("--event-merge-gap-sec", type=int, default=2, help="Merge nearby anomaly seconds into one predicted event if gap <= this value")
    p.add_argument("--event-tolerance-sec", type=int, default=5, help="Temporal tolerance for GT/pred event overlap")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory")
    p.add_argument("--prefix", default=None, help="Output file prefix (default auto)")
    return p.parse_args()


def _infer_base_date(
    base_date_arg: Optional[str],
    pred_times: Sequence[datetime],
    gt_intervals: Optional[Sequence[GtInterval]] = None,
) -> datetime:
    if base_date_arg:
        return datetime.strptime(base_date_arg, "%Y-%m-%d")
    if pred_times:
        t = min(pred_times)
        return datetime(t.year, t.month, t.day)
    if gt_intervals:
        t = min(x.start for x in gt_intervals)
        return datetime(t.year, t.month, t.day)
    raise ValueError("Cannot infer base date (no predicted anomalies and no GT). Provide --base-date YYYY-MM-DD.")


def main() -> None:
    args = _parse_args()
    if bool(args.magnetic_csv_require_all_obs2) and bool(args.magnetic_csv_require_all_obs1):
        raise SystemExit("Use at most one of --magnetic-csv-require-all-obs1 / --magnetic-csv-require-all-obs2.")
    log_file = Path(args.log_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    sensor = str(args.sensor)

    if not log_file.exists():
        raise FileNotFoundError(f"Missing log file: {log_file}")

    pred_sensor_filter = sensor if str(args.prediction_sensor_mode) == "filter" else "ALL"
    pred_points = _parse_anomaly_times_from_log(log_file, sensor_filter=pred_sensor_filter)

    if args.gt_mode == "manual_app":
        basename = str(args.manual_csv_basename).strip()
        gt_intervals = _load_gt_intervals_manual_app(basename)
        base_date = _infer_base_date(args.base_date, pred_points, gt_intervals=gt_intervals)
        exp_file: Optional[Path] = None
    else:
        exp_file = Path(args.experiment_file).resolve()
        if not exp_file.exists():
            raise FileNotFoundError(f"Missing experiment file: {exp_file}")
        base_date = _infer_base_date(args.base_date, pred_points)
        gt_intervals = _load_gt_intervals(exp_file, base_date=base_date)

    pred_events = _merge_points_into_events(pred_points, max_gap_seconds=int(args.event_merge_gap_sec))

    if args.restrict_point_seconds_to_session_predict_pairs and not args.session_dir:
        raise SystemExit(
            "--restrict-point-seconds-to-session-predict-pairs requires --session-dir "
            "(folder with per-sensor predict_input.csv and predict_out.csv)."
        )

    allowed_session: Optional[Set[pd.Timestamp]] = None
    if args.restrict_point_seconds_to_session_predict_pairs:
        sdir = Path(args.session_dir).resolve()
        if not sdir.is_dir():
            raise FileNotFoundError(f"--session-dir is not a directory: {sdir}")
        allowed_session = _allowed_seconds_from_session_predict_pairs(
            sdir, mode=str(args.predict_pairs_mode)
        )
        if not allowed_session:
            raise ValueError(
                f"No predict_input/predict_out inner-join timestamps found under {sdir} "
                "(need per-sensor folders with both CSVs)."
            )

    allowed_magnetic: Optional[Set[pd.Timestamp]] = None
    if args.magnetic_csv:
        mpath = Path(args.magnetic_csv).resolve()
        if not mpath.is_file():
            raise FileNotFoundError(f"Missing --magnetic-csv file: {mpath}")
        mag_end = _parse_optional_timestamp_arg(str(args.magnetic_csv_end or ""))
        mag_start = _parse_optional_timestamp_arg(str(getattr(args, "magnetic_csv_start", "") or ""))
        mag_skip = float(getattr(args, "magnetic_csv_skip_initial_minutes", 0.0) or 0.0)
        allowed_magnetic = _allowed_seconds_from_magnetic_csv(
            mpath,
            require_all_obs2=bool(args.magnetic_csv_require_all_obs2),
            require_all_obs1=bool(args.magnetic_csv_require_all_obs1),
            start_cap=mag_start,
            end_cap=mag_end,
            skip_initial_minutes=mag_skip,
        )

    allowed_seconds: Optional[Set[pd.Timestamp]] = None
    point_restriction = "none"
    if allowed_session is not None and allowed_magnetic is not None:
        allowed_seconds = allowed_session & allowed_magnetic
        point_restriction = "session_predict_pairs_and_magnetic_csv"
    elif allowed_session is not None:
        allowed_seconds = allowed_session
        point_restriction = "session_predict_pairs"
    elif allowed_magnetic is not None:
        allowed_seconds = allowed_magnetic
        point_restriction = "magnetic_csv_only"

    if allowed_seconds is not None and len(allowed_seconds) == 0:
        raise ValueError(
            "Allowed timestamp set is empty (e.g. no overlap between --session-dir pairs and --magnetic-csv)."
        )

    point_metrics, point_df = _point_metrics(
        gt_intervals=gt_intervals,
        pred_points=pred_points,
        tolerance_seconds=int(args.point_tolerance_sec),
        allowed_seconds=allowed_seconds,
        point_restriction=point_restriction,
    )
    event_metrics, event_df = _event_metrics(
        gt_intervals=gt_intervals,
        pred_events=pred_events,
        tolerance_seconds=int(args.event_tolerance_sec),
    )

    prefix = args.prefix or f"anomaly_eval_{sensor.replace('*', 'ALL').replace(' ', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{prefix}_summary.json"
    out_point_csv = out_dir / f"{prefix}_point_metrics.csv"
    out_event_csv = out_dir / f"{prefix}_event_metrics.csv"
    out_plot = out_dir / f"{prefix}_metrics_plot.png"
    out_cm = out_dir / f"{prefix}_confusion_matrix.png"

    point_df.to_csv(out_point_csv, index=False)
    event_df.to_csv(out_event_csv, index=False)

    cm = np.array(
        [
            [point_metrics["tn"], point_metrics["fp"]],
            [point_metrics["fn"], point_metrics["tp"]],
        ],
        dtype=int,
    )
    pred_mode_l = str(args.prediction_sensor_mode).strip().lower()
    pred_caption = "pred = OR(all sensors)" if pred_mode_l == "union_all" else f"pred = filter({sensor})"
    cm_title = f"Presence confusion ({pred_caption})"
    if args.gt_mode == "manual_app":
        cm_title += f"\nGT=app.py manual schedule: {args.manual_csv_basename}"
    if args.restrict_point_seconds_to_session_predict_pairs:
        cm_title += f"\nPoints restricted to predict_in∩predict_out ({args.predict_pairs_mode})"
    if args.magnetic_csv:
        cm_title += (
            f"\n∩ magnetic CSV ({Path(args.magnetic_csv).name})"
            if args.restrict_point_seconds_to_session_predict_pairs
            else f"\nPoints restricted to magnetic CSV ({Path(args.magnetic_csv).name})"
        )
        if str(getattr(args, "magnetic_csv_start", "") or "").strip():
            cm_title += f"\n  t >= {str(args.magnetic_csv_start).strip()}"
        if str(args.magnetic_csv_end or "").strip():
            cm_title += f"\n  t <= {str(args.magnetic_csv_end).strip()}"
        if float(args.magnetic_csv_skip_initial_minutes or 0) > 0:
            g = "OBS2" if args.magnetic_csv_require_all_obs2 else ("OBS1" if args.magnetic_csv_require_all_obs1 else "global")
            cm_title += f"\n  skip first {float(args.magnetic_csv_skip_initial_minutes)} min/sensor ({g} grid)"
    _plot_confusion_matrix_png(cm, point_metrics, out_cm, cm_title)

    summary = {
        "config": {
            "log_file": str(log_file),
            "gt_mode": str(args.gt_mode),
            "experiment_file": str(exp_file) if exp_file else None,
            "manual_csv_basename": str(args.manual_csv_basename) if args.gt_mode == "manual_app" else None,
            "session_dir": str(Path(args.session_dir).resolve()) if args.session_dir else None,
            "restrict_point_seconds_to_session_predict_pairs": bool(
                args.restrict_point_seconds_to_session_predict_pairs
            ),
            "predict_pairs_mode": str(args.predict_pairs_mode)
            if args.restrict_point_seconds_to_session_predict_pairs
            else None,
            "magnetic_csv": str(Path(args.magnetic_csv).resolve()) if args.magnetic_csv else None,
            "magnetic_csv_require_all_obs2": bool(args.magnetic_csv_require_all_obs2),
            "magnetic_csv_require_all_obs1": bool(args.magnetic_csv_require_all_obs1),
            "magnetic_csv_start": str(getattr(args, "magnetic_csv_start", "") or "").strip() or None,
            "magnetic_csv_end": str(args.magnetic_csv_end or "").strip() or None,
            "magnetic_csv_skip_initial_minutes": float(args.magnetic_csv_skip_initial_minutes or 0.0),
            "point_restriction": point_restriction,
            "sensor_filter": sensor,
            "prediction_sensor_mode": str(args.prediction_sensor_mode),
            "prediction_log_filter": pred_sensor_filter,
            "base_date": base_date.strftime("%Y-%m-%d"),
            "point_tolerance_sec": int(args.point_tolerance_sec),
            "event_merge_gap_sec": int(args.event_merge_gap_sec),
            "event_tolerance_sec": int(args.event_tolerance_sec),
        },
        "counts": {
            "n_gt_intervals": int(len(gt_intervals)),
            "n_pred_anomaly_seconds": int(len(pred_points)),
            "n_pred_events": int(len(pred_events)),
        },
        "point_level_metrics": point_metrics,
        "event_level_metrics": event_metrics,
        "confusion_matrix_2x2": cm.tolist(),
        "outputs": {
            "point_metrics_csv": str(out_point_csv),
            "event_metrics_csv": str(out_event_csv),
            "metrics_plot_png": str(out_plot),
            "confusion_matrix_png": str(out_cm),
        },
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_metrics(
        point=point_metrics,
        event=event_metrics,
        out_png=out_plot,
        title=f"Anomaly Detection Metrics ({pred_caption}; out={sensor})",
    )

    print(f"GT intervals: {len(gt_intervals)}")
    print(f"Prediction pool: {args.prediction_sensor_mode} (log sensor filter: {pred_sensor_filter})")
    print(f"Pred anomaly seconds: {len(pred_points)}")
    print(f"Pred events: {len(pred_events)}")
    print("\nPoint-level:")
    for k in ["accuracy", "precision", "recall", "f1_score", "specificity", "tp", "fp", "tn", "fn"]:
        print(f"  {k}: {point_metrics[k]}")
    print(f"  point_restriction: {point_metrics.get('point_restriction')}")
    print(f"  evaluated_points: {point_metrics.get('evaluated_points')}")
    print(f"  n_allowed_seconds: {point_metrics.get('n_allowed_seconds')}")
    print(f"  n_gt_positive_seconds_in_timeline: {point_metrics.get('n_gt_positive_seconds_in_timeline')}")
    print(f"  n_gt_positive_seconds_evaluated: {point_metrics.get('n_gt_positive_seconds_evaluated')}")
    print("\nEvent-level:")
    for k in ["precision", "recall", "f1_score", "n_gt_events", "n_pred_events", "n_gt_detected", "n_pred_true"]:
        print(f"  {k}: {event_metrics[k]}")
    print("\nSaved:")
    print(f"  {out_json}")
    print(f"  {out_point_csv}")
    print(f"  {out_event_csv}")
    print(f"  {out_plot}")
    print(f"  {out_cm}")


if __name__ == "__main__":
    main()
