#!/usr/bin/env python3
"""
Evaluate anomaly detection quality using app logs and experiment intervals.

Inputs
------
1) app.log (contains lines like: "[OBS2_2] Anomaly detected | time=2026-02-16 15:18:07 | ...")
2) Experiment_Data.csv (HHMM start/end schedule used as ground-truth anomaly windows)

Outputs
-------
- models/anomaly_eval/<prefix>_summary.json
- models/anomaly_eval/<prefix>_point_metrics.csv
- models/anomaly_eval/<prefix>_event_metrics.csv
- models/anomaly_eval/<prefix>_metrics_plot.png

Notes
-----
- Point-level metrics are computed per second over the full evaluation timeline.
- Event-level metrics are computed by comparing predicted anomaly episodes against GT intervals.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

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


def _parse_anomaly_times_from_log(log_file: Path, sensor_filter: str) -> List[datetime]:
    """ALL/* = union of anomaly seconds across all sensors in the log (OR per second)."""
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

    yt = y_true.astype(int).to_numpy()
    yp = y_pred.astype(int).to_numpy()

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate anomaly detection metrics from app.log and Experiment_Data.csv")
    p.add_argument("--log-file", required=True, help="Path to app.log")
    p.add_argument("--experiment-file", default=str(PROJECT_ROOT / "Experiment_Data.csv"), help="Path to Experiment_Data.csv")
    p.add_argument(
        "--sensor",
        default="OBS2",
        help="Output naming; use with --prediction-sensor-mode filter to restrict log lines.",
    )
    p.add_argument(
        "--prediction-sensor-mode",
        choices=("union_all", "filter"),
        default="union_all",
        help="union_all: OR across all sensors for predicted anomaly seconds. filter: use --sensor only.",
    )
    p.add_argument("--base-date", default=None, help="Base date for HHMM GT intervals: YYYY-MM-DD (default: infer from first anomaly)")
    p.add_argument("--point-tolerance-sec", type=int, default=0, help="Tolerance around each predicted anomaly second for point-level labeling")
    p.add_argument("--event-merge-gap-sec", type=int, default=2, help="Merge nearby anomaly seconds into one predicted event if gap <= this value")
    p.add_argument("--event-tolerance-sec", type=int, default=5, help="Temporal tolerance for GT/pred event overlap")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory")
    p.add_argument("--prefix", default=None, help="Output file prefix (default auto)")
    return p.parse_args()


def _infer_base_date(base_date_arg: Optional[str], pred_times: Sequence[datetime]) -> datetime:
    if base_date_arg:
        return datetime.strptime(base_date_arg, "%Y-%m-%d")
    if pred_times:
        t = min(pred_times)
        return datetime(t.year, t.month, t.day)
    raise ValueError("Cannot infer base date (no predicted anomalies). Provide --base-date YYYY-MM-DD.")


def main() -> None:
    args = _parse_args()
    log_file = Path(args.log_file).resolve()
    exp_file = Path(args.experiment_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    sensor = str(args.sensor)

    if not log_file.exists():
        raise FileNotFoundError(f"Missing log file: {log_file}")
    if not exp_file.exists():
        raise FileNotFoundError(f"Missing experiment file: {exp_file}")

    pred_sensor_filter = sensor if str(args.prediction_sensor_mode) == "filter" else "ALL"
    pred_points = _parse_anomaly_times_from_log(log_file, sensor_filter=pred_sensor_filter)
    base_date = _infer_base_date(args.base_date, pred_points)
    gt_intervals = _load_gt_intervals(exp_file, base_date=base_date)
    pred_events = _merge_points_into_events(pred_points, max_gap_seconds=int(args.event_merge_gap_sec))

    point_metrics, point_df = _point_metrics(
        gt_intervals=gt_intervals,
        pred_points=pred_points,
        tolerance_seconds=int(args.point_tolerance_sec),
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

    point_df.to_csv(out_point_csv, index=False)
    event_df.to_csv(out_event_csv, index=False)

    summary = {
        "config": {
            "log_file": str(log_file),
            "experiment_file": str(exp_file),
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
        "outputs": {
            "point_metrics_csv": str(out_point_csv),
            "event_metrics_csv": str(out_event_csv),
            "metrics_plot_png": str(out_plot),
        },
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_metrics(
        point=point_metrics,
        event=event_metrics,
        out_png=out_plot,
        title=f"Anomaly Detection Metrics ({sensor})",
    )

    print(f"GT intervals: {len(gt_intervals)}")
    print(f"Pred anomaly seconds: {len(pred_points)}")
    print(f"Pred events: {len(pred_events)}")
    print("\nPoint-level:")
    for k in ["accuracy", "precision", "recall", "f1_score", "specificity", "tp", "fp", "tn", "fn"]:
        print(f"  {k}: {point_metrics[k]}")
    print("\nEvent-level:")
    for k in ["precision", "recall", "f1_score", "n_gt_events", "n_pred_events", "n_gt_detected", "n_pred_true"]:
        print(f"  {k}: {event_metrics[k]}")
    print("\nSaved:")
    print(f"  {out_json}")
    print(f"  {out_point_csv}")
    print(f"  {out_event_csv}")
    print(f"  {out_plot}")


if __name__ == "__main__":
    main()
