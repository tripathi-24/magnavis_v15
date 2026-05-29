#!/usr/bin/env python3
"""
Build k–recall curves for six predictor families on Feb-13 (short GT) and Apr-27 (long GT).

Families: ewma, median, savgol, gru_pretrained, lstm_pretrained, attn_bilstm_fresh.

Reuses metrics from existing ``results/*/comparison_table.csv`` when available, then runs
missing (dataset, k, family) combinations.

Outputs under ``results/k_recall_curves_<timestamp>/``:
  - ``k_recall_points.csv``
  - ``k_recall_curves_feb13.png``, ``k_recall_curves_apr27.png``
  - ``k_recall_curves_combined.png``
  - ``K_RECALL_REPORT.md``

Usage::

  cd src/benchmark_feb13_2026_improved
  python run_k_recall_curves.py
  python run_k_recall_curves.py --k-values 1,1.5,2,2.5,3,3.5,4,4.5,5 --skip-app  # baselines only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

BENCH_DIR = Path(__file__).resolve().parent
SRC = BENCH_DIR.parent
PROJECT_ROOT = SRC.parent
APP_MAIN = SRC / "app.py"
OFFLINE = BENCH_DIR / "offline_statistical_baselines.py"
EVAL = PROJECT_ROOT / "tools" / "evaluate_anomaly_detection.py"
RESULTS_DIR = BENCH_DIR / "results"

# Import benchmark helpers from sibling module
sys.path.insert(0, str(BENCH_DIR))
from run_suite_improved import (  # noqa: E402
    APP_SEQUENCE,
    EVAL_SENSOR_ALL,
    OBS2_SENSORS_CSV,
    _append_metric_rows,
    _bench_python,
    _predictor_initial_split_env,
    _read_metrics,
    _run,
    _run_eval,
    _want_app_run,
    _want_baseline,
)

DEFAULT_K_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]
DEFAULT_SKIP_INITIAL_MINUTES = 0.0
DEFAULT_HISTORIC_MINUTES = 0

SCHEME_ORDER = [
    "ewma",
    "median",
    "savgol",
    "gru_pretrained",
    "lstm_pretrained",
    "attn_bilstm_fresh",
]

SCHEME_LABELS = {
    "ewma": "EWMA",
    "median": "Median",
    "savgol": "SavGol",
    "gru_pretrained": "GRU (pretrained)",
    "lstm_pretrained": "LSTM (pretrained)",
    "attn_bilstm_fresh": "Attn Bi-LSTM (fresh)",
}

# comparison_table.csv predictor column -> canonical scheme id
_PREDICTOR_ALIASES = {
    "ewma": "ewma",
    "median": "median",
    "savgol": "savgol",
    "gru_pretrained": "gru_pretrained",
    "lstm_pretrained": "lstm_pretrained",
    "attention_bi_lstm": "attn_bilstm_fresh",
    "attn_bilstm_fresh": "attn_bilstm_fresh",
}

APP_ONLY: Set[str] = {"gru_pretrained", "lstm_pretrained", "attn_bilstm_fresh"}
BASELINE_ONLY: Set[str] = {"ewma", "median", "savgol"}


@dataclass(frozen=True)
class DatasetCfg:
    key: str
    label: str
    csv_path: Path
    manual_csv_basename: str
    base_date: str
    csv_end: Optional[str]
    csv_start: str
    skip_initial_minutes: float


DATASETS: Dict[str, DatasetCfg] = {
    "feb13": DatasetCfg(
        key="feb13",
        label="Feb 13 2026 (short GT)",
        csv_path=PROJECT_ROOT / "Datafiles" / "magnetic_data_20260213_150000_to_20260213_163000.csv",
        manual_csv_basename="magnetic_data_20260213_150000_to_20260213_163000.csv",
        base_date="2026-02-13",
        csv_end="2026-02-13 16:18:30",
        csv_start="",
        skip_initial_minutes=DEFAULT_SKIP_INITIAL_MINUTES,
    ),
    "apr27": DatasetCfg(
        key="apr27",
        label="Apr 27 2026 (long GT)",
        csv_path=PROJECT_ROOT / "Datafiles" / "magnetic_data_20260426_060000_to_20260427_090000_1hz.csv",
        manual_csv_basename="magnetic_data_20260426_060000_to_20260427_090000_1hz.csv",
        base_date="2026-04-27",
        csv_end=None,
        csv_start="",
        skip_initial_minutes=DEFAULT_SKIP_INITIAL_MINUTES,
    ),
    "synthetic": DatasetCfg(
        key="synthetic",
        label="Feb 13 2026 synthetic (long GT windows)",
        csv_path=PROJECT_ROOT
        / "Datafiles"
        / "magnetic_data_20260213_150000_to_20260213_180000_synthetic_no_anomaly.csv",
        manual_csv_basename="magnetic_data_20260213_150000_to_20260213_180000_synthetic_no_anomaly.csv",
        base_date="2026-02-13",
        csv_end=None,
        csv_start="",
        skip_initial_minutes=DEFAULT_SKIP_INITIAL_MINUTES,
    ),
}


def _parse_k_from_detector(detector: str) -> Optional[float]:
    m = re.search(r"k=([\d.]+)", str(detector))
    return float(m.group(1)) if m else None


def _load_cache_from_results(results_root: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """
    Key: (dataset_key, scheme, k_str) where k_str is normalized '1.0' etc.
    Heuristic: basename contains '13Feb' or '27Apr' / 'apr27' for dataset.
    """
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if not results_root.is_dir():
        return out

    for table in results_root.rglob("comparison_table.csv"):
        path_low = str(table).lower()
        parent = table.parent.name.lower()
        if "synthetic" in parent or "synthetic" in path_low:
            continue
        if "13feb" in parent:
            dkey = "feb13"
        elif "27apr" in parent or "apr27_baseline_sweep" in path_low:
            dkey = "apr27"
        else:
            continue

        try:
            with table.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    pred = str(row.get("predictor", "")).strip().lower()
                    scheme = _PREDICTOR_ALIASES.get(pred)
                    if not scheme:
                        continue
                    k = _parse_k_from_detector(row.get("detector", ""))
                    if k is None:
                        continue
                    k_str = f"{k:g}"
                    key = (dkey, scheme, k_str)
                    out[key] = {
                        "dataset": dkey,
                        "scheme": scheme,
                        "k": k,
                        "recall": float(row["recall"]),
                        "precision": float(row["precision"]),
                        "f1_score": float(row["f1_score"]),
                        "tp": int(row["tp"]),
                        "fp": int(row["fp"]),
                        "tn": int(row["tn"]),
                        "fn": int(row["fn"]),
                        "evaluated_points": int(row["evaluated_points"]),
                        "source": str(table),
                        "status": "cached",
                    }
        except (OSError, ValueError, KeyError):
            continue

    # Apr-27 baseline sweep (no comparison_table; use sweep_results.csv)
    for sweep_csv in results_root.rglob("sweep_results.csv"):
        if "apr27_baseline_sweep" not in str(sweep_csv):
            continue
        try:
            with sweep_csv.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("status") != "ok":
                        continue
                    mode = str(row.get("mode", "")).strip()
                    if mode not in BASELINE_ONLY:
                        continue
                    k = float(row["k"])
                    k_str = f"{k:g}"
                    key = ("apr27", mode, k_str)
                    if key in out:
                        continue
                    out[key] = {
                        "dataset": "apr27",
                        "scheme": mode,
                        "k": k,
                        "recall": float(row["recall"]),
                        "precision": float(row["precision"]),
                        "f1_score": float(row["f1_score"]),
                        "tp": int(row["tp"]),
                        "fp": int(row["fp"]),
                        "tn": int(row["tn"]),
                        "fn": int(row["fn"]),
                        "evaluated_points": int(row["evaluated_points"]),
                        "source": str(sweep_csv),
                        "status": "cached",
                    }
        except (OSError, ValueError, KeyError):
            continue

    return out


def _run_baseline(
    ds: DatasetCfg,
    scheme: str,
    k: float,
    log_path: Path,
) -> int:
    cmd = [
        _bench_python(),
        str(OFFLINE),
        "--csv",
        str(ds.csv_path.resolve()),
        "--mode",
        scheme,
        "--sensors",
        OBS2_SENSORS_CSV,
        "--k",
        str(k),
        "--skip-initial-minutes",
        str(ds.skip_initial_minutes),
        "--out-log",
        str(log_path),
    ]
    if ds.csv_end:
        cmd.extend(["--csv-end", ds.csv_end])
    if ds.csv_start:
        cmd.extend(["--csv-start", ds.csv_start])
    return _run(cmd, os.environ.copy(), PROJECT_ROOT, 600)


def _run_app(
    ds: DatasetCfg,
    log_stem: str,
    extra: Dict[str, str],
    table_labels: Tuple[str, ...],
    k: float,
    log_path: Path,
    *,
    historic_minutes: int,
    train_window_minutes: int,
    fast_csv: bool,
    app_timeout: int,
    predictor_ar_closed_loop: bool = False,
) -> bool:
    sid = str(uuid.uuid4())
    env = os.environ.copy()
    env["MAGNAVIS_HEADLESS_BATCH"] = "1"
    if fast_csv:
        env["MAGNAVIS_FAST_CSV_PLAYBACK"] = "1"
    csv_rel = os.path.relpath(ds.csv_path.resolve(), PROJECT_ROOT)
    env["MAGNAVIS_BATCH_CSV"] = csv_rel if not csv_rel.startswith("..") else str(ds.csv_path.resolve())
    env["MAGNAVIS_BATCH_HISTORIC_MINUTES"] = str(int(historic_minutes))
    env["MAGNAVIS_BATCH_SENSORS"] = OBS2_SENSORS_CSV
    env["MAGNAVIS_SESSION_ID_OVERRIDE"] = sid
    env["MAGNAVIS_INITIAL_THRESHOLD_K"] = str(float(k))
    env["MAGNAVIS_INITIAL_TRAIN_WINDOW_MINUTES"] = str(int(train_window_minutes))
    if log_stem in ("gru_pretrained", "lstm_pretrained"):
        env["PREDICTOR_UPDATE_TRAINING"] = "0"
        env["PREDICTOR_SKIP_FINETUNE_ON_SESSION"] = "1"
        if predictor_ar_closed_loop and log_stem in ("gru_pretrained", "lstm_pretrained"):
            env["PREDICTOR_AR_CLOSED_LOOP"] = "1"
        else:
            env.pop("PREDICTOR_AR_CLOSED_LOOP", None)
        if log_stem == "gru_pretrained" and extra.get("PRETRAINED_GRU_MODEL_DIR"):
            env["PRETRAINED_GRU_MODEL_DIR"] = str(extra["PRETRAINED_GRU_MODEL_DIR"])
        elif log_stem == "gru_pretrained" and os.environ.get("PRETRAINED_GRU_MODEL_DIR", "").strip():
            env["PRETRAINED_GRU_MODEL_DIR"] = os.environ["PRETRAINED_GRU_MODEL_DIR"].strip()
        if log_stem == "gru_pretrained" and "PREDICTOR_GRU_DELTA_TARGET" in extra:
            env["PREDICTOR_GRU_DELTA_TARGET"] = str(extra["PREDICTOR_GRU_DELTA_TARGET"])
        elif log_stem == "gru_pretrained" and os.environ.get("PREDICTOR_GRU_DELTA_TARGET", "").strip():
            env["PREDICTOR_GRU_DELTA_TARGET"] = os.environ["PREDICTOR_GRU_DELTA_TARGET"].strip()
    else:
        env.pop("PREDICTOR_SKIP_FINETUNE_ON_SESSION", None)
        env.pop("PREDICTOR_AR_CLOSED_LOOP", None)
    if ds.csv_end:
        env["MAGNAVIS_BATCH_CSV_END"] = ds.csv_end
    else:
        env.pop("MAGNAVIS_BATCH_CSV_END", None)
    if ds.csv_start:
        env["MAGNAVIS_BATCH_CSV_START"] = ds.csv_start
    else:
        env.pop("MAGNAVIS_BATCH_CSV_START", None)
    _pp = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = str(SRC) if not _pp else f"{SRC}{os.pathsep}{_pp}"
    for ek, ev in _predictor_initial_split_env(extra, ds.skip_initial_minutes).items():
        env[ek] = ev

    rc = _run([_bench_python(), str(APP_MAIN)], env, PROJECT_ROOT, int(app_timeout))  # noqa: same signature
    session_dir = SRC / "sessions" / sid
    app_log = session_dir / "app.log"
    if not app_log.is_file():
        print(f"WARN: missing app.log for {log_stem} rc={rc}")
        return False
    shutil.copy2(app_log, log_path)
    return True


def _eval_log(
    ds: DatasetCfg,
    scheme: str,
    k: float,
    log_path: Path,
    evdir: Path,
    prefix: str,
) -> Optional[Path]:
    try:
        return _run_eval(
            log_path,
            evdir,
            prefix,
            EVAL_SENSOR_ALL,
            "union_all",
            gt_mode="manual_app",
            manual_csv_basename=ds.manual_csv_basename,
            experiment_file=None,
            base_date=ds.base_date,
            magnetic_csv=ds.csv_path.resolve(),
            magnetic_require_all_obs2=True,
            magnetic_require_all_obs1=False,
            magnetic_csv_start=ds.csv_start,
            magnetic_csv_end=ds.csv_end or "",
            magnetic_csv_skip_initial_minutes=ds.skip_initial_minutes,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"WARN: eval failed {scheme} k={k} {ds.key}: {exc}")
        return None


def _metrics_from_summary(summ: Path, ds: DatasetCfg, scheme: str, k: float, source: str) -> Dict[str, Any]:
    data = json.loads(summ.read_text(encoding="utf-8"))
    pm = data.get("point_level_metrics") or {}
    return {
        "dataset": ds.key,
        "scheme": scheme,
        "k": k,
        "recall": float(pm.get("recall", 0)),
        "precision": float(pm.get("precision", 0)),
        "f1_score": float(pm.get("f1_score", 0)),
        "tp": int(pm.get("tp", 0)),
        "fp": int(pm.get("fp", 0)),
        "tn": int(pm.get("tn", 0)),
        "fn": int(pm.get("fn", 0)),
        "evaluated_points": int(pm.get("evaluated_points", 0)),
        "source": source,
        "status": "ok",
    }


def _write_csv_points(csv_path: Path, ok_pts: List[Dict[str, Any]]) -> None:
    if not ok_pts:
        return
    fields = [
        "dataset",
        "scheme",
        "k",
        "recall",
        "precision",
        "f1_score",
        "tp",
        "fp",
        "tn",
        "fn",
        "evaluated_points",
        "status",
        "source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for p in sorted(ok_pts, key=lambda x: (x["dataset"], x["scheme"], float(x["k"]))):
            w.writerow(p)


def _plot_curves(points: List[Dict[str, Any]], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for scheme in SCHEME_ORDER:
        sub = sorted(
            [
                p
                for p in points
                if p.get("scheme") == scheme and p.get("status") in ("ok", "cached")
            ],
            key=lambda x: float(x["k"]),
        )
        if not sub:
            continue
        ks = [float(p["k"]) for p in sub]
        recalls = [float(p["recall"]) for p in sub]
        ax.plot(ks, recalls, marker="o", linewidth=2, label=SCHEME_LABELS.get(scheme, scheme))

    ax.set_xlabel("Detector k (threshold multiplier)")
    ax.set_ylabel("Point-level recall")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_report(path: Path, all_pts: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    lines = [
        "# k–recall curves (six families)",
        "",
        f"- Run: `{meta['created']}`",
        f"- k values: {meta['k_values']}",
        f"- Datasets: {', '.join(meta['datasets'])}",
        "",
    ]
    for dkey in meta["datasets"]:
        lines.append(f"## {DATASETS[dkey].label}")
        lines.append("")
        lines.append("| k | " + " | ".join(SCHEME_LABELS[s] for s in SCHEME_ORDER) + " |")
        lines.append("|---|" + "|".join(["---"] * len(SCHEME_ORDER)) + "|")
        k_vals = meta["k_values"]
        for k in k_vals:
            k_str = f"{float(k):g}"
            cells = []
            for scheme in SCHEME_ORDER:
                hit = next(
                    (
                        p
                        for p in all_pts
                        if p.get("dataset") == dkey
                        and p.get("scheme") == scheme
                        and f"{float(p['k']):g}" == k_str
                        and p.get("status") in ("ok", "cached")
                    ),
                    None,
                )
                cells.append(f"{float(hit['recall']):.3f}" if hit else "—")
            lines.append(f"| {k_str} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        default="feb13,apr27",
        help="Comma list: feb13, apr27, synthetic",
    )
    ap.add_argument(
        "--k-values",
        default=",".join(str(x) for x in DEFAULT_K_VALUES),
        help="Comma-separated k grid",
    )
    ap.add_argument("--skip-app", action="store_true", help="Only offline baselines (no TensorFlow)")
    ap.add_argument("--no-cache", action="store_true", help="Ignore existing comparison_table.csv")
    ap.add_argument(
        "--plot-cache-only",
        action="store_true",
        help="Only load prior benchmark CSVs and plot (no new runs)",
    )
    ap.add_argument(
        "--resume-runs-dir",
        type=Path,
        default=None,
        help="Also treat summaries under runs/ as completed (resume partial sweep)",
    )
    ap.add_argument(
        "--skip-initial-minutes",
        type=float,
        default=DEFAULT_SKIP_INITIAL_MINUTES,
        help="Offline trim + eval magnetic-csv-skip (0 = full timeline from file start)",
    )
    ap.add_argument(
        "--historic-minutes",
        type=int,
        default=DEFAULT_HISTORIC_MINUTES,
        help="MAGNAVIS_BATCH_HISTORIC_MINUTES for headless app (0 = no blue historic segment)",
    )
    ap.add_argument(
        "--train-window-minutes",
        type=int,
        default=0,
        help="MAGNAVIS_INITIAL_TRAIN_WINDOW_MINUTES (0 = predict-only when checkpoint exists)",
    )
    ap.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse prior benchmark CSVs (default: fresh runs only)",
    )
    ap.add_argument("--fast-csv-playback", action="store_true", default=True)
    ap.add_argument("--app-timeout-sec", type=int, default=7200)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="Write k_recall_points.csv / plots / report here (default: out-dir root)",
    )
    ap.add_argument(
        "--schemes",
        default="",
        help="Comma list subset of schemes (default: all six). Example: gru_pretrained",
    )
    ap.add_argument(
        "--runs-dataset-key",
        default="",
        help="Subfolder under runs/ (default: dataset key, e.g. apr27_gru_ar_closed_loop)",
    )
    ap.add_argument(
        "--predictor-ar-closed-loop",
        action="store_true",
        help="GRU/LSTM predict-only: closed-loop AR window (predictions feed the W-window)",
    )
    args = ap.parse_args()

    k_values = [float(x.strip()) for x in str(args.k_values).split(",") if x.strip()]
    dataset_keys = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    scheme_order = (
        [x.strip() for x in str(args.schemes).split(",") if x.strip()]
        if str(args.schemes).strip()
        else list(SCHEME_ORDER)
    )
    for s in scheme_order:
        if s not in SCHEME_ORDER:
            raise SystemExit(f"Unknown scheme {s!r}; choose from {SCHEME_ORDER}")

    cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if args.use_cache and not args.no_cache:
        cache = _load_cache_from_results(RESULTS_DIR)
        print(f"Loaded {len(cache)} cached (dataset, scheme, k) points from {RESULTS_DIR}")
    if args.resume_runs_dir is not None:
        from finalize_k_recall_curves import _load_partial_runs

        partial = _load_partial_runs(args.resume_runs_dir.resolve())
        cache.update(partial)
        print(f"Merged {len(partial)} points from resume runs dir")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (args.out_dir or (RESULTS_DIR / f"k_recall_curves_zero_hist_{ts}")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_protocol.json").write_text(
        json.dumps(
            {
                "historic_minutes": int(args.historic_minutes),
                "skip_initial_minutes": float(args.skip_initial_minutes),
                "train_window_minutes": int(args.train_window_minutes),
                "k_values": k_values,
                "pretrained_predict_only": True,
                "predictor_ar_closed_loop": bool(args.predictor_ar_closed_loop),
                "schemes": scheme_order,
                "datasets": dataset_keys,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.plot_cache_only:
        ok_pts = [dict(v) for v in cache.values()]
        meta = {
            "created": ts,
            "k_values": k_values,
            "datasets": dataset_keys,
            "n_cached": len(ok_pts),
            "n_run": 0,
            "n_fail": 0,
            "plot_cache_only": True,
        }
        csv_path = root / "k_recall_points.csv"
        _write_csv_points(csv_path, ok_pts)
        (root / "k_recall_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        for dkey in dataset_keys:
            sub = [p for p in ok_pts if p.get("dataset") == dkey]
            if sub:
                _plot_curves(sub, root / f"k_recall_curves_{dkey}.png", DATASETS[dkey].label)
        _write_report(root / "K_RECALL_REPORT.md", ok_pts, meta)
        print(f"Cache-only plots written to {root}")
        return

    all_points: List[Dict[str, Any]] = []
    n_run = 0
    n_cached = 0
    n_fail = 0

    app_models = [
        (log_stem, extra, labels)
        for log_stem, extra, labels in APP_SEQUENCE
        if log_stem in ("lstm_pretrained", "gru_pretrained", "attn_bilstm_fresh")
    ]

    for dkey in dataset_keys:
        ds = replace(
            DATASETS[dkey],
            skip_initial_minutes=float(args.skip_initial_minutes),
        )
        if not ds.csv_path.is_file():
            raise SystemExit(f"Missing CSV for {dkey}: {ds.csv_path}")

        for k in k_values:
            k_str = f"{k:g}"
            runs_key = str(args.runs_dataset_key).strip() or dkey
            for scheme in scheme_order:
                ckey = (dkey, scheme, k_str)
                if not args.no_cache and ckey in cache:
                    pt = dict(cache[ckey])
                    pt["status"] = "cached"
                    all_points.append(pt)
                    n_cached += 1
                    continue

                if scheme in APP_ONLY and args.skip_app:
                    all_points.append(
                        {"dataset": dkey, "scheme": scheme, "k": k, "status": "skipped_no_app"}
                    )
                    continue

                run_dir = root / "runs" / runs_key / f"k{k_str}" / scheme
                logs = run_dir / "logs"
                evdir = run_dir / "eval"
                logs.mkdir(parents=True, exist_ok=True)
                log_path = logs / f"{scheme}.log"

                print(f"\n=== {ds.label} | {SCHEME_LABELS[scheme]} | k={k} ===", flush=True)

                if scheme in BASELINE_ONLY:
                    rc = _run_baseline(ds, scheme, k, log_path)
                    if rc != 0:
                        all_points.append(
                            {"dataset": dkey, "scheme": scheme, "k": k, "status": "baseline_failed"}
                        )
                        n_fail += 1
                        continue
                    prefix = f"baseline_{scheme}_k{k_str}"
                else:
                    entry = next((x for x in app_models if x[0] == scheme), None)
                    if not entry:
                        all_points.append(
                            {"dataset": dkey, "scheme": scheme, "k": k, "status": "unknown_scheme"}
                        )
                        n_fail += 1
                        continue
                    log_stem, extra, _labels = entry
                    ok = _run_app(
                        ds,
                        log_stem,
                        extra,
                        _labels,
                        k,
                        log_path,
                        historic_minutes=int(args.historic_minutes),
                        train_window_minutes=int(args.train_window_minutes),
                        fast_csv=bool(args.fast_csv_playback),
                        app_timeout=int(args.app_timeout_sec),
                        predictor_ar_closed_loop=bool(args.predictor_ar_closed_loop),
                    )
                    if not ok:
                        all_points.append(
                            {"dataset": dkey, "scheme": scheme, "k": k, "status": "app_failed"}
                        )
                        n_fail += 1
                        continue
                    prefix = f"{scheme}_k{k_str}"

                summ = _eval_log(ds, scheme, k, log_path, evdir, prefix)
                if summ is None or not summ.is_file():
                    all_points.append(
                        {"dataset": dkey, "scheme": scheme, "k": k, "status": "eval_failed"}
                    )
                    n_fail += 1
                    continue

                pt = _metrics_from_summary(summ, ds, scheme, k, str(summ))
                all_points.append(pt)
                n_run += 1

    summary_root = (args.summary_dir or root).resolve()
    summary_root.mkdir(parents=True, exist_ok=True)
    csv_path = summary_root / "k_recall_points.csv"
    ok_pts = [p for p in all_points if p.get("status") in ("ok", "cached")]
    _write_csv_points(csv_path, ok_pts)

    meta = {
        "created": ts,
        "k_values": k_values,
        "datasets": dataset_keys,
        "n_cached": n_cached,
        "n_run": n_run,
        "n_fail": n_fail,
        "skip_app": bool(args.skip_app),
        "historic_minutes": int(args.historic_minutes),
        "skip_initial_minutes": float(args.skip_initial_minutes),
        "train_window_minutes": int(args.train_window_minutes),
        "predictor_ar_closed_loop": bool(args.predictor_ar_closed_loop),
        "schemes": scheme_order,
        "summary_dir": str(summary_root),
    }
    (summary_root / "k_recall_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for dkey in dataset_keys:
        sub = [p for p in ok_pts if p.get("dataset") == dkey]
        if sub:
            _plot_curves(sub, summary_root / f"k_recall_curves_{dkey}.png", DATASETS[dkey].label)

    if len(dataset_keys) >= 2:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
            for ax, dkey in zip(axes, dataset_keys[:2]):
                for scheme in SCHEME_ORDER:
                    sub = sorted(
                        [
                            p
                            for p in ok_pts
                            if p.get("dataset") == dkey
                            and p.get("scheme") == scheme
                        ],
                        key=lambda x: float(x["k"]),
                    )
                    if not sub:
                        continue
                    ax.plot(
                        [float(p["k"]) for p in sub],
                        [float(p["recall"]) for p in sub],
                        marker="o",
                        linewidth=2,
                        label=SCHEME_LABELS.get(scheme, scheme),
                    )
                ax.set_title(DATASETS[dkey].label)
                ax.set_xlabel("k")
                ax.set_ylabel("Recall")
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.35)
                ax.legend(fontsize=7)
            fig.suptitle("k–recall curves (six families)")
            fig.tight_layout()
            fig.savefig(summary_root / "k_recall_curves_combined.png", dpi=150)
            plt.close(fig)
        except ImportError:
            pass

    _write_report(summary_root / "K_RECALL_REPORT.md", ok_pts, meta)

    print(f"\nDone: {n_cached} cached + {n_run} new runs, {n_fail} failed/skipped.")
    print(f"CSV: {csv_path}")
    print(f"Report: {summary_root / 'K_RECALL_REPORT.md'}")


if __name__ == "__main__":
    main()
