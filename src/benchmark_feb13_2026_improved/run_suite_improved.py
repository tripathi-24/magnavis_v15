#!/usr/bin/env python3
"""
Feb 13 2026 **improved** benchmark (aligned preprocessing + **shared point-level grid**):

1) Offline **EWMA** (alpha=0.35 in ``offline_statistical_baselines``), **median**, **Savitzky–Golay** —
   each sensor’s series is trimmed by ``--skip-initial-minutes`` (often matched to historic load).
2) Headless ``app.py`` for the configured **APP_SEQUENCE** (LSTM / GRU / Transformer / Attention Bi-LSTM, …) —
   same ``--k``, ``--historic-minutes``, ``--csv-start`` / ``--csv-end``, and the same skip/leading-train
   policy as offline baselines via ``_predictor_initial_split_env``.

**Same evaluated point count ``n`` for every scheme:** With the default closed timeline (no
``--eval-open-timeline``), the evaluator uses ``--magnetic-csv`` plus ``--magnetic-csv-start`` /
``--magnetic-csv-end`` and ``--magnetic-csv-skip-initial-minutes`` mirroring this driver, and
``--eval-obs-grid obs1`` or ``obs2`` (``--magnetic-csv-require-all-obs1`` / ``obs2``). Every model’s
point-level row then uses the **identical** per-second allowed set; the driver calls
``_validate_equal_point_totals`` so ``TP+FP+TN+FN`` must match across all completed rows.
Use ``--eval-obs-grid none`` only if you accept a looser grid (``n`` may differ).

Usage::

  cd src/benchmark_feb13_2026_improved
  python run_suite.py

From repo root::

  python src/benchmark_feb13_2026_improved/run_suite.py

If your shell is already ``…/src/benchmark_feb13_2026_improved``, run ``python run_suite.py`` only;
do not ``cd src/benchmark_feb13_2026_improved`` again. Use a **real** ``--csv`` path (not a
``/path/to/...`` placeholder). Optional aligned eval grid::
  ``--eval-point-grid-magnetic-csv <same as --csv>``

Requires NumPy, pandas, sklearn, matplotlib; **TensorFlow** for headless ``app.py`` steps.
Subprocesses use ``sys.executable`` unless ``MAGNAVIS_BENCHMARK_PYTHON`` is set.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

BENCH_DIR = Path(__file__).resolve().parent
SRC = BENCH_DIR.parent
PROJECT_ROOT = SRC.parent
APP_MAIN = SRC / "app.py"
OFFLINE = BENCH_DIR / "offline_statistical_baselines.py"
EVAL = PROJECT_ROOT / "tools" / "evaluate_anomaly_detection.py"
DEFAULT_CSV = PROJECT_ROOT / "magnetic_data_20260213_150000_to_20260213_163000.csv"
MANUAL_BASE = "magnetic_data_20260213_150000_to_20260213_163000.csv"
DEFAULT_CSV_END = "2026-02-13 16:18:30"
DEFAULT_EXPERIMENT_FILE = PROJECT_ROOT / "Experiment_Data.csv"

OBS2_SENSORS_CSV = "OBS2_1,OBS2_2,OBS2_3"
OBS1_SENSORS_CSV = "OBS1_1,OBS1_2,OBS1_3"
EVAL_SENSOR_ALL = "ALL"

# Pretrained LSTM/GRU/Transformer load ``*_pretrained_*.keras`` from ``models/``;
# Fresh LSTM/GRU/Attention Bi-LSTM: train on first ``--skip-initial-minutes`` when >0 (leading-train env).
APP_SEQUENCE: List[Tuple[str, Dict[str, str], Tuple[str, ...]]] = [
    (
        "lstm_pretrained",
        {
            "PREDICTOR_MODEL_FAMILY": "lstm",
            "PREDICTOR_MODEL_INIT": "pretrained",
        },
        ("lstm_pretrained",),
    ),
    (
        "lstm_fresh",
        {
            "PREDICTOR_MODEL_FAMILY": "lstm",
            "PREDICTOR_MODEL_INIT": "fresh",
            "PREDICTOR_GRU_WINDOW_SIZE": "15",
        },
        ("lstm_fresh",),
    ),
    (
        "attn_bilstm_fresh",
        {
            "PREDICTOR_MODEL_FAMILY": "attn_bilstm",
            "PREDICTOR_MODEL_INIT": "fresh",
            "PREDICTOR_GRU_WINDOW_SIZE": "15",
        },
        ("attention_bi_lstm",),
    ),
    (
        "gru_pretrained",
        {
            "PREDICTOR_MODEL_FAMILY": "gru",
            "PREDICTOR_MODEL_INIT": "pretrained",
        },
        ("gru_pretrained",),
    ),
    (
        "gru_fresh",
        {
            "PREDICTOR_MODEL_FAMILY": "gru",
            "PREDICTOR_MODEL_INIT": "fresh",
            "PREDICTOR_GRU_WINDOW_SIZE": "15",
        },
        ("gru_fresh",),
    ),
    (
        "transformer_pretrained",
        {
            "PREDICTOR_MODEL_FAMILY": "transformer",
            "PREDICTOR_MODEL_INIT": "pretrained",
        },
        ("transformer_pretrained",),
    ),
]


def _bench_python() -> str:
    override = os.environ.get("MAGNAVIS_BENCHMARK_PYTHON", "").strip()
    if override:
        return override
    return sys.executable


def _predictor_initial_split_env(extra: Dict[str, str], minutes: float) -> Dict[str, str]:
    """
    Align predictor subprocess with offline/eval skip policy:
    - fresh LSTM/GRU/Attn-BiLSTM: train on first ``minutes`` only, one-step inference on the rest;
    - pretrained / transformer: drop first ``minutes`` from predict_input (no session fit on that segment).
    """
    out = dict(extra)
    out.pop("PREDICTOR_LEADING_TRAIN_MINUTES", None)
    out.pop("PREDICTOR_SKIP_INITIAL_MINUTES", None)
    if minutes <= 0:
        return out
    fam = str(out.get("PREDICTOR_MODEL_FAMILY", "")).strip().lower()
    init = str(out.get("PREDICTOR_MODEL_INIT", "fresh")).strip().lower()
    sm = str(float(minutes))
    if init == "fresh" and fam in ("lstm", "gru", "attn_bilstm"):
        out["PREDICTOR_LEADING_TRAIN_MINUTES"] = sm
    elif init == "pretrained" or fam == "transformer":
        out["PREDICTOR_SKIP_INITIAL_MINUTES"] = sm
    return out


def _assert_tensorflow(py: str) -> None:
    r = subprocess.run(
        [py, "-c", "import tensorflow as tf"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode == 0:
        return
    print(
        "\nHeadless ``app.py`` steps need TensorFlow in the benchmark interpreter.\n"
        f"Interpreter tried: {py}\n"
        "Set MAGNAVIS_BENCHMARK_PYTHON to a Python that has TensorFlow installed, or use ``--skip-app`` "
        "to run offline baselines only.\n",
        file=sys.stderr,
    )
    if r.stderr:
        print(r.stderr.strip(), file=sys.stderr)
    sys.exit(4)


def _assert_scientific_stack(py: str) -> None:
    r = subprocess.run(
        [py, "-c", "import numpy, pandas, sklearn, matplotlib"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode == 0:
        return
    req = PROJECT_ROOT / "requirements_feb_2025.txt"
    print(
        "\nThis benchmark needs NumPy, pandas, scikit-learn, and matplotlib.\n"
        f"Interpreter tried: {py}\n"
        f"See {PROJECT_ROOT / 'src' / 'benchmark_feb13_2026' / 'README.md'} for env setup.\n"
        f"pip install -r {req}\n",
        file=sys.stderr,
    )
    if r.stderr:
        print(r.stderr.strip(), file=sys.stderr)
    sys.exit(3)


def _run(cmd: List[str], env: Dict[str, str], cwd: Path, timeout: int) -> int:
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, env=env, cwd=str(cwd), timeout=timeout)
    return int(r.returncode)


def _run_eval(
    log_file: Path,
    out_dir: Path,
    prefix: str,
    sensor: str,
    pred_mode: str,
    *,
    gt_mode: str,
    manual_csv_basename: str,
    experiment_file: Optional[Path],
    base_date: str,
    magnetic_csv: Optional[Path] = None,
    magnetic_require_all_obs2: bool = False,
    magnetic_require_all_obs1: bool = False,
    magnetic_csv_start: str = "",
    magnetic_csv_end: str = "",
    magnetic_csv_skip_initial_minutes: float = 0.0,
    point_tolerance_sec: int = 0,
    event_merge_gap_sec: int = 2,
    event_tolerance_sec: int = 5,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    py = _bench_python()
    cmd = [
        py,
        str(EVAL),
        "--log-file",
        str(log_file),
        "--gt-mode",
        str(gt_mode).strip(),
        "--sensor",
        sensor,
        "--base-date",
        str(base_date).strip(),
        "--prediction-sensor-mode",
        pred_mode,
        "--out-dir",
        str(out_dir),
        "--prefix",
        prefix,
    ]
    if str(gt_mode).strip().lower() == "manual_app":
        cmd.extend(["--manual-csv-basename", str(manual_csv_basename).strip()])
    else:
        if experiment_file is None or not experiment_file.is_file():
            raise FileNotFoundError(f"--experiment-file must exist for gt-mode experiment: {experiment_file}")
        cmd.extend(["--experiment-file", str(experiment_file.resolve())])
    cmd.extend(
        [
            "--point-tolerance-sec",
            str(int(point_tolerance_sec)),
            "--event-merge-gap-sec",
            str(int(event_merge_gap_sec)),
            "--event-tolerance-sec",
            str(int(event_tolerance_sec)),
        ]
    )
    if magnetic_csv is not None:
        cmd.extend(["--magnetic-csv", str(magnetic_csv.resolve())])
        if magnetic_require_all_obs2:
            cmd.append("--magnetic-csv-require-all-obs2")
        if magnetic_require_all_obs1:
            cmd.append("--magnetic-csv-require-all-obs1")
        if str(magnetic_csv_start or "").strip():
            cmd.extend(["--magnetic-csv-start", str(magnetic_csv_start).strip()])
        if str(magnetic_csv_end or "").strip():
            cmd.extend(["--magnetic-csv-end", str(magnetic_csv_end).strip()])
        if float(magnetic_csv_skip_initial_minutes or 0.0) > 0:
            cmd.extend(
                ["--magnetic-csv-skip-initial-minutes", str(float(magnetic_csv_skip_initial_minutes))]
            )
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    return out_dir / f"{prefix}_summary.json"


def _read_metrics(summary_path: Path) -> Tuple[Dict[str, Any], str]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    pm = data.get("point_level_metrics") or {}
    return pm, data.get("outputs", {}).get("confusion_matrix_png", "")


def _only_set(raw: str) -> Optional[Set[str]]:
    if not raw.strip():
        return None
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _want_baseline(mode: str, only: Optional[Set[str]]) -> bool:
    if only is None:
        return True
    return mode.lower() in only


def _want_app_run(log_stem: str, table_labels: Tuple[str, ...], only: Optional[Set[str]]) -> bool:
    if only is None:
        return True
    if log_stem.lower() in only:
        return True
    for lab in table_labels:
        if lab.lower() in only:
            return True
    return False


def _append_metric_rows(
    rows: List[Dict[str, Any]],
    predictor_labels: Tuple[str, ...],
    k: float,
    pm: Dict[str, Any],
    cm: str,
    summ: Path,
) -> None:
    base = {
        "detector": f"standard k={k}",
        "tp": pm.get("tp"),
        "fp": pm.get("fp"),
        "tn": pm.get("tn"),
        "fn": pm.get("fn"),
        "evaluated_points": pm.get("evaluated_points"),
        "recall": pm.get("recall"),
        "precision": pm.get("precision"),
        "f1_score": pm.get("f1_score"),
        "specificity": pm.get("specificity"),
        "accuracy": pm.get("accuracy"),
        "summary_json": str(summ),
        "confusion_matrix_png": cm,
    }
    for lab in predictor_labels:
        r = dict(base)
        r["predictor"] = lab
        rows.append(r)


def _validate_equal_point_totals(rows: List[Dict[str, Any]]) -> None:
    """Ensure TP+FP+TN+FN is identical for every completed row (same ``n`` on the magnetic eval grid)."""
    totals: List[int] = []
    for r in rows:
        try:
            tp, fp, tn, fn = int(r["tp"]), int(r["fp"]), int(r["tn"]), int(r["fn"])
        except (TypeError, ValueError):
            continue
        totals.append(tp + fp + tn + fn)
    if len(totals) < 2:
        return
    if len(set(totals)) != 1:
        raise SystemExit(
            "ERROR: point-level totals TP+FP+TN+FN differ across predictors (they must equal the same n). "
            f"Distinct totals: {sorted(set(totals))}. Use --eval-obs-grid obs1 or obs2 with the same "
            "--csv-start/--csv-end/--skip-initial-minutes for every scheme, do not pass --eval-open-timeline, "
            "and ensure every model run completed evaluation."
        )


def main() -> None:
    if not APP_MAIN.is_file():
        print(f"Cannot find {APP_MAIN}", file=sys.stderr)
        sys.exit(2)

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Feb 13 magnetic export CSV")
    ap.add_argument("--k", type=float, default=4.0, help="Detector k (AnomalyDetector on |error|)")
    ap.add_argument("--historic-minutes", type=int, default=62, help="Initial historic window for headless CSV replay")
    ap.add_argument(
        "--train-window-minutes",
        type=int,
        default=0,
        help="Predictor training window (minutes); 0 = all loaded historic data",
    )
    ap.add_argument("--app-timeout-sec", type=int, default=7200, help="Per app.py subprocess wall clock")
    ap.add_argument("--batch-sensors", default=OBS2_SENSORS_CSV, help="MAGNAVIS_BATCH_SENSORS")
    ap.add_argument("--offline-sensors", default=OBS2_SENSORS_CSV, help="Comma list for offline baselines")
    ap.add_argument(
        "--skip-initial-minutes",
        type=float,
        default=62.0,
        help="Offline baselines trim + evaluator --magnetic-csv-skip-initial-minutes (per OBS tag when "
        "--eval-obs-grid obs1/obs2); keeps TP+FP+TN+FN on the same n for every scheme (default 62, match historic load).",
    )
    ap.add_argument("--csv-end", default=DEFAULT_CSV_END, help="Truncate CSV at this timestamp inclusive")
    ap.add_argument(
        "--csv-start",
        default="",
        help="Keep rows with timestamp >= this value (offline, evaluator magnetic grid, headless MAGNAVIS_BATCH_CSV_START).",
    )
    ap.add_argument("--no-csv-end", action="store_true")
    ap.add_argument(
        "--manual-csv-basename",
        default=MANUAL_BASE,
        help="GT table basename for --gt-mode manual_app (must match app.py / evaluate_anomaly_detection).",
    )
    ap.add_argument(
        "--base-date",
        default="2026-02-13",
        help="Evaluator --base-date (YYYY-MM-DD): for manual_app, calendar day for HHMM cells; "
        "for experiment, base date for interval parsing (see evaluate_anomaly_detection).",
    )
    ap.add_argument(
        "--gt-mode",
        choices=("manual_app", "experiment"),
        default="manual_app",
        help="manual_app: GT from known magnetic CSV basename table (keep in sync in app.py + evaluate). "
        "experiment: GT intervals from --experiment-file.",
    )
    ap.add_argument(
        "--experiment-file",
        type=Path,
        default=None,
        metavar="CSV",
        help="When --gt-mode experiment: Experiment_Data-style CSV (default: <repo>/Experiment_Data.csv).",
    )
    ap.add_argument(
        "--point-tolerance-sec",
        type=int,
        default=0,
        help="Evaluator: seconds tolerance around each predicted anomaly for point-level labeling.",
    )
    ap.add_argument(
        "--event-merge-gap-sec",
        type=int,
        default=2,
        help="Evaluator: merge predicted anomaly seconds into one event if gap <= this many seconds.",
    )
    ap.add_argument(
        "--event-tolerance-sec",
        type=int,
        default=5,
        help="Evaluator: temporal tolerance (seconds) for GT vs predicted event overlap.",
    )
    ap.add_argument(
        "--eval-obs-grid",
        choices=("obs2", "obs1", "none"),
        default="obs2",
        help="Magnetic CSV eval grid: seconds where all three OBS2 or OBS1 sensors have a row, or no triple filter.",
    )
    ap.add_argument("--prediction-sensor-mode", default="union_all", choices=("union_all", "filter"))
    ap.add_argument(
        "--eval-point-grid-magnetic-csv",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "Override CSV for evaluator --magnetic-csv. Default: same as --csv, with --csv-start/--csv-end and "
            "--skip-initial-minutes mirrored into evaluate_anomaly_detection.py so every model shares "
            "the same evaluated seconds (see --eval-obs-grid)."
        ),
    )
    ap.add_argument(
        "--eval-open-timeline",
        action="store_true",
        help="Legacy: do not pass --magnetic-csv to the evaluator (point grid = GT ∪ pred span; totals differ per model).",
    )
    ap.add_argument("--skip-app", action="store_true", help="Only offline baselines + eval")
    ap.add_argument(
        "--fast-csv-playback",
        action="store_true",
        help=(
            "For headless app.py runs: set MAGNAVIS_FAST_CSV_PLAYBACK=1 so CSV ingest is not real-time–paced "
            "(short data timer + large sim_step; matplotlib redraw paused until end of file). Same pipeline, much faster. "
            "Also skips the optional GRU-pretrained headless UI PNG snapshot (otherwise forces a heavy full-window render)."
        ),
    )
    ap.add_argument(
        "--only",
        default="",
        help="Comma list: ewma, median, savgol, lstm_pretrained, lstm_fresh, attn_bilstm_fresh, gru_pretrained, "
        "gru_fresh, transformer_pretrained (or matching log stems)",
    )
    args = ap.parse_args()

    _bench_py = _bench_python()
    _assert_scientific_stack(_bench_py)
    if not args.skip_app:
        _assert_tensorflow(_bench_py)

    magnetic_csv_path = args.csv.expanduser().resolve()
    if not magnetic_csv_path.is_file():
        print(
            "ERROR: --csv must point to an existing magnetic export file.\n"
            f"  You passed: {args.csv}\n"
            f"  Resolved: {magnetic_csv_path}\n\n"
            "Documentation used `/path/to/magnetic_data_....csv` only as an example — "
            "use your real path. Omit --csv to use the repo default:\n"
            f"  {DEFAULT_CSV}\n\n"
            "Directory hint: if your shell is already\n"
            f"  …/src/benchmark_feb13_2026_improved\n"
            "do not run `cd src/benchmark_feb13_2026_improved` (that path does not exist from there). "
            "Just run:  python run_suite.py  [options]\n",
            file=sys.stderr,
        )
        sys.exit(2)

    experiment_path = (args.experiment_file or DEFAULT_EXPERIMENT_FILE).expanduser().resolve()
    if str(args.gt_mode).strip() == "experiment" and not experiment_path.is_file():
        print(
            "ERROR: --gt-mode experiment requires an existing experiment CSV.\n"
            f"  Resolved: {experiment_path}\n"
            "Pass --experiment-file /path/to/your.csv (columns per tools/evaluate_anomaly_detection.py).\n",
            file=sys.stderr,
        )
        sys.exit(2)
    experiment_file_for_eval: Optional[Path] = (
        experiment_path if str(args.gt_mode).strip() == "experiment" else None
    )

    eval_grid_csv: Optional[Path] = args.eval_point_grid_magnetic_csv
    if eval_grid_csv is not None:
        eg = eval_grid_csv.expanduser().resolve()
        if not eg.is_file():
            print(
                "ERROR: --eval-point-grid-magnetic-csv must point to an existing file.\n"
                f"  Resolved: {eg}",
                file=sys.stderr,
            )
            sys.exit(2)
        eval_grid_csv = eg
    if args.eval_open_timeline:
        eval_grid_csv = None
    else:
        eval_grid_csv = eval_grid_csv or magnetic_csv_path

    only = _only_set(args.only)
    csv_end = "" if args.no_csv_end else str(args.csv_end).strip()
    csv_start = str(args.csv_start or "").strip()
    skip_min = float(args.skip_initial_minutes)
    ev_grid = str(args.eval_obs_grid).strip().lower()
    req_obs2 = ev_grid == "obs2"
    req_obs1 = ev_grid == "obs1"
    eval_prefix_obs = "OBS2" if req_obs2 else ("OBS1" if req_obs1 else "ANY")
    manual_bn = str(args.manual_csv_basename).strip()
    base_date = str(args.base_date).strip()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = BENCH_DIR / "results" / ts
    logs = root / "logs"
    evroot = root / "eval"
    logs.mkdir(parents=True, exist_ok=True)

    csv_rel = os.path.relpath(magnetic_csv_path, PROJECT_ROOT)
    csv_arg = str(magnetic_csv_path) if csv_rel.startswith("..") else csv_rel

    rows: List[Dict[str, Any]] = []

    for mode in ("ewma", "median", "savgol"):
        if not _want_baseline(mode, only):
            continue
        log_copy = logs / f"{mode}_baseline_app.log"
        off_cmd = [
            _bench_python(),
            str(OFFLINE),
            "--csv",
            str(magnetic_csv_path),
            "--mode",
            mode,
            "--sensors",
            str(args.offline_sensors).strip(),
            "--k",
            str(args.k),
            "--skip-initial-minutes",
            str(skip_min),
            "--out-log",
            str(log_copy),
        ]
        if csv_end:
            off_cmd.extend(["--csv-end", csv_end])
        if csv_start:
            off_cmd.extend(["--csv-start", csv_start])
        rc = _run(off_cmd, os.environ.copy(), PROJECT_ROOT, 600)
        if rc != 0:
            print(f"WARN: baseline {mode} failed rc={rc}")
            continue
        summ = _run_eval(
            log_copy,
            evroot / f"baseline_{mode}",
            f"baseline_{mode}_{eval_prefix_obs}_all",
            EVAL_SENSOR_ALL,
            str(args.prediction_sensor_mode),
            gt_mode=str(args.gt_mode),
            manual_csv_basename=manual_bn,
            experiment_file=experiment_file_for_eval,
            base_date=base_date,
            magnetic_csv=eval_grid_csv,
            magnetic_require_all_obs2=bool(eval_grid_csv is not None and req_obs2),
            magnetic_require_all_obs1=bool(eval_grid_csv is not None and req_obs1),
            magnetic_csv_start=csv_start,
            magnetic_csv_end=csv_end,
            magnetic_csv_skip_initial_minutes=skip_min if eval_grid_csv else 0.0,
            point_tolerance_sec=int(args.point_tolerance_sec),
            event_merge_gap_sec=int(args.event_merge_gap_sec),
            event_tolerance_sec=int(args.event_tolerance_sec),
        )
        pm, cm = _read_metrics(summ)
        _append_metric_rows(rows, (mode,), float(args.k), pm, cm, summ)

    if not args.skip_app:
        for log_stem, extra, table_labels in APP_SEQUENCE:
            if not _want_app_run(log_stem, table_labels, only):
                continue
            sid = str(uuid.uuid4())
            env = os.environ.copy()
            env["MAGNAVIS_HEADLESS_BATCH"] = "1"
            if args.fast_csv_playback:
                env["MAGNAVIS_FAST_CSV_PLAYBACK"] = "1"
            env["MAGNAVIS_BATCH_CSV"] = csv_arg
            env["MAGNAVIS_BATCH_HISTORIC_MINUTES"] = str(int(args.historic_minutes))
            env["MAGNAVIS_BATCH_SENSORS"] = str(args.batch_sensors).strip()
            env["MAGNAVIS_SESSION_ID_OVERRIDE"] = sid
            env["MAGNAVIS_INITIAL_THRESHOLD_K"] = str(float(args.k))
            env["MAGNAVIS_INITIAL_TRAIN_WINDOW_MINUTES"] = str(int(args.train_window_minutes))
            if csv_end:
                env["MAGNAVIS_BATCH_CSV_END"] = csv_end
            else:
                env.pop("MAGNAVIS_BATCH_CSV_END", None)
            if csv_start:
                env["MAGNAVIS_BATCH_CSV_START"] = csv_start
            else:
                env.pop("MAGNAVIS_BATCH_CSV_START", None)
            _pp = env.get("PYTHONPATH", "").strip()
            env["PYTHONPATH"] = str(SRC) if not _pp else f"{SRC}{os.pathsep}{_pp}"
            for k, v in _predictor_initial_split_env(extra, skip_min).items():
                env[k] = v
            # GRU pretrained: optional HD full-window PNG before csv_end (heavy Qt/matplotlib — skip when
            # --fast-csv-playback so benchmark stays on the fast headless + ingest path only).
            if log_stem == "gru_pretrained" and csv_end and not args.fast_csv_playback:
                snap_path = logs / f"{log_stem}_headless_ui_before_csv_end.png"
                env["MAGNAVIS_HEADLESS_SNAPSHOT_PNG"] = str(snap_path)
                env["MAGNAVIS_HEADLESS_SNAPSHOT_BEFORE"] = str(csv_end).strip()
                env.setdefault("MAGNAVIS_HEADLESS_SNAPSHOT_WINDOW_SEC", "90")
                env.setdefault("MAGNAVIS_UI_SNAPSHOT_SCALE", "3")
            rc = _run([_bench_python(), str(APP_MAIN)], env, PROJECT_ROOT, int(args.app_timeout_sec))
            session_dir = SRC / "sessions" / sid
            app_log = session_dir / "app.log"
            if not app_log.is_file():
                print(f"WARN: missing app.log for {log_stem} (rc={rc}) path={app_log}")
                continue
            dest = logs / f"{log_stem}_app.log"
            shutil.copy2(app_log, dest)
            summ = _run_eval(
                dest,
                evroot / log_stem,
                f"{log_stem}_{EVAL_SENSOR_ALL}",
                EVAL_SENSOR_ALL,
                str(args.prediction_sensor_mode),
                gt_mode=str(args.gt_mode),
                manual_csv_basename=manual_bn,
                experiment_file=experiment_file_for_eval,
                base_date=base_date,
                magnetic_csv=eval_grid_csv,
                magnetic_require_all_obs2=bool(eval_grid_csv is not None and req_obs2),
                magnetic_require_all_obs1=bool(eval_grid_csv is not None and req_obs1),
                magnetic_csv_start=csv_start,
                magnetic_csv_end=csv_end,
                magnetic_csv_skip_initial_minutes=skip_min if eval_grid_csv else 0.0,
                point_tolerance_sec=int(args.point_tolerance_sec),
                event_merge_gap_sec=int(args.event_merge_gap_sec),
                event_tolerance_sec=int(args.event_tolerance_sec),
            )
            pm, cm = _read_metrics(summ)
            labels = table_labels if only is None else tuple(
                lab for lab in table_labels if lab.lower() in only
            ) or table_labels
            _append_metric_rows(rows, labels, float(args.k), pm, cm, summ)

    equal_point_totals_verified = False
    if (
        not args.eval_open_timeline
        and eval_grid_csv is not None
        and ev_grid in ("obs1", "obs2")
        and len(rows) >= 2
    ):
        _validate_equal_point_totals(rows)
        equal_point_totals_verified = True

    table_csv = root / "comparison_table.csv"
    table_md = root / "comparison_table.md"
    fieldnames = [
        "predictor",
        "detector",
        "tp",
        "fp",
        "tn",
        "fn",
        "evaluated_points",
        "recall",
        "precision",
        "f1_score",
        "specificity",
        "accuracy",
        "summary_json",
        "confusion_matrix_png",
    ]
    with table_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    grid_desc = (
        "GT∪pred open timeline"
        if args.eval_open_timeline
        else (
            f"magnetic CSV grid (eval-obs-grid={ev_grid}; csv_start={csv_start or '(none)'}; "
            f"csv_end={csv_end or '(none)'}; skip={skip_min})"
        )
    )
    lines = [
        "# Feb 13 2026 improved benchmark",
        "",
        f"- CSV: `{magnetic_csv_path.name}`",
        f"- k={args.k}; historic load: {args.historic_minutes} min; offline skip first: {skip_min} min/sensor",
        f"- App sequence: {', '.join(x[0] for x in APP_SEQUENCE)}. Train window min: {args.train_window_minutes}",
        f"- GT: mode=`{args.gt_mode}`"
        + (
            f", experiment_file=`{experiment_path}`"
            if str(args.gt_mode).strip() == "experiment"
            else f", manual_csv_basename=`{manual_bn}` (must match app.py / evaluate tables)"
        ),
        f"- Evaluator: base_date=`{base_date}`; point_tolerance_sec={args.point_tolerance_sec}; "
        f"event_merge_gap_sec={args.event_merge_gap_sec}; event_tolerance_sec={args.event_tolerance_sec}",
        f"- CSV window: start `{csv_start or '(none)'}`, end `{csv_end or '(none)'}`",
        f"- Point-level eval: `{grid_desc}`",
        (
            f"- **Equal n check:** `TP+FP+TN+FN` verified identical across {len(rows)} rows "
            f"({'yes' if equal_point_totals_verified else 'skipped or single row'})."
        ),
        "",
        "| Predictor | Detector | TP | FP | TN | FN | n | Recall | Precision | F1 | Specificity | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {predictor} | {detector} | {tp} | {fp} | {tn} | {fn} | {n} | {recall:.4f} | {precision:.4f} | {f1_score:.4f} | {specificity:.4f} | {accuracy:.4f} |".format(
                predictor=r["predictor"],
                detector=r["detector"],
                tp=r["tp"],
                fp=r["fp"],
                tn=r["tn"],
                fn=r["fn"],
                n=r.get("evaluated_points", ""),
                recall=float(r["recall"] or 0),
                precision=float(r["precision"] or 0),
                f1_score=float(r["f1_score"] or 0),
                specificity=float(r["specificity"] or 0),
                accuracy=float(r["accuracy"] or 0),
            )
        )
    lines.append("")
    table_md.write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "benchmark": "feb13_2026_improved",
        "created": ts,
        "csv": str(magnetic_csv_path),
        "rows": len(rows),
        "output_dir": str(root),
        "k": float(args.k),
        "historic_minutes": int(args.historic_minutes),
        "skip_initial_minutes_offline": skip_min,
        "train_window_minutes": int(args.train_window_minutes),
        "app_sequence": [x[0] for x in APP_SEQUENCE],
        "csv_end": csv_end or None,
        "csv_start": csv_start or None,
        "manual_csv_basename": manual_bn,
        "base_date": base_date,
        "gt_mode": str(args.gt_mode),
        "experiment_file": str(experiment_path) if str(args.gt_mode).strip() == "experiment" else None,
        "point_tolerance_sec": int(args.point_tolerance_sec),
        "event_merge_gap_sec": int(args.event_merge_gap_sec),
        "event_tolerance_sec": int(args.event_tolerance_sec),
        "eval_obs_grid": ev_grid,
        "eval_open_timeline": bool(args.eval_open_timeline),
        "eval_point_grid_magnetic_csv": str(eval_grid_csv) if eval_grid_csv else None,
        "eval_magnetic_require_all_obs2": bool(eval_grid_csv is not None and req_obs2),
        "eval_magnetic_require_all_obs1": bool(eval_grid_csv is not None and req_obs1),
        "eval_magnetic_csv_end_applied": (csv_end or None) if eval_grid_csv else None,
        "eval_magnetic_csv_start_applied": (csv_start or None) if eval_grid_csv else None,
        "eval_magnetic_skip_initial_minutes_applied": float(skip_min) if eval_grid_csv else None,
        "equal_point_totals_verified": bool(equal_point_totals_verified),
        "fast_csv_playback": bool(args.fast_csv_playback),
        "evaluated_points_by_row": [
            {"predictor": r.get("predictor"), "evaluated_points": r.get("evaluated_points")}
            for r in rows
        ],
    }
    ns = [r.get("evaluated_points") for r in rows]
    try:
        ints = [int(x) for x in ns if x is not None and str(x).strip() != ""]
        meta["shared_evaluated_points"] = ints[0] if ints and len(set(ints)) == 1 else None
        meta["all_schemes_same_evaluated_n"] = bool(ints) and len(set(ints)) == 1
    except (TypeError, ValueError):
        meta["shared_evaluated_points"] = None
        meta["all_schemes_same_evaluated_n"] = False
    (root / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done. Wrote {table_csv} and {table_md} under {root}")


if __name__ == "__main__":
    main()
