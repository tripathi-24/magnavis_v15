#!/usr/bin/env python3
"""
Feb 13 2026 benchmark driver:

1) Offline statistical baselines (EWMA / median / Savitzky–Golay) → synthetic logs → evaluate_anomaly_detection.py
2) For each deep-learning predictor, runs ``src/app.py`` in MAGNAVIS_HEADLESS_BATCH mode (Qt) then evaluates app.log

Default profile (Feb 13 CSV): k=4, 62 minutes historic, train on all loaded data (0 = all),
OBS2_1–OBS2_3, sequence length W=15 for fresh **LSTM** and fresh **GRU**.

**Run order:** EWMA → median → Savitzky–Golay (offline) → vanilla **LSTM** (fresh) → **Attention
Bi-LSTM** (fresh) → **Transformer** (pretrained) → **GRU** (fresh) → **GRU** (pretrained).

By default, magnetic CSV rows are **truncated at ``2026-02-13 16:18:30`` inclusive** for every model
(``MAGNAVIS_BATCH_CSV_END`` in ``app.py``, ``--csv-end`` for offline baselines).

Usage (with your usual Python env):

  From ``src/``: ``cd benchmark_feb13_2026`` then ``python run_suite.py``.
  From repo root: ``python src/benchmark_feb13_2026/run_suite.py``.
  For the **improved** subset (deep RNN, offline trim): ``cd src/benchmark_feb13_2026_improved``
  then ``python run_suite.py`` (different driver; supports the same ``--eval-point-grid-magnetic-csv``).

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
# Inclusive upper bound on CSV ``timestamp`` for Feb 13 benchmark (app + offline baselines).
DEFAULT_CSV_END = "2026-02-13 16:18:30"

# Feb 13 2026 benchmark: three OBS2 sensors, fixed historic and detector (see README / argparse defaults).
OBS2_SENSORS_CSV = "OBS2_1,OBS2_2,OBS2_3"
EVAL_SENSOR_ALL = "ALL"

# (subprocess log stem / eval folder, env extras, comparison-table predictor name(s))
APP_SEQUENCE: List[Tuple[str, Dict[str, str], Tuple[str, ...]]] = [
    (
        "lstm_fresh",
        {
            "PREDICTOR_MODEL_FAMILY": "lstm",
            "PREDICTOR_MODEL_INIT": "fresh",
            "PREDICTOR_GRU_WINDOW_SIZE": "15",
        },
        ("lstm",),
    ),
    (
        "attn_bilstm_fresh",
        {"PREDICTOR_MODEL_FAMILY": "attn_bilstm", "PREDICTOR_MODEL_INIT": "fresh"},
        ("attention_bi_lstm",),
    ),
    (
        "pretrained_keras_forecaster",
        {"PREDICTOR_MODEL_FAMILY": "transformer", "PREDICTOR_MODEL_INIT": "pretrained"},
        ("pretrained_keras_forecaster",),
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
        "gru_pretrained",
        {"PREDICTOR_MODEL_FAMILY": "gru", "PREDICTOR_MODEL_INIT": "pretrained"},
        ("gru_pretrained",),
    ),
]


def _bench_python() -> str:
    override = os.environ.get("MAGNAVIS_BENCHMARK_PYTHON", "").strip()
    if override:
        return override
    return sys.executable


def _predictor_initial_split_env(extra: Dict[str, str], minutes: float) -> Dict[str, str]:
    """See benchmark_feb13_2026_improved/run_suite_improved.py (same semantics)."""
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
        "\nThis benchmark needs NumPy, pandas, scikit-learn, and matplotlib in the Python you use.\n"
        f"Interpreter tried: {py}\n"
        "Install project dependencies, then rerun from src/benchmark_feb13_2026/:\n\n"
        f"  cd {PROJECT_ROOT}\n"
        "  source .venv/bin/activate   # or your own env\n"
        f"  pip install -r {req.name}\n\n"
        f"Requirements path: {req}\n",
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
    magnetic_csv: Optional[Path] = None,
    magnetic_require_all_obs2: bool = False,
    magnetic_csv_end: str = "",
    magnetic_csv_skip_initial_minutes: float = 0.0,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    py = _bench_python()
    cmd = [
        py,
        str(EVAL),
        "--log-file",
        str(log_file),
        "--gt-mode",
        "manual_app",
        "--manual-csv-basename",
        MANUAL_BASE,
        "--sensor",
        sensor,
        "--base-date",
        "2026-02-13",
        "--prediction-sensor-mode",
        pred_mode,
        "--out-dir",
        str(out_dir),
        "--prefix",
        prefix,
    ]
    if magnetic_csv is not None:
        cmd.extend(["--magnetic-csv", str(magnetic_csv.resolve())])
        if magnetic_require_all_obs2:
            cmd.append("--magnetic-csv-require-all-obs2")
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


def main() -> None:
    if not APP_MAIN.is_file():
        print(
            "Cannot find app.py next to this folder (expected src/app.py).\n"
            "Run from the benchmark directory:\n"
            f"  cd {BENCH_DIR}\n"
            "  python run_suite.py",
            file=sys.stderr,
        )
        sys.exit(2)

    _assert_scientific_stack(_bench_python())

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Feb 13 magnetic export CSV")
    ap.add_argument("--k", type=float, default=4.0, help="Detector k (AnomalyDetector on |error|)")
    ap.add_argument("--historic-minutes", type=int, default=62, help="Initial historic window for headless CSV replay")
    ap.add_argument(
        "--train-window-minutes",
        type=int,
        default=0,
        help="Predictor training window (minutes); 0 = all loaded historic data (matches app spinbox)",
    )
    ap.add_argument("--app-timeout-sec", type=int, default=7200, help="Per app.py subprocess wall clock")
    ap.add_argument(
        "--batch-sensors",
        default=OBS2_SENSORS_CSV,
        help="MAGNAVIS_BATCH_SENSORS for headless app (comma-separated)",
    )
    ap.add_argument(
        "--offline-sensors",
        default=OBS2_SENSORS_CSV,
        help="Comma list passed to offline_statistical_baselines --sensors",
    )
    ap.add_argument(
        "--csv-end",
        default=DEFAULT_CSV_END,
        help="Truncate CSV at this timestamp inclusive for app + offline baselines (pandas-parsable)",
    )
    ap.add_argument(
        "--no-csv-end",
        action="store_true",
        help="Do not truncate by time (use full CSV span)",
    )
    ap.add_argument(
        "--prediction-sensor-mode",
        default="union_all",
        choices=("union_all", "filter"),
        help="For app logs and multi-sensor baselines: OR across sensors (union_all)",
    )
    ap.add_argument(
        "--eval-point-grid-magnetic-csv",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "Passed to tools/evaluate_anomaly_detection.py as --magnetic-csv: point-level TP+FP+TN+FN "
            "use the same 1 Hz timestamps (from that file) for every model. Often set to the same path as --csv."
        ),
    )
    ap.add_argument(
        "--eval-magnetic-require-all-obs2",
        action="store_true",
        help="With --eval-point-grid-magnetic-csv: only seconds with OBS2_1, OBS2_2, and OBS2_3 rows.",
    )
    ap.add_argument("--skip-app", action="store_true", help="Only offline baselines + aggregate existing eval/")
    ap.add_argument(
        "--only",
        default="",
        help="Comma list: ewma,median,savgol,lstm,attention_bi_lstm,pretrained_keras_forecaster,gru_fresh,gru_pretrained (+log stems lstm_fresh, attn_bilstm_fresh, …)",
    )
    args = ap.parse_args()

    magnetic_csv_path = args.csv.expanduser().resolve()
    if not magnetic_csv_path.is_file():
        print(
            "ERROR: --csv must point to an existing magnetic export file.\n"
            f"  You passed: {args.csv}\n"
            f"  Resolved: {magnetic_csv_path}\n\n"
            "Use a real path, not a `/path/to/...` placeholder. Omit --csv for the repo default:\n"
            f"  {DEFAULT_CSV}\n\n"
            "If you are already in this folder, run `python run_suite.py` without `cd src/...`.\n",
            file=sys.stderr,
        )
        sys.exit(2)

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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = BENCH_DIR / "results" / ts
    logs = root / "logs"
    evroot = root / "eval"
    logs.mkdir(parents=True, exist_ok=True)

    only = _only_set(args.only)

    csv_end = "" if args.no_csv_end else str(args.csv_end).strip()

    csv_rel = os.path.relpath(magnetic_csv_path, PROJECT_ROOT)
    if csv_rel.startswith(".."):
        csv_arg = str(magnetic_csv_path)
    else:
        csv_arg = csv_rel

    rows: List[Dict[str, Any]] = []

    # --- Offline baselines: one merged log per mode (OBS2_1 + OBS2_2 + OBS2_3) ---
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
            "--out-log",
            str(log_copy),
        ]
        if csv_end:
            off_cmd.extend(["--csv-end", csv_end])
        rc = _run(
            off_cmd,
            os.environ.copy(),
            PROJECT_ROOT,
            600,
        )
        if rc != 0:
            print(f"WARN: baseline {mode} failed rc={rc}")
            continue
        summ = _run_eval(
            log_copy,
            evroot / f"baseline_{mode}",
            f"baseline_{mode}_OBS2_all",
            EVAL_SENSOR_ALL,
            str(args.prediction_sensor_mode),
            magnetic_csv=eval_grid_csv,
            magnetic_require_all_obs2=bool(args.eval_magnetic_require_all_obs2),
            magnetic_csv_end=csv_end if eval_grid_csv else "",
            magnetic_csv_skip_initial_minutes=0.0,
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
            _pp = env.get("PYTHONPATH", "").strip()
            env["PYTHONPATH"] = str(SRC) if not _pp else f"{SRC}{os.pathsep}{_pp}"
            for k, v in _predictor_initial_split_env(extra, 62.0).items():
                env[k] = v
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
                magnetic_csv=eval_grid_csv,
                magnetic_require_all_obs2=bool(args.eval_magnetic_require_all_obs2),
                magnetic_csv_end=csv_end if eval_grid_csv else "",
                magnetic_csv_skip_initial_minutes=0.0,
            )
            pm, cm = _read_metrics(summ)
            if only is None:
                labels: Tuple[str, ...] = table_labels
            else:
                picked = [lab for lab in table_labels if lab.lower() in only]
                labels = tuple(picked) if picked else table_labels
            _append_metric_rows(rows, labels, float(args.k), pm, cm, summ)

    # --- Write comparison tables ---
    table_csv = root / "comparison_table.csv"
    table_md = root / "comparison_table.md"
    fieldnames = [
        "predictor",
        "detector",
        "tp",
        "fp",
        "tn",
        "fn",
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
    lines = [
        "# Feb 13 2026 benchmark (manual GT from app.py)",
        "",
        f"- CSV: `{magnetic_csv_path.name}`",
        f"- k={args.k}; historic load: {args.historic_minutes} min; predictor train window: {args.train_window_minutes} min (0 = all loaded data)",
        f"- Sensors: `{args.batch_sensors}` (app) / `{args.offline_sensors}` (offline baselines); eval: `{EVAL_SENSOR_ALL}` + `{args.prediction_sensor_mode}`",
        f"- CSV time cap: `{csv_end or '(none — full file)'}`",
        "- Fresh **LSTM** and fresh **GRU** use `PREDICTOR_GRU_WINDOW_SIZE=15` (sequence length W).",
        "",
        "| Predictor | Detector | TP | FP | TN | FN | Recall | Precision | F1 | Specificity | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {predictor} | {detector} | {tp} | {fp} | {tn} | {fn} | {recall:.4f} | {precision:.4f} | {f1_score:.4f} | {specificity:.4f} | {accuracy:.4f} |".format(
                predictor=r["predictor"],
                detector=r["detector"],
                tp=r["tp"],
                fp=r["fp"],
                tn=r["tn"],
                fn=r["fn"],
                recall=float(r["recall"] or 0),
                precision=float(r["precision"] or 0),
                f1_score=float(r["f1_score"] or 0),
                specificity=float(r["specificity"] or 0),
                accuracy=float(r["accuracy"] or 0),
            )
        )
    lines.append("")
    lines.append("Confusion matrix PNG paths are in `comparison_table.csv` (`confusion_matrix_png` column).")
    table_md.write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "created": ts,
        "csv": str(magnetic_csv_path),
        "rows": len(rows),
        "output_dir": str(root),
        "k": float(args.k),
        "historic_minutes": int(args.historic_minutes),
        "train_window_minutes": int(args.train_window_minutes),
        "batch_sensors": str(args.batch_sensors).strip(),
        "offline_sensors": str(args.offline_sensors).strip(),
        "eval_sensor_tag": EVAL_SENSOR_ALL,
        "prediction_sensor_mode": str(args.prediction_sensor_mode),
        "csv_end": csv_end or None,
        "eval_point_grid_magnetic_csv": str(eval_grid_csv) if eval_grid_csv else None,
        "eval_magnetic_require_all_obs2": bool(args.eval_magnetic_require_all_obs2),
    }
    (root / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done. Wrote {table_csv} and {table_md} under {root}")


if __name__ == "__main__":
    main()
