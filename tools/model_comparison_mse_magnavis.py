#!/usr/bin/env python3
"""
Build a grouped bar chart (Validation MSE vs Test MSE) similar to ``model_comparison_mse.png``,
for Magnavis predictors and offline baselines on a magnetic export CSV.

**Default deep-learning jobs** (unless ``--skip-deep-learning``): Attention Bi-LSTM (fresh),
LSTM (pretrained), GRU (pretrained) — aligned with the Feb. 13 benchmark comparison set.
Unless ``--deep-only`` is set, statistical baselines EWMA, median, and Savitzky--Golay are included.
``--deep-only`` plots MSE for Attention Bi-LSTM (fresh), LSTM (pretrained), and GRU (pretrained) only.
``--pretrained-only`` plots MSE for **LSTM (pretrained)** and **GRU (pretrained)** only (no baselines, no Attention Bi-LSTM).
Checkpoints are read from ``models/lstm_pretrained_<sensor>.keras`` and ``models/gru_pretrained_<sensor>.keras``;
those folders do not store Keras training histories, so this script estimates **forecast MSE** on the chosen magnetic window rather than reading loss logs.

Designed for **very large** CSVs (streaming read with optional time bounds and row cap).

Typical usage (October 2025 dataset, first week — adjust ``--time-end`` / ``--max-points`` as needed)::

  cd /path/to/magnavis_v13_Polish
  python tools/model_comparison_mse_magnavis.py \\
    --csv magnetic_data_20251001_000000_to_20251015_234500.csv \\
    --sensor OBS2_1 \\
    --time-start \"2025-10-01 00:00:00\" \\
    --time-end \"2025-10-08 00:00:00\" \\
    --max-points 80000 \\
    --out model_comparison_mse_magnavis.png

Requires: numpy, pandas, matplotlib, scipy; **TensorFlow** for LSTM / Attention-BiLSTM / GRU / pretrained runs.

Use the same interpreter as your Magnavis venv (TensorFlow is often **not** on the system Python), e.g.::

  export MAGNAVIS_BENCHMARK_PYTHON=/path/to/.venv/bin/python
  python tools/model_comparison_mse_magnavis.py ...
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
OFFLINE = SRC / "benchmark_feb13_2026_improved" / "offline_statistical_baselines.py"


def _sensor_mask(series: pd.Series, sensor_tag: str) -> pd.Series:
    """Match ``sensor_id``: suffix mode for tags like ``_1`` (avoids ``_10`` matching ``_1``)."""
    s = series.astype(str)
    st = str(sensor_tag)
    if st.startswith("_") and len(st) <= 6:
        return s.str.endswith(st)
    return s.str.contains(st, regex=False)


def _bench_python() -> str:
    """Prefer venv / benchmark interpreter (TensorFlow)."""
    override = os.environ.get("MAGNAVIS_BENCHMARK_PYTHON", "").strip()
    if override:
        return override
    venv_py = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def _load_offline_module():
    spec = importlib.util.spec_from_file_location("offline_statistical_baselines", OFFLINE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_mag_series_streaming(
    csv_path: Path,
    sensor_tag: str,
    time_start: Optional[pd.Timestamp],
    time_end: Optional[pd.Timestamp],
    chunksize: int,
    assume_chronological: bool,
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """Load |B| series for rows matching ``sensor_tag`` and optional ``[time_start, time_end]``."""
    usecols = ["sensor_id", "timestamp", "b_x", "b_y", "b_z"]
    parts: List[pd.DataFrame] = []
    n_read = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=int(chunksize), low_memory=False):
        n_read += len(chunk)
        if n_read % (chunksize * 10) == 0:
            print(f"  … scanned {n_read:,} CSV rows", flush=True)
        tchunk = pd.to_datetime(chunk["timestamp"], errors="coerce")
        if (
            assume_chronological
            and time_end is not None
            and tchunk.notna().any()
            and bool((tchunk.dropna() > time_end).all())
        ):
            break
        m = _sensor_mask(chunk["sensor_id"], sensor_tag)
        if not bool(m.any()):
            continue
        sub = chunk.loc[m].copy()
        sub["t"] = pd.to_datetime(sub["timestamp"], errors="coerce")
        sub = sub.dropna(subset=["t"])
        if time_start is not None:
            sub = sub[sub["t"] >= time_start]
        if time_end is not None:
            sub = sub[sub["t"] <= time_end]
        if sub.empty:
            continue
        bx = pd.to_numeric(sub["b_x"], errors="coerce")
        by = pd.to_numeric(sub["b_y"], errors="coerce")
        bz = pd.to_numeric(sub["b_z"], errors="coerce")
        sub["mag"] = np.sqrt(bx * bx + by * by + bz * bz)
        sub = sub.dropna(subset=["mag"])
        parts.append(sub[["t", "mag"]])

    if not parts:
        raise SystemExit(
            f"No data for sensor_id containing {sensor_tag!r} in the given time window.\n"
            f"Hint: large exports often use session-style ids (e.g. …_1); try --sensor \"_1\" or a unique substring."
        )
    df = pd.concat(parts, ignore_index=True).sort_values("t")
    df = df.groupby("t", as_index=False)["mag"].mean().sort_values("t")
    ts = pd.DatetimeIndex(pd.to_datetime(df["t"]))
    y = df["mag"].to_numpy(dtype=np.float64)
    return ts, y


def _trim_skip(ts: pd.DatetimeIndex, y: np.ndarray, skip_minutes: float) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    if skip_minutes <= 0 or len(ts) == 0:
        return ts, y
    cutoff = ts[0] + pd.Timedelta(minutes=float(skip_minutes))
    m = ts >= cutoff
    return ts[m], y[m]


def _subsample(ts: pd.DatetimeIndex, y: np.ndarray, max_points: int) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    if max_points <= 0 or len(ts) <= max_points:
        return ts, y
    idx = np.unique(np.linspace(0, len(ts) - 1, num=max_points, dtype=np.int64))
    return ts[idx], y[idx]


def _segment_mse(y: np.ndarray, p: np.ndarray, i0: int, i1: int) -> float:
    sl = slice(i0, i1)
    d = y[sl] - p[sl]
    return float(np.mean(d * d))


def _resolve_pretrained(project_root: Path, checkpoint_sensor_id: str, family: str) -> Optional[Path]:
    """Resolve ``models/<family>_pretrained_<sensor>.keras`` for gru / lstm / transformer."""
    models = project_root / "models"
    fam = str(family).strip().lower()
    if fam == "transformer":
        p = models / f"transformer_pretrained_{checkpoint_sensor_id}.keras"
    elif fam == "lstm":
        p = models / f"lstm_pretrained_{checkpoint_sensor_id}.keras"
    else:
        p = models / f"gru_pretrained_{checkpoint_sensor_id}.keras"
    return p if p.is_file() else None


def _run_predictor(
    work: Path,
    ts: pd.DatetimeIndex,
    y: np.ndarray,
    *,
    family: str,
    init: str,
    pretrained_path: Optional[Path],
    project_root: Path,
    epochs: int,
    gru_window: int,
) -> pd.DataFrame:
    inp = work / "predict_input.csv"
    pd.DataFrame({"x": ts, "y": y}).to_csv(inp, index=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PREDICTOR_MODEL_FAMILY"] = family
    env["PREDICTOR_MODEL_INIT"] = init
    env["PREDICTOR_GRU_WINDOW_SIZE"] = str(int(gru_window))
    env["PREDICTOR_EPOCHS_PER_UPDATE"] = str(int(epochs))
    env["PREDICTOR_N_FUTURE"] = "200"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    env.setdefault("TF_NUM_INTEROP_THREADS", "1")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    rt = work / f"runtime_{family}.keras"
    env["PREDICTOR_CHECKPOINT_PATH"] = str(rt)
    if init == "fresh":
        env["PREDICTOR_UPDATE_TRAINING"] = "1"
        env["PREDICTOR_LEADING_TRAIN_MINUTES"] = "62"
        env.pop("PREDICTOR_SKIP_INITIAL_MINUTES", None)
        env.pop("PRETRAINED_MODEL_PATH", None)
    else:
        env["PREDICTOR_UPDATE_TRAINING"] = "0"
        env["PREDICTOR_SKIP_INITIAL_MINUTES"] = "62"
        env.pop("PREDICTOR_LEADING_TRAIN_MINUTES", None)
        if pretrained_path is not None:
            env["PRETRAINED_MODEL_PATH"] = str(pretrained_path)
        else:
            env.pop("PRETRAINED_MODEL_PATH", None)

    if rt.exists():
        rt.unlink()
    for pat in ("*_scaler.pkl", "*_predictor_meta.json"):
        for f in work.glob(pat):
            try:
                f.unlink()
            except OSError:
                pass

    cmd = [_bench_python(), str(project_root / "src" / "predictor_ai.py"), str(inp)]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(work), env=env, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        print(r.stdout[-4000:] if r.stdout else "", file=sys.stderr)
        print(r.stderr[-4000:] if r.stderr else "", file=sys.stderr)
        raise RuntimeError(f"predictor_ai failed rc={r.returncode} for family={family} init={init}")
    out = work / "predict_out.csv"
    if not out.is_file():
        raise FileNotFoundError(f"Missing {out}")
    return pd.read_csv(out)


def _align_predictions(ts: pd.DatetimeIndex, y: np.ndarray, pred_df: pd.DataFrame) -> np.ndarray:
    """Map predictions onto ``ts`` (NaN where unknown)."""
    base = pd.DataFrame({"t": pd.to_datetime(ts), "y": y})
    pred_df = pred_df.copy()
    pred_df["x"] = pd.to_datetime(pred_df["x"])
    merged = base.merge(pred_df.rename(columns={"y": "p"}), left_on="t", right_on="x", how="left")
    return merged["p"].to_numpy(dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "magnetic_data_20251001_000000_to_20251015_234500.csv",
        help="Magnetic export CSV (large files are streamed).",
    )
    ap.add_argument(
        "--sensor",
        default="_1",
        help="Substring matched against ``sensor_id`` (default \"_1\" for first component in session-style exports). "
        "Use ``OBS2_1`` for observatory-tagged CSVs.",
    )
    ap.add_argument(
        "--pretrained-sensor",
        default="OBS2_1",
        help="Sensor tag embedded in bundled ``models/*_pretrained_<id>.keras`` filenames (default OBS2_1).",
    )
    ap.add_argument(
        "--time-start",
        default="2025-10-08 00:00:00",
        help="Inclusive lower time bound (large Oct 2025 export may start after the filename date).",
    )
    ap.add_argument(
        "--time-end",
        default="2025-10-15 23:45:00",
        help="Inclusive upper time bound.",
    )
    ap.add_argument("--skip-initial-minutes", type=float, default=62.0)
    ap.add_argument("--max-points", type=int, default=100_000, help="Uniform subsample cap after load (0 = no cap).")
    ap.add_argument("--val-fraction", type=float, default=0.7, help="Chronological split for validation vs test MSE.")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument(
        "--assume-chronological-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop reading once a chunk's timestamps are all after --time-end (unsafe if rows are not time-sorted).",
    )
    ap.add_argument("--gru-window", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=25, help="Epochs per in-session fit (predictor_ai).")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "model_comparison_mse_magnavis.png")
    ap.add_argument("--csv-metrics", type=Path, default=None, help="Optional path to write MSE table CSV.")
    ap.add_argument(
        "--skip-deep-learning",
        action="store_true",
        help="Only EWMA / Median / Savitzky–Golay (no predictor_ai / TensorFlow).",
    )
    ap.add_argument(
        "--deep-only",
        action="store_true",
        help="Only Attention Bi-LSTM (fresh), LSTM (pretrained), GRU (pretrained) — no EWMA/Median/Savitzky–Golay.",
    )
    ap.add_argument(
        "--pretrained-only",
        action="store_true",
        help="Only LSTM (pretrained) and GRU (pretrained) — no baselines, no Attention Bi-LSTM.",
    )
    args = ap.parse_args()
    if args.deep_only and args.skip_deep_learning:
        raise SystemExit("Choose at most one of --deep-only and --skip-deep-learning.")
    if args.pretrained_only and args.skip_deep_learning:
        raise SystemExit("Choose at most one of --pretrained-only and --skip-deep-learning.")
    if args.deep_only and args.pretrained_only:
        raise SystemExit("Choose at most one of --deep-only and --pretrained-only.")

    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise SystemExit("matplotlib is required. Install project deps or activate your venv.") from e

    csv_path = args.csv if args.csv.is_absolute() else (PROJECT_ROOT / args.csv).resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    t0 = pd.to_datetime(args.time_start)
    t1 = pd.to_datetime(args.time_end)
    if t1 <= t0:
        raise SystemExit("--time-end must be after --time-start")

    print(f"Loading {csv_path.name} for sensor {args.sensor!r} in [{t0}, {t1}] …", flush=True)
    ts, y = _load_mag_series_streaming(
        csv_path, args.sensor, t0, t1, args.chunksize, bool(args.assume_chronological_csv)
    )
    print(f"  Loaded {len(y):,} points after time filter.", flush=True)
    ts, y = _subsample(ts, y, int(args.max_points))
    if int(args.max_points) > 0:
        print(f"  After max-points cap: {len(y):,} points.", flush=True)

    ts = pd.DatetimeIndex(pd.to_datetime(ts))
    y_arr = np.asarray(y, dtype=np.float64)
    skip_m = float(args.skip_initial_minutes)
    if skip_m > 0.0:
        cutoff = pd.Timestamp(ts[0]) + pd.Timedelta(minutes=skip_m)
        eval_mask = np.asarray(ts >= cutoff, dtype=bool)
        print(
            f"  MSE uses timestamps >= {cutoff} (--skip-initial-minutes={skip_m}); "
            "predictors receive the full window so internal burn-in matches the benchmark.",
            flush=True,
        )
    else:
        eval_mask = np.ones(len(ts), dtype=bool)
    n_eval = int(np.sum(eval_mask))
    if n_eval < 50:
        raise SystemExit(
            f"Too few evaluation points ({n_eval}) after skip-initial; widen --time-end or lower --skip-initial-minutes."
        )
    print(f"  Evaluation segment: {n_eval:,} points.", flush=True)

    def pair_mse_on_mask(pred: np.ndarray) -> Tuple[float, float]:
        ye = y_arr[eval_mask]
        pe = np.asarray(pred, dtype=np.float64)[eval_mask]
        n_e = len(ye)
        split_e = max(1, min(n_e - 1, int(float(args.val_fraction) * n_e)))
        return _segment_mse(ye, pe, 0, split_e), _segment_mse(ye, pe, split_e, n_e)

    results: Dict[str, Tuple[float, float]] = {}
    if not args.deep_only and not args.pretrained_only:
        offline = _load_offline_module()
        pred_ewma = offline._predict_ewma_shifted(y_arr)
        pred_med = offline._predict_median_shifted(y_arr)
        pred_sg = offline._predict_savgol_shifted(y_arr)
        results = {
            "EWMA": pair_mse_on_mask(pred_ewma),
            "Median": pair_mse_on_mask(pred_med),
            "Savitzky–Golay": pair_mse_on_mask(pred_sg),
        }
    elif args.deep_only:
        print("  --deep-only: skipping EWMA, median, and Savitzky–Golay.", flush=True)
    else:
        print("  --pretrained-only: skipping baselines and Attention Bi-LSTM.", flush=True)

    dl_jobs: List[Tuple[str, str, str, Optional[Path]]] = []
    if not args.skip_deep_learning:
        if args.pretrained_only:
            dl_jobs = [
                (
                    "LSTM (pretrained)",
                    "lstm",
                    "pretrained",
                    _resolve_pretrained(PROJECT_ROOT, str(args.pretrained_sensor), "lstm"),
                ),
                (
                    "GRU (pretrained)",
                    "gru",
                    "pretrained",
                    _resolve_pretrained(PROJECT_ROOT, str(args.pretrained_sensor), "gru"),
                ),
            ]
        else:
            # Same forecaster families as benchmark_feb13_2026_improved (no fresh LSTM/GRU, no Transformer here).
            dl_jobs = [
                ("Attention Bi-LSTM", "attn_bilstm", "fresh", None),
                (
                    "LSTM (pretrained)",
                    "lstm",
                    "pretrained",
                    _resolve_pretrained(PROJECT_ROOT, str(args.pretrained_sensor), "lstm"),
                ),
                (
                    "GRU (pretrained)",
                    "gru",
                    "pretrained",
                    _resolve_pretrained(PROJECT_ROOT, str(args.pretrained_sensor), "gru"),
                ),
            ]

    for label, fam, init, ck in dl_jobs:
        if init == "pretrained" and ck is None:
            print(f"SKIP {label}: missing checkpoint under models/.", flush=True)
            continue
        tmp = Path(tempfile.mkdtemp(prefix="mse_magnavis_"))
        try:
            print(f"Running {label} …", flush=True)
            pred_df = _run_predictor(
                tmp,
                ts,
                y_arr,
                family=fam,
                init=init,
                pretrained_path=ck,
                project_root=PROJECT_ROOT,
                epochs=int(args.epochs),
                gru_window=int(args.gru_window),
            )
            aligned = _align_predictions(ts, y_arr, pred_df)
            ok = np.isfinite(aligned)
            mask = eval_mask & ok
            if int(np.sum(mask)) < max(50, n_eval // 10):
                print(f"WARN {label}: only {int(np.sum(mask))} finite aligned preds in eval segment; check predictor.", flush=True)
            if int(np.sum(mask)) < 2:
                print(f"SKIP {label}: insufficient predictions.", flush=True)
                continue
            # split on masked indices: preserve chronological split by position in original series
            idx = np.flatnonzero(mask)
            split_m = max(1, min(len(idx) - 1, int(float(args.val_fraction) * len(idx))))
            i_val = idx[:split_m]
            i_test = idx[split_m:]
            val_mse = float(np.mean((y_arr[i_val] - aligned[i_val]) ** 2))
            test_mse = float(np.mean((y_arr[i_test] - aligned[i_test]) ** 2))
            results[label] = (val_mse, test_mse)
            print(f"  {label}: val_mse={val_mse:.6f} test_mse={test_mse:.6f}", flush=True)
        except Exception as e:
            print(f"FAIL {label}: {e}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if len(results) < 1:
        raise SystemExit("No models produced metrics.")

    labels = list(results.keys())
    val = [results[k][0] for k in labels]
    tst = [results[k][1] for k in labels]

    import matplotlib.pyplot as plt

    x = np.arange(len(labels))
    w = 0.36
    fig_w = 11.0
    if args.pretrained_only and len(labels) == 2:
        fig_w = 6.0
    elif (args.deep_only or args.pretrained_only) and len(labels) <= 3:
        fig_w = 7.5
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))
    ax.bar(x - w / 2, val, width=w, label="Validation MSE", color="#4472C4", edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, tst, width=w, label="Test MSE", color="#ED7D31", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("MSE (nT²)")
    if args.pretrained_only:
        title = "Pretrained LSTM vs GRU: Validation vs Test MSE (Magnavis)"
    elif args.deep_only:
        title = "Deep models: Validation vs Test MSE (Magnavis)"
    else:
        title = "Model comparison: Validation vs Test MSE (Magnavis)"
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = args.out if args.out.is_absolute() else (PROJECT_ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close()
    print(f"Wrote {out}", flush=True)

    if args.csv_metrics:
        mpath = args.csv_metrics if args.csv_metrics.is_absolute() else (PROJECT_ROOT / args.csv_metrics)
        pd.DataFrame({"model": labels, "val_mse": val, "test_mse": tst}).to_csv(mpath, index=False)
        print(f"Wrote {mpath}", flush=True)


if __name__ == "__main__":
    main()
