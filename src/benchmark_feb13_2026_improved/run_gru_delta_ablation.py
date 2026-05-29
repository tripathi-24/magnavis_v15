#!/usr/bin/env python3
"""Train GRU absolute vs Δ-target checkpoints and benchmark Apr-27 k-recall.

Usage (repo root)::

  .venv/bin/python src/benchmark_feb13_2026_improved/run_gru_delta_ablation.py --train --benchmark
  .venv/bin/python src/benchmark_feb13_2026_improved/run_gru_delta_ablation.py --benchmark-only

Outputs under ``results/gru_delta_ablation_<timestamp>/``:
  - ``models/absolute/``, ``models/delta/`` (when --train)
  - ``k_recall_points.csv``, ``GRU_DELTA_ABLATION.md``
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train_gru_pretrained.py"
K_RECALL = BENCH_DIR / "run_k_recall_curves.py"
BASELINE_CSV = (
    BENCH_DIR
    / "results"
    / "k_recall_curves_zero_hist_20260522_170707"
    / "k_recall_points.csv"
)
DEFAULT_TRAIN_CSV = (
    PROJECT_ROOT
    / "Datafiles"
    / "magnetic_data_20251201_000000_to_20251231_234500.csv"
)
K_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]

VARIANTS = (
    {
        "key": "gru_absolute_retrain",
        "label": "GRU absolute (retrained)",
        "train_delta": False,
        "infer_delta_env": "0",
        "model_subdir": "absolute",
    },
    {
        "key": "gru_delta_retrain",
        "label": "GRU Δ-target (retrained)",
        "train_delta": True,
        "infer_delta_env": "1",
        "model_subdir": "delta",
    },
)


def _py() -> str:
    return sys.executable


def _run(cmd: List[str], cwd: Path) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _train_variant(
    root: Path,
    train_csv: Path,
    *,
    delta: bool,
    epochs: int,
    sensors: List[str],
    subdir: str,
    subsample_every: int,
) -> Path:
    out_dir = root / "models" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        _py(),
        str(TRAIN_SCRIPT),
        str(train_csv),
        str(out_dir),
        "--epochs",
        str(epochs),
        "--window-size",
        "15",
        "--sensors",
        *sensors,
        "--subsample-every",
        str(max(1, int(subsample_every))),
    ]
    if delta:
        cmd.append("--delta-target")
    log_path = root / "logs" / f"train_{subdir}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$", " ".join(cmd), f"→ {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as logf:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True, stdout=logf, stderr=subprocess.STDOUT)
    return out_dir


def _benchmark_variant(root: Path, variant: Dict[str, Any]) -> Path:
    model_dir = root / "models" / variant["model_subdir"]
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Missing model dir: {model_dir}")

    out_dir = root / "runs" / variant["key"]
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = {
        "PREDICTOR_MODEL_FAMILY": "gru",
        "PREDICTOR_MODEL_INIT": "pretrained",
        "PRETRAINED_GRU_MODEL_DIR": str(model_dir.resolve()),
        "PREDICTOR_GRU_DELTA_TARGET": str(variant["infer_delta_env"]),
    }
    cmd = [
        _py(),
        str(K_RECALL),
        "--datasets",
        "apr27",
        "--schemes",
        "gru_pretrained",
        "--k-values",
        ",".join(str(k) for k in K_VALUES),
        "--no-cache",
        "--out-dir",
        str(out_dir),
        "--summary-dir",
        str(out_dir),
        "--runs-dataset-key",
        variant["key"],
    ]
    env = dict(**{k: v for k, v in extra.items()})
    # run_k_recall reads APP_SEQUENCE extra from run_suite; pass via subprocess env hack:
    # inject into a tiny wrapper by setting os.environ in child — use env on subprocess instead.
    import os

    child_env = os.environ.copy()
    for k, v in extra.items():
        child_env[k] = v
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(BENCH_DIR), check=True, env=child_env)
    return out_dir / "k_recall_points.csv"


def _load_baseline() -> Dict[float, Dict[str, float]]:
    out: Dict[float, Dict[str, float]] = {}
    if not BASELINE_CSV.is_file():
        return out
    with BASELINE_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("dataset") != "apr27" or row.get("scheme") != "gru_pretrained":
                continue
            k = float(row["k"])
            out[k] = {
                "recall": float(row["recall"]),
                "precision": float(row["precision"]),
                "f1_score": float(row["f1_score"]),
            }
    return out


def _load_points(path: Path, dataset_key: str) -> Dict[float, Dict[str, float]]:
    out: Dict[float, Dict[str, float]] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("dataset") != dataset_key:
                continue
            k = float(row["k"])
            out[k] = {
                "recall": float(row["recall"]),
                "precision": float(row["precision"]),
                "f1_score": float(row["f1_score"]),
            }
    return out


def _write_report(
    root: Path,
    baseline: Dict[float, Dict[str, float]],
    variant_rows: List[tuple[str, Dict[float, Dict[str, float]]]],
) -> None:
    lines = [
        "# GRU absolute vs Δ-target ablation (Apr-27 long GT)",
        "",
        f"**Output:** `{root.name}`",
        "",
        "## Protocol",
        "",
        "- Dataset: Apr-27 long sustained-offset GT (`apr27`)",
        "- Zero-historic, predict-only, OBS2 union grid",
        "- k ∈ {1, 2, 3, 4, 5}",
        "",
        "## Recall vs k (Apr-27)",
        "",
        "| k | Baseline GRU (bundled `models/`) | "
        + " | ".join(label for label, _ in variant_rows)
        + " |",
        "|---|" + "|".join(["---:"] * (1 + len(variant_rows))) + "|",
    ]
    for k in K_VALUES:
        b = baseline.get(k, {})
        cells = [f"{100.0 * b.get('recall', 0.0):.1f}%" if b else "—"]
        for _, pts in variant_rows:
            v = pts.get(k, {})
            cells.append(f"{100.0 * v.get('recall', 0.0):.1f}%" if v else "—")
        lines.append(f"| {k:g} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **Baseline** uses legacy bundled `models/gru_pretrained_*.keras` (no meta sidecar → absolute inference).",
            "- **Absolute retrain** explicitly saves `gru_delta_y: false` and runs with `PREDICTOR_GRU_DELTA_TARGET=0`.",
            "- **Δ-target retrain** saves `gru_delta_y: true` and runs with `PREDICTOR_GRU_DELTA_TARGET=1`.",
            "- If Δ-target recall cliff at k≥4 **disappears** with absolute retrain but **returns** with Δ-target, delta integration is the main culprit.",
            "",
        ]
    )
    (root / "GRU_DELTA_ABLATION.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="GRU absolute vs delta-target Apr-27 ablation")
    ap.add_argument("--train", action="store_true", help="Retrain absolute and delta GRU (OBS2 sensors)")
    ap.add_argument("--benchmark", action="store_true", help="Run Apr-27 k-recall for both variants")
    ap.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Skip training; benchmark existing models under out-dir/models/",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument(
        "--subsample-every",
        type=int,
        default=6,
        help="Train on every Nth sample (faster; default 6 ≈ 13k steps/epoch vs 81k)",
    )
    ap.add_argument(
        "--sensors",
        default="OBS2_1,OBS2_2,OBS2_3",
        help="Comma-separated sensor IDs to train (default: OBS2 trio for benchmark grid)",
    )
    args = ap.parse_args()

    do_train = args.train or (not args.benchmark and not args.benchmark_only)
    do_bench = args.benchmark or args.benchmark_only or (not args.train)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (args.out_dir or (BENCH_DIR / "results" / f"gru_delta_ablation_{ts}")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    sensors = [x.strip() for x in str(args.sensors).split(",") if x.strip()]

    if do_train:
        for v in VARIANTS:
            _train_variant(
                root,
                args.train_csv.resolve(),
                delta=bool(v["train_delta"]),
                epochs=int(args.epochs),
                sensors=sensors,
                subdir=str(v["model_subdir"]),
                subsample_every=int(args.subsample_every),
            )

    variant_rows: List[tuple[str, Dict[float, Dict[str, float]]]] = []
    merged: List[Dict[str, Any]] = []
    if do_bench:
        for v in VARIANTS:
            pts_path = _benchmark_variant(root, v)
            pts = _load_points(pts_path, v["key"])
            variant_rows.append((v["label"], pts))
            for k, m in pts.items():
                merged.append(
                    {
                        "dataset": v["key"],
                        "scheme": "gru_pretrained",
                        "k": k,
                        **m,
                        "status": "ok",
                        "source": str(pts_path),
                    }
                )

        out_csv = root / "k_recall_points.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            fields = ["dataset", "scheme", "k", "recall", "precision", "f1_score", "status", "source"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(merged)

    baseline = _load_baseline()
    if variant_rows:
        _write_report(root, baseline, variant_rows)

    print(f"\nDone. Results: {root}")


if __name__ == "__main__":
    main()
