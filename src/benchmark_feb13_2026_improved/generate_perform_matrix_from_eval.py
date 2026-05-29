#!/usr/bin/env python3
"""
Build ``perform_matrix_heatmap_GRU_4.png`` and ``perform_matrix_table_GRU_4.png`` (same style as
``benchmark_feb13_2026/generate_perform_matrix.py``) from **eval** summaries under a timestamped run:

  ``results/<run>/eval/<model>/*_summary.json``

Outputs are written under ``benchmark_feb13_2026_improved/results/``:

  - ``perform_matrix_heatmap_GRU_4.png``
  - ``perform_matrix_table_GRU_4.png``
  - ``perform_matrix_GRU_4.csv`` (numeric copy of the table)

All figure text uses **bold** and **larger** font sizes for readability.

Usage::

  cd src/benchmark_feb13_2026_improved
  python generate_perform_matrix_from_eval.py
  python generate_perform_matrix_from_eval.py --results-dir results/20260422_133547
  python generate_perform_matrix_from_eval.py --results-dir results/20260426_063929 \\
      --out-dir results/20260426_063929
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = BENCH_DIR / "results"

# eval/ subfolder name -> row label (heatmap / table)
DISPLAY_NAMES: Dict[str, str] = {
    "baseline_ewma": "EWMA",
    "baseline_median": "Median",
    "baseline_savgol": "Savitzky–Golay",
    "lstm_fresh": "LSTM (fresh)",
    "lstm_fresh_improved": "LSTM fresh (8×LSTM, 4×Dense, W=30)",
    "lstm_pretrained": "LSTM (pretrained)",
    "attn_bilstm_fresh": "Attention Bi-LSTM (fresh)",
    "pretrained_keras_forecaster": "Pretrained Keras forecaster",
    "gru_fresh": "GRU (fresh)",
    "gru_fresh_improved": "GRU fresh (8×GRU, 4×Dense, W=30)",
    "gru_pretrained": "GRU (pretrained)",
}

# Preferred row order; unknown dirs sort after these alphabetically
_ROW_ORDER: List[str] = [
    "baseline_ewma",
    "baseline_median",
    "baseline_savgol",
    "lstm_fresh",
    "lstm_fresh_improved",
    "lstm_pretrained",
    "attn_bilstm_fresh",
    "pretrained_keras_forecaster",
    "gru_fresh",
    "gru_fresh_improved",
    "gru_pretrained",
]

METRIC_COLS = ("recall", "precision", "f1_score", "specificity", "accuracy")

# Figure typography (bold + larger than default)
_HEAT_TICK_FS = 12
_HEAT_ANN_FS = 12
_HEAT_SUBTITLE_FS = 14
_HEAT_SUPTITLE_FS = 15
_TABLE_CELL_FS = 11
_TABLE_AX_TITLE_FS = 14
_TABLE_SUPTITLE_FS = 15
# Header row (column labels) background
_TABLE_HEADER_FACE = "#c8e6c9"


def _latest_results_dir() -> Path:
    if not RESULTS_ROOT.is_dir():
        raise SystemExit(f"No results directory: {RESULTS_ROOT}")
    subs = [p for p in RESULTS_ROOT.iterdir() if p.is_dir()]
    if not subs:
        raise SystemExit(f"No timestamped runs under {RESULTS_ROOT}")
    return max(subs, key=lambda p: p.stat().st_mtime)


def _display_name(predictor: str) -> str:
    return DISPLAY_NAMES.get(predictor.strip(), predictor.strip().replace("_", " ").title())


def _pick_summary_json(eval_subdir: Path) -> Optional[Path]:
    cands = sorted(eval_subdir.glob("*_summary.json"))
    if not cands:
        return None
    for c in cands:
        n = c.name.upper()
        if "ALL" in n or "OBS2_ALL" in n:
            return c
    return cands[0]


def _collect_rows_from_eval(run_dir: Path) -> List[Dict[str, Any]]:
    eval_root = run_dir / "eval"
    if not eval_root.is_dir():
        raise SystemExit(f"Missing eval directory: {eval_root}")

    rows: List[Dict[str, Any]] = []
    for sub in sorted(eval_root.iterdir()):
        if not sub.is_dir():
            continue
        summ = _pick_summary_json(sub)
        if summ is None:
            continue
        data = json.loads(summ.read_text(encoding="utf-8"))
        pm = data.get("point_level_metrics") or {}
        key = sub.name
        rows.append(
            {
                "predictor": key,
                "tp": pm.get("tp", ""),
                "fp": pm.get("fp", ""),
                "tn": pm.get("tn", ""),
                "fn": pm.get("fn", ""),
                "recall": pm.get("recall", ""),
                "precision": pm.get("precision", ""),
                "f1_score": pm.get("f1_score", ""),
                "specificity": pm.get("specificity", ""),
                "accuracy": pm.get("accuracy", ""),
            }
        )

    if not rows:
        raise SystemExit(f"No *_summary.json found under {eval_root}")

    def _sort_key(r: Dict[str, Any]) -> tuple[int, str]:
        p = str(r.get("predictor", ""))
        try:
            return (_ROW_ORDER.index(p), p)
        except ValueError:
            return (len(_ROW_ORDER), p)

    rows.sort(key=_sort_key)
    return rows


def _write_perform_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "model",
        "tp",
        "fp",
        "tn",
        "fn",
        "recall",
        "precision",
        "f1_score",
        "specificity",
        "accuracy",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            pred = str(r.get("predictor", ""))
            w.writerow(
                {
                    "model": _display_name(pred),
                    "tp": r.get("tp", ""),
                    "fp": r.get("fp", ""),
                    "tn": r.get("tn", ""),
                    "fn": r.get("fn", ""),
                    "recall": r.get("recall", ""),
                    "precision": r.get("precision", ""),
                    "f1_score": r.get("f1_score", ""),
                    "specificity": r.get("specificity", ""),
                    "accuracy": r.get("accuracy", ""),
                }
            )


def _metrics_matrix_Z(rows: List[Dict[str, Any]]) -> tuple[np.ndarray, List[str]]:
    labels = [_display_name(str(r.get("predictor", ""))) for r in rows]
    n = len(labels)
    Z = np.zeros((n, len(METRIC_COLS)), dtype=float)
    for i, r in enumerate(rows):
        for j, c in enumerate(METRIC_COLS):
            try:
                Z[i, j] = float(r.get(c) or 0.0)
            except (TypeError, ValueError):
                Z[i, j] = 0.0
    return Z, labels


def _plot_metrics_heatmap(rows: List[Dict[str, Any]], out_png: Path, title: str) -> None:
    Z, labels = _metrics_matrix_Z(rows)
    n = len(labels)
    norm = Normalize(vmin=0.0, vmax=1.0)
    fig_h = max(4.5, 0.58 * n + 2.8)
    fig_w = max(8.0, 1.15 * len(METRIC_COLS) + 4.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(Z, aspect="auto", cmap="RdYlGn", norm=norm, interpolation="nearest")
    ax.set_xticks(np.arange(len(METRIC_COLS)))
    ax.set_xticklabels(
        [c.replace("_", " ").title() for c in METRIC_COLS],
        rotation=25,
        ha="right",
        fontsize=_HEAT_TICK_FS,
        fontweight="bold",
    )
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=_HEAT_TICK_FS, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=_HEAT_TICK_FS)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    ax.set_title(
        "Point metrics (0–1)",
        fontsize=_HEAT_SUBTITLE_FS,
        fontweight="bold",
        pad=12,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=_HEAT_TICK_FS)
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight("bold")
    for i in range(n):
        for j in range(len(METRIC_COLS)):
            ax.text(
                j,
                i,
                f"{Z[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=_HEAT_ANN_FS,
                fontweight="bold",
            )
    fig.suptitle(title, fontsize=_HEAT_SUPTITLE_FS, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _model_column_width_fraction(labels: List[str], n_cols: int) -> List[float]:
    max_len = max((len(s) for s in labels), default=12)
    w0_full = min(0.58, max(0.22, 0.07 + max_len * 0.012))
    # Slightly wider first column than before (was 0.5 * w0_full).
    w0 = min(0.42, w0_full * 0.62)
    rest = (1.0 - w0) / max(1, n_cols - 1)
    return [w0] + [rest] * (n_cols - 1)


def _plot_numeric_table(rows: List[Dict[str, Any]], out_png: Path, title: str) -> None:
    _, labels = _metrics_matrix_Z(rows)
    n = len(labels)
    col_labels = ["Model", "TP", "FP", "TN", "FN", "Recall", "Prec.", "F1", "Spec.", "Acc."]
    n_cols = len(col_labels)
    col_widths = _model_column_width_fraction(labels, n_cols)
    table_data: List[List[str]] = []
    for r in rows:
        table_data.append(
            [
                _display_name(str(r.get("predictor", ""))),
                str(r.get("tp", "")),
                str(r.get("fp", "")),
                str(r.get("tn", "")),
                str(r.get("fn", "")),
                f'{float(r.get("recall") or 0):.4f}',
                f'{float(r.get("precision") or 0):.4f}',
                f'{float(r.get("f1_score") or 0):.4f}',
                f'{float(r.get("specificity") or 0):.4f}',
                f'{float(r.get("accuracy") or 0):.4f}',
            ]
        )
    max_len = max((len(s) for s in labels), default=12)
    fig_w = max(11.5, 3.4 + max_len * 0.06 + n_cols * 0.85)
    fig_h = max(3.5, 0.42 * n + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(_TABLE_CELL_FS)
    table.scale(1.05, 1.5)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor(_TABLE_HEADER_FACE)
        kw: Dict[str, Any] = {"weight": "bold", "fontsize": _TABLE_CELL_FS}
        if col_idx == 0:
            kw["ha"] = "left"
        cell.set_text_props(**kw)
    ax.set_title("Counts & metrics", fontsize=_TABLE_AX_TITLE_FS, fontweight="bold", pad=14)
    fig.suptitle(title, fontsize=_TABLE_SUPTITLE_FS, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _title_with_run_meta(run_dir: Path, base: str) -> str:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        return base
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return base
    bits = [
        f"k={meta.get('k')}",
        f"historic={meta.get('historic_minutes')} min",
        f"skip={meta.get('skip_initial_minutes_offline')} min",
    ]
    ce = meta.get("csv_end")
    if ce:
        bits.append(f"t≤{ce}")
    if meta.get("eval_magnetic_require_all_obs2"):
        bits.append("grid=OBS2_1∩2∩3")
    csv_name = meta.get("eval_point_grid_magnetic_csv") or meta.get("csv")
    if isinstance(csv_name, str) and csv_name:
        bits.append(Path(csv_name).name)
    return f"{base}\n({'; '.join(bits)})"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build perform_matrix_*_GRU_4.png from eval summaries under results/<run>/eval/"
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Run folder containing eval/ (default: latest under results/)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Write CSV/PNG outputs here. Default: this folder's results/ with filenames *_GRU_4.* "
            "Pass e.g. results/20260426_063929 to place figures next to that run's eval/."
        ),
    )
    args = ap.parse_args()
    if args.results_dir:
        rd = Path(args.results_dir)
        run_dir = rd.resolve() if rd.is_absolute() else (BENCH_DIR / rd).resolve()
    else:
        run_dir = _latest_results_dir()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    rows = _collect_rows_from_eval(run_dir)
    title = _title_with_run_meta(run_dir, f"Feb 13 improved — eval aggregate — {run_dir.name}")

    if args.out_dir is not None:
        od = Path(args.out_dir)
        out_dir = od.resolve() if od.is_absolute() else (BENCH_DIR / od).resolve()
        name_suffix = ""
    else:
        out_dir = RESULTS_ROOT
        name_suffix = "_GRU_4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"perform_matrix{name_suffix}.csv"
    out_heat = out_dir / f"perform_matrix_heatmap{name_suffix}.png"
    out_table = out_dir / f"perform_matrix_table{name_suffix}.png"

    _write_perform_csv(rows, out_csv)
    _plot_metrics_heatmap(rows, out_heat, title)
    _plot_numeric_table(rows, out_table, title)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_heat}")
    print(f"Wrote {out_table}")
    print(f"(source: {run_dir / 'eval'})")


if __name__ == "__main__":
    main()
