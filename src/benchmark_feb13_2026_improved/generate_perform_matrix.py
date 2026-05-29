#!/usr/bin/env python3
"""
Build performance figures from ``comparison_table.csv`` produced by ``run_suite.py`` in this folder.

Writes into the chosen run directory (default: latest under ``results/``):
  - ``perform_matrix_heatmap.png``
  - ``perform_matrix_table.png``
  - ``perform_matrix.csv``
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = BENCH_DIR / "results"

DISPLAY_NAMES: Dict[str, str] = {
    "ewma": "EWMA (skip initial window)",
    "median": "Median (skip initial window)",
    "savgol": "Savitzky–Golay (skip initial window)",
    "lstm_fresh_improved": "LSTM fresh (8×LSTM, 4×Dense, W=30)",
    "gru_fresh_improved": "GRU fresh (8×GRU, 4×Dense, W=30)",
    "lstm_pretrained": "LSTM (pretrained)",
    "attention_bi_lstm": "Attention Bi-LSTM (fresh)",
    "gru_pretrained": "GRU (pretrained)",
}

METRIC_COLS = ("recall", "precision", "f1_score", "specificity", "accuracy")


def _latest_results_dir() -> Path:
    if not RESULTS_ROOT.is_dir():
        raise SystemExit(f"No results directory: {RESULTS_ROOT}")
    subs = [p for p in RESULTS_ROOT.iterdir() if p.is_dir()]
    if not subs:
        raise SystemExit(f"No timestamped runs under {RESULTS_ROOT}")
    return max(subs, key=lambda p: p.stat().st_mtime)


def _display_name(predictor: str) -> str:
    return DISPLAY_NAMES.get(predictor.strip(), predictor.strip())


def _read_comparison_table(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    if n == 0:
        raise SystemExit("comparison_table.csv has no rows")
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
    fig_h = max(4.0, 0.55 * n + 2.5)
    fig_w = max(7.5, 1.1 * len(METRIC_COLS) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(Z, aspect="auto", cmap="RdYlGn", norm=norm, interpolation="nearest")
    ax.set_xticks(np.arange(len(METRIC_COLS)))
    ax.set_xticklabels([c.replace("_", " ").title() for c in METRIC_COLS], rotation=25, ha="right")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels)
    ax.set_title("Point metrics (0–1)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(n):
        for j in range(len(METRIC_COLS)):
            ax.text(j, i, f"{Z[i, j]:.3f}", ha="center", va="center", color="black", fontsize=8)
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _model_column_width_fraction(labels: List[str], n_cols: int) -> List[float]:
    max_len = max((len(s) for s in labels), default=12)
    w0_full = min(0.55, max(0.20, 0.06 + max_len * 0.011))
    w0 = 0.5 * w0_full
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
    fig_w = max(11.0, 3.2 + max_len * 0.055 + n_cols * 0.85)
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
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if col_idx == 0:
            cell.set_text_props(ha="left")
        if row_idx == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("Counts & metrics")
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate perform_matrix figures from comparison_table.csv")
    ap.add_argument("--results-dir", type=Path, default=None, help="Run folder (default: latest results/)")
    args = ap.parse_args()
    run_dir = args.results_dir.resolve() if args.results_dir else _latest_results_dir()
    csv_path = run_dir / "comparison_table.csv"
    if not csv_path.is_file():
        raise SystemExit(f"Missing {csv_path}")
    rows = _read_comparison_table(csv_path)
    title = f"Feb 13 improved benchmark — {run_dir.name}"
    out_csv = run_dir / "perform_matrix.csv"
    out_heat = run_dir / "perform_matrix_heatmap.png"
    out_table = run_dir / "perform_matrix_table.png"
    _write_perform_csv(rows, out_csv)
    _plot_metrics_heatmap(rows, out_heat, title)
    _plot_numeric_table(rows, out_table, title)
    print(f"Wrote {out_csv}\nWrote {out_heat}\nWrote {out_table}")


if __name__ == "__main__":
    main()
