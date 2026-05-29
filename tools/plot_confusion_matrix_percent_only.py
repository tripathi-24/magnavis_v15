#!/usr/bin/env python3
"""
Plot a confusion matrix from summary JSON with percentage-only annotations.

Expected JSON shape:
  summary["point_level_metrics"] has keys: tn, fp, fn, tp
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _plot_confusion_percent_only(matrix: np.ndarray, out_png: Path, title: str, point_metrics: dict) -> None:
    total = float(matrix.sum())
    if total <= 0:
        raise ValueError("Confusion matrix total is zero; cannot compute percentages.")

    pct = matrix / total

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.weight": "bold",
        }
    )

    # Compact but readable canvas so matrix + footer fit cleanly.
    fig, ax = plt.subplots(figsize=(7.6, 6.8), dpi=300, facecolor="white")
    im = ax.imshow(pct, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=max(0.5, float(pct.max())))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12, width=1.0)
    cbar.set_label("Percentage", fontsize=14, fontweight="bold")

    classes = ["Normal (0)", "Anomaly (1)"]
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(classes, fontsize=14, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=16, fontweight="bold")
    ax.set_ylabel("Actual Label", fontsize=16, fontweight="bold")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=10)

    # Percentage-only annotations (no raw counts).
    thresh = float(pct.max()) * 0.58
    for i in range(2):
        for j in range(2):
            val = pct[i, j]
            txt = f"{val * 100:.2f}%"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=19,
                fontweight="bold",
                color="white" if val > thresh else "#111111",
            )

    ax.set_aspect("equal")
    # Restore bottom metrics strip (point-level).
    acc = float(point_metrics.get("accuracy", 0.0))
    prec = float(point_metrics.get("precision", 0.0))
    rec = float(point_metrics.get("recall", 0.0))
    f1 = float(point_metrics.get("f1_score", 0.0))
    spec = float(point_metrics.get("specificity", 0.0))
    footer = (
        f"Accuracy: {acc * 100:.2f}%   |   Precision: {prec * 100:.2f}%   |   "
        f"Recall: {rec * 100:.2f}%   |   F1: {f1 * 100:.2f}%   |   Specificity: {spec * 100:.2f}%"
    )
    fig.text(0.5, 0.03, footer, ha="center", va="bottom", fontsize=12.5, fontweight="bold", color="#111111")

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot percent-only confusion matrix from summary JSON.")
    parser.add_argument("--summary-json", required=True, help="Path to *_summary.json containing point_level_metrics")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--title", default="Presence-only Confusion Matrix", help="Figure title")
    args = parser.parse_args()

    summary_path = Path(args.summary_json).resolve()
    out_path = Path(args.out).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    p = summary.get("point_level_metrics", {})
    tn = int(p.get("tn", 0))
    fp = int(p.get("fp", 0))
    fn = int(p.get("fn", 0))
    tp = int(p.get("tp", 0))

    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    _plot_confusion_percent_only(cm, out_path, title=str(args.title), point_metrics=p)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

