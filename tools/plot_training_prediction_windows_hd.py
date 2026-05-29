#!/usr/bin/env python3
"""
Create an HD PNG figure that explains periodic retraining + prediction-only windows.

The figure is illustrative (not from live run logs) and is meant for documentation/demo.
"""

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def build_synthetic_series(total_minutes: int, seed: int = 42):
    """Generate a smooth synthetic magnetic-field series with visible dots."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, total_minutes + 1, dtype=float)

    baseline = 48000.0
    trend = 0.03 * t
    wave1 = 6.5 * np.sin(2 * np.pi * t / 65.0)
    wave2 = 2.8 * np.sin(2 * np.pi * t / 14.0)
    anomaly1 = 9.0 * np.exp(-0.5 * ((t - 95.0) / 6.0) ** 2)
    anomaly2 = -7.0 * np.exp(-0.5 * ((t - 182.0) / 7.5) ** 2)
    noise = rng.normal(0.0, 0.7, size=t.shape[0])

    actual = baseline + trend + wave1 + wave2 + anomaly1 + anomaly2 + noise

    # Predicted: slightly smoothed and lightly lagging proxy.
    pred = actual.copy()
    for i in range(2, len(pred)):
        pred[i] = 0.70 * pred[i - 1] + 0.25 * pred[i - 2] + 0.05 * pred[i]
    pred += 0.35 * np.sin(2 * np.pi * t / 37.0) - 0.20

    return t, actual, pred


def make_figure(
    out_png: str,
    retrain_interval_min: int = 60,
    total_minutes: int = 240,
    training_window_visual_min: int = 8,
):
    t, actual, pred = build_synthetic_series(total_minutes=total_minutes)

    # Typography + contrast settings for publication-style clarity.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#111111",
            "axes.linewidth": 1.4,
            "xtick.color": "#111111",
            "ytick.color": "#111111",
        }
    )

    # Bright, high-contrast palette.
    # Match application_temp.py line colors for visual uniformity:
    # - Actual/realtime line: color=[0.1, 0.7, 0.2]
    # - Prediction line: color=[0.3, 0.1, 0.4]
    c_actual = "#1AB233"     # light green (application_temp.py actual/realtime)
    c_pred = "#4D1A66"       # purple (application_temp.py prediction)
    c_train = "#FFB300"      # high-contrast amber
    c_predict_window = "#00BCD4"  # cyan for predict-only background windows
    c_grid = "#A8B3C3"
    c_text = "#111111"
    c_bg = "#FFFFFF"

    # Slightly smaller canvas, but still high-resolution output.
    fig = plt.figure(figsize=(16, 9), dpi=300, facecolor=c_bg)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4.6, 1.8], hspace=0.22)
    ax_ts = fig.add_subplot(gs[0])
    ax_mode = fig.add_subplot(gs[1], sharex=ax_ts)

    # Background windows.
    cycles = list(range(0, total_minutes + 1, retrain_interval_min))
    for start in cycles:
        train_end = min(start + training_window_visual_min, total_minutes)
        cycle_end = min(start + retrain_interval_min, total_minutes)
        if train_end > start:
            ax_ts.axvspan(start, train_end, color=c_train, alpha=0.34, zorder=0)
            ax_mode.broken_barh(
                [(start, train_end - start)],
                (1.15, 0.70),
                facecolors=c_train,
                edgecolors="#8A6A00",
                linewidth=1.4,
                alpha=0.98,
                zorder=2,
            )
        if cycle_end > train_end:
            ax_ts.axvspan(train_end, cycle_end, color=c_predict_window, alpha=0.20, zorder=0)
            ax_mode.broken_barh(
                [(train_end, cycle_end - train_end)],
                (0.20, 0.70),
                facecolors=c_predict_window,
                edgecolors="#00697A",
                linewidth=1.4,
                alpha=0.98,
                zorder=2,
            )

    # Time-series lines + dots.
    ax_ts.plot(t, actual, color=c_actual, linewidth=2.6, alpha=0.98, zorder=3)
    ax_ts.scatter(
        t,
        actual,
        s=21,
        color=c_actual,
        edgecolor="#111111",
        linewidths=0.45,
        alpha=0.98,
        zorder=4,
        label="Actual data points",
    )
    ax_ts.plot(t, pred, color=c_pred, linewidth=2.4, alpha=0.98, zorder=3)
    ax_ts.scatter(
        t,
        pred,
        s=18,
        color=c_pred,
        edgecolor="#111111",
        linewidths=0.40,
        alpha=0.97,
        zorder=4,
        label="Predicted data points",
    )

    # Vertical guides at retrain boundaries.
    for x in cycles:
        ax_ts.axvline(x=x, color="#555", linestyle="--", linewidth=0.9, alpha=0.35, zorder=1)
        ax_mode.axvline(x=x, color="#555", linestyle="--", linewidth=0.9, alpha=0.35, zorder=1)

    # Labels and annotations.
    ax_ts.set_title(
        "Periodic Training + Continuous Prediction Windows",
        fontsize=26,
        weight="bold",
        color=c_text,
        pad=12,
    )
    ax_ts.set_ylabel("Magnetic Field (nT)", fontsize=18, color=c_text, fontweight="bold")
    ax_ts.grid(True, which="major", color=c_grid, linestyle="-", linewidth=0.9, alpha=0.92)
    ax_ts.tick_params(axis="both", labelsize=14, colors=c_text, width=1.2)

    y_mid = float(np.nanmedian(actual))
    ax_ts.annotate(
        f"Training refresh window (once every T={retrain_interval_min} min)",
        xy=(training_window_visual_min * 0.65, y_mid + 8),
        xytext=(22, y_mid + 19),
        arrowprops=dict(arrowstyle="->", color="#7A5A00", lw=1.5),
        fontsize=16,
        fontweight="bold",
        color="#7A5A00",
        bbox=dict(boxstyle="round,pad=0.25", fc="#FFF4C4", ec="#B88A00", alpha=0.95),
        zorder=5,
    )
    ax_ts.annotate(
        "Prediction-only window\n(reuse latest trained checkpoint)",
        xy=(retrain_interval_min * 0.55, y_mid - 10),
        xytext=(78, y_mid - 22),
        arrowprops=dict(arrowstyle="->", color="#0C5F31", lw=1.5),
        fontsize=14,
        fontweight="bold",
        color="#0C5F31",
        bbox=dict(boxstyle="round,pad=0.25", fc="#D9FBE8", ec="#199D4E", alpha=0.95),
        zorder=5,
    )

    # Mode timeline panel.
    ax_mode.set_ylim(0.0, 2.1)
    ax_mode.set_yticks([0.55, 1.50])
    ax_mode.set_yticklabels(["Predict-only", "Train + Predict"], fontsize=14, color=c_text, fontweight="bold")
    ax_mode.set_xlabel("Elapsed Time (minutes)", fontsize=18, color=c_text, fontweight="bold")
    ax_mode.grid(True, axis="x", color=c_grid, linestyle="-", linewidth=0.9, alpha=0.92)
    ax_mode.tick_params(axis="x", labelsize=14, colors=c_text, width=1.2)
    ax_mode.tick_params(axis="y", length=0, colors=c_text)
    ax_mode.set_xlim(0, total_minutes)
    ax_mode.set_title(
        "Scheduler State Timeline",
        fontsize=17,
        fontweight="bold",
        color=c_text,
        pad=6,
    )

    # Legend.
    legend_items = [
        Line2D([0], [0], color=c_actual, lw=2.2, marker="o", markersize=6, label="Actual time-series"),
        Line2D([0], [0], color=c_pred, lw=2.0, marker="o", markersize=6, label="Predicted data"),
        Patch(facecolor=c_train, edgecolor="#8A6A00", label="Training refresh window"),
        Patch(facecolor=c_predict_window, edgecolor="#00697A", label="Prediction-only window"),
    ]
    ax_ts.legend(
        handles=legend_items,
        loc="lower right",
        bbox_to_anchor=(0.995, 0.02),
        prop={"family": "Arial", "weight": "bold", "size": 14},
        frameon=True,
        framealpha=0.98,
        edgecolor="#111111",
        facecolor="#FFFFFF",
    )

    start_label = datetime(2026, 2, 22, 15, 30, 0)
    fig.text(
        0.01,
        0.01,
        (
            f"Illustration of latest scheduling logic in application_temp.py | "
            f"Example start: {start_label.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Retrain interval T={retrain_interval_min} min"
        ),
        fontsize=12.5,
        fontweight="bold",
        color="#111111",
    )

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate HD training/prediction window figure.")
    parser.add_argument(
        "--out",
        default=os.path.join("models", "anomaly_eval", "training_prediction_windows_hd.png"),
        help="Output PNG path",
    )
    parser.add_argument("--interval", type=int, default=60, help="Retrain interval in minutes (T)")
    parser.add_argument("--duration", type=int, default=240, help="Total timeline duration in minutes")
    parser.add_argument(
        "--train-window-visual",
        type=int,
        default=8,
        help="Visual width (minutes) for each training-refresh window",
    )
    args = parser.parse_args()

    make_figure(
        out_png=args.out,
        retrain_interval_min=max(1, int(args.interval)),
        total_minutes=max(30, int(args.duration)),
        training_window_visual_min=max(1, int(args.train_window_visual)),
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

