#!/usr/bin/env python3
"""
Generate a Train_Predict_Process-like figure for GRU internals.

Output: models/anomaly_eval/GRU_internal_architecture_similar.png
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
from matplotlib.lines import Line2D


def _series_fn(x):
    return 38.0 + 9.0 * np.sin(x / 4.8) + 2.2 * np.cos(x / 1.9)


def _draw_labeled_box(ax, x, y, w, h, text, fc, ec, fs=11, lw=2.0, bold=True):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.8",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold" if bold else "normal",
        color="#111111",
        zorder=3,
    )


def make_figure(out_png: str):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 13,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    fig = plt.figure(figsize=(17, 10), dpi=300, facecolor="white")
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.35, 1.0], hspace=0.16)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    # ----------------------------
    # TOP PANEL: Train/Predict window style
    # ----------------------------
    ax_top.set_xlim(0, 100)
    ax_top.set_ylim(0, 80)
    ax_top.axis("off")

    # Time arrow and Y-axis arrow
    ax_top.annotate(
        "",
        xy=(93, 75),
        xytext=(8, 75),
        arrowprops=dict(arrowstyle="->", lw=2.2, color="#2E6FB7"),
    )
    ax_top.text(50.5, 76.2, "Time", ha="center", va="bottom", fontsize=22, fontweight="bold")
    ax_top.annotate(
        "",
        xy=(3.8, 66),
        xytext=(3.8, 18),
        arrowprops=dict(arrowstyle="<->", lw=2.2, color="#2E6FB7"),
    )
    ax_top.text(
        1.4,
        42,
        "Magnetic Field (nT)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    # Background zones
    c_train_zone = "#AFCBFF"
    c_pred_zone = "#BDF9CC"
    c_window = "#1F5BB5"
    c_obs = "#4D4D4D"
    c_target = "#FF9F1A"

    train_rect = Rectangle((14, 19), 46, 44, facecolor=c_train_zone, edgecolor="#2B64C9", linewidth=2.2, alpha=0.95)
    pred_rect = Rectangle((68, 19), 26, 44, facecolor=c_pred_zone, edgecolor="#2E9C53", linewidth=2.2, alpha=0.98)
    ax_top.add_patch(train_rect)
    ax_top.add_patch(pred_rect)

    ax_top.text(21, 64.9, "Training: input window W=15", fontsize=18, fontweight="bold")
    ax_top.text(66, 64.9, "Prediction: input window W=15", fontsize=18, fontweight="bold")

    # Synthetic magnetic-field curve
    xx = np.linspace(5, 98, 500)
    yy = _series_fn(xx)
    ax_top.plot(xx, yy, color="#333333", linewidth=2.0, alpha=0.9, zorder=1)

    # Sliding window boxes
    tw_x, tw_y, tw_w, tw_h = 18, 24, 20, 34
    pw_x, pw_y, pw_w, pw_h = 69.5, 24, 18, 34
    ax_top.add_patch(Rectangle((tw_x, tw_y), tw_w, tw_h, fill=False, edgecolor=c_window, linewidth=2.2))
    # First prediction window shown with light opaque fill for clarity.
    ax_top.add_patch(
        Rectangle((pw_x, pw_y), pw_w, pw_h, facecolor="#A9EDC2", edgecolor="#237E52", linewidth=2.6, alpha=0.55, zorder=2)
    )
    ax_top.plot([tw_x + 2, tw_x + 2], [tw_y, tw_y + tw_h], color=c_window, ls="--", lw=2.0, alpha=0.8)
    ax_top.plot([tw_x + tw_w + 1.5, tw_x + tw_w + 1.5], [tw_y, tw_y + tw_h], color=c_window, ls="--", lw=2.0, alpha=0.8)
    ax_top.plot([pw_x + 2, pw_x + 2], [pw_y, pw_y + pw_h], color="#237E52", ls="--", lw=2.0, alpha=0.8)
    ax_top.plot([pw_x + pw_w + 1.5, pw_x + pw_w + 1.5], [pw_y, pw_y + pw_h], color="#237E52", ls="--", lw=2.0, alpha=0.8)

    ax_top.text(tw_x + 4, 20, "slide by 1 point", color="navy", fontsize=14.5, fontweight="bold")
    ax_top.text(pw_x + 5, 20, "slide by 1 point", color="navy", fontsize=14.5, fontweight="bold")

    # Observed/seed points
    x_obs = np.array([16, 20, 24, 28, 32, 35, 38, 42, 45, 48, 52, 55, 58, 69, 72, 74, 76, 78, 80, 82, 85.5])
    y_obs = _series_fn(x_obs)
    ax_top.scatter(x_obs, y_obs, s=65, color=c_obs, edgecolors="none", zorder=4)

    # Target/predicted points
    x_tgt = tw_x + tw_w
    y_tgt = float(_series_fn(x_tgt))
    # Predicted point is shown at the right edge of the first prediction window.
    x_pred = pw_x + pw_w
    y_pred = float(_series_fn(x_pred))
    ax_top.scatter([x_tgt, x_pred], [y_tgt, y_pred], s=120, color=c_target, edgecolors="none", zorder=5)
    ax_top.text(x_tgt + 2.2, y_tgt - 0.8, "Output: next point", fontsize=15.5, color="navy", fontweight="bold")
    # Move label farther up and left into a clear area (with arrow to point).
    ax_top.annotate(
        "Predicted next point",
        xy=(x_pred, y_pred),
        xytext=(x_pred - 12.5, y_pred + 14.5),
        arrowprops=dict(arrowstyle="->", color="navy", lw=1.8),
        fontsize=15.5,
        color="navy",
        fontweight="bold",
        zorder=6,
    )

    # Concise inputs/outputs text (details moved to documentation)
    ax_top.text(
        14.5,
        11.2,
        "x_t features: [mag_scaled, sin_day, cos_day, sin_year, cos_year]  ->  X: (B, 15, 5)",
        fontsize=12.8,
        color="#0F172A",
        fontweight="bold",
    )
    ax_top.text(
        14.5,
        7.0,
        "Output: y_(t+1) = next predicted magnetic-field point",
        fontsize=12.8,
        color="#0F172A",
        fontweight="bold",
    )

    legend_handles = [
        Patch(facecolor=c_train_zone, edgecolor="#2B64C9", label="Training window"),
        Patch(facecolor=c_pred_zone, edgecolor="#2E9C53", label="Prediction window"),
        Line2D([0], [0], marker="o", linestyle="None", color=c_obs, markersize=8, label="Observed/seed point"),
        Line2D([0], [0], marker="o", linestyle="None", color=c_target, markersize=9, label="Target/predicted point"),
        Patch(facecolor="white", edgecolor=c_window, label="Sliding window box"),
    ]
    ax_top.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.13, -0.08),
        ncol=3,
        frameon=False,
        prop={"family": "Arial", "weight": "bold", "size": 13.8},
        handlelength=1.6,
        handletextpad=0.5,
    )

    # ----------------------------
    # BOTTOM PANEL: GRU internals + stack used in code
    # ----------------------------
    ax_bot.set_xlim(0, 100)
    ax_bot.set_ylim(0, 44)
    ax_bot.axis("off")

    ax_bot.text(
        50,
        41.6,
        "GRU Internal Architecture Used in Predictor",
        ha="center",
        va="center",
        fontsize=21,
        fontweight="bold",
        color="#111111",
    )

    # Pipeline boxes
    _draw_labeled_box(
        ax_bot,
        4,
        24.5,
        18,
        11.5,
        "Input X\n(B, 15, 5)\nB=batch\n15=steps, 5=features",
        fc="#E7F0FF",
        ec="#2B64C9",
        fs=12.6,
    )
    _draw_labeled_box(
        ax_bot,
        26,
        24.5,
        16,
        11.5,
        "GRU Layer 1\nunits=32\nreturn_sequences=True",
        fc="#FFF5D6",
        ec="#AA7A00",
        fs=12.4,
    )
    _draw_labeled_box(
        ax_bot,
        45.5,
        24.5,
        13.8,
        11.5,
        "GRU Layer 2\nunits=16",
        fc="#FFF5D6",
        ec="#AA7A00",
        fs=12.4,
    )
    _draw_labeled_box(
        ax_bot,
        62.3,
        24.5,
        12.6,
        11.5,
        "Dense(16)\nReLU",
        fc="#F0E9FF",
        ec="#6A3FB4",
        fs=12.6,
    )
    _draw_labeled_box(
        ax_bot,
        78.1,
        24.5,
        15.5,
        11.5,
        "Dense(1)\nOutput y_(t+1)\nmagnetic field",
        fc="#E9FCEB",
        ec="#208B46",
        fs=12.4,
    )

    # Flow arrows
    def flow_arrow(x0, x1):
        ax_bot.annotate(
            "",
            xy=(x1, 30.2),
            xytext=(x0, 30.2),
            arrowprops=dict(arrowstyle="->", lw=2.3, color="#1F2937"),
        )

    flow_arrow(22.2, 26)
    flow_arrow(42.2, 45.5)
    flow_arrow(59.4, 62.3)
    flow_arrow(75.0, 78.1)

    # GRU cell equations panel
    eq_box = FancyBboxPatch(
        (6, 3.3),
        88,
        16.2,
        boxstyle="round,pad=0.03,rounding_size=0.8",
        facecolor="#F8FAFC",
        edgecolor="#334155",
        linewidth=1.8,
    )
    ax_bot.add_patch(eq_box)
    ax_bot.text(8.0, 17.0, "Inside each GRU unit at time step t:", fontsize=14.6, fontweight="bold", color="#0F172A")
    ax_bot.text(8.0, 13.0, "r_t = sigmoid(W_r x_t + U_r h_(t-1) + b_r)         (reset gate)", fontsize=13.0, color="#0F172A", fontweight="bold")
    ax_bot.text(8.0, 9.8, "z_t = sigmoid(W_z x_t + U_z h_(t-1) + b_z)         (update gate)", fontsize=13.0, color="#0F172A", fontweight="bold")
    ax_bot.text(8.0, 6.6, "h~_t = tanh(W_h x_t + U_h (r_t * h_(t-1)) + b_h)    (candidate state)", fontsize=13.0, color="#0F172A", fontweight="bold")
    ax_bot.text(8.0, 3.9, "h_t = (1 - z_t) * h_(t-1) + z_t * h~_t               (new hidden output)", fontsize=13.0, color="#0F172A", fontweight="bold")

    # Short overall description at bottom-right of the lowest box.
    ax_bot.text(
        92.8,
        4.2,
        "Top: train/predict windows with sliding input.\nBottom: GRU stack and gate equations for y_(t+1).",
        ha="right",
        va="bottom",
        fontsize=11.0,
        fontweight="bold",
        color="#1E293B",
    )

    # Figure title
    fig.suptitle(
        "GRU Train/Predict Process + Internal Architecture",
        fontsize=26,
        fontweight="bold",
        y=0.985,
        color="#111111",
    )

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate GRU internal architecture figure in Train_Predict style.")
    parser.add_argument(
        "--out",
        default=os.path.join("models", "anomaly_eval", "GRU_internal_architecture_similar.png"),
        help="Output PNG path",
    )
    args = parser.parse_args()

    make_figure(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

