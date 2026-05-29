#!/usr/bin/env python3
"""
Generate LSTM internal-architecture figure matching GRU_internal_architecture (1).png style.

Uses the same matplotlib layout as plot_gru_internal_architecture_similar.py.
LSTM stack from src/predictor_ai.py (MODEL_FAMILY_LSTM default): 48→32→16 + Dense head.

Outputs: PNG (300 dpi), SVG (Inkscape), PDF.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
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


def make_figure(out_base: str, window_w: int = 15) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 13,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )

    fig = plt.figure(figsize=(17, 10), dpi=300, facecolor="white")
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.35, 1.0], hspace=0.16)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    c_train_zone = "#AFCBFF"
    c_pred_zone = "#BDF9CC"
    c_window = "#1F5BB5"
    c_obs = "#4D4D4D"
    c_target = "#FF9F1A"

    # ---- TOP PANEL (identical structure to GRU figure) ----
    ax_top.set_xlim(0, 100)
    ax_top.set_ylim(0, 80)
    ax_top.axis("off")

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

    ax_top.add_patch(
        Rectangle((14, 19), 46, 44, facecolor=c_train_zone, edgecolor="#2B64C9", linewidth=2.2, alpha=0.95)
    )
    ax_top.add_patch(
        Rectangle((68, 19), 26, 44, facecolor=c_pred_zone, edgecolor="#2E9C53", linewidth=2.2, alpha=0.98)
    )

    ax_top.text(21, 64.9, f"Training: input window W={window_w}", fontsize=18, fontweight="bold")
    ax_top.text(66, 64.9, f"Prediction: input window W={window_w}", fontsize=18, fontweight="bold")

    xx = np.linspace(5, 98, 500)
    yy = _series_fn(xx)
    ax_top.plot(xx, yy, color="#333333", linewidth=2.0, alpha=0.9, zorder=1)

    tw_x, tw_y, tw_w, tw_h = 18, 24, 20, 34
    pw_x, pw_y, pw_w, pw_h = 69.5, 24, 18, 34
    ax_top.add_patch(Rectangle((tw_x, tw_y), tw_w, tw_h, fill=False, edgecolor=c_window, linewidth=2.2))
    ax_top.add_patch(
        Rectangle(
            (pw_x, pw_y),
            pw_w,
            pw_h,
            facecolor="#A9EDC2",
            edgecolor="#237E52",
            linewidth=2.6,
            alpha=0.55,
            zorder=2,
        )
    )
    ax_top.plot([tw_x + 2, tw_x + 2], [tw_y, tw_y + tw_h], color=c_window, ls="--", lw=2.0, alpha=0.8)
    ax_top.plot(
        [tw_x + tw_w + 1.5, tw_x + tw_w + 1.5],
        [tw_y, tw_y + tw_h],
        color=c_window,
        ls="--",
        lw=2.0,
        alpha=0.8,
    )
    ax_top.plot([pw_x + 2, pw_x + 2], [pw_y, pw_y + pw_h], color="#237E52", ls="--", lw=2.0, alpha=0.8)
    ax_top.plot(
        [pw_x + pw_w + 1.5, pw_x + pw_w + 1.5],
        [pw_y, pw_y + pw_h],
        color="#237E52",
        ls="--",
        lw=2.0,
        alpha=0.8,
    )

    ax_top.text(tw_x + 4, 20, "slide by 1 point", color="navy", fontsize=14.5, fontweight="bold")
    ax_top.text(pw_x + 5, 20, "slide by 1 point", color="navy", fontsize=14.5, fontweight="bold")

    x_obs = np.array([16, 20, 24, 28, 32, 35, 38, 42, 45, 48, 52, 55, 58, 69, 72, 74, 76, 78, 80, 82, 85.5])
    y_obs = _series_fn(x_obs)
    ax_top.scatter(x_obs, y_obs, s=65, color=c_obs, edgecolors="none", zorder=4)

    x_tgt = tw_x + tw_w
    y_tgt = float(_series_fn(x_tgt))
    x_pred = pw_x + pw_w
    y_pred = float(_series_fn(x_pred))
    ax_top.scatter([x_tgt, x_pred], [y_tgt, y_pred], s=120, color=c_target, edgecolors="none", zorder=5)
    ax_top.text(x_tgt + 2.2, y_tgt - 0.8, "Output: next point", fontsize=15.5, color="navy", fontweight="bold")
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

    ax_top.text(
        14.5,
        11.2,
        f"x_t features: [mag_scaled, sin_day, cos_day, sin_year, cos_year]  ->  X: (B, {window_w}, 5)",
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

    # ---- BOTTOM PANEL: LSTM stack (predictor_ai.py) ----
    ax_bot.set_xlim(0, 100)
    ax_bot.set_ylim(0, 44)
    ax_bot.axis("off")

    ax_bot.text(
        50,
        41.6,
        "LSTM Internal Architecture Used in Predictor",
        ha="center",
        va="center",
        fontsize=21,
        fontweight="bold",
        color="#111111",
    )

    # Six boxes (narrower than GRU five-box row to fit three LSTM layers)
    pipeline = [
        (2.0, 13.0, "Input X\n(B, 15, 5)\nB=batch\n15=steps, 5=features", "#E7F0FF", "#2B64C9", 11.8),
        (16.2, 12.0, "LSTM Layer 1\nunits=48\nreturn_sequences=True", "#FFF5D6", "#AA7A00", 11.2),
        (29.4, 12.0, "LSTM Layer 2\nunits=32\nreturn_sequences=True", "#FFF5D6", "#AA7A00", 11.2),
        (42.6, 11.0, "LSTM Layer 3\nunits=16", "#FFF5D6", "#AA7A00", 11.2),
        (55.0, 11.0, "Dense(16)\nReLU", "#F0E9FF", "#6A3FB4", 11.8),
        (67.5, 12.5, "Dense(1)\nOutput y_(t+1)\nmagnetic field", "#E9FCEB", "#208B46", 11.2),
    ]
    box_y, box_h = 24.5, 11.5
    for x, w, label, fc, ec, fs in pipeline:
        _draw_labeled_box(ax_bot, x, box_y, w, box_h, label, fc, ec, fs=fs)

    def flow_arrow(x0, x1):
        ax_bot.annotate(
            "",
            xy=(x1, 30.2),
            xytext=(x0, 30.2),
            arrowprops=dict(arrowstyle="->", lw=2.3, color="#1F2937"),
        )

    flow_arrow(15.2, 16.2)
    flow_arrow(28.4, 29.4)
    flow_arrow(41.6, 42.6)
    flow_arrow(53.6, 55.0)
    flow_arrow(66.5, 67.5)

    ax_bot.text(
        50,
        22.8,
        "Optional Dropout (rate=0.05) between LSTM blocks when PREDICTOR_GRU_DROPOUT>0",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#334155",
    )

    eq_box = FancyBboxPatch(
        (6, 2.0),
        88,
        17.8,
        boxstyle="round,pad=0.03,rounding_size=0.8",
        facecolor="#F8FAFC",
        edgecolor="#334155",
        linewidth=1.8,
    )
    ax_bot.add_patch(eq_box)
    ax_bot.text(
        8.0,
        18.2,
        "Inside each LSTM unit at time step t:",
        fontsize=14.6,
        fontweight="bold",
        color="#0F172A",
    )
    eq_y = [14.8, 12.4, 10.0, 7.6, 5.2, 2.8]
    eq_lines = [
        "f_t = sigmoid(W_f x_t + U_f h_(t-1) + b_f)         (forget gate)",
        "i_t = sigmoid(W_i x_t + U_i h_(t-1) + b_i)         (input gate)",
        "C~_t = tanh(W_C x_t + U_C h_(t-1) + b_C)           (candidate cell state)",
        "C_t = f_t * C_(t-1) + i_t * C~_t                 (cell state)",
        "o_t = sigmoid(W_o x_t + U_o h_(t-1) + b_o)         (output gate)",
        "h_t = o_t * tanh(C_t)                             (hidden output)",
    ]
    for y, line in zip(eq_y, eq_lines):
        ax_bot.text(8.0, y, line, fontsize=12.5, color="#0F172A", fontweight="bold")

    ax_bot.text(
        92.8,
        4.2,
        "Top: train/predict windows with sliding input.\nBottom: LSTM stack and gate equations for y_(t+1).",
        ha="right",
        va="bottom",
        fontsize=11.0,
        fontweight="bold",
        color="#1E293B",
    )

    fig.suptitle(
        "LSTM Train/Predict Process + Internal Architecture",
        fontsize=26,
        fontweight="bold",
        y=0.985,
        color="#111111",
    )

    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    try:
        fig.savefig(out_base + ".pdf", bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LSTM architecture figure (GRU-matched matplotlib style)."
    )
    parser.add_argument(
        "--out",
        default=os.path.join("models", "anomaly_eval", "LSTM_internal_architecture_plotly"),
        help="Output path without extension",
    )
    parser.add_argument("--window", type=int, default=15)
    args = parser.parse_args()
    base = os.path.splitext(args.out)[0]
    make_figure(base, window_w=args.window)
    print(f"Saved: {base}.png")
    print(f"Saved: {base}.svg")
    if os.path.isfile(base + ".pdf"):
        print(f"Saved: {base}.pdf")


if __name__ == "__main__":
    main()
