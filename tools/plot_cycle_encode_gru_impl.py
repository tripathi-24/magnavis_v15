#!/usr/bin/env python3
"""
Improve cycle-encoding GRU I/O figure based on current implementation.

Implements what predictor_ai.py uses when use_yearly_cycle=True:
  x_t = [mag_scaled, sin_day, cos_day, sin_year, cos_year]
and window input to GRU:
  X shape = (B, 15, 5)
with output:
  y_(t+1) predicted magnetic field.
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch


def _draw_cycle(
    ax,
    cx,
    cy,
    r,
    angle_rad,
    title,
    point_label,
    color_point,
    point_dx=-10.0,
    point_dy=3.2,
    add_phase_labels=False,
):
    # Circle and axes
    ax.add_patch(Circle((cx, cy), r, fill=False, lw=3.0, ec="#2F78C4"))
    ax.plot([cx - r, cx + r], [cy, cy], color="#B4B4B4", lw=1.8)
    ax.plot([cx, cx], [cy - r, cy + r], color="#B4B4B4", lw=1.8)

    # Point and radius
    px = cx + r * math.cos(angle_rad)
    py = cy + r * math.sin(angle_rad)
    ax.plot([cx, px], [cy, py], color="#2F78C4", ls="--", lw=2.0, alpha=0.9)
    ax.scatter([px], [py], s=180, color=color_point, edgecolors="white", linewidths=1.2, zorder=5)

    # Labels
    ax.text(cx, cy + r + 4.3, title, ha="center", va="bottom", fontsize=16.5, fontweight="bold")
    ax.text(cx + r + 1.8, cy - 0.2, "cos", color="#C00000", fontsize=13.5, fontweight="bold")
    ax.text(cx - 0.3, cy + r + 0.8, "sin", color="#C00000", fontsize=13.5, fontweight="bold")
    if add_phase_labels:
        # Angle markers on axes: 0, pi/2, pi, 3pi/2.
        # Keep these inside the circle to avoid overlap with other labels.
        ax.text(cx + r - 2.2, cy + 0.8, "0", color="#C00000", fontsize=13.2, fontweight="bold")
        ax.text(cx + 3.0, cy + r - 2.2, "π/2", color="#C00000", fontsize=13.2, fontweight="bold")
        ax.text(cx - r + 1.0, cy + 0.8, "π", color="#C00000", fontsize=13.2, fontweight="bold")
        ax.text(cx - 2.4, cy - r + 1.1, "3π/2", color="#C00000", fontsize=13.2, fontweight="bold")
    ax.text(px + point_dx, py + point_dy, point_label, fontsize=13.8, fontweight="bold", color="#111111")
    return px, py


def _add_box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#1B7F5C", fs=13):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.8",
        facecolor=fc,
        edgecolor=ec,
        linewidth=2.6,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + 1.8, y + h - 2.8, text, fontsize=fs, fontweight="bold", va="top", color="#111111")
    return box


def make_figure(out_png: str, hour: int = 8, day_of_year: int = 53):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 14,
            "font.weight": "bold",
        }
    )

    day_angle = 2.0 * math.pi * (hour / 24.0)
    year_angle = 2.0 * math.pi * (day_of_year / 365.25)
    sin_day = math.sin(day_angle)
    cos_day = math.cos(day_angle)
    sin_year = math.sin(year_angle)
    cos_year = math.cos(year_angle)

    fig = plt.figure(figsize=(18, 10), dpi=300, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 191)
    ax.set_ylim(0, 95)
    ax.axis("off")

    fig.suptitle(
        "Cyclic Time Encoding -> GRU Inputs and Output (Implementation-Aligned)",
        fontsize=27,
        fontweight="bold",
        y=0.98,
    )

    # Left: two cycle encodings used by predictor_ai.py
    px1, py1 = _draw_cycle(
        ax,
        cx=22,
        cy=63,
        r=16,
        angle_rad=day_angle,
        title="Time-of-day cycle (24h)",
        point_label=f"t = {hour:02d}:00",
        color_point="#FF9F1A",
        add_phase_labels=True,
    )
    px2, py2 = _draw_cycle(
        ax,
        cx=22,
        cy=24,
        r=12,
        angle_rad=year_angle,
        title="Day-of-year cycle",
        point_label=f"DOY = {day_of_year}",
        color_point="#8B5CF6",
        point_dx=0.6,
        point_dy=4.8,
    )

    # Arrow from cycles to feature vector box
    ax.annotate(
        "",
        xy=(62, 56),
        xytext=(px1 + 3.0, py1 - 1.0),
        arrowprops=dict(arrowstyle="->", lw=3.0, color="#F97316"),
    )
    ax.annotate(
        "",
        xy=(62, 44),
        xytext=(px2 + 3.0, py2 + 1.0),
        arrowprops=dict(arrowstyle="->", lw=3.0, color="#8B5CF6"),
    )

    # Feature vector at one timestep
    feature_text = (
        "Sample x_t (one timestep)\n\n"
        "1) mag_scaled\n"
        f"2) sin_day  = {sin_day: .3f}\n"
        f"3) cos_day  = {cos_day: .3f}\n"
        f"4) sin_year = {sin_year: .3f}\n"
        f"5) cos_year = {cos_year: .3f}"
    )
    # Slightly reduce top/bottom extent of green box for cleaner layout.
    _add_box(ax, 63, 36.0, 36, 31.0, feature_text, fc="#ECFDF5", ec="#047857", fs=13.6)

    # Window shape box
    ax.annotate("", xy=(106, 52), xytext=(99.5, 52), arrowprops=dict(arrowstyle="->", lw=2.6, color="#2563EB"))
    _add_box(
        ax,
        105.5,
        43,
        26,
        18,
        "Build sliding\nwindow\n\nX shape = (B, 15, 5)\n(B=batch, W=15)",
        fc="#EFF6FF",
        ec="#1D4ED8",
        fs=13.0,
    )

    # GRU stack box (aligned with current implementation)
    ax.annotate("", xy=(136.0, 52), xytext=(131.8, 52), arrowprops=dict(arrowstyle="->", lw=2.6, color="#2563EB"))
    _add_box(
        ax,
        136.0,
        36,
        31.0,
        33,
        "GRU model\n\nGRU(32,\nreturn_sequences=True)\n-> GRU(16)\n-> Dense(16, relu)\n-> Dense(1)",
        fc="#F5F3FF",
        ec="#6D28D9",
        fs=12.8,
    )

    # Output box
    ax.annotate("", xy=(170.0, 52), xytext=(167.6, 52), arrowprops=dict(arrowstyle="->", lw=2.6, color="#2563EB"))
    _add_box(
        ax,
        170.0,
        42.2,
        18.6,
        17.3,
        "Output\n\ny_(t+1)\n(next\nmagnetic\nvalue)",
        fc="#FEF2F2",
        ec="#DC2626",
        fs=13.4,
    )

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot improved cycle encoding + GRU I/O figure.")
    parser.add_argument(
        "--out",
        default=os.path.join("models", "anomaly_eval", "Cycle_encode (1).png"),
        help="Output PNG path",
    )
    parser.add_argument("--hour", type=int, default=8, help="Example hour for daily cycle marker")
    parser.add_argument("--doy", type=int, default=53, help="Example day-of-year for yearly cycle marker")
    args = parser.parse_args()

    hour = min(max(int(args.hour), 0), 23)
    doy = min(max(int(args.doy), 1), 366)
    make_figure(args.out, hour=hour, day_of_year=doy)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

