#!/usr/bin/env python3
"""
High-definition Plotly figure: LSTM train/predict windows + internal architecture.

Mirrors the layout of GRU_internal_architecture (1).png and
tools/plot_gru_internal_architecture_similar.py, using the vanilla LSTM stack
from src/predictor_ai.py (AttnBiLSTMPredictor.build_model, MODEL_FAMILY_LSTM).

Output (default):
  models/anomaly_eval/LSTM_internal_architecture_plotly.svg  (Inkscape-editable)
  models/anomaly_eval/LSTM_internal_architecture_plotly.png
  models/anomaly_eval/LSTM_internal_architecture_plotly.pdf
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import plotly.graph_objects as go


def _series_fn(x: np.ndarray) -> np.ndarray:
    return 38.0 + 9.0 * np.sin(x / 4.8) + 2.2 * np.cos(x / 1.9)


def _ax_to_paper(x: float, y: float, *, y_top: bool = True) -> tuple[float, float]:
    """Map top-panel coords (0–100 × 0–80) to Plotly paper [0,1]."""
    px = 0.04 + (x / 100.0) * 0.92
    if y_top:
        py = 0.54 + (y / 80.0) * 0.42
    else:
        py = 0.02 + (y / 44.0) * 0.48
    return px, py


def _add_rect(
    fig: go.Figure,
    x0: float,
    y0: float,
    w: float,
    h: float,
    *,
    fillcolor: str,
    linecolor: str,
    linewidth: float = 2.0,
    y_top: bool = True,
    layer: str = "below",
) -> None:
    x1, y1 = x0 + w, y0 + h
    p0 = _ax_to_paper(x0, y0, y_top=y_top)
    p1 = _ax_to_paper(x1, y1, y_top=y_top)
    fig.add_shape(
        type="rect",
        x0=min(p0[0], p1[0]),
        y0=min(p0[1], p1[1]),
        x1=max(p0[0], p1[0]),
        y1=max(p0[1], p1[1]),
        fillcolor=fillcolor,
        line=dict(color=linecolor, width=linewidth),
        layer=layer,
    )


def _add_text(
    fig: go.Figure,
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    bold: bool = True,
    color: str = "#111111",
    ha: str = "center",
    y_top: bool = True,
) -> None:
    px, py = _ax_to_paper(x, y, y_top=y_top)
    fig.add_annotation(
        x=px,
        y=py,
        text=text.replace("\n", "<br>"),
        showarrow=False,
        xref="paper",
        yref="paper",
        xanchor=ha,
        yanchor="middle",
        font=dict(
            family="Arial, Helvetica, sans-serif",
            size=size,
            color=color,
        ),
    )


def _add_arrow_line(
    fig: go.Figure,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    color: str = "#1F2937",
    width: float = 2.3,
    y_top: bool = True,
    arrow: bool = True,
    export_scale: int = 2,
    fig_width: int = 5100,
    fig_height: int = 3000,
) -> None:
    p0 = _ax_to_paper(x0, y0, y_top=y_top)
    p1 = _ax_to_paper(x1, y1, y_top=y_top)
    if arrow:
        w_px = fig_width * export_scale
        h_px = fig_height * export_scale
        fig.add_annotation(
            x=p1[0],
            y=p1[1],
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=width,
            arrowcolor=color,
            ax=int((p0[0] - p1[0]) * w_px),
            ay=int((p0[1] - p1[1]) * h_px),
            xref="paper",
            yref="paper",
            axref="pixel",
            ayref="pixel",
        )
    else:
        fig.add_shape(
            type="line",
            x0=p0[0],
            y0=p0[1],
            x1=p1[0],
            y1=p1[1],
            line=dict(color=color, width=width),
            layer="above",
        )


def make_figure(out_png: str, out_pdf: str | None = None, window_w: int = 15) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=5100,
        height=3000,
        margin=dict(l=20, r=20, t=120, b=30),
        title=dict(
            text="LSTM Train/Predict Process + Internal Architecture",
            x=0.5,
            font=dict(family="Arial, Helvetica, sans-serif", size=52, color="#111111"),
        ),
    )

    # ---- colours (match GRU reference) ----
    c_train_zone = "#AFCBFF"
    c_pred_zone = "#BDF9CC"
    c_window = "#1F5BB5"
    c_obs = "#4D4D4D"
    c_target = "#FF9F1A"

    # ---- TOP: train / predict windows ----
    _add_rect(fig, 14, 19, 46, 44, fillcolor=c_train_zone, linecolor="#2B64C9", linewidth=2.2)
    _add_rect(fig, 68, 19, 26, 44, fillcolor=c_pred_zone, linecolor="#2E9C53", linewidth=2.2)
    _add_text(fig, 37, 64.9, f"Training: input window W={window_w}", size=36)
    _add_text(fig, 81, 64.9, f"Prediction: input window W={window_w}", size=36)

    xx = np.linspace(5, 98, 500)
    yy = _series_fn(xx)
    px = [_ax_to_paper(x, float(y), y_top=True)[0] for x, y in zip(xx, yy)]
    py = [_ax_to_paper(x, float(y), y_top=True)[1] for x, y in zip(xx, yy)]
    fig.add_trace(
        go.Scatter(
            x=px,
            y=py,
            mode="lines",
            line=dict(color="#333333", width=3),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    tw_x, tw_y, tw_w, tw_h = 18, 24, 20, 34
    pw_x, pw_y, pw_w, pw_h = 69.5, 24, 18, 34
    _add_rect(fig, tw_x, tw_y, tw_w, tw_h, fillcolor="rgba(0,0,0,0)", linecolor=c_window, linewidth=2.2)
    _add_rect(
        fig,
        pw_x,
        pw_y,
        pw_w,
        pw_h,
        fillcolor="#A9EDC2",
        linecolor="#237E52",
        linewidth=2.6,
    )

    x_obs = np.array(
        [16, 20, 24, 28, 32, 35, 38, 42, 45, 48, 52, 55, 58, 69, 72, 74, 76, 78, 80, 82, 85.5]
    )
    y_obs = _series_fn(x_obs)
    ox = [_ax_to_paper(float(x), float(y), y_top=True)[0] for x, y in zip(x_obs, y_obs)]
    oy = [_ax_to_paper(float(x), float(y), y_top=True)[1] for x, y in zip(x_obs, y_obs)]
    fig.add_trace(
        go.Scatter(
            x=ox,
            y=oy,
            mode="markers",
            marker=dict(size=14, color=c_obs),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    x_tgt = tw_x + tw_w
    y_tgt = float(_series_fn(x_tgt))
    x_pred = pw_x + pw_w
    y_pred = float(_series_fn(x_pred))
    tx = [_ax_to_paper(x_tgt, y_tgt, y_top=True)[0], _ax_to_paper(x_pred, y_pred, y_top=True)[0]]
    ty = [_ax_to_paper(x_tgt, y_tgt, y_top=True)[1], _ax_to_paper(x_pred, y_pred, y_top=True)[1]]
    fig.add_trace(
        go.Scatter(
            x=tx,
            y=ty,
            mode="markers",
            marker=dict(size=22, color=c_target),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    _add_text(fig, x_tgt + 2.2, y_tgt - 0.8, "Output: next point", size=31, ha="left", color="navy")
    _add_text(fig, x_pred - 12.5, y_pred + 14.5, "Predicted next point", size=31, ha="center", color="navy")
    _add_arrow_line(
        fig,
        x_pred - 12.5,
        y_pred + 14.5,
        x_pred,
        y_pred,
        color="navy",
        width=2.0,
    )

    _add_text(fig, tw_x + 4, 20, "slide by 1 point", size=29, color="navy")
    _add_text(fig, pw_x + 5, 20, "slide by 1 point", size=29, color="navy")

    _add_text(
        fig,
        14.5,
        11.2,
        f"x_t features: [mag_scaled, sin_day, cos_day, sin_year, cos_year]  ->  X: (B, {window_w}, 5)",
        size=26,
        ha="left",
    )
    _add_text(
        fig,
        14.5,
        7.0,
        "Output: y_(t+1) = next predicted magnetic-field point",
        size=26,
        ha="left",
    )

    # Time / field axis labels
    _add_arrow_line(fig, 8, 75, 93, 75, color="#2E6FB7", width=2.5)
    _add_text(fig, 50.5, 76.2, "Time", size=44)
    _add_text(fig, 1.4, 42, "Magnetic Field (nT)", size=36, ha="center")

    # Legend (top panel)
    leg_y = 8
    _add_rect(fig, 13, leg_y - 2, 3.5, 3, fillcolor=c_train_zone, linecolor="#2B64C9", linewidth=1.5)
    _add_text(fig, 18, leg_y, "Training window", size=24, ha="left", bold=False)
    _add_rect(fig, 33, leg_y - 2, 3.5, 3, fillcolor=c_pred_zone, linecolor="#2E9C53", linewidth=1.5)
    _add_text(fig, 38, leg_y, "Prediction window", size=24, ha="left", bold=False)
    fig.add_trace(
        go.Scatter(
            x=[_ax_to_paper(52, leg_y, y_top=True)[0]],
            y=[_ax_to_paper(52, leg_y, y_top=True)[1]],
            mode="markers",
            marker=dict(size=12, color=c_obs),
            showlegend=False,
        )
    )
    _add_text(fig, 55, leg_y, "Observed/seed point", size=24, ha="left", bold=False)
    fig.add_trace(
        go.Scatter(
            x=[_ax_to_paper(72, leg_y, y_top=True)[0]],
            y=[_ax_to_paper(72, leg_y, y_top=True)[1]],
            mode="markers",
            marker=dict(size=14, color=c_target),
            showlegend=False,
        )
    )
    _add_text(fig, 75, leg_y, "Target/predicted point", size=24, ha="left", bold=False)

    # ---- BOTTOM: LSTM stack (predictor_ai.py default) ----
    _add_text(
        fig,
        50,
        41.6,
        "LSTM Internal Architecture Used in Predictor (app.py)",
        size=42,
        y_top=False,
    )

    boxes = [
        (2.5, 24.5, 14.5, 11.5, "Input X\n(B, 15, 5)\nB=batch\n15=steps, 5=features", "#E7F0FF", "#2B64C9"),
        (
            18.5,
            24.5,
            14.5,
            11.5,
            "LSTM Layer 1\nunits=48\nreturn_sequences=True",
            "#FFF5D6",
            "#AA7A00",
        ),
        (
            34.5,
            24.5,
            14.5,
            11.5,
            "LSTM Layer 2\nunits=32\nreturn_sequences=True",
            "#FFF5D6",
            "#AA7A00",
        ),
        (50.5, 24.5, 13.5, 11.5, "LSTM Layer 3\nunits=16", "#FFF5D6", "#AA7A00"),
        (66.0, 24.5, 11.5, 11.5, "Dense(16)\nReLU", "#F0E9FF", "#6A3FB4"),
        (79.5, 24.5, 14.5, 11.5, "Dense(1)\nOutput y_(t+1)\nmagnetic field", "#E9FCEB", "#208B46"),
    ]
    for x, y, w, h, label, fc, ec in boxes:
        _add_rect(fig, x, y, w, h, fillcolor=fc, linecolor=ec, linewidth=2.0, y_top=False)
        _add_text(fig, x + w / 2, y + h / 2, label, size=22, y_top=False)

    for x0, x1 in [(17.2, 18.5), (33.2, 34.5), (49.2, 50.5), (64.0, 66.0), (77.8, 79.5)]:
        _add_arrow_line(fig, x0, 30.2, x1, 30.2, y_top=False, arrow=True)

    _add_text(
        fig,
        50,
        21.5,
        "Optional Dropout (rate=0.05) between LSTM blocks when PREDICTOR_GRU_DROPOUT>0",
        size=18,
        y_top=False,
        bold=False,
        color="#334155",
    )

    # LSTM gate equations panel
    _add_rect(fig, 6, 3.3, 88, 16.2, fillcolor="#F8FAFC", linecolor="#334155", linewidth=1.8, y_top=False)
    _add_text(
        fig,
        8.0,
        17.0,
        "Inside each LSTM unit at time step t:",
        size=29,
        ha="left",
        y_top=False,
    )
    eq_lines = [
        (16.2, "f_t = sigmoid(W_f x_t + U_f h_(t-1) + b_f)         (forget gate)"),
        (14.0, "i_t = sigmoid(W_i x_t + U_i h_(t-1) + b_i)         (input gate)"),
        (11.8, "C~_t = tanh(W_C x_t + U_C h_(t-1) + b_C)           (candidate cell state)"),
        (9.6, "C_t = f_t * C_(t-1) + i_t * C~_t                 (cell state)"),
        (7.4, "o_t = sigmoid(W_o x_t + U_o h_(t-1) + b_o)         (output gate)"),
        (5.2, "h_t = o_t * tanh(C_t)                             (hidden output)"),
    ]
    for y, line in eq_lines:
        _add_text(fig, 8.0, y, line, size=24, ha="left", y_top=False, bold=True)

    _add_text(
        fig,
        92.8,
        4.2,
        "Top: train/predict windows with sliding input.\nBottom: LSTM stack and gate equations for y_(t+1).",
        size=20,
        ha="right",
        y_top=False,
        bold=True,
        color="#1E293B",
    )

    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LSTM internal architecture (Plotly HD).")
    parser.add_argument(
        "--out",
        default=os.path.join("models", "anomaly_eval", "LSTM_internal_architecture_plotly.svg"),
        help="Output path (.svg recommended for Inkscape; .png/.pdf also written alongside)",
    )
    parser.add_argument("--window", type=int, default=15, help="Sequence window W shown in diagram")
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Skip SVG/PDF export (PNG only)",
    )
    args = parser.parse_args()

    base = os.path.splitext(args.out)[0]
    out_svg = base + ".svg"
    out_png = base + ".png"
    out_pdf = base + ".pdf"
    os.makedirs(os.path.dirname(out_svg) or ".", exist_ok=True)

    fig = make_figure(out_svg, out_pdf, window_w=args.window)
    if not args.png_only:
        fig.write_image(out_svg, width=5100, height=3000)
        print(f"Saved: {out_svg}")
    fig.write_image(out_png, scale=2, width=5100, height=3000)
    print(f"Saved: {out_png}")
    try:
        fig.write_image(out_pdf, width=5100, height=3000)
        print(f"Saved: {out_pdf}")
    except Exception as exc:
        print(f"PDF skipped: {exc}")


if __name__ == "__main__":
    main()
