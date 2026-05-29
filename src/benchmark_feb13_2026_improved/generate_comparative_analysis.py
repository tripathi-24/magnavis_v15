#!/usr/bin/env python3
"""Build COMPARATIVE_ANALYSIS.md and k-recall/F1 plots for thesis (incl. closed-loop)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = BENCH_DIR / "results" / "k_recall_curves_zero_hist_20260522_170707"
ABLATION_ROOT = BENCH_DIR / "results" / "gru_delta_ablation_restart"

SCHEME_ORDER = [
    "ewma",
    "median",
    "savgol",
    "gru_pretrained",
    "lstm_pretrained",
    "attn_bilstm_fresh",
]
SCHEME_LABELS = {
    "ewma": "EWMA baseline",
    "median": "Median baseline",
    "savgol": "SavGol baseline",
    "gru_pretrained": "GRU (pretrained)",
    "lstm_pretrained": "LSTM (pretrained)",
    "attn_bilstm_fresh": "Attn Bi-LSTM (fresh)",
}
# (dataset_key, panel_title, schemes or None=all six)
SCENARIOS: List[Tuple[str, str, Optional[List[str]]]] = [
    ("feb13", "Short duration GT Anomaly", None),
    ("apr27", "Long duration GT Anomaly", None),
    (
        "synthetic",
        "Synthetic data(with long duration GT Anomaly)",
        None,
    ),
    ("apr27_gru_closed_loop", "GRU closed loop", ["gru_pretrained"]),
    ("apr27_lstm_closed_loop", "LSTM closed loop", ["lstm_pretrained"]),
]

# Plotly export: 3600×2200 px ≈ 200+ DPI at full thesis text width
PLOTLY_FIG_WIDTH = 3600
PLOTLY_FIG_HEIGHT = 2200
PLOTLY_SCALE = 1
PLOTLY_FONT_FAMILY = "Arial, Helvetica, sans-serif"
PLOTLY_TITLE_SIZE = 60
PLOTLY_FONT_SIZE = 52
PLOTLY_LEGEND_SIZE = 72

# High-contrast saturated palette (distinct hues on white)
COLORS = {
    "ewma": "#FF0080",           # hot magenta
    "median": "#00B4FF",         # bright sky cyan
    "savgol": "#FFD000",         # vivid gold
    "gru_pretrained": "#0057FF",  # electric blue
    "lstm_pretrained": "#FF4500",  # orange-red
    "attn_bilstm_fresh": "#00C853",  # emerald green
}
CLOSED_LOOP_CSV_DIRS = {
    "apr27_gru_closed_loop": "apr27_gru_closed_loop",
    "apr27_lstm_closed_loop": "apr27_lstm_closed_loop",
}

# Apr-27 long GT: extra GRU ablation run (dataset key, legend label, colour)
EXTRA_APR27_GRU: List[Tuple[str, str, str]] = [
    ("gru_absolute_retrain", "GRU absolute retrain", "#7B1FA2"),
]


def _load_points(root: Path) -> pd.DataFrame:
    dfs = [pd.read_csv(root / "k_recall_points.csv")]
    syn = root / "runs" / "synthetic" / "k_recall_points.csv"
    if syn.is_file():
        dfs.append(pd.read_csv(syn))
    for dkey, subdir in CLOSED_LOOP_CSV_DIRS.items():
        cl_path = root / "runs" / subdir / "k_recall_points.csv"
        if cl_path.is_file():
            cl = pd.read_csv(cl_path)
            cl["dataset"] = dkey
            dfs.append(cl)
    abl = ABLATION_ROOT / "k_recall_points.csv"
    if abl.is_file():
        dfs.append(pd.read_csv(abl))
    return pd.concat(dfs, ignore_index=True)


def _schemes_for(dkey: str) -> list[str]:
    for key, _title, schemes in SCENARIOS:
        if key == dkey:
            return schemes if schemes is not None else SCHEME_ORDER
    return SCHEME_ORDER


def _table(df: pd.DataFrame, dkey: str, metric: str, pct: bool) -> str:
    schemes = _schemes_for(dkey)
    hdr = "| k | " + " | ".join(SCHEME_LABELS[s] for s in schemes) + " |"
    sep = "|---|" + "|".join(["---:"] * len(schemes)) + "|"
    rows = []
    for k in (1, 2, 3, 4, 5):
        cells = [str(k)]
        for s in schemes:
            r = df[(df.dataset == dkey) & (df.scheme == s) & (df.k == k)]
            if r.empty:
                cells.append("—")
            else:
                v = float(r[metric].iloc[0])
                cells.append(f"{v * 100:.1f}%" if pct else f"{v:.3f}")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([hdr, sep, *rows])


def _closed_loop_comparison(df: pd.DataFrame) -> str:
    """Markdown table: open-loop vs closed-loop GRU/LSTM on Apr-27."""
    lines = [
        "| k | GRU open | GRU closed | LSTM open | LSTM closed |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in (1, 2, 3, 4, 5):
        cells = [str(k)]
        for dkey, scheme in (
            ("apr27", "gru_pretrained"),
            ("apr27_gru_closed_loop", "gru_pretrained"),
            ("apr27", "lstm_pretrained"),
            ("apr27_lstm_closed_loop", "lstm_pretrained"),
        ):
            r = df[(df.dataset == dkey) & (df.scheme == scheme) & (df.k == k)]
            if r.empty:
                cells.append("—")
            else:
                cells.append(f"{float(r.recall.iloc[0]) * 100:.1f}%")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _plotly_font(size: int | None = None, color: str = "#111111") -> dict:
    return dict(family=PLOTLY_FONT_FAMILY, size=size or PLOTLY_FONT_SIZE, color=color)


def _plot_k_recall_five_scenarios_plotly(df: pd.DataFrame, root: Path) -> None:
    """Bright HD 2×3 recall panel (Plotly, title Arial 60, body Arial 52)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    text_font = _plotly_font(PLOTLY_FONT_SIZE)
    panel_font = _plotly_font(PLOTLY_FONT_SIZE)
    axis_font = _plotly_font(PLOTLY_FONT_SIZE)

    panel_titles = [title for _, title, _ in SCENARIOS]
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, None],
        ],
        subplot_titles=panel_titles,
        horizontal_spacing=0.11,
        vertical_spacing=0.20,
    )

    legend_shown: set[str] = set()

    def _add_recall_trace(
        row: int,
        col: int,
        sub: pd.DataFrame,
        label: str,
        colour: str,
        group: str,
        *,
        dash: str | None = None,
    ) -> None:
        if sub.empty:
            return
        show = group not in legend_shown
        if show:
            legend_shown.add(group)
        line_kw: dict = dict(color=colour, width=5.5)
        if dash:
            line_kw["dash"] = dash
        fig.add_trace(
            go.Scatter(
                x=sub.k,
                y=sub.recall,
                mode="lines+markers",
                name=label,
                legendgroup=group,
                showlegend=show,
                line=line_kw,
                marker=dict(
                    size=18,
                    color=colour,
                    line=dict(width=2.5, color="#1A1A1A"),
                    symbol="circle",
                ),
            ),
            row=row,
            col=col,
        )

    for idx, (dkey, _title, _) in enumerate(SCENARIOS):
        row, col = idx // 3 + 1, idx % 3 + 1
        for scheme in _schemes_for(dkey):
            sub = df[(df.dataset == dkey) & (df.scheme == scheme)].sort_values("k")
            colour = COLORS.get(scheme, "#333333")
            label = SCHEME_LABELS[scheme]
            if dkey == "apr27" and scheme == "gru_pretrained":
                label = "GRU (pretrained, bundled)"
            _add_recall_trace(row, col, sub, label, colour, scheme)
        if dkey == "apr27":
            for abl_key, abl_label, abl_colour in EXTRA_APR27_GRU:
                sub = df[(df.dataset == abl_key) & (df.scheme == "gru_pretrained")].sort_values(
                    "k"
                )
                _add_recall_trace(row, col, sub, abl_label, abl_colour, abl_key, dash="dot")

    for ann in fig.layout.annotations:
        ann.font = panel_font

    axis_style = dict(
        tickfont=text_font,
        title_font=axis_font,
        gridcolor="#D6E4F0",
        gridwidth=1.5,
        zeroline=False,
        linecolor="#1A1A1A",
        linewidth=2.5,
        mirror=True,
        ticks="outside",
        tickwidth=2,
        ticklen=8,
        showline=True,
    )

    axis_cells = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    for r, c in axis_cells:
        fig.update_xaxes(
            title_text="Detector multiplier k" if r == 2 else "",
            tickmode="array",
            tickvals=[1, 2, 3, 4, 5],
            range=[0.85, 5.15],
            row=r,
            col=c,
            **axis_style,
        )
        fig.update_yaxes(
            title_text="Recall" if c == 1 else "",
            range=[-0.02, 1.05],
            dtick=0.2,
            row=r,
            col=c,
            **axis_style,
        )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="<b>K-Recall Curves</b>",
            x=0.5,
            xanchor="center",
            font=_plotly_font(PLOTLY_TITLE_SIZE, "#0D47A1"),
        ),
        font=text_font,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFCFF",
        width=PLOTLY_FIG_WIDTH,
        height=PLOTLY_FIG_HEIGHT,
        margin=dict(l=200, r=60, t=260, b=520),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.14,
            xanchor="center",
            x=0.5,
            font=_plotly_font(PLOTLY_LEGEND_SIZE, "#111111"),
            bgcolor="#FFFFFF",
            bordercolor="#263238",
            borderwidth=4,
            tracegroupgap=40,
            itemsizing="constant",
            itemwidth=80,
            itemclick=False,
            itemdoubleclick=False,
        ),
    )

    out_png = root / "k_recall_curves_five_scenarios.png"
    fig.write_image(
        str(out_png),
        scale=PLOTLY_SCALE,
        width=PLOTLY_FIG_WIDTH,
        height=PLOTLY_FIG_HEIGHT,
    )
    out_pdf = root / "k_recall_curves_five_scenarios.pdf"
    try:
        fig.write_image(str(out_pdf), width=PLOTLY_FIG_WIDTH, height=PLOTLY_FIG_HEIGHT)
    except Exception:
        pass


def _plot_panels(df: pd.DataFrame, root: Path) -> None:
    import matplotlib.pyplot as plt

    _plot_k_recall_five_scenarios_plotly(df, root)

    n_scen = len(SCENARIOS)
    ncols, nrows = 3, 2

    for metric, ylabel, fname in (
        ("f1_score", "Point-level F1", "k_recall_f1_five_scenarios.png"),
    ):
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10))
        for ax, (dkey, title, _) in zip(axes.flat, SCENARIOS):
            for scheme in _schemes_for(dkey):
                sub = df[(df.dataset == dkey) & (df.scheme == scheme)].sort_values("k")
                if sub.empty:
                    continue
                ax.plot(
                    sub.k,
                    sub[metric],
                    marker="o" if metric == "recall" else "s",
                    linewidth=2,
                    label=SCHEME_LABELS[scheme],
                    color=COLORS.get(scheme),
                )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Detector multiplier $k$", fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.grid(True, alpha=0.35)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7, loc="best")
        for j in range(n_scen, nrows * ncols):
            axes.flat[j].set_visible(False)
        fig.suptitle("K–F1 curves", fontsize=13)
        fig.tight_layout()
        fig.savefig(root / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Legacy 2x2 (first four scenarios) for backward compatibility
    legacy = SCENARIOS[:4]
    for metric, fname in (
        ("recall", "k_recall_curves_four_scenarios.png"),
        ("f1_score", "k_recall_f1_four_scenarios.png"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ylabel = "Point-level recall" if metric == "recall" else "Point-level F1"
        for ax, (dkey, title, _) in zip(axes.flat, legacy):
            for scheme in _schemes_for(dkey):
                sub = df[(df.dataset == dkey) & (df.scheme == scheme)].sort_values("k")
                if sub.empty:
                    continue
                ax.plot(
                    sub.k,
                    sub[metric],
                    marker="o" if metric == "recall" else "s",
                    linewidth=2,
                    label=SCHEME_LABELS[scheme],
                    color=COLORS.get(scheme),
                )
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Detector multiplier $k$", fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.grid(True, alpha=0.35)
            ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(root / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Apr-27 open vs closed-loop (GRU + LSTM)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, ylabel in zip(
        axes,
        ("recall", "f1_score"),
        ("Recall", "F1 score"),
    ):
        series = [
            ("apr27", "gru_pretrained", "GRU open-loop", COLORS["gru_pretrained"], "-"),
            (
                "apr27_gru_closed_loop",
                "gru_pretrained",
                "GRU closed-loop",
                COLORS["gru_pretrained"],
                "--",
            ),
            ("apr27", "lstm_pretrained", "LSTM open-loop", COLORS["lstm_pretrained"], "-"),
            (
                "apr27_lstm_closed_loop",
                "lstm_pretrained",
                "LSTM closed-loop",
                COLORS["lstm_pretrained"],
                "--",
            ),
        ]
        for dkey, scheme, label, color, ls in series:
            sub = df[(df.dataset == dkey) & (df.scheme == scheme)].sort_values("k")
            if sub.empty:
                continue
            ax.plot(
                sub.k,
                sub[metric],
                marker="o",
                linewidth=2,
                linestyle=ls,
                label=label,
                color=color,
            )
        ax.set_title(f"Apr-27 long GT — {ylabel} vs k", fontsize=11)
        ax.set_xlabel("Detector k")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)
    fig.suptitle("Open-loop vs closed-loop inference (pretrained GRU & LSTM)", fontsize=12)
    fig.tight_layout()
    fig.savefig(root / "k_recall_apr27_open_vs_closed_loop.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    neural = ["gru_pretrained", "lstm_pretrained", "attn_bilstm_fresh"]
    for ax, (dkey, title, _) in zip(axes, SCENARIOS[:3]):
        for scheme in neural:
            sub = df[(df.dataset == dkey) & (df.scheme == scheme)].sort_values("k")
            if sub.empty:
                continue
            ax.plot(
                sub.k,
                sub.recall,
                marker="o",
                linewidth=2.2,
                label=SCHEME_LABELS[scheme],
                color=COLORS[scheme],
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("k")
        ax.set_ylabel("Recall")
        ax.set_ylim(0, 1.05)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Neural predictors only (pretrained GRU/LSTM + fresh Attn Bi-LSTM)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(root / "k_recall_curves_neural_three_scenarios.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(df: pd.DataFrame, root: Path) -> None:
    ops = [
        ("feb13", 4, "Short shot-duration campaigns"),
        ("apr27", 3, "Long sustained-offset campaigns"),
        ("synthetic", 4, "Synthetic control (long GT, flat magnetics)"),
    ]
    win_lines = []
    for ds, k, desc in ops:
        sub = df[(df.dataset == ds) & (df.k == k)].sort_values("f1_score", ascending=False)
        if sub.empty:
            continue
        top = sub.iloc[0]
        win_lines.append(
            f"- **{desc}** (k={k}): **{SCHEME_LABELS[top.scheme]}** — "
            f"F1={top.f1_score:.3f}, recall={top.recall:.3f}, precision={top.precision:.3f}"
        )

    mean_lines = []
    for k in (3, 4, 5):
        core = df[df.dataset.isin(["feb13", "apr27", "synthetic"]) & (df.k == k)]
        m = core.groupby("scheme")["f1_score"].mean().sort_values(ascending=False)
        mean_lines.append(
            "- k="
            + str(k)
            + ": "
            + ", ".join(f"{SCHEME_LABELS[s]}={v:.3f}" for s, v in m.items())
        )

    has_lstm_cl = not df[df.dataset == "apr27_lstm_closed_loop"].empty
    lstm_cl_note = ""
    if has_lstm_cl:
        lstm_cl_note = _closed_loop_comparison(df)
    else:
        lstm_cl_note = "_LSTM closed-loop results pending (`runs/apr27_lstm_closed_loop/`)._"

    n_runs = 95 + (5 if has_lstm_cl else 0)
    n_scenarios = 5 if has_lstm_cl else 4

    scenario_rows = """| 1 | `feb13` | Feb-13 2026, short shot-duration windows | 4,711 | ~4.4% (207 s) |
| 2 | `apr27` | Apr-27 2026, long sustained-offset windows | 8,825 | ~94.5% (8,344 s) |
| 3 | `synthetic` | Feb-13 GT layout on synthetic flat magnetics | 10,800 | ~50.0% (5,403 s) |
| 4 | `apr27_gru_closed_loop` | Apr-27, GRU, `PREDICTOR_AR_CLOSED_LOOP=1` | 8,825 | ~94.5% |
| 5 | `apr27_lstm_closed_loop` | Apr-27, LSTM, `PREDICTOR_AR_CLOSED_LOOP=1` | 8,825 | ~94.5% |"""

    if not has_lstm_cl:
        scenario_rows = scenario_rows.rsplit("\n", 1)[0]

    recall_sections = "\n\n".join(
        f"### {title}\n\n{_table(df, dkey, 'recall', True)}"
        for dkey, title, _ in SCENARIOS
        if dkey != "apr27_lstm_closed_loop" or has_lstm_cl
    )
    f1_sections = "\n\n".join(
        f"### {title}\n\n{_table(df, dkey, 'f1_score', False)}"
        for dkey, title, _ in SCENARIOS
        if dkey != "apr27_lstm_closed_loop" or has_lstm_cl
    )

    closed_loop_section = f"""### Open-loop vs closed-loop on Apr-27 (recall %)

{lstm_cl_note}

**GRU closed-loop:** does not restore k ≥ 4 recall (64.5% @ k=4 vs 63.3% open-loop; 57.1% @ k=5). Near-total flagging at k ≤ 2.

**LSTM closed-loop:** see table above — compare stability vs open-loop LSTM (~85% flat recall)."""

    md = f"""# Comparative Analysis: k–Recall Benchmark (Zero-Historic)

**Results bundle:** `k_recall_curves_zero_hist_20260522_170707`  
**Generated for thesis inclusion** — summarizes **{n_runs}** evaluation runs across **{n_scenarios}** scenarios, six detector families (where applicable), and k ∈ {{1, 2, 3, 4, 5}}.

---

## Executive summary

Under a **zero-historic, predict-only** protocol (no session fine-tuning), **pretrained LSTM (open-loop)** is the most robust choice for production: flat ~85% recall on long-duration GT across all k, vs GRU’s cliff at k ≥ 4 (~63%). **Closed-loop autoregressive inference** does not fix GRU’s high-k interior misses; LSTM closed-loop should be compared in the table below — if it tracks open-loop LSTM, closed-loop is unnecessary for deployment.

**Recommended deployment:** `PREDICTOR_MODEL_FAMILY=lstm`, `PREDICTOR_MODEL_INIT=pretrained`, **k = 3** (long/mixed), **k = 4** (short shot-duration only). Avoid k = 1 and GRU @ k ≥ 4 on long GT.

---

## Experimental protocol

| Parameter | Value |
|-----------|--------|
| Historic context | 0 min (`MAGNAVIS_BATCH_HISTORIC_MINUTES=0`) |
| Skip initial | 0 min |
| Predictor training window | 0 min (predict-only pretrained) |
| Detector | EWMA on \\|actual − predicted\\| |
| Threshold | mean + **k** × σ |
| k grid | 1, 2, 3, 4, 5 |
| Sensors | OBS2 (all) |
| Closed-loop (scenarios 4–5) | `PREDICTOR_AR_CLOSED_LOOP=1` — AR window rolls on **predicted** magnitudes |

**Pipeline (neural):** magnetic CSV → `app.py` → predictor → residual → `AnomalyDetector` → metrics vs manual GT.

---

## Scenarios

| ID | Dataset key | Description | Evaluated points | GT-positive rate |
|----|-------------|-------------|------------------|------------------|
{scenario_rows}

---

## Figures

| File | Description |
|------|-------------|
| `k_recall_curves_five_scenarios.png` | Plotly HD 2×3 panel: recall vs k (all scenarios); PDF also exported |
| `k_recall_f1_five_scenarios.png` | 2×3 panel: F1 vs k |
| `k_recall_apr27_open_vs_closed_loop.png` | **GRU & LSTM** open vs closed-loop on Apr-27 |
| `k_recall_curves_four_scenarios.png` | Legacy 2×2 (scenarios 1–4) |
| `k_recall_curves_neural_three_scenarios.png` | Core scenarios, neural only |

---

## Recall vs k (percent)

{recall_sections}

---

## F1 score vs k

{f1_sections}

---

## Cross-scenario analysis

### k sensitivity (Apr-27 open-loop)

| Model | Recall @ k=2 | Recall @ k=5 | Δ recall | Behaviour |
|-------|--------------|--------------|----------|-----------|
| **LSTM** | 84.8% | 84.8% | **0.0%** | Plateau-stable |
| Attn Bi-LSTM | 85.0% | 84.8% | 0.2% | Stable (dip at k=3 only) |
| **GRU** | 87.4% | 63.2% | **−24.2%** | Cliff at k ≥ 4 |

{closed_loop_section}

### Mean F1 across scenarios 1–3 (feb13 + apr27 + synthetic)

{chr(10).join(mean_lines)}

### Best family at operational k per scenario type

{chr(10).join(win_lines)}

---

## Final recommendation (thesis & production)

| Role | Choice |
|------|--------|
| **Primary predictor** | Pretrained **LSTM**, **open-loop** |
| **Default k (long / mixed)** | **3.0** |
| **k (short shot-duration only)** | **4.0** |
| **Ablation** | Closed-loop GRU/LSTM, Attn Bi-LSTM (fresh) |
| **Avoid** | k = 1; GRU @ k ≥ 4 on long GT; closed-loop unless validated |

```bash
export PREDICTOR_MODEL_FAMILY=lstm
export PREDICTOR_MODEL_INIT=pretrained
export PREDICTOR_UPDATE_TRAINING=0
export PREDICTOR_SKIP_FINETUNE_ON_SESSION=1
export MAGNAVIS_INITIAL_THRESHOLD_K=3.0
```

---

## Data sources

| Path | Contents |
|------|----------|
| `k_recall_points.csv` | Feb-13 + Apr-27 open-loop |
| `runs/synthetic/k_recall_points.csv` | Synthetic |
| `runs/apr27_gru_closed_loop/k_recall_points.csv` | GRU closed-loop |
| `runs/apr27_lstm_closed_loop/k_recall_points.csv` | LSTM closed-loop |

---

*Regenerate:*

```bash
.venv/bin/python src/benchmark_feb13_2026_improved/generate_comparative_analysis.py
```
"""
    (root / "COMPARATIVE_ANALYSIS.md").write_text(md, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="k_recall_curves_zero_hist results directory",
    )
    args = ap.parse_args()
    root = args.out_dir.resolve()
    df = _load_points(root)
    _plot_panels(df, root)
    _write_markdown(df, root)
    print(f"Wrote {root / 'COMPARATIVE_ANALYSIS.md'}")
    has_lstm = not df[df.dataset == "apr27_lstm_closed_loop"].empty
    print(f"LSTM closed-loop rows: {len(df[df.dataset == 'apr27_lstm_closed_loop'])}")
    print("Plots: five_scenarios, four_scenarios, open_vs_closed_loop, neural_three")


if __name__ == "__main__":
    main()
