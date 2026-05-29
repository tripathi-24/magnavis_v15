#!/usr/bin/env python3
"""
Plot **Keras ``model.fit`` histories** for pretrained LSTM / GRU checkpoints.

Those curves are **not** stored inside ``*.keras`` weights. They are written as JSON when you run
``src/train_lstm_pretrained.py`` or ``src/train_gru_pretrained.py`` (since the training scripts were
updated to dump ``<output_model_dir>/training_histories/<stem>_fit_history.json`` after each fit).

This script reads those JSON files and exports a **high-resolution** static PNG via Plotly + Kaleido.

Install::

  pip install plotly kaleido

Example::

  python tools/plot_pretrained_fit_history_plotly.py \\
    --hist-dir models/training_histories \\
    --sensor OBS2_1 \\
    --out models/pretrained_training_curves_OBS2_1.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_hist(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _epochs_and_series(hist: Dict[str, Any], key: str) -> Tuple[List[int], List[float]]:
    h = hist.get("history") or {}
    series = h.get(key)
    if not series:
        return [], []
    n = len(series)
    return list(range(1, n + 1)), [float(x) for x in series]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--hist-dir",
        type=Path,
        default=None,
        help="Directory containing *_fit_history.json (default: <repo>/models/training_histories).",
    )
    ap.add_argument(
        "--sensor",
        default="OBS2_1",
        help="Canonical sensor tag in filenames, e.g. OBS2_1 (matches lstm_pretrained_OBS2_1_fit_history.json).",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    ap.add_argument("--width", type=int, default=1920, help="Figure width in pixels (default 1920).")
    ap.add_argument("--height", type=int, default=1080, help="Figure height in pixels (default 1080).")
    args = ap.parse_args()

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise SystemExit(
            "plotly is required. Install with:  pip install plotly kaleido\n"
            "(kaleido is needed for PNG export.)"
        ) from e

    project_root = Path(__file__).resolve().parents[1]
    hist_dir = args.hist_dir
    if hist_dir is None:
        hist_dir = project_root / "models" / "training_histories"
    else:
        hist_dir = hist_dir if hist_dir.is_absolute() else (project_root / hist_dir).resolve()

    tag = str(args.sensor).strip()
    lstm_p = hist_dir / f"lstm_pretrained_{tag}_fit_history.json"
    gru_p = hist_dir / f"gru_pretrained_{tag}_fit_history.json"

    missing = [str(p) for p in (lstm_p, gru_p) if not p.is_file()]
    if len(missing) == 2:
        raise SystemExit(
            "No fit-history JSON files found.\n\n"
            f"Expected (after you run or re-run training with the updated train scripts):\n"
            f"  {lstm_p}\n"
            f"  {gru_p}\n\n"
            "The ``*.keras`` checkpoints alone do not record per-epoch loss; the training scripts now "
            "write ``training_histories/*_fit_history.json`` next to the models they save.\n"
            f"Example training:\n"
            f"  python src/train_lstm_pretrained.py your_data.csv models/ --epochs 50 --sensors {tag}\n"
            f"  python src/train_gru_pretrained.py your_data.csv models/ --epochs 50 --sensors {tag}\n"
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("LSTM (pretrained) — training loss", "GRU (pretrained) — training loss"),
        horizontal_spacing=0.08,
    )

    def add_panel(col: int, path: Path, label: str) -> None:
        if not path.is_file():
            fig.add_annotation(
                row=1,
                col=col,
                text=f"Missing file:<br><b>{path.name}</b><br><br>Re-run training with the updated<br>train script to emit JSON here.",
                showarrow=False,
                font=dict(size=14, color="#c0392b"),
                align="center",
            )
            return
        data = _load_hist(path)
        title = data.get("checkpoint_stem", path.stem)
        for metric, dash, color in (
            ("loss", "solid", "#4472C4"),
            ("val_loss", "dash", "#ED7D31"),
        ):
            if metric not in (data.get("history") or {}):
                continue
            xs, ys = _epochs_and_series(data, metric)
            if not xs:
                continue
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=f"{label} {metric}",
                    line=dict(color=color, dash=dash),
                    legendgroup=label,
                    hovertemplate=f"{title}<br>epoch=%{{x}}<br>{metric}=%{{y:.6f}}<extra></extra>",
                ),
                row=1,
                col=col,
            )

    add_panel(1, lstm_p, "LSTM")
    add_panel(2, gru_p, "GRU")

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)

    fig.update_layout(
        title_text=f"Pretrained LSTM vs GRU — fit history ({tag})",
        height=int(args.height),
        width=int(args.width),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=100, b=60),
    )

    out = args.out if args.out.is_absolute() else (project_root / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(out), width=int(args.width), height=int(args.height), scale=1)
    except ValueError as e:
        if "kaleido" in str(e).lower() or "orca" in str(e).lower():
            raise SystemExit(
                "PNG export failed. Install kaleido:  pip install kaleido\n"
                f"Original error: {e}"
            ) from e
        raise
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
