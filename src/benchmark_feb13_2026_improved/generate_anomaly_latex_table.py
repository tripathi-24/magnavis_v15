#!/usr/bin/env python3
"""Generate anomaly_detection_comparative_table.tex from zero-hist k-recall runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = BENCH_DIR / "results" / "k_recall_curves_zero_hist_20260522_170707" / "runs"

MODELS = [
    ("ewma", "EWMA", "ewma"),
    ("median", "Median", "median"),
    ("savgol", "Savitzky--Golay", "savgol"),
    ("attn_bilstm_fresh", "Attention Bi-LSTM", "attn_bilstm_fresh"),
    ("gru_pretrained", "GRU", "gru_pretrained"),
    ("lstm_pretrained", "LSTM", "lstm_pretrained"),
]

# Split across two pages: baselines + Attention Bi-LSTM | GRU + LSTM.
MODEL_SPLIT_INDEX = 4

BASE_SCENARIOS: List[Tuple[str, str]] = [
    ("Shot_Duration_Anomalies", "Short Duration Anomaly"),
    ("Long_Duration_Anomalies", "Long Duration Anomaly"),
    ("synthetic", "Synthetic Data"),
]

CLOSED_LOOP_EXTRA: dict[str, Tuple[str, str]] = {
    "gru_pretrained": ("apr27_gru_closed_loop", "GRU Closed-loop"),
    "lstm_pretrained": ("apr27_lstm_closed_loop", "LSTM Closed-loop"),
}

K_VALUES = [1, 2, 3, 4, 5]
ROWS_PER_SCEN = len(K_VALUES)
NUM_COLS = 12
# Scenario rules: span cols 2–12 so col-1 model \multirow stays visually merged.
SCENARIO_RULE = rf"\cline{{2-{NUM_COLS}}}"

CAPTION_COMMON = (
    r"Point-level anomaly detection across six schemes, three open-loop scenarios "
    r"(short / long / synthetic GT), and GRU/LSTM closed-loop on Apr-27 long GT "
    r"(\texttt{PREDICTOR\_AR\_CLOSED\_LOOP=1}), for $k \in \{1,\ldots,5\}$ "
    r"(zero-historic, predict-only benchmark). "
    r"Short Duration: 4{,}711 s; Long Duration: 8{,}825 s; Synthetic: 10{,}800 s."
)
CAPTION_PARTS = (
    r" Part~I: EWMA, Median, Savitzky--Golay, and Attention Bi-LSTM.",
    r" Part~II: GRU and LSTM (including Apr-27 closed-loop).",
)
LABEL_PARTS = (
    r"tab:anomaly_k_recall_zero_hist_I",
    r"tab:anomaly_k_recall_zero_hist_II",
)

TABULAR_HEADER = [
    r"\hline",
    r"\textbf{Model} & \textbf{Scenario} & $\mathbf{K}$ & \textbf{TP} & \textbf{FP} & "
    r"\textbf{TN} & \textbf{FN} & \textbf{Recall} & \textbf{Precision} & \textbf{F1} & "
    r"\textbf{Specificity} & \textbf{Accuracy} \\",
    r"\hline",
]


def scenarios_for_model(scheme_key: str) -> List[Tuple[str, str]]:
    out = list(BASE_SCENARIOS)
    if scheme_key in CLOSED_LOOP_EXTRA:
        out.append(CLOSED_LOOP_EXTRA[scheme_key])
    return out


def load_metrics(root: Path, run_dir: str, scheme: str, k: int):
    kdir = root / run_dir / f"k{k}" / scheme / "eval"
    matches = list(kdir.glob("*_summary.json")) if kdir.is_dir() else []
    if not matches:
        return None
    pm = json.loads(matches[0].read_text(encoding="utf-8"))["point_level_metrics"]
    return {
        "tp": int(pm["tp"]),
        "fp": int(pm["fp"]),
        "tn": int(pm["tn"]),
        "fn": int(pm["fn"]),
        "recall": float(pm["recall"]),
        "precision": float(pm["precision"]),
        "f1": float(pm["f1_score"]),
        "specificity": float(pm.get("specificity", 0)),
        "accuracy": float(pm.get("accuracy", 0)),
    }


def metrics_cells(m) -> str:
    return (
        f"{m['tp']} & {m['fp']} & {m['tn']} & {m['fn']} & "
        f"{m['recall']:.4f} & {m['precision']:.4f} & {m['f1']:.4f} & "
        f"{m['specificity']:.4f} & {m['accuracy']:.4f}"
    )


def build_tabular_rows(
    root: Path, models: Sequence[Tuple[str, str, str]]
) -> Tuple[List[str], int]:
    """Emit tabular body rows; scenario separators use \\cline{2-12} (not full \\hline)."""
    lines: List[str] = []
    n_cells = 0
    for mi, (scheme_key, model_label, scheme_dir) in enumerate(models):
        scenarios = scenarios_for_model(scheme_key)
        rows_per_model = len(scenarios) * ROWS_PER_SCEN
        if mi > 0:
            lines.append(r"\hline")
        for si, (run_dir, scen_label) in enumerate(scenarios):
            if si > 0:
                lines.append(SCENARIO_RULE)
            for ki, k in enumerate(K_VALUES):
                m = load_metrics(root, run_dir, scheme_dir, k)
                if m is None:
                    raise FileNotFoundError(
                        f"Missing: {model_label} / {scen_label} / k={k}"
                    )
                parts: List[str] = []
                if ki == 0 and si == 0:
                    # One merged cell for the whole model block (cols 2+ use cline only).
                    parts.append(
                        f"\\multirow{{{rows_per_model}}}{{*}}{{{model_label}}}"
                    )
                else:
                    parts.append("")
                if ki == 0:
                    parts.append(f"\\multirow{{{ROWS_PER_SCEN}}}{{*}}{{{scen_label}}}")
                else:
                    parts.append("")
                parts.append(str(k))
                parts.append(metrics_cells(m))
                lines.append(" & ".join(parts) + r" \\")
                n_cells += 1
    return lines, n_cells


def build_table_part(
    root: Path,
    models: Sequence[Tuple[str, str, str]],
    caption_suffix: str,
    label: str,
    placement: str = "t",
) -> Tuple[List[str], int]:
    body, n_cells = build_tabular_rows(root, models)
    lines = [
        rf"\begin{{table*}}[{placement}]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{{CAPTION_COMMON}{caption_suffix}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{|l|l|c|r|r|r|r|c|c|c|c|c|}",
        *TABULAR_HEADER,
        *body,
        r"\hline",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ]
    return lines, n_cells


def build_latex(root: Path, *, split: bool = True) -> str:
    # Preamble (add to thesis): \usepackage{multirow}
    preamble = [
        r"% Requires: \usepackage{multirow}",
        r"% Model column: \multirow spans all scenario×k rows; scenario rules use",
        r"% \cline{2-12} so horizontal lines do not cut through the merged model cell.",
    ]
    if not split:
        lines, n = build_table_part(
            root,
            MODELS,
            (
                r" EWMA, Median, Savitzky--Golay, Attention Bi-LSTM, GRU, and LSTM: "
                r"offline baselines vs.\ Magnavis predictor--residual pipeline."
            ),
            r"tab:anomaly_k_recall_zero_hist",
        )
        return "\n".join(preamble + lines + [f"% {n} data cells"]) + "\n"

    part1_models = MODELS[:MODEL_SPLIT_INDEX]
    part2_models = MODELS[MODEL_SPLIT_INDEX:]
    lines1, n1 = build_table_part(
        root, part1_models, CAPTION_PARTS[0], LABEL_PARTS[0], placement="t"
    )
    lines2, n2 = build_table_part(
        root, part2_models, CAPTION_PARTS[1], LABEL_PARTS[1], placement="p"
    )
    out = preamble + lines1 + [""] + lines2
    out.append(f"% {n1 + n2} data cells ({n1} + {n2})")
    return "\n".join(out) + "\n"


def validate(root: Path) -> int:
    n = 0
    for scheme_key, label, scheme_dir in MODELS:
        for run_dir, scen in scenarios_for_model(scheme_key):
            for k in K_VALUES:
                if load_metrics(root, run_dir, scheme_dir, k) is None:
                    raise FileNotFoundError(f"Missing: {label} / {scen} / k={k}")
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_ROOT)
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_ROOT.parent / "anomaly_detection_comparative_table.tex",
    )
    ap.add_argument(
        "--single-table",
        action="store_true",
        help="Emit one table* (may overflow one page). Default: two-part split.",
    )
    args = ap.parse_args()
    root = args.runs_dir.resolve()
    n = validate(root)
    out = args.out.resolve()
    out.write_text(build_latex(root, split=not args.single_table), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Validated {n} cells")
    print(f"Split: {not args.single_table}")


if __name__ == "__main__":
    main()
