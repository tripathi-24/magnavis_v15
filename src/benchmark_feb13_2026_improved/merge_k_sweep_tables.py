#!/usr/bin/env python3
"""Merge comparison_table.csv from multiple k-sweep run folders into one table."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _k_from_detector(det: str) -> float | None:
    m = re.search(r"k\s*=\s*([\d.]+)", str(det), re.I)
    return float(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Pairs label:path/to/comparison_table.csv (e.g. k2:results/foo/comparison_table.csv)",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    frames = []
    for spec in args.inputs:
        if ":" not in spec:
            raise SystemExit(f"Bad --inputs entry (need label:path): {spec!r}")
        label, path = spec.split(":", 1)
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"Missing: {p}")
        df = pd.read_csv(p)
        k_val = label.replace("k", "", 1) if label.lower().startswith("k") else label
        try:
            k_float = float(k_val)
        except ValueError:
            k_float = _k_from_detector(df["detector"].iloc[0]) if "detector" in df.columns else None
        df.insert(0, "k", k_float)
        df.insert(1, "run_label", label)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparison_table_all_k.csv"
    md_path = out_dir / "comparison_table_all_k.md"
    out.to_csv(csv_path, index=False)

    lines = [
        "# Synthetic CSV benchmark — k sweep (2, 3, 4, 5)",
        "",
        "CSV: `magnetic_data_20260213_150000_to_20260213_180000_synthetic_no_anomaly.csv`",
        "",
        "| k | Predictor | TP | FP | TN | FN | n | Recall | Precision | F1 | Specificity | Accuracy |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"| {row['k']:.0f} | {row['predictor']} | {int(row['tp'])} | {int(row['fp'])} | "
            f"{int(row['tn'])} | {int(row['fn'])} | {int(row['evaluated_points'])} | "
            f"{row['recall']:.4f} | {row['precision']:.4f} | {row['f1_score']:.4f} | "
            f"{row['specificity']:.4f} | {row['accuracy']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
