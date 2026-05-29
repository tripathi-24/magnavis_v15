#!/usr/bin/env python3
"""Merge cached benchmarks + partial run summaries, plot k–recall curves."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from run_k_recall_curves import (  # noqa: E402
    DATASETS,
    DEFAULT_K_VALUES,
    RESULTS_DIR,
    SCHEME_LABELS,
    SCHEME_ORDER,
    _load_cache_from_results,
    _plot_curves,
    _write_csv_points,
    _write_report,
)

RUNS_RE = re.compile(
    r"/runs/(?P<ds>feb13|apr27|synthetic)/k(?P<k>[\d.]+)/(?P<scheme>[\w]+)/eval/(?P<prefix>.+)_summary\.json$"
)


def _load_partial_runs(runs_root: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if not runs_root.is_dir():
        return out
    for summ in runs_root.rglob("*_summary.json"):
        m = RUNS_RE.search(str(summ))
        if not m:
            continue
        ds = m.group("ds")
        k = float(m.group("k"))
        scheme = m.group("scheme")
        data = json.loads(summ.read_text(encoding="utf-8"))
        pm = data.get("point_level_metrics") or {}
        k_str = f"{k:g}"
        out[(ds, scheme, k_str)] = {
            "dataset": ds,
            "scheme": scheme,
            "k": k,
            "recall": float(pm.get("recall", 0)),
            "precision": float(pm.get("precision", 0)),
            "f1_score": float(pm.get("f1_score", 0)),
            "tp": int(pm.get("tp", 0)),
            "fp": int(pm.get("fp", 0)),
            "tn": int(pm.get("tn", 0)),
            "fn": int(pm.get("fn", 0)),
            "evaluated_points": int(pm.get("evaluated_points", 0)),
            "source": str(summ),
            "status": "ok",
        }
    return out


def merge_points(
    cache: Dict[Tuple[str, str, str], Dict[str, Any]],
    partial: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[str, str, str], Dict[str, Any]] = dict(cache)
    for key, pt in partial.items():
        merged[key] = {**pt, "status": pt.get("status", "ok")}
    return list(merged.values())


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=BENCH_DIR / "results" / "k_recall_curves_20260522_114418" / "runs",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=BENCH_DIR / "results" / "k_recall_curves_merged",
    )
    args = ap.parse_args()

    cache = _load_cache_from_results(RESULTS_DIR)
    partial = _load_partial_runs(args.runs_dir.resolve())
    pts = merge_points(cache, partial)

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_csv_points(out / "k_recall_points.csv", pts)

    meta = {
        "created": "merged",
        "k_values": DEFAULT_K_VALUES,
        "datasets": ["feb13", "apr27"],
        "n_cache": len(cache),
        "n_partial": len(partial),
        "n_merged": len(pts),
    }
    (out / "k_recall_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for dkey in ("feb13", "apr27"):
        sub = [p for p in pts if p.get("dataset") == dkey]
        if sub:
            _plot_curves(sub, out / f"k_recall_curves_{dkey}.png", DATASETS[dkey].label)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        for ax, dkey in zip(axes, ("feb13", "apr27")):
            for scheme in SCHEME_ORDER:
                sub = sorted(
                    [p for p in pts if p.get("dataset") == dkey and p.get("scheme") == scheme],
                    key=lambda x: float(x["k"]),
                )
                if not sub:
                    continue
                ax.plot(
                    [float(p["k"]) for p in sub],
                    [float(p["recall"]) for p in sub],
                    marker="o",
                    linewidth=2,
                    label=SCHEME_LABELS.get(scheme, scheme),
                )
            ax.set_title(DATASETS[dkey].label)
            ax.set_xlabel("k")
            ax.set_ylabel("Recall")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.35)
            ax.legend(fontsize=7)
        fig.suptitle("k–recall curves (six families)")
        fig.tight_layout()
        fig.savefig(out / "k_recall_curves_combined.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    _write_report(out / "K_RECALL_REPORT.md", pts, meta)

    # Coverage table
    lines = ["# Coverage", ""]
    for dkey in ("feb13", "apr27"):
        lines.append(f"## {dkey}")
        for scheme in SCHEME_ORDER:
            have = {f"{float(p['k']):g}" for p in pts if p["dataset"] == dkey and p["scheme"] == scheme}
            missing = [f"{k:g}" for k in DEFAULT_K_VALUES if f"{k:g}" not in have]
            lines.append(f"- **{scheme}**: {len(have)}/9 k values" + (f" (missing: {', '.join(missing)})" if missing else " ✓"))
        lines.append("")
    (out / "COVERAGE.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Merged {len(pts)} points -> {out}")
    print(f"  cache={len(cache)} partial={len(partial)}")


if __name__ == "__main__":
    main()
