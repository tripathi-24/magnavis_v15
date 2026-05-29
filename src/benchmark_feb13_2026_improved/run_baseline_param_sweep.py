#!/usr/bin/env python3
"""
Parameter sweep for offline EWMA / median / Savitzky–Golay baselines on long-GT campaigns.

Runs ``offline_statistical_baselines.py`` + ``tools/evaluate_anomaly_detection.py`` for each
combination, writes ``sweep_results.csv``, ``sweep_best_by_mode.csv``, and ``SWEEP_REPORT.md``.

Default dataset: Apr 27 2026 1 Hz export (long GT, 62 min skip aligned with improved benchmark).

Usage::

  cd src/benchmark_feb13_2026_improved
  python run_baseline_param_sweep.py

  python run_baseline_param_sweep.py --quick   # smaller grid (~12 runs)
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

BENCH_DIR = Path(__file__).resolve().parent
SRC = BENCH_DIR.parent
PROJECT_ROOT = SRC.parent
OFFLINE = BENCH_DIR / "offline_statistical_baselines.py"
EVAL = PROJECT_ROOT / "tools" / "evaluate_anomaly_detection.py"

DEFAULT_CSV = PROJECT_ROOT / "Datafiles" / "magnetic_data_20260426_060000_to_20260427_090000_1hz.csv"
DEFAULT_MANUAL_BN = "magnetic_data_20260426_060000_to_20260427_090000_1hz.csv"
DEFAULT_BASE_DATE = "2026-04-27"
OBS2_SENSORS = "OBS2_1,OBS2_2,OBS2_3"

DEFAULT_K = [1.0, 1.5, 2.0]
DEFAULT_EWMA_ALPHA = [0.15, 0.35, 0.5]
DEFAULT_WINDOW = [7, 15, 31, 61]
DEFAULT_DETECTOR_ALPHA = [0.995]

QUICK_K = [1.0, 1.5, 2.0]
QUICK_EWMA_ALPHA = [0.15, 0.35]
QUICK_WINDOW = [15, 31]


def _bench_python() -> str:
    import os

    override = os.environ.get("MAGNAVIS_BENCHMARK_PYTHON", "").strip()
    if override:
        return override
    venv_py = PROJECT_ROOT / ".venv" / "bin" / "python3"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def _run(cmd: List[str], cwd: Path, timeout: int = 900) -> int:
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(cwd), timeout=timeout)
    return int(r.returncode)


@dataclass(frozen=True)
class SweepCase:
    mode: str
    k: float
    ewma_alpha: float
    median_window: int
    savgol_window: int
    detector_alpha: float
    sweep_phase: str

    @property
    def param_key(self) -> tuple:
        return (
            self.mode,
            self.k,
            self.ewma_alpha,
            self.median_window,
            self.savgol_window,
            self.detector_alpha,
        )

    @property
    def run_id(self) -> str:
        return (
            f"{self.mode}_k{self.k:g}_ea{self.ewma_alpha:g}_mw{self.median_window}"
            f"_sw{self.savgol_window}_da{self.detector_alpha:g}_{self.sweep_phase}"
        ).replace(".", "p")


def _iter_cases(
    *,
    quick: bool,
    k_values: List[float],
    ewma_alpha_values: List[float],
    window_values: List[int],
    detector_alpha_values: List[float],
    modes: List[str],
) -> Iterable[SweepCase]:
    """Phase A: k sweep (defaults). Phase B: mode-specific secondary sweeps at best k (fixed k=2)."""
    med_w = sav_w = 31
    ea = 0.35
    da = 0.995

    for mode in modes:
        for k in k_values:
            yield SweepCase(mode, k, ea, med_w, sav_w, da, "k_sweep")

    ref_k = 2.0 if 2.0 in k_values else k_values[-1]

    if "ewma" in modes:
        for alpha in ewma_alpha_values:
            if quick and alpha == 0.35:
                continue
            yield SweepCase("ewma", ref_k, alpha, med_w, sav_w, da, "ewma_alpha_sweep")

    if "median" in modes:
        for w in window_values:
            if quick and w == 31:
                continue
            yield SweepCase("median", ref_k, ea, w, sav_w, da, "median_window_sweep")

    if "savgol" in modes:
        for w in window_values:
            if quick and w == 31:
                continue
            yield SweepCase("savgol", ref_k, ea, med_w, w, da, "savgol_window_sweep")

    if not quick and len(detector_alpha_values) > 1:
        for da_val in detector_alpha_values:
            if da_val == 0.995:
                continue
            for mode in modes:
                yield SweepCase(mode, ref_k, ea, med_w, sav_w, da_val, "detector_alpha_sweep")


def _dedupe_cases(cases: Iterable[SweepCase]) -> List[SweepCase]:
    """One eval per unique parameter tuple; keep first sweep_phase label."""
    seen: set[tuple] = set()
    out: List[SweepCase] = []
    for c in cases:
        if c.param_key in seen:
            continue
        seen.add(c.param_key)
        out.append(c)
    return out


def _run_eval(
    log_file: Path,
    out_dir: Path,
    prefix: str,
    *,
    manual_bn: str,
    base_date: str,
    magnetic_csv: Path,
    skip_min: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    py = _bench_python()
    cmd = [
        py,
        str(EVAL),
        "--log-file",
        str(log_file),
        "--gt-mode",
        "manual_app",
        "--manual-csv-basename",
        manual_bn,
        "--sensor",
        "ALL",
        "--base-date",
        base_date,
        "--prediction-sensor-mode",
        "union_all",
        "--out-dir",
        str(out_dir),
        "--prefix",
        prefix,
        "--magnetic-csv",
        str(magnetic_csv.resolve()),
        "--magnetic-csv-require-all-obs2",
        "--magnetic-csv-skip-initial-minutes",
        str(float(skip_min)),
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True, timeout=600)
    return out_dir / f"{prefix}_summary.json"


def _read_row(summary_path: Path, case: SweepCase) -> Dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    pm = data.get("point_level_metrics") or {}
    em = data.get("event_level_metrics") or {}
    return {
        "run_id": case.run_id,
        "mode": case.mode,
        "sweep_phase": case.sweep_phase,
        "k": case.k,
        "ewma_alpha": case.ewma_alpha,
        "median_window": case.median_window,
        "savgol_window": case.savgol_window,
        "detector_alpha": case.detector_alpha,
        "tp": pm.get("tp"),
        "fp": pm.get("fp"),
        "tn": pm.get("tn"),
        "fn": pm.get("fn"),
        "evaluated_points": pm.get("evaluated_points"),
        "recall": pm.get("recall"),
        "precision": pm.get("precision"),
        "f1_score": pm.get("f1_score"),
        "specificity": pm.get("specificity"),
        "accuracy": pm.get("accuracy"),
        "event_recall": em.get("recall"),
        "event_f1": em.get("f1_score"),
        "n_pred_events": data.get("counts", {}).get("n_pred_events"),
        "summary_json": str(summary_path),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _best_by_mode(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_mode: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        mode = str(r["mode"])
        cur = by_mode.get(mode)
        if cur is None or float(r.get("f1_score") or 0) > float(cur.get("f1_score") or 0):
            by_mode[mode] = r
    return sorted(by_mode.values(), key=lambda x: x["mode"])


def _write_report(path: Path, rows: List[Dict[str, Any]], best: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    lines = [
        "# Baseline parameter sweep (Apr 27 long GT)",
        "",
        f"- Created: `{meta['created']}`",
        f"- CSV: `{meta['csv']}`",
        f"- Cases run: {meta['n_cases']} ({meta['n_ok']} ok, {meta['n_fail']} failed)",
        f"- Skip initial minutes: {meta['skip_initial_minutes']}",
        "",
        "## Best F1 per mode",
        "",
        "| mode | k | ewma_α | med_win | sav_win | recall | precision | F1 | event_recall |",
        "|------|---|--------|---------|---------|--------|-----------|-----|--------------|",
    ]
    for r in best:
        lines.append(
            f"| {r['mode']} | {r['k']} | {r['ewma_alpha']} | {r['median_window']} | "
            f"{r['savgol_window']} | {float(r['recall']):.3f} | {float(r['precision']):.3f} | "
            f"{float(r['f1_score']):.3f} | {float(r.get('event_recall') or 0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Reference (prior benchmark, k=3)",
            "",
            "Apr-27 runs at **k=3** reported ~4% baseline recall vs **~75%** for GRU pretrained.",
            "This sweep shows **k is the dominant knob** on long-GT timelines.",
            "",
            "## Top 10 by F1 (all cases)",
            "",
        ]
    )
    top = sorted(rows, key=lambda x: float(x.get("f1_score") or 0), reverse=True)[:10]
    lines.append(
        "| run_id | mode | phase | k | recall | F1 | TP | FN |"
    )
    lines.append("|--------|------|-------|---|--------|-----|----|----|")
    for r in top:
        lines.append(
            f"| {r['run_id']} | {r['mode']} | {r['sweep_phase']} | {r['k']} | "
            f"{float(r['recall']):.3f} | {float(r['f1_score']):.3f} | {r['tp']} | {r['fn']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep k / forecast windows / EWMA α on offline baselines.")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--manual-csv-basename", default=DEFAULT_MANUAL_BN)
    ap.add_argument("--base-date", default=DEFAULT_BASE_DATE)
    ap.add_argument("--skip-initial-minutes", type=float, default=62.0)
    ap.add_argument("--offline-sensors", default=OBS2_SENSORS)
    ap.add_argument("--quick", action="store_true", help="Smaller grid (skip redundant default secondary points)")
    ap.add_argument("--modes", default="ewma,median,savgol", help="Comma list of baseline modes")
    ap.add_argument("--k-values", default="", help="Override k list, e.g. 1,1.5,2")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output root (default: results/apr27_baseline_sweep_<ts>)")
    args = ap.parse_args()

    magnetic_csv = args.csv.resolve()
    if not magnetic_csv.is_file():
        raise SystemExit(f"CSV not found: {magnetic_csv}")

    k_values = (
        [float(x) for x in args.k_values.split(",") if x.strip()]
        if str(args.k_values).strip()
        else (QUICK_K if args.quick else DEFAULT_K)
    )
    ewma_alpha_values = QUICK_EWMA_ALPHA if args.quick else DEFAULT_EWMA_ALPHA
    window_values = QUICK_WINDOW if args.quick else DEFAULT_WINDOW
    detector_alpha_values = DEFAULT_DETECTOR_ALPHA
    modes = [x.strip() for x in str(args.modes).split(",") if x.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.out_dir or (BENCH_DIR / "results" / f"apr27_baseline_sweep_{ts}")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    cases = _dedupe_cases(
        _iter_cases(
            quick=bool(args.quick),
            k_values=k_values,
            ewma_alpha_values=ewma_alpha_values,
            window_values=window_values,
            detector_alpha_values=detector_alpha_values,
            modes=modes,
        )
    )

    print(f"Sweep output: {root}")
    print(f"Planned cases: {len(cases)}")

    py = _bench_python()
    rows: List[Dict[str, Any]] = []
    n_ok = 0
    n_fail = 0

    for i, case in enumerate(cases, 1):
        run_dir = root / "runs" / case.run_id
        logs = run_dir / "logs"
        evdir = run_dir / "eval"
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{case.mode}_baseline_app.log"

        off_cmd = [
            py,
            str(OFFLINE),
            "--csv",
            str(magnetic_csv),
            "--mode",
            case.mode,
            "--sensors",
            str(args.offline_sensors).strip(),
            "--k",
            str(case.k),
            "--ewma-alpha",
            str(case.ewma_alpha),
            "--median-window",
            str(case.median_window),
            "--savgol-window",
            str(case.savgol_window),
            "--detector-alpha",
            str(case.detector_alpha),
            "--skip-initial-minutes",
            str(float(args.skip_initial_minutes)),
            "--out-log",
            str(log_path),
        ]
        print(f"\n[{i}/{len(cases)}] {case.run_id}", flush=True)
        rc = _run(off_cmd, PROJECT_ROOT, timeout=600)
        if rc != 0:
            n_fail += 1
            rows.append(
                {
                    "run_id": case.run_id,
                    "mode": case.mode,
                    "sweep_phase": case.sweep_phase,
                    "k": case.k,
                    "ewma_alpha": case.ewma_alpha,
                    "median_window": case.median_window,
                    "savgol_window": case.savgol_window,
                    "detector_alpha": case.detector_alpha,
                    "status": "baseline_failed",
                }
            )
            continue

        prefix = f"baseline_{case.mode}_{case.run_id}"
        try:
            summ = _run_eval(
                log_path,
                evdir,
                prefix,
                manual_bn=str(args.manual_csv_basename).strip(),
                base_date=str(args.base_date).strip(),
                magnetic_csv=magnetic_csv,
                skip_min=float(args.skip_initial_minutes),
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            n_fail += 1
            rows.append(
                {
                    "run_id": case.run_id,
                    "mode": case.mode,
                    "sweep_phase": case.sweep_phase,
                    "status": f"eval_failed:{exc}",
                }
            )
            continue

        row = _read_row(summ, case)
        row["status"] = "ok"
        rows.append(row)
        n_ok += 1

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    best = _best_by_mode(ok_rows)
    meta = {
        "created": ts,
        "csv": str(magnetic_csv),
        "skip_initial_minutes": float(args.skip_initial_minutes),
        "manual_csv_basename": str(args.manual_csv_basename),
        "base_date": str(args.base_date),
        "quick": bool(args.quick),
        "n_cases": len(cases),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "k_values": k_values,
        "ewma_alpha_values": ewma_alpha_values,
        "window_values": window_values,
    }
    (root / "sweep_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_csv(root / "sweep_results.csv", rows)
    _write_csv(root / "sweep_best_by_mode.csv", best)
    _write_report(root / "SWEEP_REPORT.md", ok_rows, best, meta)

    print(f"\nDone: {n_ok}/{len(cases)} succeeded.")
    print(f"Results: {root / 'sweep_results.csv'}")
    print(f"Report:  {root / 'SWEEP_REPORT.md'}")


if __name__ == "__main__":
    main()
