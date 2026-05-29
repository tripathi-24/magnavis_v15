# Feb 13 2026 improved benchmark

This folder mirrors `benchmark_feb13_2026` but runs a **fixed comparison set** with **aligned preprocessing** and a **shared point-level evaluation grid**.

## What runs (default)

1. **EWMA** (alpha 0.35 in `offline_statistical_baselines.py`), **median**, **Savitzky–Golay** — each sensor’s series is trimmed by `--skip-initial-minutes` (default **62**), matching the historic load window.

2. Headless `app.py` for:
   - **LSTM (pretrained)** — loads `lstm_pretrained_*.keras` from `models/`; first **62 min** skipped on the predict path (`PREDICTOR_SKIP_INITIAL_MINUTES`).
   - **Attention Bi-LSTM (fresh)** — trains on the first **62 min** only (`PREDICTOR_LEADING_TRAIN_MINUTES`), **W=15**.
   - **GRU (pretrained)** — loads `gru_pretrained_*.keras`; same **62 min** skip as LSTM pretrained.

Shared settings: **`--k` 4**, **`--historic-minutes` 62**, **`--csv-end`** `2026-02-13 16:18:30`, sensors **OBS2_1–OBS2_3**, default CSV `magnetic_data_20260213_150000_to_20260213_163000.csv`.

## Same *n* for TP, FP, TN, FN

By default the evaluator is called with **`--magnetic-csv`** (same export as `--csv`), **`--magnetic-csv-require-all-obs2`**, **`--magnetic-csv-end`**, and **`--magnetic-csv-skip-initial-minutes`** equal to **`--skip-initial-minutes`**.

So every scheme is scored on the **same per-second set** of timestamps (seconds where all three OBS2 sensors have a row, after the warmup cut). For each row, **`TP + FP + TN + FN = evaluated_points`** (the same integer *n*).

Do **not** pass **`--eval-open-timeline`** if you need that guarantee (open timeline uses GT∪pred span and totals can differ per model).

After a successful multi-model run, the driver checks that **TP+FP+TN+FN** is identical across rows and **exits with an error** if not.

## Run

```bash
cd src/benchmark_feb13_2026_improved
python run_suite.py
```

From repo root:

```bash
python src/benchmark_feb13_2026_improved/run_suite.py \
  --csv magnetic_data_20260213_150000_to_20260213_163000.csv \
  --k 4 --historic-minutes 62 --skip-initial-minutes 62
```

(`run_suite.py` delegates to `run_suite_improved.py`.)

Use **`MAGNAVIS_BENCHMARK_PYTHON`** if your default `python` lacks TensorFlow (required for headless `app.py`).

Subset only:

```bash
python run_suite.py --only ewma,median,lstm_pretrained
```

Offline only (no TensorFlow):

```bash
python run_suite.py --skip-app
```

## Baseline parameter sweep (Apr 27 long GT)

Tune detector **k**, forecast **EWMA α**, **median/SavGol window**, and detector EWMA α:

```bash
cd src/benchmark_feb13_2026_improved
# uses repo .venv/bin/python3 when present (needs matplotlib for eval)
python run_baseline_param_sweep.py
python run_baseline_param_sweep.py --quick --k-values 1,1.5,2
```

Outputs under `results/apr27_baseline_sweep_<timestamp>/`: `sweep_results.csv`, `sweep_best_by_mode.csv`, `SWEEP_REPORT.md`.

### k–recall curves (six families × two datasets)

```bash
python run_k_recall_curves.py
python finalize_k_recall_curves.py --runs-dir results/k_recall_curves_<ts>/runs
```

Outputs: `k_recall_points.csv`, `k_recall_curves_feb13.png`, `k_recall_curves_apr27.png`, `K_RECALL_REPORT.md`.

Offline baselines accept the new flags directly:

```bash
python offline_statistical_baselines.py --mode ewma --csv ... --sensors OBS2_1,OBS2_2,OBS2_3 \
  --k 1.5 --ewma-alpha 0.15 --median-window 15 --savgol-window 31 --detector-alpha 0.995 \
  --skip-initial-minutes 62 --out-log /tmp/ewma.log
```

## After a run

```bash
python generate_perform_matrix.py --results-dir results/<timestamp>
```

Aggregate eval summaries into thesis-style figures:

```bash
python generate_perform_matrix_from_eval.py --results-dir results/<timestamp>
```

Write the same-style table and heatmap **into** that run folder (uses `run_meta.json` for the figure title):

```bash
python generate_perform_matrix_from_eval.py \
  --results-dir results/20260426_063929 \
  --out-dir results/20260426_063929
```

Produces `perform_matrix_table.png`, `perform_matrix_heatmap.png`, and `perform_matrix.csv` there. Omit `--out-dir` to keep writing `*_GRU_4.*` under `results/` only.

## Note

Headless Qt + TensorFlow runs are heavy; use the same environment as the main Feb-13 benchmark. Ensure **`models/lstm_pretrained_OBS2_*.keras`** and **`models/gru_pretrained_OBS2_*.keras`** exist for the three batch sensors.
