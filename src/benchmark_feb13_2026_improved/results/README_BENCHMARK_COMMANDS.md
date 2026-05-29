# Feb 13 improved benchmark — command reference

This folder holds **timestamped runs** (`YYYYMMDD_HHMMSS/`) from `run_suite.py`. Each run contains `logs/`, `eval/`, `comparison_table.csv`, `comparison_table.md`, and `run_meta.json`.

---

## 1. Run the full benchmark

From the **repository root**:

```bash
python src/benchmark_feb13_2026_improved/run_suite.py \
  --csv magnetic_data_20260213_150000_to_20260213_163000.csv \
  --k 4 \
  --historic-minutes 62 \
  --skip-initial-minutes 62 \
  --csv-end "2026-02-13 16:18:30"
```

From **`src/benchmark_feb13_2026_improved/`** (same options, paths relative to here):

```bash
cd src/benchmark_feb13_2026_improved
python run_suite.py
```

**Defaults** (if you omit flags): same Feb‑13 CSV, `k=4`, historic **62** min, skip **62** min per sensor for offline baselines and for the evaluator magnetic grid, CSV end `2026-02-13 16:18:30`, sensors **OBS2_1,OBS2_2,OBS2_3**.

**Interpreter:** set `MAGNAVIS_BENCHMARK_PYTHON` to a Python that has **TensorFlow** (and the rest of the app stack) if `python` does not.

**Offline only** (no TensorFlow / no headless `app.py`):

```bash
python run_suite.py --skip-app
```

**Subset of models** (comma-separated, case-insensitive):

```bash
python run_suite.py --only ewma,median,lstm_pretrained
```

**Useful flags**

| Flag | Role |
|------|------|
| `--no-csv-end` | Do not truncate the magnetic CSV at `MAGNAVIS_BATCH_CSV_END` / default end time. |
| `--eval-open-timeline` | Evaluator does **not** use `--magnetic-csv` grid; point totals **can differ** per model (avoid if you need identical *n*). |
| `--eval-point-grid-magnetic-csv PATH` | Override the CSV used only for the evaluator’s allowed-second grid (default: same as `--csv`). |
| `--train-window-minutes N` | Predictor training window (minutes); `0` = all loaded historic data. |
| `--app-timeout-sec` | Wall clock per headless `app.py` run (default 7200). |

**What runs (default app sequence):** EWMA, median, Savgol (offline), then **LSTM (pretrained)**, **Attention Bi‑LSTM (fresh, W=15)**, **GRU (pretrained)** — see `run_meta.json` → `app_sequence` in each run.

**Same *n* for TP+FP+TN+FN:** With the default magnetic eval grid (do **not** pass `--eval-open-timeline`), every row in `comparison_table.csv` uses the same evaluated seconds; `TP+FP+TN+FN` should match `evaluated_points`. The driver checks this when ≥2 rows complete.

---

## 2. Figures from `comparison_table.csv` (one run folder)

Still under **`src/benchmark_feb13_2026_improved/`**:

```bash
python generate_perform_matrix.py --results-dir results/20260426_080127
```

Writes into that timestamped directory (script default: latest run if you omit `--results-dir` — check script help).

---

## 3. Figures from `eval/*/`*_summary.json* (heatmap + numeric table)

**Default output** (files under this **`results/`** folder with `*_GRU_4` in the name):

```bash
python generate_perform_matrix_from_eval.py
python generate_perform_matrix_from_eval.py --results-dir results/20260426_080127
```

Produces `perform_matrix_GRU_4.csv`, `perform_matrix_heatmap_GRU_4.png`, `perform_matrix_table_GRU_4.png` **here** in `results/` (not inside the timestamp subfolder).

**Write the same-style outputs inside a specific run folder** (title picks up `run_meta.json` when present):

```bash
python generate_perform_matrix_from_eval.py \
  --results-dir results/20260426_080127 \
  --out-dir results/20260426_080127
```

That creates **`perform_matrix.csv`**, **`perform_matrix_heatmap.png`**, **`perform_matrix_table.png`** next to that run’s `eval/` and `logs/`.

---

## 4. Preconditions for headless app steps

- **`models/lstm_pretrained_OBS2_*.keras`** (+ scalers) for LSTM pretrained.  
- **`models/gru_pretrained_OBS2_*.keras`** (+ scalers) for GRU pretrained.  
- Ground-truth / manual schedule for the Feb‑13 CSV is wired in `tools/evaluate_anomaly_detection.py` and `src/app.py` (basename `magnetic_data_20260213_150000_to_20260213_163000.csv`).

---

## 5. More documentation

- **`../README.md`** — overview of the improved benchmark and the shared evaluation grid.  
- **`../run_suite_improved.py`** — full argparse and `APP_SEQUENCE`.  
- Parent benchmark: **`../../benchmark_feb13_2026/README.md`**.
