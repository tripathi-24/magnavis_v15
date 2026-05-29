# Feb 13 2026 model benchmark (under `src/`)

Harness for the Feb 13 magnetic CSV (`magnetic_data_20260213_150000_to_20260213_163000.csv` at the **repository root**) using the same manual GT intervals as `tools/evaluate_anomaly_detection.py`.

## Scripts

| File | Role |
|------|------|
| `run_suite.py` | Offline baselines (EWMA → median → Savitzky–Golay on **OBS2_1–3** merged logs), then headless `app.py` in order: **Vanilla LSTM (fresh, W=15)** → Attention–BiLSTM (fresh) → Transformer (pretrained) → GRU (fresh, W=15) → GRU (pretrained). Writes `results/<timestamp>/comparison_table.csv` (+ `.md`). |
| `offline_statistical_baselines.py` | Builds synthetic `app.log` lines via `Anomaly_detector.py`; `--sensors` merges per-sensor anomalies. |
| `generate_perform_matrix.py` | After a run, builds `perform_matrix_heatmap.png`, `perform_matrix_table.png`, and `perform_matrix.csv` from `comparison_table.csv`. |

## Default settings (Feb 13 CSV)

- **k** = 4 (`MAGNAVIS_INITIAL_THRESHOLD_K`)
- **Historic load** = 62 minutes (`MAGNAVIS_BATCH_HISTORIC_MINUTES`)
- **Predictor training window** = 0 minutes → train on **all** loaded data (`MAGNAVIS_INITIAL_TRAIN_WINDOW_MINUTES`; same meaning as the app spinbox)
- **Sensors** = `OBS2_1,OBS2_2,OBS2_3` for both offline baselines and headless app (`MAGNAVIS_BATCH_SENSORS`)
- **CSV time cap** = `2026-02-13 16:18:30` inclusive on the magnetic `timestamp` column (`MAGNAVIS_BATCH_CSV_END` / `--csv-end`). Use `--no-csv-end` for the full file.
- **Evaluation** = `--sensor ALL` with `--prediction-sensor-mode union_all` (a second is positive if **any** OBS2 sensor logs an anomaly)
- **Vanilla LSTM** is a stacked unidirectional LSTM in `predictor_ai.py` (`PREDICTOR_MODEL_FAMILY=lstm`), distinct from the **Attention Bi-LSTM** graph (`attn_bilstm`).

Override with flags: `--k`, `--historic-minutes`, `--train-window-minutes`, `--batch-sensors`, `--offline-sensors`, etc.

## Run

Use **whatever Python environment already works for `src/app.py`** (conda, venv, etc.). Subprocesses use the same interpreter as `run_suite.py` (`sys.executable`), unless you set `MAGNAVIS_BENCHMARK_PYTHON`.

Use your real clone path, not a placeholder. From **inside the repository**:

**If you are already in `src/`** (your prompt shows `… magnavis_v13_Polish/src %`):

```bash
cd benchmark_feb13_2026
python run_suite.py
```

**If you are at the repository root** (`magnavis_v13_Polish/`):

```bash
cd src/benchmark_feb13_2026
python run_suite.py
```

**Without changing directory** (from repository root):

```bash
python src/benchmark_feb13_2026/run_suite.py
```

**Performance matrix** (matplotlib; reads `comparison_table.csv` from a run folder):

```bash
cd src/benchmark_feb13_2026
python generate_perform_matrix.py
# or: python generate_perform_matrix.py --results-dir results/20260422_104826
```

Options: `--skip-app`, `--only ewma,lstm,pretrained_keras_forecaster,…`, `--prediction-sensor-mode union_all|filter`. Headless servers may need `export QT_QPA_PLATFORM=offscreen`.

Prior benchmark outputs live under `results/` (gitignored). Delete that folder anytime to start clean; the suite creates a new timestamped subfolder on each run.

## Dependencies

Same stack as the main app (TensorFlow / Keras for DL runs; **scipy** for Savitzky–Golay). If imports fail, install from repo root: `pip install -r requirements_feb_2025.txt` into the env you use to run the commands above.

**Cornish–Fisher** is not implemented here; the table uses the app’s standard k·σ detector only.
