# Script Doc: `src/app.py` (primary DB/CSV runtime)

## Purpose

`app.py` is the primary Magnavis runtime for DB/CSV-driven operation (class `ApplicationTemp`).  
`application_temp.py` is a small compatibility launcher that executes `app.py`.  
It orchestrates:
- multi-sensor ingestion
- predictor execution (pre-trained GRU, fresh GRU, LSTM, Transformer, or Attention-BiLSTM)
- adaptive anomaly detection

## How to run

```bash
python src/app.py
# equivalent:
python src/application_temp.py
```

## Where it sits in the pipeline

1. Reads historical and incremental data from DB or CSV.
2. Writes per-sensor predictor input files.
3. Launches predictor subprocess for each active sensor.
4. Reads predictor outputs and computes residual anomalies.
5. Updates time-series charts and logs.

## Inputs

### User/runtime inputs

- Mode: real-time DB, simulation DB, or CSV playback
- Initial history window (minutes)
- Active sensor list (multi-sensor selection)
- Runtime controls: threshold multiplier **k**, predictor training window (minutes), retrain interval
- Predictor mode (startup dialogs + env defaults):
  - **family:** `gru`, `lstm`, `transformer`, or `attn_bilstm` (`PREDICTOR_MODEL_FAMILY`)
  - **init:** `pretrained` or `fresh` (`PREDICTOR_MODEL_INIT`; fresh GRU also prompts for sequence window **W**)
- Magnetic trace **smoothing:** first-order low-pass (EMA) on plotted / detector-facing series; strength controlled by `MAGNAVIS_LOWPASS_ALPHA` (see below)

### Data dependencies

- DB path: `src/data_convert_db_now.py`
- CSV path: loaded file with time-synchronized magnetic channels (`sensor_id`, `timestamp`, `b_x`, `b_y`, `b_z`)

## Outputs and artifacts

- UI time-series charts
- Session files under `src/sessions/<session_id>/<sensor_id>/`:
  - `predict_input.csv`
  - `predict_out.csv`
  - `predict_stdout.log`
  - `predict_stderr.log`
- App-level logs and status messages

## Step-by-step functionality

### 1) Startup configuration

- Closes the splash screen before modal prompts so dialogs are visible.
- Prompts user for predictor family and initialization mode (and fresh GRU **W** presets when applicable).
- Prompts for historical window, optional **k** (threshold multiplier), and predictor training window.
- Resolves sensor set from DB schema or CSV columns.

### 2) Data loading and update loop

- Fetches initial history (blue segment).
- Polls/advances incremental data (green segment).
- Maintains per-sensor buffers for downstream predictor/anomaly operations.

### 3) Predictor orchestration

- Writes updated `predict_input.csv`.
- Starts `src/predictor_ai.py` with model-family/env settings.
- Reads `predict_out.csv` when generation completes.
- Re-triggers predictor when actual-data horizon exceeds predicted horizon.

### 4) Anomaly detection

- Calls `AnomalyDetector` to compare actual vs predicted timeline.
- Uses adaptive EWMA behavior (detector-side implementation).
- Keeps anomaly stream contiguous by catch-up scheduling.
- Logs newly detected anomalies with timestamp and magnitude.

## Environment variables: predictor subprocess

The app copies the current process environment and sets or defaults keys consumed by `predictor_ai.py`:

| Variable | Typical role |
|----------|----------------|
| `PREDICTOR_MODEL_FAMILY` | `gru`, `lstm`, `transformer`, or `attn_bilstm` |
| `PREDICTOR_MODEL_INIT` | `pretrained` or `fresh` |
| `PREDICTOR_UPDATE_TRAINING` | `1` train+predict, `0` predict-only |
| `PREDICTOR_CHECKPOINT_PATH` | Per-sensor `predictor_runtime_<family>.keras` in the session folder |
| `PRETRAINED_MODEL_PATH` | Resolved checkpoint or bundled pretrained `.keras` when applicable |
| `TRAIN_WINDOW_MINUTES` | From startup “training window” when set |
| `PREDICTOR_N_FUTURE` | Forecast steps sized from data span (see cap below) |
| `PREDICTOR_GRU_WINDOW_SIZE` | Set for **fresh GRU** to match chosen **W** |
| `PREDICTOR_EPOCHS_PER_UPDATE` | Default **40** if unset (`setdefault` before spawn) |
| `PREDICTOR_ONLINE_FIT_EACH_STEP` | Default **0** (leave off unless debugging) |
| `PREDICTOR_RETRAIN_INTERVAL_MINUTES` | Per-sensor retrain cadence |
| `PRETRAINED_GRU_MODEL_DIR` / `PRETRAINED_MODEL_DIR` / `PRETRAINED_TRANSFORMER_MODEL_DIR` | Search paths for bundled models |
| `OMP_NUM_THREADS`, `TF_NUM_INTRAOP_THREADS`, `TF_NUM_INTEROP_THREADS` | Default **1** each to limit CPU contention |

## Environment variables: host process (`app.py`)

These are read when `ApplicationTemp` starts (export before launching the app):

| Variable | Role |
|----------|------|
| `MAGNAVIS_LOWPASS_ALPHA` | EMA weight on **new** samples; default **0.45** (higher = less smoothing). Clamped to `[0.01, 1]`. |
| `MAGNAVIS_PREDICTOR_RAW_MAG` | If `1` / `true` / `yes` / `on`, `predict_input.csv` uses **raw** magnitude (aligned with `x`) while the UI still uses the low-pass series. |
| `PREDICTOR_N_FUTURE_CAP` | Optional integer ≥ **100**: caps `PREDICTOR_N_FUTURE` before spawn (shorter runs, less autoregression). |
| `PREDICTOR_GRU_WINDOW_SIZE` | May preset fresh GRU **W** when used together with your workflow |
| `MAGNAVIS_DISABLE_GROUND_TRUTH` | Disables GT overlay loading |
| `MAGNAVIS_INITIAL_THRESHOLD_K` | Optional startup default for **k** (alias: `INITIAL_ANOMALY_THRESHOLD_K`) |
| `MAGNAVIS_INITIAL_TRAIN_WINDOW_MINUTES` | Optional startup default for training window |
| `GT_REFERENCE_SESSION_ID`, `GT_FALLBACK_TERMINAL_LOGS` | Ground-truth reference session discovery |

## Failure modes and troubleshooting

- No prediction output:
  - check `predict_stderr.log`
  - verify TensorFlow environment and CSV schema (`x`, `y`)
- No anomalies:
  - verify predictor output overlaps actual timeline
  - tune threshold multiplier / EWMA memory settings

## Example scenarios

### Example A: replay historical DB window

1. Launch script.
2. Select simulation mode with a chosen start date.
3. Select OBS sensors.
4. Observe sequential prediction + anomaly behavior before moving to realtime.

### Example B: CSV offline validation

1. Launch script.
2. Select CSV mode and a prepared multi-sensor file.
3. Let playback reach end-of-file.
4. Confirm final forced catch-up pass processes remaining anomalies.
