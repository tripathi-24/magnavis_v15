# Script Doc: `src/predictor_ai.py`

## Purpose

Generate near-future magnetic predictions from historical scalar series using sequence models.

Supported families:
- Attention-BiLSTM (`attn_bilstm`)
- GRU (`gru`)
- Vanilla stacked LSTM (`lstm`)
- Transformer (`transformer`, pretrained checkpoints only; fresh training not implemented)

This script is usually executed by `app.py` (`ApplicationTemp`) as a subprocess (also reachable via the thin `application_temp.py` launcher).

## Direct execution

```bash
python src/predictor_ai.py path/to/predict_input.csv
```

Required input CSV columns:
- `x` (timestamp)
- `y` (scalar magnetic value)

## What it produces

- `predict_out.csv` beside input file
- Optional model checkpoint (`.keras`) and scaler (`.pkl`) when checkpoint settings are enabled

## Core processing flow

1. Read and validate `predict_input.csv`.
2. Parse and normalize timestamp/value series.
3. Build time-feature matrix (cyclic day/year features when enabled).
4. Build windowed training samples.
5. Select model family (`attn_bilstm`, `gru`, `lstm`, or `transformer`).
6. Load pretrained checkpoint if configured.
7. Train/update model (based on flags and available data).
8. Forecast `n_future` points autoregressively.
9. Write `predict_out.csv`.

## Key classes and functionality

- `AttnBiLSTMPredictor` builds the model graph and training / forecast loop.
- `create_windowed_dataset(...)` converts tabular series into `(B, W, F)` tensors.
- `forecast(...)` performs iterative next-step prediction.
- `save_model(...)` / `load_model(...)` manage checkpoint + scaler consistency.

### GRU family graph (current `predictor_ai`)

When `PREDICTOR_MODEL_FAMILY=gru`, the default stack is **four GRU layers** (`48 → 32 → 24 → 16` units) with **Dropout** after the first two GRU layers (rate `PREDICTOR_GRU_DROPOUT`, default **0.05**), then `Dense(16, relu)` and `Dense(1)`. **Fresh** runs use **`StandardScaler`** on magnitude (instead of squeezing a narrow nT band into MinMax `[0,1]`) and, by default, train on **one-step Δ(scaled magnitude)** so the head does not collapse to the mean; this mode is recorded in `*_predictor_meta.json` beside the checkpoint so reloads match. Bundled `gru_pretrained_*.keras` still use their pickled **MinMax** scaler and **absolute** targets (no meta file). Older two-layer checkpoints will **not** load into this graph; delete the session `predictor_runtime_gru*` files or retrain.

## Environment variables

Read by `predictor_ai.py` when launched (the DB/CSV app sets many of these automatically):

| Variable | Role |
|----------|------|
| `PREDICTOR_MODEL_FAMILY` | `attn_bilstm`, `gru`, `lstm`, or `transformer` |
| `PREDICTOR_MODEL_INIT` | `pretrained` or `fresh` (app-driven) |
| `PRETRAINED_MODEL_PATH` | Optional path to a `.keras` model to load before predict |
| `PREDICTOR_CHECKPOINT_PATH` | Session runtime checkpoint path (read/write when training) |
| `TRAIN_WINDOW_MINUTES` | If set, restrict training rows to the last *N* minutes |
| `PREDICTOR_N_FUTURE` | Autoregressive forecast length (steps); usually set by `app.py` from data span |
| `PREDICTOR_UPDATE_TRAINING` | `1` / `0` — train+predict vs predict-only |
| `PREDICTOR_GRU_WINDOW_SIZE` | Sequence window **W** for GRU (default **15**; fresh GRU uses app-selected **W**) |
| `PREDICTOR_EPOCHS_PER_UPDATE` | `fit` epochs per training invocation (default **40**, max **500**) |
| `PREDICTOR_ONLINE_FIT_EACH_STEP` | Default **0**; if `1`, fits on every autoregressive step (very slow) |
| `PREDICTOR_RETRAIN_INTERVAL_MINUTES` | How often the app may schedule a training run (app sets this) |
| `PREDICTOR_GRU_DELTA_TARGET` | Default **1** (`0` to disable): for **fresh** GRU only, train/predict **Δ** scaled magnitude; ignored for checkpoints that ship a `*_predictor_meta.json` / legacy absolute models |
| `PREDICTOR_GRU_DROPOUT` | Dropout rate between first two GRU blocks (**0.05** default; set **0** to disable) |
| `PREDICTOR_LEARNING_RATE` | Adam LR; default **0.003** for GRU, **0.001** for other families |

Host-only (read in `app.py`, not required inside this script):

- `PREDICTOR_N_FUTURE_CAP` — optional upper bound on computed `PREDICTOR_N_FUTURE` before the subprocess starts.
- `MAGNAVIS_PREDICTOR_RAW_MAG` — if enabled, `predict_input.csv` **y** column uses unsmoothed magnitude while plots still use the low-pass series.

## Failure modes and troubleshooting

- Empty/invalid CSV:
  - verify `x` and `y` columns are present and non-null
- Training crash:
  - check TensorFlow install and compatible dependency versions
- No output file:
  - inspect subprocess stderr log in sensor session folder
- Poor prediction quality or very flat `predict_out.csv`:
  - delete stale session `predictor_runtime_gru.keras`, matching `*_scaler.pkl`, and `*_predictor_meta.json`, then rerun train+predict so **StandardScaler + Δ-target** (defaults) apply
  - increase `PREDICTOR_EPOCHS_PER_UPDATE` or training data span; try `MAGNAVIS_PREDICTOR_RAW_MAG=1` (see `src_application_temp.md`)
  - inspect variance in `predict_input.csv` **y** vs residuals in logs
- Checkpoint load errors after upgrading:
  - remove mismatched `predictor_runtime_*.keras` in the session folder so a fresh graph is trained

## Practical examples

### Example A: manual local run

```bash
python src/predictor_ai.py src/sessions/test_session/OBS2_1/predict_input.csv
```

Expected output:
- `src/sessions/test_session/OBS2_1/predict_out.csv`

### Example B: force GRU family from caller

```bash
PREDICTOR_MODEL_FAMILY=gru python src/predictor_ai.py src/sessions/test_session/OBS2_1/predict_input.csv
```

### Example C: use pretrained checkpoint

```bash
PRETRAINED_MODEL_PATH=models/gru_pretrained_OBS2_1.keras python src/predictor_ai.py src/sessions/test_session/OBS2_1/predict_input.csv
```

