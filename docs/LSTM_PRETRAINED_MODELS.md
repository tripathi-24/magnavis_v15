# LSTM pre-trained sequence predictors

Six checkpoints (one per canonical sensor) can be produced with **`src/train_lstm_pretrained.py`**, using the same CSV layout and feature stack as **`src/train_gru_pretrained.py`** (scaled \(\|B\|\) plus daily/yearly cyclic time features; supervised one-step windows).

## Artifacts (under `models/` or any output directory)

| File | Role |
|------|------|
| `lstm_pretrained_<SENSOR>.keras` | Keras model (`AttnBiLSTMPredictor` with `MODEL_FAMILY_LSTM`) |
| `lstm_pretrained_<SENSOR>_scaler.pkl` | Fitted `StandardScaler` for magnitude |

`<SENSOR>` is one of `OBS1_1`, `OBS1_2`, `OBS1_3`, `OBS2_1`, `OBS2_2`, `OBS2_3`. Session-prefixed `sensor_id` values in exports are merged to these names during training (same canonicalisation as the GRU trainer).

## Training (two CSVs, six sensors)

From the repository root, with a Python environment that has **TensorFlow** and the usual scientific stack:

```bash
python src/train_lstm_pretrained.py \
  "magnetic_data_20251201_000000_to_20251231_234500.csv,magnetic_data_20251231_234500_to_20260131_234500.csv" \
  models/ \
  --sensors OBS1_1 OBS1_2 OBS1_3 OBS2_1 OBS2_2 OBS2_3 \
  --epochs 50 \
  --window-size 15
```

Large multi-month files are memory-heavy; use a machine with sufficient RAM or trim CSVs if loading fails.

## Runtime (GUI)

Choose **“Pretrained LSTM”** at startup. The app resolves `lstm_pretrained_<OBS>.keras` under `models/` (or `PRETRAINED_MODEL_DIR` / `PRETRAINED_LSTM_MODEL_DIR`). Sequence length **W** is read from the checkpoint input shape (same as pretrained GRU).

## Architecture note

Checkpoints match **`predictor_ai.py`** `build_model` for **`MODEL_FAMILY_LSTM`** at train time (default: three stacked LSTM layers + dense head, unless `MAGNAVIS_DEEP_RNN_BENCHMARK=1`). After changing `predictor_ai.py`, retrain bundled `.keras` files if shapes diverge.
