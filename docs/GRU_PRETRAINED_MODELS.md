# GRU pre-trained sequence predictors (production bundle)

This note describes the **six canonical GRU checkpoints** produced by `src/train_gru_pretrained.py`, how they relate to sensors, and how the DB/CSV app (`src/app.py`, launched via `src/application_temp.py`) loads them at runtime.

## Artifacts (under `models/`)

For each canonical sensor ID, training writes a **pair** of files:

| File | Role |
|------|------|
| `gru_pretrained_<SENSOR>.keras` | Keras model (`AttnBiLSTMPredictor` with `MODEL_FAMILY_GRU`) |
| `gru_pretrained_<SENSOR>_scaler.pkl` | Fitted scaler for total-field magnitude, saved next to the model |

`<SENSOR>` is one of:

- `OBS1_1`, `OBS1_2`, `OBS1_3`
- `OBS2_1`, `OBS2_2`, `OBS2_3`

Session-prefixed IDs in raw CSVs (for example `S20260202_124213_OBS2_1`) are **merged** into these six names during training and at load time.

## What the model predicts

- **Input signal:** total magnetic field magnitude  
  \(\sqrt{b_x^2 + b_y^2 + b_z^2}\) in nT, from columns `sensor_id`, `timestamp`, `b_x`, `b_y`, `b_z` in the export CSVs.
- **Features:** scaled magnitude plus **cyclic time features** (daily sine/cosine; by default **yearly** sine/cosine as well), matching the feature layout used in `predictor_ai.AttnBiLSTMPredictor` for pre-trained runs.
- **Sequence length:** `window_size=15` (CLI: `--window-size`).
- **Architecture:** training calls `AttnBiLSTMPredictor.build_model` in `predictor_ai.py`, so the saved graph matches the **current** GRU definition in that file (today: **four GRU layers** with inter-layer dropout and a dense head). If you upgrade `predictor_ai.py` after training, old `.keras` files may no longer load; re-run `train_gru_pretrained.py` and refresh `models/gru_pretrained_*.keras`.
- **Training:** one model per canonical sensor; `validation_split=0.1` inside `model.fit`.

Training data are **not** anomaly labels: any `Experiment_Data*.csv` ground-truth files are for evaluation overlays, not for this pretraining script.

## Typical training command (repo root)

Adjust CSV paths to your exports. Example combining two monthly magnetic dumps:

```bash
cd "/path/to/magnavis_v13_Polish"
PYTHONUNBUFFERED=1 "src/.venv/bin/python3" -u "src/train_gru_pretrained.py" \
  "magnetic_data_20251201_000000_to_20251231_234500.csv,magnetic_data_20251231_234500_to_20260131_234500.csv" \
  "models" \
  --epochs 25
```

Use `PYTHONUNBUFFERED=1` or `python -u` so progress prints promptly on long runs. Models are written **after each sensor** finishes.

## Using the bundle in `application_temp.py`

1. Place (or keep) the twelve files under the project `models/` directory, **or** point the app at another directory that contains them.
2. At startup, choose **GRU (pretrained)** when prompted for the prediction model family (see `ApplicationTemp` in `src/app.py`).
3. **Optional environment variables** (predictor subprocess is started from `ApplicationTemp.start_prediction_process_for_sensor` in `src/app.py`):
   - `PRETRAINED_GRU_MODEL_DIR` — directory to search for `gru_pretrained_*.keras` (highest priority for GRU).
   - `PRETRAINED_MODEL_DIR` — fallback if the GRU-specific variable is unset.
   - If neither is set, the code defaults to `<project_root>/models` when that folder exists.

Resolution order for a given `sensor_id` prefers `gru_pretrained_<canonical_OBS>.keras` (and the matching `_scaler.pkl` loaded via `predictor_ai` when the model is loaded).

## Related documentation

- Training script details: `docs/scripts/src_train_gru_pretrained.md`
- Predictor behavior: `docs/scripts/src_predictor_ai.md`
- App wiring: `docs/scripts/src_application_temp.md`
- **Vanilla LSTM (pretrained) bundle:** `docs/LSTM_PRETRAINED_MODELS.md` and `src/train_lstm_pretrained.py` (same CSV pipeline, `lstm_pretrained_<SENSOR>.keras`).
