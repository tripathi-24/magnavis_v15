# Script Doc: `src/train_lstm_pretrained.py`

## Purpose

Offline pretraining utility for **per-sensor vanilla LSTM** sequence predictors (`predictor_ai.AttnBiLSTMPredictor` with `MODEL_FAMILY_LSTM`).  
One checkpoint is trained per **canonical** sensor (`OBS1_1` … `OBS2_3`).

Mirrors `src/train_gru_pretrained.py` layout and CSV conventions.

## How to run

```bash
python src/train_lstm_pretrained.py \
  "magnetic_data_20251201_000000_to_20251231_234500.csv,magnetic_data_20251231_234500_to_20260131_234500.csv" \
  models/ \
  --sensors OBS1_1 OBS1_2 OBS1_3 OBS2_1 OBS2_2 OBS2_3 \
  --epochs 50 \
  --window-size 15
```

## Inputs

- CSV files with magnetic timeseries (single- or multi-sensor).
- CLI: `--epochs`, `--window-size`, `--sensors`, file list or `--folder` glob.

## Outputs

- Per sensor in the output directory:
  - `lstm_pretrained_<sensor_id>.keras`
  - `lstm_pretrained_<sensor_id>_scaler.pkl`
  - optional `lstm_pretrained_<sensor_id>_predictor_meta.json` when meta is written by trainer

See also: `docs/LSTM_PRETRAINED_MODELS.md`.

## Main functionality

1. Loads and normalizes input datasets (canonical `OBS#_#` sensor IDs).
2. Trains one LSTM predictor per sensor via `AttnBiLSTMPredictor` / `MODEL_FAMILY_LSTM`.
3. Saves model + scaler for **Pretrained LSTM** mode in `src/app.py`.

## Caveats

- Large CSVs are memory-heavy; trim or stage exports if loading fails.
- Checkpoint graph must match `predictor_ai.py` `build_model` for `lstm` at train time; retrain after architecture changes.

## Runtime linkage

Launch `python src/app.py`, choose **Pretrained LSTM**; the app resolves `lstm_pretrained_<OBS>.keras` under `models/` (or `PRETRAINED_LSTM_MODEL_DIR` / `PRETRAINED_MODEL_DIR`).
