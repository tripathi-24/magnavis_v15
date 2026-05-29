# Script Doc: `src/train_gru_pretrained.py`

## Purpose

Offline pretraining utility for **per-sensor GRU** sequence predictors (`predictor_ai.AttnBiLSTMPredictor` with `MODEL_FAMILY_GRU`).  
One checkpoint is trained per **canonical** sensor (`OBS1_1` … `OBS2_3`).

## How to run

```bash
python src/train_gru_pretrained.py "file1.csv,file2.csv" models/ --epochs 25
# or
python src/train_gru_pretrained.py dummy models/ --folder "Large Files" --epochs 50
```

You can also pass explicit files or globs.

## Inputs

- CSV files containing magnetic timeseries (single or multi-sensor formats).
- Training hyperparameters from CLI options.

## Outputs

- Per-sensor files in the output directory:
  - `gru_pretrained_<sensor_id>.keras`
  - `gru_pretrained_<sensor_id>_scaler.pkl`

See also: `docs/GRU_PRETRAINED_MODELS.md`.

## Main functionality

1. Loads and normalizes input datasets.
2. Groups data by canonical sensor IDs (`_canonical_sensor_id`).
3. Trains one GRU predictor per sensor (`validation_split=0.1`).
4. Saves model and scaler for the DB/CSV app (`src/app.py`, **GRU pretrained** mode).

## Caveats

- Training runtime is large for multi-month exports; use unbuffered Python output for live logs.
- Data continuity and timestamp quality strongly affect model quality.
- The saved Keras graph is whatever `predictor_ai.AttnBiLSTMPredictor.build_model` produces **at train time**; after upgrading `predictor_ai.py`, retrain bundled checkpoints so layer shapes match.

## Example usage

Pretrain before app runtime, then launch `python src/app.py` (or `python src/application_temp.py`) and choose **GRU (pretrained)** so each sensor can load `gru_pretrained_<OBS>.keras` from `models/` (or `PRETRAINED_GRU_MODEL_DIR`).

For the same workflow with **vanilla LSTM**, use `src/train_lstm_pretrained.py` and `docs/LSTM_PRETRAINED_MODELS.md` (outputs `lstm_pretrained_<OBS>.keras`; in the GUI choose **Pretrained LSTM**).

