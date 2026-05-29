# Magnavis v15 (thesis)

Magnavis is a magnetic-field analytics project for:
- time-series visualization
- sequence prediction
- residual-based anomaly detection

This README is an overview.  
For script internals, use canonical docs in `docs/scripts/`.

**Publishing this project for thesis work:** see [`docs/THESIS_REPOSITORY.md`](docs/THESIS_REPOSITORY.md) (what to commit, data layout, new GitHub remote, reproducibility). Magnetic CSVs live in `Datafiles/` and are not versioned; see [`Datafiles/README.md`](Datafiles/README.md).

## Canonical documentation rule

- One script -> one canonical doc owner
- Canonical mapping index: `docs/CANONICAL_SCRIPT_DOC_INDEX.md`
- Overview docs must link, not duplicate script internals

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_feb_2025.txt
```

## Primary runtime entry points

### 1) Base application (USGS workflow)

```bash
python src/application.py
```

Canonical docs:
- `docs/scripts/src_application.md`
- `docs/scripts/src_data_convert_now.md`

### 2) Primary DB/CSV application (recommended)

```bash
python src/app.py
```

The file `src/application_temp.py` remains as a thin compatibility launcher that runs `app.py`.

Canonical docs:
- `docs/scripts/src_application_temp.md` (covers `src/app.py` behavior)
- `docs/scripts/src_data_convert_db_now.md`
- `docs/scripts/src_predictor_ai.md`
- `docs/scripts/src_Anomaly_detector.md`

## End-to-end model workflow

### Sequence predictor pretraining

```bash
python src/train_gru_pretrained.py dummy models/ --folder "Large Files" --epochs 50
```

Canonical doc:
- `docs/scripts/src_train_gru_pretrained.md`

Bundled checkpoints under `models/`: `docs/GRU_PRETRAINED_MODELS.md`, `docs/LSTM_PRETRAINED_MODELS.md`

### Anomaly evaluation from app logs

```bash
python tools/evaluate_anomaly_detection.py --log-file src/sessions/<session_id>/app.log
```

Canonical doc:
- `docs/scripts/tools_evaluate_anomaly_detection.md`

## Additional utilities

- Streamlit DB data fetch app: `src/Get_Data_09Dec25.py`
  - Canonical doc: `docs/scripts/src_Get_Data_09Dec25.md`
- Real-time CSV->DB watcher: `src/real_time_file_watcher_db_updater.py`
  - Canonical doc: `docs/scripts/src_real_time_file_watcher_db_updater.md`
- Live serial acquisition UI: `src/real_time_compensation.py`
  - Canonical doc: `docs/scripts/src_real_time_compensation.md`
- LSTM pretraining: `src/train_lstm_pretrained.py` — see `docs/LSTM_PRETRAINED_MODELS.md` and `docs/scripts/src_train_lstm_pretrained.md`
- Other utilities (benchmarks, plotting): `docs/OTHER_PYTHON_SCRIPTS.md`

## Troubleshooting quick checklist

- Predictor fails: check `predict_stderr.log` in sensor session folder
- No anomalies: verify `predict_out.csv` coverage and threshold settings
- Slow DB mode: review `docs/scripts/src_data_convert_db_now.md`

## DB/CSV app tuning (optional)

The primary DB/CSV runtime (`src/app.py`) supports extra environment variables for low-pass smoothing, predictor input (raw vs smoothed magnitude), forecast caps, and per-update training epochs. See **`docs/scripts/src_application_temp.md`** (host vs subprocess tables) and **`docs/scripts/src_predictor_ai.md`** for the full list.
