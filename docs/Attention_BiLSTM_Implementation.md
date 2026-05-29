# Attention-BiLSTM Integration Note

This file is an overview note only.  
Canonical implementation details are in:

- `docs/scripts/src_predictor_ai.md`
- `docs/Attention_BiLSTM_Architecture.md`
- `docs/scripts/src_application_temp.md`
- `docs/scripts/src_train_gru_pretrained.md`

## Current integration summary

- Runtime supports `attn_bilstm`, `gru`, `lstm`, and `transformer` (`PREDICTOR_MODEL_FAMILY`).
- Model family and init (`pretrained` / `fresh`) are chosen at startup in `src/app.py` and passed as environment variables to `predictor_ai.py`.
- `src/application_temp.py` is a thin launcher that runs `app.py`.
- Detection uses predictor residuals and `AnomalyDetector` EWMA thresholding (no direction-finding stage).

## Minimal example (conceptual)

1. App writes `predict_input.csv`.
2. App sets `PREDICTOR_MODEL_FAMILY=attn_bilstm` (or `gru`, `lstm`, `transformer`).
3. `predictor_ai.py` runs and writes `predict_out.csv`.
4. `Anomaly_detector.py` compares actual vs predicted and flags anomalies.

For full behavior, arguments, edge cases, and artifacts, use canonical script docs listed above.
