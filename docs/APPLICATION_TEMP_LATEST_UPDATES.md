# `application_temp.py` Latest Updates

Overview-only update log for major behavior changes in the DB/CSV runtime.

Canonical doc:
- `docs/scripts/src_application_temp.md`

## Current highlights

- Startup model family selection (`attn_bilstm` / `gru` / `transformer`) and init mode (`pretrained` / `fresh`), with fresh GRU sequence window **W** presets.
- Splash closed before startup modals so prompts are not hidden.
- Startup prompts for **k** (threshold multiplier), predictor training window (minutes), and related defaults via env (`MAGNAVIS_INITIAL_*`).
- Sequential prediction catch-up to keep anomaly comparisons contiguous.
- EWMA-based adaptive thresholding in anomaly detection.
- Predictor tuning: higher default **`PREDICTOR_EPOCHS_PER_UPDATE`**, optional **`MAGNAVIS_LOWPASS_ALPHA`**, **`MAGNAVIS_PREDICTOR_RAW_MAG`**, **`PREDICTOR_N_FUTURE_CAP`**; GRU stack includes dropout between recurrent blocks (see `docs/scripts/src_predictor_ai.md`).
- Offline evaluator: **`--prediction-sensor-mode union_all`** (default) vs **`filter`** — documented in `docs/scripts/tools_evaluate_anomaly_detection.md`.

## Related canonical docs

- Predictor details: `docs/scripts/src_predictor_ai.md`
- Anomaly detection details: `docs/scripts/src_Anomaly_detector.md`

## Quick run example

```bash
python src/application_temp.py
```

If behavior differs from this note, trust and update the canonical docs first.

