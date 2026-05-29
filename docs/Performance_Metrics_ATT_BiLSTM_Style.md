# Performance Metrics (Report-Style Overview)

This document is a presentation overview. It is **not** the canonical owner of script behavior.

Canonical script docs:
- `docs/scripts/tools_evaluate_anomaly_detection.md`
- `docs/Benchmark_Models_and_Baselines.md`

## Recommended metric groups

1. **Anomaly detection metrics**
   - Point-level precision/recall/F1
   - Event-level precision/recall/F1
   - False alarm rate and miss rate
2. **Predictor / forecast quality** (optional, separate from detection)
   - Residual MAE / RMSE on held-out magnitude
   - Horizon stability of `predict_out.csv`
3. **Runtime metrics**
   - Inference latency per cycle
   - End-to-end detection delay

## Reproducible example commands

```bash
# Feb 13 benchmark suite (baselines + deep models + shared detector)
cd src/benchmark_feb13_2026_improved
python run_suite.py

# Anomaly detection evaluation from app logs
python tools/evaluate_anomaly_detection.py --log-file src/sessions/<session_id>/app.log
```

Use this file for report structure and narration; use canonical and topic docs for exact script behavior and flags.
