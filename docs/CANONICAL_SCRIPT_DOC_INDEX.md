# Canonical Script Documentation Index

This index enforces a **one-script-one-document** rule for primary runtime and tooling scripts.

- **Canonical** means the single source of truth for a script's full behavior.
- **Topic** docs describe assets or cross-cutting behavior (models, benchmarks, architecture).
- **Supplementary** scripts are listed in `docs/OTHER_PYTHON_SCRIPTS.md` (plotting utilities, benchmarks helpers, legacy one-offs).

## Canonical ownership table

| Script | Canonical doc | Status |
|---|---|---|
| `src/application.py` | `docs/scripts/src_application.md` | canonical |
| `src/app.py` | `docs/scripts/src_application_temp.md` | canonical |
| `src/application_temp.py` | `docs/scripts/src_application_temp.md` | shim → runs `app.py` |
| `src/predictor_ai.py` | `docs/scripts/src_predictor_ai.md` | canonical |
| `src/Anomaly_detector.py` | `docs/scripts/src_Anomaly_detector.md` | canonical |
| `src/data_convert_db_now.py` | `docs/scripts/src_data_convert_db_now.md` | canonical |
| `src/data_convert_now.py` | `docs/scripts/src_data_convert_now.md` | canonical |
| `src/train_gru_pretrained.py` | `docs/scripts/src_train_gru_pretrained.md` | canonical |
| `src/train_lstm_pretrained.py` | `docs/scripts/src_train_lstm_pretrained.md` | canonical |
| `src/Get_Data_09Dec25.py` | `docs/scripts/src_Get_Data_09Dec25.md` | canonical |
| `src/real_time_file_watcher_db_updater.py` | `docs/scripts/src_real_time_file_watcher_db_updater.md` | canonical |
| `src/real_time_compensation.py` | `docs/scripts/src_real_time_compensation.md` | canonical |
| `src/equivalent_source_interpolation.py` | `docs/scripts/src_equivalent_source_interpolation.md` | canonical |
| `tools/evaluate_anomaly_detection.py` | `docs/scripts/tools_evaluate_anomaly_detection.md` | canonical |
| `tools/Anomaly_Detection_Evaluation/evaluate_anomaly_detection.py` | `docs/scripts/tools_evaluate_anomaly_detection.md` | alias of canonical |

## Removed scripts (docs retired)

These paths no longer exist in the repository; do not add new references to them.

| Former script | Former doc | Notes |
|---|---|---|
| `src/application_temp_fast.py` | `docs/scripts/src_application_temp_fast.md` | removed — use `src/app.py` |
| `generate_performance_plots.py` | `docs/scripts/generate_performance_plots.md` | removed — use benchmark matrices / `tools/plot_*` |
| `src/anomaly_direction.py` | `docs/scripts/src_anomaly_direction.md` | removed — direction finding retired |
| `src/physics_informed.py` | `docs/scripts/src_physics_informed.md` | removed from tree |
| `tools/*direction*.py` | `docs/scripts/tools_*direction*.md` | removed — direction training retired |

## Topic docs (not tied to a single script)

| Topic | Doc |
|---|---|
| GRU pre-trained `models/gru_pretrained_*.keras` | `docs/GRU_PRETRAINED_MODELS.md` |
| LSTM pre-trained `models/lstm_pretrained_*.keras` | `docs/LSTM_PRETRAINED_MODELS.md` |
| GRU input tensor / window conventions | `docs/GRU_INPUT_TENSOR_DETAILS.md` |
| Attention Bi-LSTM graph | `docs/Attention_BiLSTM_Architecture.md` |
| Transformer graph | `docs/Transformer_Architecture.md` |
| Feb 13 benchmark harness | `docs/Benchmark_Models_and_Baselines.md` |
| DB fetch performance notes | `docs/DATABASE_OPTIMIZATION_GUIDE.md` |
| Recent `app.py` behavior highlights | `docs/APPLICATION_TEMP_LATEST_UPDATES.md` |
| Short Attention-BiLSTM integration note | `docs/Attention_BiLSTM_Implementation.md` |
| Report-style metrics overview | `docs/Performance_Metrics_ATT_BiLSTM_Style.md` |

## Non-canonical usage rule

- `README.md` and overview docs under `docs/*.md` must summarize only and link to canonical or topic docs for internals.
- If script behavior changes, update the canonical doc first, then adjust overview docs.

## Scripts without a dedicated canonical doc

See **`docs/OTHER_PYTHON_SCRIPTS.md`** for plotting utilities, benchmark helpers, Streamlit analysis (`app_14_Nov.py`), and legacy modules.
