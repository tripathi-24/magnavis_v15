# Other Python scripts (no canonical doc)

These scripts exist in the repository but do not have a separate `docs/scripts/*.md` owner. Use inline docstrings and the linked topic docs where noted.

## Benchmark harness (`src/benchmark_feb13_2026*`)

| Script | Role | See also |
|---|---|---|
| `src/benchmark_feb13_2026_improved/run_suite.py` | Entry: delegates to `run_suite_improved.main()` | `docs/Benchmark_Models_and_Baselines.md` |
| `src/benchmark_feb13_2026_improved/run_suite_improved.py` | Main Feb 13 driver (baselines + headless `app.py`) | same |
| `src/benchmark_feb13_2026_improved/offline_statistical_baselines.py` | EWMA / median / Savgol synthetic logs | same |
| `src/benchmark_feb13_2026_improved/generate_perform_matrix.py` | Build comparison tables from latest run dir | same |
| `src/benchmark_feb13_2026_improved/generate_perform_matrix_from_eval.py` | Matrices from `eval/` subfolders | same |
| `src/benchmark_feb13_2026_improved/merge_k_sweep_tables.py` | Merge k-sweep CSV outputs | same |
| `src/benchmark_feb13_2026/run_suite.py` | Older suite entry (parent folder) | same |
| `src/benchmark_feb13_2026/offline_statistical_baselines.py` | Older baseline copy | same |
| `src/benchmark_feb13_2026/generate_perform_matrix.py` | Older matrix generator | same |

## Tools (`tools/`)

| Script | Role |
|---|---|
| `tools/plot_confusion_matrix_percent_only.py` | Confusion-matrix figures from eval outputs |
| `tools/plot_training_prediction_windows_hd.py` | HD training/prediction window figures |
| `tools/plot_gru_internal_architecture_similar.py` | GRU architecture diagram helper |
| `tools/plot_cycle_encode_gru_impl.py` | Cyclic encoding / GRU illustration |
| `tools/plot_pretrained_fit_history_plotly.py` | Plotly fit-history charts |
| `tools/model_comparison_mse_magnavis.py` | MSE comparison plots across models |
| `tools/resample_magnetic_csv_to_1hz.py` | Resample magnetic CSV to 1 Hz |
| `tools/make_application_temp_deep_dive_slides.py` | PowerPoint deep-dive (may mention retired direction stack) |
| `tools/make_expanded_deep_dive_slides.py` | Expanded slide deck generator |
| `tools/download_papers.py` | Paper download helper |
| `tools/generate_line_by_line_docs.py` | Doc generation utility |

## `src/` utilities and legacy

| Script | Role |
|---|---|
| `src/app_14_Nov.py` | Standalone Streamlit magnetic analysis UI (not `app.py`) |
| `src/data_convert.py` | Early/simple DB sample pull |
| `src/test_db.py` | Ad-hoc MySQL query helper |
| `src/add_data_to_mesh.py` | VTK mesh + magnetic field demo |
| `src/debug_docstrings.py` | Docstring debug helper |
| `update_predictor.py` | Standalone predictor graph experiment (root) |

Canonical docs for the **main pipeline** remain in `docs/CANONICAL_SCRIPT_DOC_INDEX.md`.
