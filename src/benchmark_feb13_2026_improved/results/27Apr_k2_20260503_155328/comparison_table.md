# Feb 13 2026 improved benchmark

- CSV: `magnetic_data_20260426_060000_to_20260427_090000_1hz.csv`
- k=2.0; historic load: 62 min; offline skip first: 62.0 min/sensor
- App sequence: lstm_pretrained, lstm_fresh, attn_bilstm_fresh, gru_pretrained, gru_fresh, transformer_pretrained. Train window min: 0
- GT: mode=`manual_app`, manual_csv_basename=`magnetic_data_20260426_060000_to_20260427_090000_1hz.csv` (must match app.py / evaluate tables)
- Evaluator: base_date=`2026-04-27`; point_tolerance_sec=0; event_merge_gap_sec=2; event_tolerance_sec=5
- CSV window: start `(none)`, end `(none)`
- Point-level eval: `magnetic CSV grid (eval-obs-grid=obs2; csv_start=(none); csv_end=(none); skip=62.0)`
- **Equal n check:** `TP+FP+TN+FN` verified identical across 6 rows (yes).

| Predictor | Detector | TP | FP | TN | FN | n | Recall | Precision | F1 | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ewma | standard k=2.0 | 1612 | 59 | 304 | 3130 | 5105 | 0.3399 | 0.9647 | 0.5027 | 0.8375 | 0.3753 |
| median | standard k=2.0 | 1672 | 71 | 292 | 3070 | 5105 | 0.3526 | 0.9593 | 0.5157 | 0.8044 | 0.3847 |
| savgol | standard k=2.0 | 1294 | 69 | 294 | 3448 | 5105 | 0.2729 | 0.9494 | 0.4239 | 0.8099 | 0.3111 |
| lstm_pretrained | standard k=2.0 | 3533 | 323 | 40 | 1209 | 5105 | 0.7450 | 0.9162 | 0.8218 | 0.1102 | 0.6999 |
| attention_bi_lstm | standard k=2.0 | 4104 | 363 | 0 | 638 | 5105 | 0.8655 | 0.9187 | 0.8913 | 0.0000 | 0.8039 |
| gru_pretrained | standard k=2.0 | 3914 | 336 | 27 | 828 | 5105 | 0.8254 | 0.9209 | 0.8706 | 0.0744 | 0.7720 |
