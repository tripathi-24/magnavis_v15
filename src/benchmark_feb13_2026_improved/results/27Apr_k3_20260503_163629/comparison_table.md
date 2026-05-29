# Feb 13 2026 improved benchmark

- CSV: `magnetic_data_20260426_060000_to_20260427_090000_1hz.csv`
- k=3.0; historic load: 62 min; offline skip first: 62.0 min/sensor
- App sequence: lstm_pretrained, lstm_fresh, attn_bilstm_fresh, gru_pretrained, gru_fresh, transformer_pretrained. Train window min: 0
- GT: mode=`manual_app`, manual_csv_basename=`magnetic_data_20260426_060000_to_20260427_090000_1hz.csv` (must match app.py / evaluate tables)
- Evaluator: base_date=`2026-04-27`; point_tolerance_sec=0; event_merge_gap_sec=2; event_tolerance_sec=5
- CSV window: start `(none)`, end `(none)`
- Point-level eval: `magnetic CSV grid (eval-obs-grid=obs2; csv_start=(none); csv_end=(none); skip=62.0)`
- **Equal n check:** `TP+FP+TN+FN` verified identical across 6 rows (yes).

| Predictor | Detector | TP | FP | TN | FN | n | Recall | Precision | F1 | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ewma | standard k=3.0 | 202 | 22 | 341 | 4540 | 5105 | 0.0426 | 0.9018 | 0.0814 | 0.9394 | 0.1064 |
| median | standard k=3.0 | 229 | 31 | 332 | 4513 | 5105 | 0.0483 | 0.8808 | 0.0916 | 0.9146 | 0.1099 |
| savgol | standard k=3.0 | 179 | 37 | 326 | 4563 | 5105 | 0.0377 | 0.8287 | 0.0722 | 0.8981 | 0.0989 |
| lstm_pretrained | standard k=3.0 | 3533 | 323 | 40 | 1209 | 5105 | 0.7450 | 0.9162 | 0.8218 | 0.1102 | 0.6999 |
| attention_bi_lstm | standard k=3.0 | 3543 | 322 | 41 | 1199 | 5105 | 0.7472 | 0.9167 | 0.8233 | 0.1129 | 0.7021 |
| gru_pretrained | standard k=3.0 | 3533 | 323 | 40 | 1209 | 5105 | 0.7450 | 0.9162 | 0.8218 | 0.1102 | 0.6999 |
