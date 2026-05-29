# Feb 13 2026 improved benchmark

- CSV: `magnetic_data_20260426_060000_to_20260427_090000_1hz.csv`
- k=4.0; historic load: 62 min; offline skip first: 62.0 min/sensor
- App sequence: lstm_pretrained, lstm_fresh, attn_bilstm_fresh, gru_pretrained, gru_fresh, transformer_pretrained. Train window min: 0
- GT: mode=`manual_app`, manual_csv_basename=`magnetic_data_20260426_060000_to_20260427_090000_1hz.csv` (must match app.py / evaluate tables)
- Evaluator: base_date=`2026-04-27`; point_tolerance_sec=0; event_merge_gap_sec=2; event_tolerance_sec=5
- CSV window: start `(none)`, end `(none)`
- Point-level eval: `magnetic CSV grid (eval-obs-grid=obs2; csv_start=(none); csv_end=(none); skip=62.0)`
- **Equal n check:** `TP+FP+TN+FN` verified identical across 9 rows (yes).

| Predictor | Detector | TP | FP | TN | FN | n | Recall | Precision | F1 | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ewma | standard k=4.0 | 25 | 17 | 346 | 4717 | 5105 | 0.0053 | 0.5952 | 0.0105 | 0.9532 | 0.0727 |
| median | standard k=4.0 | 43 | 25 | 338 | 4699 | 5105 | 0.0091 | 0.6324 | 0.0179 | 0.9311 | 0.0746 |
| savgol | standard k=4.0 | 43 | 36 | 327 | 4699 | 5105 | 0.0091 | 0.5443 | 0.0178 | 0.9008 | 0.0725 |
| lstm_pretrained | standard k=4.0 | 3533 | 323 | 40 | 1209 | 5105 | 0.7450 | 0.9162 | 0.8218 | 0.1102 | 0.6999 |
| lstm_fresh | standard k=4.0 | 1245 | 10 | 353 | 3497 | 5105 | 0.2625 | 0.9920 | 0.4152 | 0.9725 | 0.3130 |
| attention_bi_lstm | standard k=4.0 | 3714 | 363 | 0 | 1028 | 5105 | 0.7832 | 0.9110 | 0.8423 | 0.0000 | 0.7275 |
| gru_pretrained | standard k=4.0 | 3533 | 323 | 40 | 1209 | 5105 | 0.7450 | 0.9162 | 0.8218 | 0.1102 | 0.6999 |
| gru_fresh | standard k=4.0 | 11 | 15 | 348 | 4731 | 5105 | 0.0023 | 0.4231 | 0.0046 | 0.9587 | 0.0703 |
| transformer_pretrained | standard k=4.0 | 0 | 0 | 363 | 4742 | 5105 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0711 |
