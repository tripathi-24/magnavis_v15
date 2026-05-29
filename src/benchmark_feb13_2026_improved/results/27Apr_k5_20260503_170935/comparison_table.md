# Feb 13 2026 improved benchmark

- CSV: `magnetic_data_20260426_060000_to_20260427_090000_1hz.csv`
- k=5.0; historic load: 62 min; offline skip first: 62.0 min/sensor
- App sequence: lstm_pretrained, lstm_fresh, attn_bilstm_fresh, gru_pretrained, gru_fresh, transformer_pretrained. Train window min: 0
- GT: mode=`manual_app`, manual_csv_basename=`magnetic_data_20260426_060000_to_20260427_090000_1hz.csv` (must match app.py / evaluate tables)
- Evaluator: base_date=`2026-04-27`; point_tolerance_sec=0; event_merge_gap_sec=2; event_tolerance_sec=5
- CSV window: start `(none)`, end `(none)`
- Point-level eval: `magnetic CSV grid (eval-obs-grid=obs2; csv_start=(none); csv_end=(none); skip=62.0)`
- **Equal n check:** `TP+FP+TN+FN` verified identical across 6 rows (yes).

| Predictor | Detector | TP | FP | TN | FN | n | Recall | Precision | F1 | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ewma | standard k=5.0 | 8 | 17 | 346 | 4734 | 5105 | 0.0017 | 0.3200 | 0.0034 | 0.9532 | 0.0693 |
| median | standard k=5.0 | 22 | 24 | 339 | 4720 | 5105 | 0.0046 | 0.4783 | 0.0092 | 0.9339 | 0.0707 |
| savgol | standard k=5.0 | 27 | 36 | 327 | 4715 | 5105 | 0.0057 | 0.4286 | 0.0112 | 0.9008 | 0.0693 |
| lstm_pretrained | standard k=5.0 | 3533 | 190 | 173 | 1209 | 5105 | 0.7450 | 0.9490 | 0.8347 | 0.4766 | 0.7260 |
| attention_bi_lstm | standard k=5.0 | 3534 | 324 | 39 | 1208 | 5105 | 0.7453 | 0.9160 | 0.8219 | 0.1074 | 0.6999 |
| gru_pretrained | standard k=5.0 | 3533 | 169 | 194 | 1209 | 5105 | 0.7450 | 0.9543 | 0.8368 | 0.5344 | 0.7301 |
