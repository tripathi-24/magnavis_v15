# Feb 13 2026 improved benchmark

- CSV: `magnetic_data_20260213_150000_to_20260213_180000_synthetic_no_anomaly.csv`
- k=2.0; historic load: 62 min; offline skip first: 0.0 min/sensor
- App sequence: lstm_pretrained, lstm_fresh, attn_bilstm_fresh, gru_pretrained, gru_fresh, transformer_pretrained. Train window min: 0
- GT: mode=`manual_app`, manual_csv_basename=`magnetic_data_20260213_150000_to_20260213_180000_synthetic_no_anomaly.csv` (must match app.py / evaluate tables)
- Evaluator: base_date=`2026-02-13`; point_tolerance_sec=0; event_merge_gap_sec=2; event_tolerance_sec=5
- CSV window: start `(none)`, end `(none)`
- Point-level eval: `magnetic CSV grid (eval-obs-grid=obs2; csv_start=(none); csv_end=(none); skip=0.0)`
- **Equal n check:** `TP+FP+TN+FN` verified identical across 6 rows (yes).

| Predictor | Detector | TP | FP | TN | FN | n | Recall | Precision | F1 | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ewma | standard k=2.0 | 75 | 4958 | 439 | 5328 | 10800 | 0.0139 | 0.0149 | 0.0144 | 0.0813 | 0.0476 |
| median | standard k=2.0 | 51 | 5045 | 352 | 5352 | 10800 | 0.0094 | 0.0100 | 0.0097 | 0.0652 | 0.0373 |
| savgol | standard k=2.0 | 93 | 4834 | 563 | 5310 | 10800 | 0.0172 | 0.0189 | 0.0180 | 0.1043 | 0.0607 |
| lstm_pretrained | standard k=2.0 | 3602 | 2420 | 2977 | 1801 | 10800 | 0.6667 | 0.5981 | 0.6305 | 0.5516 | 0.6092 |
| attention_bi_lstm | standard k=2.0 | 3602 | 762 | 4635 | 1801 | 10800 | 0.6667 | 0.8254 | 0.7376 | 0.8588 | 0.7627 |
| gru_pretrained | standard k=2.0 | 3602 | 2140 | 3257 | 1801 | 10800 | 0.6667 | 0.6273 | 0.6464 | 0.6035 | 0.6351 |
