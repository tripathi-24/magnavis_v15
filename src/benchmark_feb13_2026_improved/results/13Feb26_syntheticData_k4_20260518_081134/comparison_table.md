# Feb 13 2026 improved benchmark

- CSV: `magnetic_data_20260213_150000_to_20260213_180000_synthetic_no_anomaly.csv`
- k=4.0; historic load: 62 min; offline skip first: 0.0 min/sensor
- App sequence: lstm_pretrained, lstm_fresh, attn_bilstm_fresh, gru_pretrained, gru_fresh, transformer_pretrained. Train window min: 0
- GT: mode=`manual_app`, manual_csv_basename=`magnetic_data_20260213_150000_to_20260213_180000_synthetic_no_anomaly.csv` (must match app.py / evaluate tables)
- Evaluator: base_date=`2026-02-13`; point_tolerance_sec=0; event_merge_gap_sec=2; event_tolerance_sec=5
- CSV window: start `(none)`, end `(none)`
- Point-level eval: `magnetic CSV grid (eval-obs-grid=obs2; csv_start=(none); csv_end=(none); skip=0.0)`
- **Equal n check:** `TP+FP+TN+FN` verified identical across 6 rows (yes).

| Predictor | Detector | TP | FP | TN | FN | n | Recall | Precision | F1 | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ewma | standard k=4.0 | 63 | 4495 | 902 | 5340 | 10800 | 0.0117 | 0.0138 | 0.0126 | 0.1671 | 0.0894 |
| median | standard k=4.0 | 51 | 4503 | 894 | 5352 | 10800 | 0.0094 | 0.0112 | 0.0102 | 0.1656 | 0.0875 |
| savgol | standard k=4.0 | 93 | 4508 | 889 | 5310 | 10800 | 0.0172 | 0.0202 | 0.0186 | 0.1647 | 0.0909 |
| lstm_pretrained | standard k=4.0 | 3602 | 516 | 4881 | 1801 | 10800 | 0.6667 | 0.8747 | 0.7566 | 0.9044 | 0.7855 |
| attention_bi_lstm | standard k=4.0 | 3602 | 2998 | 2399 | 1801 | 10800 | 0.6667 | 0.5458 | 0.6002 | 0.4445 | 0.5556 |
| gru_pretrained | standard k=4.0 | 3602 | 829 | 4568 | 1801 | 10800 | 0.6667 | 0.8129 | 0.7326 | 0.8464 | 0.7565 |
