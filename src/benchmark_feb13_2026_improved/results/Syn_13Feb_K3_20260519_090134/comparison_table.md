# Feb 13 2026 improved benchmark

- CSV: `magnetic_data_20260213_150000_to_20260213_180000_synthetic_no_anomaly.csv`
- k=3.0; historic load: 62 min; offline skip first: 0.0 min/sensor
- App sequence: lstm_pretrained, lstm_fresh, attn_bilstm_fresh, gru_pretrained, gru_fresh, transformer_pretrained. Train window min: 0
- GT: mode=`manual_app`, manual_csv_basename=`magnetic_data_20260213_150000_to_20260213_180000_synthetic_no_anomaly.csv` (must match app.py / evaluate tables)
- Evaluator: base_date=`2026-02-13`; point_tolerance_sec=0; event_merge_gap_sec=2; event_tolerance_sec=5
- CSV window: start `(none)`, end `(none)`
- Point-level eval: `magnetic CSV grid (eval-obs-grid=obs2; csv_start=(none); csv_end=(none); skip=0.0)`
- **Equal n check:** `TP+FP+TN+FN` verified identical across 6 rows (yes).

| Predictor | Detector | TP | FP | TN | FN | n | Recall | Precision | F1 | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ewma | standard k=3.0 | 68 | 4525 | 872 | 5335 | 10800 | 0.0126 | 0.0148 | 0.0136 | 0.1616 | 0.0870 |
| median | standard k=3.0 | 51 | 4562 | 835 | 5352 | 10800 | 0.0094 | 0.0111 | 0.0102 | 0.1547 | 0.0820 |
| savgol | standard k=3.0 | 93 | 4553 | 844 | 5310 | 10800 | 0.0172 | 0.0200 | 0.0185 | 0.1564 | 0.0868 |
| lstm_pretrained | standard k=3.0 | 3602 | 879 | 4518 | 1801 | 10800 | 0.6667 | 0.8038 | 0.7289 | 0.8371 | 0.7519 |
| attention_bi_lstm | standard k=3.0 | 3601 | 2998 | 2399 | 1802 | 10800 | 0.6665 | 0.5457 | 0.6001 | 0.4445 | 0.5556 |
| gru_pretrained | standard k=3.0 | 3602 | 1851 | 3546 | 1801 | 10800 | 0.6667 | 0.6606 | 0.6636 | 0.6570 | 0.6619 |
