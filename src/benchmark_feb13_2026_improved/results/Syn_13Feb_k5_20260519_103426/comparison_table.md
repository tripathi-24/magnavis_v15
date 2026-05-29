# Feb 13 2026 improved benchmark

- CSV: `magnetic_data_20260213_150000_to_20260213_180000_synthetic_no_anomaly.csv`
- k=5.0; historic load: 62 min; offline skip first: 0.0 min/sensor
- App sequence: lstm_pretrained, lstm_fresh, attn_bilstm_fresh, gru_pretrained, gru_fresh, transformer_pretrained. Train window min: 0
- GT: mode=`manual_app`, manual_csv_basename=`magnetic_data_20260213_150000_to_20260213_180000_synthetic_no_anomaly.csv` (must match app.py / evaluate tables)
- Evaluator: base_date=`2026-02-13`; point_tolerance_sec=0; event_merge_gap_sec=2; event_tolerance_sec=5
- CSV window: start `(none)`, end `(none)`
- Point-level eval: `magnetic CSV grid (eval-obs-grid=obs2; csv_start=(none); csv_end=(none); skip=0.0)`
- **Equal n check:** `TP+FP+TN+FN` verified identical across 6 rows (yes).

| Predictor | Detector | TP | FP | TN | FN | n | Recall | Precision | F1 | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ewma | standard k=5.0 | 47 | 4302 | 1095 | 5356 | 10800 | 0.0087 | 0.0108 | 0.0096 | 0.2029 | 0.1057 |
| median | standard k=5.0 | 51 | 4307 | 1090 | 5352 | 10800 | 0.0094 | 0.0117 | 0.0104 | 0.2020 | 0.1056 |
| savgol | standard k=5.0 | 93 | 4386 | 1011 | 5310 | 10800 | 0.0172 | 0.0208 | 0.0188 | 0.1873 | 0.1022 |
| lstm_pretrained | standard k=5.0 | 3602 | 15 | 5382 | 1801 | 10800 | 0.6667 | 0.9959 | 0.7987 | 0.9972 | 0.8319 |
| attention_bi_lstm | standard k=5.0 | 0 | 2996 | 2401 | 5403 | 10800 | 0.0000 | 0.0000 | 0.0000 | 0.4449 | 0.2223 |
| gru_pretrained | standard k=5.0 | 3602 | 478 | 4919 | 1801 | 10800 | 0.6667 | 0.8828 | 0.7597 | 0.9114 | 0.7890 |
