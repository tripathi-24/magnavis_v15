# Feb 13 2026 improved benchmark

- CSV: `magnetic_data_20260213_150000_to_20260213_163000.csv`
- k=2.0; historic load: 62 min; offline skip first: 62.0 min/sensor
- App sequence: lstm_pretrained, lstm_fresh, attn_bilstm_fresh, gru_pretrained, gru_fresh, transformer_pretrained. Train window min: 0
- GT: mode=`manual_app`, manual_csv_basename=`magnetic_data_20260213_150000_to_20260213_163000.csv` (must match app.py / evaluate tables)
- Evaluator: base_date=`2026-02-13`; point_tolerance_sec=0; event_merge_gap_sec=2; event_tolerance_sec=5
- CSV window: start `(none)`, end `2026-02-13 16:18:30`
- Point-level eval: `magnetic CSV grid (eval-obs-grid=obs2; csv_start=(none); csv_end=2026-02-13 16:18:30; skip=62.0)`
- **Equal n check:** `TP+FP+TN+FN` verified identical across 6 rows (yes).

| Predictor | Detector | TP | FP | TN | FN | n | Recall | Precision | F1 | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ewma | standard k=2.0 | 205 | 500 | 284 | 2 | 991 | 0.9903 | 0.2908 | 0.4496 | 0.3622 | 0.4934 |
| median | standard k=2.0 | 207 | 527 | 257 | 0 | 991 | 1.0000 | 0.2820 | 0.4400 | 0.3278 | 0.4682 |
| savgol | standard k=2.0 | 207 | 381 | 403 | 0 | 991 | 1.0000 | 0.3520 | 0.5208 | 0.5140 | 0.6155 |
| lstm_pretrained | standard k=2.0 | 187 | 210 | 574 | 20 | 991 | 0.9034 | 0.4710 | 0.6192 | 0.7321 | 0.7679 |
| attention_bi_lstm | standard k=2.0 | 206 | 525 | 259 | 1 | 991 | 0.9952 | 0.2818 | 0.4392 | 0.3304 | 0.4692 |
| gru_pretrained | standard k=2.0 | 191 | 72 | 712 | 16 | 991 | 0.9227 | 0.7262 | 0.8128 | 0.9082 | 0.9112 |
