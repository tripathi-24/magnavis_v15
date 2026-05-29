# Feb 13 2026 improved benchmark

- CSV: `magnetic_data_20260213_150000_to_20260213_163000.csv`
- k=3.0; historic load: 62 min; offline skip first: 62.0 min/sensor
- App sequence: lstm_pretrained, lstm_fresh, attn_bilstm_fresh, gru_pretrained, gru_fresh, transformer_pretrained. Train window min: 0
- GT: mode=`manual_app`, manual_csv_basename=`magnetic_data_20260213_150000_to_20260213_163000.csv` (must match app.py / evaluate tables)
- Evaluator: base_date=`2026-02-13`; point_tolerance_sec=0; event_merge_gap_sec=2; event_tolerance_sec=5
- CSV window: start `(none)`, end `2026-02-13 16:18:30`
- Point-level eval: `magnetic CSV grid (eval-obs-grid=obs2; csv_start=(none); csv_end=2026-02-13 16:18:30; skip=62.0)`
- **Equal n check:** `TP+FP+TN+FN` verified identical across 6 rows (yes).

| Predictor | Detector | TP | FP | TN | FN | n | Recall | Precision | F1 | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ewma | standard k=3.0 | 197 | 84 | 700 | 10 | 991 | 0.9517 | 0.7011 | 0.8074 | 0.8929 | 0.9051 |
| median | standard k=3.0 | 205 | 174 | 610 | 2 | 991 | 0.9903 | 0.5409 | 0.6997 | 0.7781 | 0.8224 |
| savgol | standard k=3.0 | 206 | 102 | 682 | 1 | 991 | 0.9952 | 0.6688 | 0.8000 | 0.8699 | 0.8961 |
| lstm_pretrained | standard k=3.0 | 183 | 16 | 768 | 24 | 991 | 0.8841 | 0.9196 | 0.9015 | 0.9796 | 0.9596 |
| attention_bi_lstm | standard k=3.0 | 201 | 345 | 439 | 6 | 991 | 0.9710 | 0.3681 | 0.5339 | 0.5599 | 0.6458 |
| gru_pretrained | standard k=3.0 | 188 | 19 | 765 | 19 | 991 | 0.9082 | 0.9082 | 0.9082 | 0.9758 | 0.9617 |
