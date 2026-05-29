# Feb 13 2026 improved benchmark

- CSV: `magnetic_data_20260213_150000_to_20260213_163000.csv`
- k=5.0; historic load: 62 min; offline skip first: 62.0 min/sensor
- App sequence: lstm_pretrained, lstm_fresh, attn_bilstm_fresh, gru_pretrained, gru_fresh, transformer_pretrained. Train window min: 0
- GT: mode=`manual_app`, manual_csv_basename=`magnetic_data_20260213_150000_to_20260213_163000.csv` (must match app.py / evaluate tables)
- Evaluator: base_date=`2026-02-13`; point_tolerance_sec=0; event_merge_gap_sec=2; event_tolerance_sec=5
- CSV window: start `(none)`, end `2026-02-13 16:18:30`
- Point-level eval: `magnetic CSV grid (eval-obs-grid=obs2; csv_start=(none); csv_end=2026-02-13 16:18:30; skip=62.0)`
- **Equal n check:** `TP+FP+TN+FN` verified identical across 6 rows (yes).

| Predictor | Detector | TP | FP | TN | FN | n | Recall | Precision | F1 | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ewma | standard k=5.0 | 184 | 25 | 759 | 23 | 991 | 0.8889 | 0.8804 | 0.8846 | 0.9681 | 0.9516 |
| median | standard k=5.0 | 196 | 46 | 738 | 11 | 991 | 0.9469 | 0.8099 | 0.8731 | 0.9413 | 0.9425 |
| savgol | standard k=5.0 | 202 | 54 | 730 | 5 | 991 | 0.9758 | 0.7891 | 0.8726 | 0.9311 | 0.9405 |
| lstm_pretrained | standard k=5.0 | 152 | 11 | 773 | 55 | 991 | 0.7343 | 0.9325 | 0.8216 | 0.9860 | 0.9334 |
| attention_bi_lstm | standard k=5.0 | 196 | 90 | 694 | 11 | 991 | 0.9469 | 0.6853 | 0.7951 | 0.8852 | 0.8981 |
| gru_pretrained | standard k=5.0 | 183 | 16 | 768 | 24 | 991 | 0.8841 | 0.9196 | 0.9015 | 0.9796 | 0.9596 |
