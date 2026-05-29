# Feb 13 2026 improved benchmark

- CSV: `magnetic_data_20260213_150000_to_20260213_163000.csv`
- k=4.0; historic load: 62 min; offline skip first: 62.0 min/sensor
- App sequence: lstm_pretrained, lstm_fresh, attn_bilstm_fresh, gru_pretrained, gru_fresh, transformer_pretrained. Train window min: 0
- GT: mode=`manual_app`, manual_csv_basename=`magnetic_data_20260213_150000_to_20260213_163000.csv` (must match app.py / evaluate tables)
- Evaluator: base_date=`2026-02-13`; point_tolerance_sec=0; event_merge_gap_sec=2; event_tolerance_sec=5
- CSV window: start `(none)`, end `2026-02-13 16:18:30`
- Point-level eval: `magnetic CSV grid (eval-obs-grid=obs2; csv_start=(none); csv_end=2026-02-13 16:18:30; skip=62.0)`
- **Equal n check:** `TP+FP+TN+FN` verified identical across 6 rows (yes).

| Predictor | Detector | TP | FP | TN | FN | n | Recall | Precision | F1 | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ewma | standard k=4.0 | 191 | 29 | 755 | 16 | 991 | 0.9227 | 0.8682 | 0.8946 | 0.9630 | 0.9546 |
| median | standard k=4.0 | 201 | 63 | 721 | 6 | 991 | 0.9710 | 0.7614 | 0.8535 | 0.9196 | 0.9304 |
| savgol | standard k=4.0 | 203 | 72 | 712 | 4 | 991 | 0.9807 | 0.7382 | 0.8423 | 0.9082 | 0.9233 |
| lstm_pretrained | standard k=4.0 | 176 | 11 | 773 | 31 | 991 | 0.8502 | 0.9412 | 0.8934 | 0.9860 | 0.9576 |
| attention_bi_lstm | standard k=4.0 | 196 | 186 | 598 | 11 | 991 | 0.9469 | 0.5131 | 0.6655 | 0.7628 | 0.8012 |
| gru_pretrained | standard k=4.0 | 185 | 17 | 767 | 22 | 991 | 0.8937 | 0.9158 | 0.9046 | 0.9783 | 0.9606 |
