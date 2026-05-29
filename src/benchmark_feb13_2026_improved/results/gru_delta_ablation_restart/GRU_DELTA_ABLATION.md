# GRU absolute vs Δ-target ablation (Apr-27 long GT)

**Output:** `gru_delta_ablation_restart`  
**Completed:** 2026-05-28

## Protocol

- Training CSV: `Datafiles/magnetic_data_20251201_000000_to_20251231_234500.csv`
- Sensors: OBS2_1, OBS2_2, OBS2_3
- Subsample: every 6th row; 8 epochs per sensor
- Benchmark: Apr-27 long GT, zero-historic predict-only, k ∈ {1…5}

## Recall vs k (Apr-27)

| k | Baseline GRU (`models/`) | Absolute retrain | Δ-target retrain |
|---:|---:|---:|---:|
| 1 | 98.3% | 84.8% | 94.5% |
| 2 | 87.4% | 84.8% | 85.3% |
| 3 | 85.4% | 84.8% | 47.0% |
| 4 | 63.3% | 84.8% | 4.5% |
| 5 | 63.2% | 84.8% | 0.3% |

## F1 vs k (Apr-27)

| k | Baseline | Absolute retrain | Δ-target retrain |
|---:|---:|---:|---:|
| 1 | 0.964 | 0.893 | 0.945 |
| 2 | 0.908 | 0.893 | 0.895 |
| 3 | 0.896 | 0.893 | 0.634 |
| 4 | 0.756 | 0.893 | 0.087 |
| 5 | 0.765 | 0.893 | 0.005 |

## Conclusion

**Δ-target integration is the main culprit for the high-k recall cliff on long GT.**

- **Δ-target retrain** (train + infer both use Δ): severe cliff from k=3 onward (47% → 4.5% → 0.3%).
- **Absolute retrain** (train + infer both use level): **flat ~84.8% recall** across k=1–5 — same stability pattern as LSTM open-loop in the main benchmark.
- **Baseline bundled GRU** still shows the historical cliff (85% @ k=3 → 63% @ k≥4), likely due to older checkpoints / train–infer mismatch vs the new explicit absolute bundle.

## Recommendation

1. For long sustained-offset campaigns: use **absolute-target GRU** or **LSTM**; avoid Δ-target GRU at k≥3.
2. If using Δ-target GRU: cap at **k≤2** on long GT (only ~85% recall there).
3. Prefer **LSTM + k≈3** for production unless absolute GRU is retrained and validated on your deployment CSVs.

## Artifacts

| Path | Contents |
|------|----------|
| `models/absolute/` | `gru_delta_y: false` checkpoints |
| `models/delta/` | `gru_delta_y: true` checkpoints |
| `runs/gru_absolute_retrain/k_recall_points.csv` | Absolute benchmark |
| `runs/gru_delta_retrain/k_recall_points.csv` | Δ-target benchmark |
| `runs/*/k_recall_curves_apr27.png` | Recall curves |
