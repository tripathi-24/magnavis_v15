# Comparative Analysis: k–Recall Benchmark (Zero-Historic)

**Results bundle:** `k_recall_curves_zero_hist_20260522_170707`  
**Generated for thesis inclusion** — summarizes **100** evaluation runs across **5** scenarios, six detector families (where applicable), and k ∈ {1, 2, 3, 4, 5}.

---

## Executive summary

Under a **zero-historic, predict-only** protocol (no session fine-tuning), **pretrained LSTM (open-loop)** is the most robust choice for production: flat ~85% recall on long-duration GT across all k, vs GRU’s cliff at k ≥ 4 (~63%). **Closed-loop autoregressive inference** does not fix GRU’s high-k interior misses; LSTM closed-loop should be compared in the table below — if it tracks open-loop LSTM, closed-loop is unnecessary for deployment.

**Recommended deployment:** `PREDICTOR_MODEL_FAMILY=lstm`, `PREDICTOR_MODEL_INIT=pretrained`, **k = 3** (long/mixed), **k = 4** (short shot-duration only). Avoid k = 1 and GRU @ k ≥ 4 on long GT.

---

## Experimental protocol

| Parameter | Value |
|-----------|--------|
| Historic context | 0 min (`MAGNAVIS_BATCH_HISTORIC_MINUTES=0`) |
| Skip initial | 0 min |
| Predictor training window | 0 min (predict-only pretrained) |
| Detector | EWMA on \|actual − predicted\| |
| Threshold | mean + **k** × σ |
| k grid | 1, 2, 3, 4, 5 |
| Sensors | OBS2 (all) |
| Closed-loop (scenarios 4–5) | `PREDICTOR_AR_CLOSED_LOOP=1` — AR window rolls on **predicted** magnitudes |

**Pipeline (neural):** magnetic CSV → `app.py` → predictor → residual → `AnomalyDetector` → metrics vs manual GT.

---

## Scenarios

| ID | Dataset key | Description | Evaluated points | GT-positive rate |
|----|-------------|-------------|------------------|------------------|
| 1 | `feb13` | Feb-13 2026, short shot-duration windows | 4,711 | ~4.4% (207 s) |
| 2 | `apr27` | Apr-27 2026, long sustained-offset windows | 8,825 | ~94.5% (8,344 s) |
| 3 | `synthetic` | Feb-13 GT layout on synthetic flat magnetics | 10,800 | ~50.0% (5,403 s) |
| 4 | `apr27_gru_closed_loop` | Apr-27, GRU, `PREDICTOR_AR_CLOSED_LOOP=1` | 8,825 | ~94.5% |
| 5 | `apr27_lstm_closed_loop` | Apr-27, LSTM, `PREDICTOR_AR_CLOSED_LOOP=1` | 8,825 | ~94.5% |

---

## Figures

| File | Description |
|------|-------------|
| `k_recall_curves_five_scenarios.png` | Plotly HD 2×3 panel: recall vs k (all scenarios); PDF also exported |
| `k_recall_f1_five_scenarios.png` | 2×3 panel: F1 vs k |
| `k_recall_apr27_open_vs_closed_loop.png` | **GRU & LSTM** open vs closed-loop on Apr-27 |
| `k_recall_curves_four_scenarios.png` | Legacy 2×2 (scenarios 1–4) |
| `k_recall_curves_neural_three_scenarios.png` | Core scenarios, neural only |

---

## Recall vs k (percent)

### Short duration GT Anomaly

| k | EWMA baseline | Median baseline | SavGol baseline | GRU (pretrained) | LSTM (pretrained) | Attn Bi-LSTM (fresh) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 100.0% | 100.0% | 100.0% | 97.6% | 98.6% | 100.0% |
| 2 | 96.6% | 99.0% | 99.5% | 92.3% | 89.9% | 99.0% |
| 3 | 94.2% | 98.6% | 99.0% | 90.3% | 86.5% | 96.1% |
| 4 | 90.8% | 97.1% | 98.1% | 88.4% | 85.0% | 91.8% |
| 5 | 87.0% | 94.7% | 94.2% | 88.4% | 73.4% | 90.8% |

### Long duration GT Anomaly

| k | EWMA baseline | Median baseline | SavGol baseline | GRU (pretrained) | LSTM (pretrained) | Attn Bi-LSTM (fresh) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 99.4% | 99.4% | 96.9% | 98.3% | 84.8% | 88.9% |
| 2 | 26.1% | 31.2% | 24.6% | 87.4% | 84.8% | 85.0% |
| 3 | 2.7% | 3.3% | 3.2% | 85.4% | 84.8% | 64.0% |
| 4 | 0.5% | 0.6% | 0.9% | 63.3% | 84.8% | 84.7% |
| 5 | 0.2% | 0.4% | 0.7% | 63.2% | 84.8% | 84.8% |

### Synthetic data(with long duration GT Anomaly)

| k | EWMA baseline | Median baseline | SavGol baseline | GRU (pretrained) | LSTM (pretrained) | Attn Bi-LSTM (fresh) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.6% | 0.9% | 1.7% | 100.0% | 100.0% | 99.9% |
| 2 | 1.4% | 0.9% | 1.7% | 100.0% | 100.0% | 99.9% |
| 3 | 1.3% | 0.9% | 1.7% | 100.0% | 100.0% | 100.0% |
| 4 | 1.2% | 0.9% | 1.7% | 100.0% | 100.0% | 66.6% |
| 5 | 0.9% | 0.9% | 1.7% | 100.0% | 100.0% | 66.5% |

### GRU closed loop

| k | GRU (pretrained) |
|---|---:|
| 1 | 99.6% |
| 2 | 99.5% |
| 3 | 85.0% |
| 4 | 64.5% |
| 5 | 57.1% |

### LSTM closed loop

| k | LSTM (pretrained) |
|---|---:|
| 1 | 99.6% |
| 2 | 94.1% |
| 3 | 85.0% |
| 4 | 84.8% |
| 5 | 59.0% |

---

## F1 score vs k

### Short duration GT Anomaly

| k | EWMA baseline | Median baseline | SavGol baseline | GRU (pretrained) | LSTM (pretrained) | Attn Bi-LSTM (fresh) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.085 | 0.085 | 0.087 | 0.098 | 0.097 | 0.087 |
| 2 | 0.249 | 0.206 | 0.265 | 0.171 | 0.280 | 0.306 |
| 3 | 0.708 | 0.636 | 0.688 | 0.477 | 0.806 | 0.873 |
| 4 | 0.864 | 0.839 | 0.834 | 0.849 | 0.884 | 0.945 |
| 5 | 0.865 | 0.873 | 0.859 | 0.888 | 0.819 | 0.940 |

### Long duration GT Anomaly

| k | EWMA baseline | Median baseline | SavGol baseline | GRU (pretrained) | LSTM (pretrained) | Attn Bi-LSTM (fresh) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.969 | 0.969 | 0.957 | 0.964 | 0.893 | 0.914 |
| 2 | 0.410 | 0.470 | 0.391 | 0.908 | 0.893 | 0.895 |
| 3 | 0.052 | 0.063 | 0.061 | 0.896 | 0.893 | 0.757 |
| 4 | 0.009 | 0.012 | 0.019 | 0.756 | 0.893 | 0.893 |
| 5 | 0.005 | 0.008 | 0.013 | 0.765 | 0.893 | 0.902 |

### Synthetic data(with long duration GT Anomaly)

| k | EWMA baseline | Median baseline | SavGol baseline | GRU (pretrained) | LSTM (pretrained) | Attn Bi-LSTM (fresh) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.016 | 0.009 | 0.017 | 0.703 | 0.702 | 0.698 |
| 2 | 0.014 | 0.010 | 0.018 | 0.730 | 0.736 | 0.709 |
| 3 | 0.014 | 0.010 | 0.019 | 0.747 | 0.819 | 0.774 |
| 4 | 0.013 | 0.010 | 0.019 | 0.821 | 0.939 | 0.533 |
| 5 | 0.010 | 0.010 | 0.019 | 0.845 | 0.992 | 0.533 |

### GRU closed loop

| k | GRU (pretrained) |
|---|---:|
| 1 | 0.970 |
| 2 | 0.969 |
| 3 | 0.894 |
| 4 | 0.763 |
| 5 | 0.721 |

### LSTM closed loop

| k | LSTM (pretrained) |
|---|---:|
| 1 | 0.970 |
| 2 | 0.945 |
| 3 | 0.894 |
| 4 | 0.893 |
| 5 | 0.737 |

---

## Cross-scenario analysis

### k sensitivity (Apr-27 open-loop)

| Model | Recall @ k=2 | Recall @ k=5 | Δ recall | Behaviour |
|-------|--------------|--------------|----------|-----------|
| **LSTM** | 84.8% | 84.8% | **0.0%** | Plateau-stable |
| Attn Bi-LSTM | 85.0% | 84.8% | 0.2% | Stable (dip at k=3 only) |
| **GRU** | 87.4% | 63.2% | **−24.2%** | Cliff at k ≥ 4 |

### Open-loop vs closed-loop on Apr-27 (recall %)

| k | GRU open | GRU closed | LSTM open | LSTM closed |
|---|---:|---:|---:|---:|
| 1 | 98.3% | 99.6% | 84.8% | 99.6% |
| 2 | 87.4% | 99.5% | 84.8% | 94.1% |
| 3 | 85.4% | 85.0% | 84.8% | 85.0% |
| 4 | 63.3% | 64.5% | 84.8% | 84.8% |
| 5 | 63.2% | 57.1% | 84.8% | 59.0% |

**GRU closed-loop:** does not restore k ≥ 4 recall (64.5% @ k=4 vs 63.3% open-loop; 57.1% @ k=5). Near-total flagging at k ≤ 2.

**LSTM closed-loop:** see table above — compare stability vs open-loop LSTM (~85% flat recall).

### Mean F1 across scenarios 1–3 (feb13 + apr27 + synthetic)

- k=3: LSTM (pretrained)=0.839, Attn Bi-LSTM (fresh)=0.801, GRU (pretrained)=0.707, EWMA baseline=0.258, SavGol baseline=0.256, Median baseline=0.236
- k=4: LSTM (pretrained)=0.906, GRU (pretrained)=0.809, Attn Bi-LSTM (fresh)=0.790, EWMA baseline=0.295, SavGol baseline=0.290, Median baseline=0.287
- k=5: LSTM (pretrained)=0.901, GRU (pretrained)=0.833, Attn Bi-LSTM (fresh)=0.792, Median baseline=0.297, SavGol baseline=0.297, EWMA baseline=0.293

### Best family at operational k per scenario type

- **Short shot-duration campaigns** (k=4): **Attn Bi-LSTM (fresh)** — F1=0.945, recall=0.918, precision=0.974
- **Long sustained-offset campaigns** (k=3): **GRU (pretrained)** — F1=0.896, recall=0.854, precision=0.943
- **Synthetic control (long GT, flat magnetics)** (k=4): **LSTM (pretrained)** — F1=0.939, recall=1.000, precision=0.885

---

## Final recommendation (thesis & production)

| Role | Choice |
|------|--------|
| **Primary predictor** | Pretrained **LSTM**, **open-loop** |
| **Default k (long / mixed)** | **3.0** |
| **k (short shot-duration only)** | **4.0** |
| **Ablation** | Closed-loop GRU/LSTM, Attn Bi-LSTM (fresh) |
| **Avoid** | k = 1; GRU @ k ≥ 4 on long GT; closed-loop unless validated |

```bash
export PREDICTOR_MODEL_FAMILY=lstm
export PREDICTOR_MODEL_INIT=pretrained
export PREDICTOR_UPDATE_TRAINING=0
export PREDICTOR_SKIP_FINETUNE_ON_SESSION=1
export MAGNAVIS_INITIAL_THRESHOLD_K=3.0
```

---

## Data sources

| Path | Contents |
|------|----------|
| `k_recall_points.csv` | Feb-13 + Apr-27 open-loop |
| `runs/synthetic/k_recall_points.csv` | Synthetic |
| `runs/apr27_gru_closed_loop/k_recall_points.csv` | GRU closed-loop |
| `runs/apr27_lstm_closed_loop/k_recall_points.csv` | LSTM closed-loop |

---

*Regenerate:*

```bash
.venv/bin/python src/benchmark_feb13_2026_improved/generate_comparative_analysis.py
```
