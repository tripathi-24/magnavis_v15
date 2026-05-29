# Why GRU Underperforms on Long-Duration Ground-Truth Anomalies

**Context:** Zero-historic, predict-only benchmark (`k_recall_curves_zero_hist_20260522_170707`)  
**Dataset:** Apr-27 2026 — long sustained-offset manual GT (~94.5% positive seconds)  
**Compared predictors:** Pretrained GRU vs pretrained LSTM (open-loop), EWMA residual detector with \(k \in \{1,\ldots,5\}\)

---

## 1. Summary

Pretrained GRU is **not uniformly poor** on long-duration ground truth (GT). At detector multiplier **\(k = 3\)**, GRU matches or slightly exceeds LSTM on recall and F1. The failure mode is **sharp and threshold-specific**: when **\(k \ge 4\)**, GRU recall on Apr-27 drops from ~85% to ~63%, while LSTM stays flat at ~85% across \(k = 2\)–\(5\).

This document explains **what** is observed, **why** it likely happens in the Magnavis pipeline, **what evidence supports that mechanism**, and **how to mitigate** it in deployment and future work.

---

## 2. What the benchmark shows

### 2.1 Apr-27 (long GT) — recall vs \(k\)

| \(k\) | GRU recall | LSTM recall | GRU false negatives (of ~8,344 GT-positive s) |
|------:|-----------:|------------:|-----------------------------------------------:|
| 1 | 98.3% | 84.8% | 139 |
| 2 | 87.4% | 84.8% | 1,053 |
| 3 | **85.4%** | **84.8%** | 1,219 |
| 4 | **63.3%** | **84.8%** | **3,063** |
| 5 | **63.2%** | **84.8%** | **3,069** |

Between \(k = 3\) and \(k = 4\), GRU loses roughly **1,800** correctly flagged anomaly seconds; LSTM loses **none** (identical true-positive / false-negative counts for \(k = 2\)–\(5\)).

### 2.2 Precision and F1 at the cliff

At \(k = 4\) on Apr-27:

- GRU precision remains **~94%** — the model is not flooding the timeline with spurious alarms.
- The problem is **missed GT-positive seconds** (high false negatives), not random false positives.
- GRU F1 falls from **0.896** (\(k=3\)) to **0.756** (\(k=4\)); LSTM F1 stays **~0.893**.

So the cliff is **interior under-detection** during long events, not a precision collapse.

### 2.3 Contrast with other scenarios

| Scenario | GRU behaviour | Interpretation |
|----------|---------------|----------------|
| **Feb-13 (short GT, ~4.4% positive)** | Recall degrades gradually with \(k\); competitive at \(k=4\) | Sparse, short anomalies; different residual geometry |
| **Apr-27 (long GT, ~94.5% positive)** | Stable ~85% until \(k \ge 4\), then cliff | Dominated by sustained offset tracking |
| **Synthetic (long GT windows, flat magnetics)** | ~100% recall all \(k\) | Long label layout alone is not sufficient to cause the cliff |
| **GRU closed-loop AR on Apr-27** | ~64% @ \(k=4\) (vs ~63% open-loop) | Autoregressive roll-forward does not fix high-\(k\) misses |

Offline smoothers (EWMA / median / Savitzky–Golay **on raw magnetics**, not predictor residuals) collapse on long GT for \(k \ge 3\); neural predictors are required for this regime.

---

## 3. Detection pipeline (where GRU and LSTM differ)

Magnavis flags anomalies from **prediction residuals**, not from raw field magnitude alone:

```
magnetic CSV  →  predictor (GRU or LSTM)  →  ŷ_t
                      ↓
              residual e_t = |B_t − ŷ_t|
                      ↓
              EWMA on |e|  →  threshold T_t = μ_t + k · σ_t
                      ↓
              flag if |e_t| > T_t
```

**Implication:** Any predictor that **tracks** a sustained level shift will **shrink** \(|B - \hat B|\) in the interior of long events. A stricter threshold (larger \(k\)) then drops those seconds below \(T_t\) even though they remain GT-positive.

---

## 4. Delta-target GRU explained (plain language)

This section explains **what “delta target” means** in Magnavis and why it matters for long anomalies.

### 4.1 Two ways to predict the next field value

At each second the predictor sees recent history and forecasts the **next** scaled magnetic magnitude.

| Mode | What the network learns | How the code builds the forecast |
|------|-------------------------|----------------------------------|
| **Absolute (level)** | “The next value will be about **X**.” | Use the network output directly as \(\hat{y}_t\). |
| **Delta (Δ)** | “The next value will be about **+Δ** compared to now.” | \(\hat{y}_t = y_{t-1} + \widehat{\Delta y}_t\) |

**Example:** if the current scaled magnitude is 0.42 and the model predicts Δ = +0.01, the forecast becomes 0.43.

In `predictor_ai.py`, fresh GRU builds default to **Δ-target** when `PREDICTOR_GRU_DELTA_TARGET=1`. LSTM always uses **absolute** level.

### 4.2 Why Δ-target is used at all

Magnetic data often changes slowly. Training on **steps** (differences) can help the model focus on **local variation** instead of always outputting a number close to “whatever we just saw.” That can work well on **normal**, slowly varying data.

### 4.3 Why Δ-target hurts on long sustained anomalies

On Apr-27, manual GT marks **long plateaus** where the field stays offset for many minutes (~94.5% of seconds are GT-positive).

A Δ-trained GRU is good at **small steps**. During a plateau it can keep predicting **small positive deltas**, so the forecast **creeps upward** and partially **follows** the anomaly level.

Then the **residual** \(|B - \hat{B}|\) becomes **small in the middle** of the event—even though the field is still anomalous. The EWMA detector only fires when the residual is **large** enough (above \(\mu + k\sigma\)). Small interior residuals → **missed GT seconds**, especially when **\(k\)** is high (stricter threshold).

**Simple analogy:** instead of guessing tomorrow’s temperature directly, you guess “+1°C from today” every day. During a long heatwave you might catch up slowly; your “surprise” vs forecast stays low in the middle of the heatwave, so you under-alarm.

### 4.4 What our ablation confirmed (May 2026)

We retrained GRU on Dec-2025 data with **matched** train and inference modes, then benchmarked on Apr-27 long GT. See `results/gru_delta_ablation_restart/GRU_DELTA_ABLATION.md`.

| \(k\) | Bundled GRU | Absolute retrain | Δ-target retrain |
|------:|------------:|-----------------:|-----------------:|
| 3 | 85.4% | **84.8%** | 47.0% |
| 4 | 63.3% | **84.8%** | 4.5% |
| 5 | 63.2% | **84.8%** | 0.3% |

**Takeaways:**

- **Δ-target (train + infer):** severe cliff from \(k \ge 3\) — confirms Δ integration as the main driver when used consistently.
- **Absolute retrain (train + infer):** **flat ~85% recall** for all \(k\) — same stability as LSTM; **no cliff**.
- **Bundled GRU:** still shows a cliff at \(k \ge 4\), but milder than fresh Δ-target — likely older checkpoints and/or train–infer mismatch.

### 4.5 Practical rule of thumb

| Campaign type | Safer GRU setting |
|---------------|-------------------|
| Long sustained-offset GT (Apr-27 style) | **Absolute** target, or use **LSTM**; avoid Δ-target at \(k \ge 3\) |
| Short shot-duration GT (Feb-13 style) | Either mode can work; tune \(k\) on validation |
| Production default | **LSTM + \(k \approx 3\)** unless absolute GRU is retrained and validated |

---

## 5. Proposed mechanism (technical summary)

### 5.1 Long GT is almost entirely “anomaly time”

On Apr-27, **~94.5%** of evaluated seconds are GT-positive (8,344 of 8,825). Point-level recall therefore measures how well the pipeline flags **the bulk of the timeline during sustained offsets**, not rare short spikes. Predictor behaviour **inside long plateaus** dominates the metric.

### 5.2 GRU: delta-target integration vs LSTM: absolute level

In `predictor_ai.py`, **fresh** GRU sessions can train and predict one-step **Δ(scaled magnitude)**:

\[
\hat{y}_t = y_{t-1} + \widehat{\Delta y}_t
\]

when `PREDICTOR_GRU_DELTA_TARGET=1` (default for new GRU builds). **LSTM** uses **absolute** scaled magnitude as the supervised target.

**Why this matters on sustained offsets**

- A **delta-trained** recurrent model is well suited to **local** step changes seen in normal pretraining data.
- During a **long plateau** (sustained shot-related offset), the network can **inch** the prediction toward the new level through a sequence of small predicted deltas, especially when recent windows still resemble “normal” variation.
- That **partial tracking** reduces \(|B - \hat{B}|\) in the **middle** of long GT segments.
- **LSTM** (absolute target) tends to leave a **more persistent residual offset** on the same segments, so the same seconds remain above \(T_t\) for \(k = 2\)–\(5\).

This explains:

1. **High GRU recall at low \(k\)** — even small residual bumps exceed a loose threshold.
2. **GRU cliff at \(k \ge 4\)** — many interior seconds sit in a **marginal band** just below \(\mu + k\sigma\) when \(k\) increases.
3. **Flat LSTM recall** — residuals stay consistently above threshold across \(k\).

### 5.3 Note on bundled pretrained GRU checkpoints

Production `models/gru_pretrained_*.keras` files (April 2026 bundle) were produced by `train_gru_pretrained.py`, which supervises **absolute** magnitude and often ships **without** `*_predictor_meta.json`. At load time, missing meta implies **absolute inference** (legacy behaviour).

The bundled models still show a recall cliff at \(k \ge 4\), while **fresh absolute retrain** does not. That suggests the cliff is not only “GRU vs LSTM architecture,” but strongly tied to **Δ-target when train and inference both use it**, and possibly to **older checkpoint / training-data mismatch** for the bundled files.

### 5.4 Closed-loop autoregressive (AR) inference does not explain the cliff

**AR (autoregressive) closed-loop** means the sliding input window is rolled forward using **predicted** magnitudes after bootstrap, not only measured values (`PREDICTOR_AR_CLOSED_LOOP=1`).

On Apr-27:

| \(k\) | GRU open-loop | GRU closed-loop |
|------:|--------------:|----------------:|
| 4 | 63.3% | 64.5% |
| 5 | 63.2% | 57.1% |

Closed-loop slightly changes low-\(k\) behaviour (near-total flagging at \(k \le 2\)) but **does not restore** high-\(k\) recall. The dominant issue is **predictor–residual coupling during sustained offsets**, not AR drift alone.

---

## 6. Supporting evidence (checklist)

| Observation | Supports mechanism |
|-------------|-------------------|
| GRU precision ~94% at \(k=4\) despite low recall | Interior misses, not alarm flooding |
| ~1,800 extra FN when \(k\) goes 3 → 4 | Marginal residuals below stricter threshold |
| LSTM identical TP/FN for \(k=2\)–\(5\) | Stable residual offset on long GT |
| Synthetic long GT + flat \(B(t)\): both models ~100% | Not label geometry alone; real Apr-27 field shape matters |
| Closed-loop GRU ≈ open-loop at \(k \ge 4\) | Not fixed by AR roll-forward |
| GRU best F1 on long GT at \(k=3\) (0.896) | Operational sweet spot exists before cliff |
| Δ-target retrain: 47% @ \(k=3\), 4.5% @ \(k=4\) | Δ integration strongly worsens long-GT recall |
| Absolute retrain: flat ~85% all \(k\) | Cliff removed when level target used consistently |

---

## 7. Mitigation strategies

### 7.1 Deployment (no retraining)

| Action | Rationale |
|--------|-----------|
| Use **LSTM** for long sustained-offset campaigns | Flat ~85% recall across \(k\) |
| If GRU is retained, cap **\(k \le 3\)** on long GT | Recall ~85%, best GRU F1 at \(k=3\) |
| **Avoid GRU with \(k \ge 4\)** on Apr-27-like data | Cliff region |
| Prefer **open-loop** LSTM unless closed-loop is validated | LSTM closed-loop degrades at \(k=5\) |

Example environment:

```bash
export PREDICTOR_MODEL_FAMILY=lstm
export PREDICTOR_MODEL_INIT=pretrained
export PREDICTOR_UPDATE_TRAINING=0
export PREDICTOR_SKIP_FINETUNE_ON_SESSION=1
export MAGNAVIS_INITIAL_THRESHOLD_K=3.0
```

### 7.2 Detector-side (moderate effort)

- **Per-family \(k\):** e.g. \(k_{\mathrm{GRU}} = 3\), \(k_{\mathrm{LSTM}} = 3\)–\(4\) depending on campaign type.
- **Session routing:** short shot-duration GT (Feb-13 style) vs long offset GT (Apr-27 style) with different defaults.
- **Temporal post-processing:** morphological closing on flag sequences to bridge interior gaps when onset/offset are detected.
- **Residual statistic:** shorter EWMA \(\alpha\) or robust MAD band so \(\sigma_t\) does not adapt down during long elevated \(|e|\).

### 7.3 Model / training (strongest long-term)

1. **Use absolute-target GRU** for long campaigns (retrain with `gru_delta_y: false` in meta; `PREDICTOR_GRU_DELTA_TARGET=0` at runtime). Ablation shows flat ~85% recall across \(k\).
2. **Do not deploy Δ-target GRU** on Apr-27-style long GT at \(k \ge 3\) without revalidation.
3. **Include sustained-offset segments** in pretraining or a small labelled fine-tune set.
4. **Ensemble:** require LSTM or absolute GRU flag on long campaigns.

Checkpoints from ablation: `results/gru_delta_ablation_restart/models/absolute/` and `models/delta/`.

### 7.4 Recommended further analysis

1. **Residual timeline plot** on one Apr-27 event: \(|B - \hat{B}|\) for absolute GRU vs Δ-target GRU vs LSTM with thresholds at \(k=3\) and \(k=4\).
2. **Event-level recall** (segment-based) to confirm whole-block misses vs edge-only detection.

---

## 8. Thesis-ready wording

> On long sustained-offset ground truth, GRU with **Δ-target** training and inference shows a severe recall collapse at \(k \ge 3\) (47% at \(k=3\), 4.5% at \(k=4\)), whereas **absolute-target** retrained GRU achieves flat ~85% recall across \(k=1\)–\(5\), matching LSTM stability. The mechanism is **partial tracking of sustained level shifts**: Δ-integrated forecasts creep toward the anomaly plateau, shrinking \(|B-\hat{B}|\) in event interiors so stricter EWMA thresholds miss GT-positive seconds. Bundled pretrained GRU shows an intermediate cliff (~63% at \(k \ge 4\)). For production on long campaigns, use **LSTM or absolute-target GRU at \(k \approx 3\)**; avoid Δ-target GRU at \(k \ge 3\) on this regime.

---

## 9. Bottom line

| Question | Answer |
|----------|--------|
| Is GRU “bad on long GT”? | **No** — it is **fragile at high \(k\)** when using **Δ-target**; **absolute** retrain is stable (~85%). |
| What is delta-target GRU? | Predicts **change** per step, then adds to last value — can **track** long plateaus and shrink residuals. |
| Is the detector alone at fault? | **Unlikely** — same EWMA detector; LSTM and absolute GRU stay stable. |
| Is closed-loop AR the fix? | **No** — high-\(k\) recall stays ~63% for bundled GRU. |
| What should we deploy today? | **LSTM + \(k \approx 3\)**, or **absolute-target GRU** retrain; **not Δ-target GRU** at \(k \ge 3\) on long GT. |
| What proves the delta hypothesis? | **Completed ablation** in `gru_delta_ablation_restart/GRU_DELTA_ABLATION.md`. |

---

## References in this repository

- Full tables: `COMPARATIVE_ANALYSIS.md` (same folder)
- Figures: `k_recall_curves_five_scenarios.png`, `k_recall_apr27_open_vs_closed_loop.png`
- Predictor logic: `src/predictor_ai.py` (`PREDICTOR_GRU_DELTA_TARGET`, `PREDICTOR_AR_CLOSED_LOOP`)
- Ablation results: `results/gru_delta_ablation_restart/GRU_DELTA_ABLATION.md`
- Ablation driver: `src/benchmark_feb13_2026_improved/run_gru_delta_ablation.py`

*Last updated: May 2026 (includes GRU absolute vs Δ-target ablation).*
