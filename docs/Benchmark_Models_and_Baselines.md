# Benchmark models and statistical baselines

This note documents how **Magnavis** compares **offline statistical baselines** (EWMA, rolling median, Savitzky–Golay) with **deep-learning predictors** (LSTM, Attention Bi-LSTM, Transformer, GRU) in the Feb 13 benchmark harnesses. It consolidates the intended behaviour of each scheme and how scores are produced.

## Where this lives in the repo

| Piece | Location |
|--------|-----------|
| Offline baselines (EWMA / median / Savgol) | `src/benchmark_feb13_2026_improved/offline_statistical_baselines.py` (and sibling `src/benchmark_feb13_2026/offline_statistical_baselines.py`) |
| Full benchmark driver | `src/benchmark_feb13_2026/run_suite.py` |
| “Improved” subset (deep RNN, aligned skip) | `src/benchmark_feb13_2026_improved/run_suite_improved.py` (invoked via `run_suite.py` in that folder) |
| Residuals, adaptive threshold, anomaly flags | `src/Anomaly_detector.py` — `AnomalyDetector.calculate_differences` |
| Neural architectures and training/inference | `src/predictor_ai.py` |
| Evaluation vs manual GT | `tools/evaluate_anomaly_detection.py` |

Architecture detail for specific nets:

- `docs/Attention_BiLSTM_Architecture.md`
- `docs/Transformer_Architecture.md`
- `docs/GRU_PRETRAINED_MODELS.md` (pretrained GRU assets)

---

## What is being compared (shared pipeline)

1. **Target signal** — For each sensor row in the magnetic CSV, the scalar **total-field magnitude** is  
   \(\|B\| = \sqrt{b_x^2 + b_y^2 + b_z^2}\) (nanoTesla). See `_mag_series_from_export` in `offline_statistical_baselines.py`.

2. **Forecast** — Each method produces a **sequence of predicted** \(\|B\|\) values aligned in time with the actuals (offline baselines on a 1 Hz-style grid from the export; deep models via the headless app path).

3. **Anomaly detection (same for every method)** — Actuals and predictions are passed to **`AnomalyDetector.calculate_differences`**, which interpolates predictions onto actual timestamps, forms **errors** (residuals), maintains **error history**, and applies the **same adaptive thresholding** (including multiplier `k`, EWMA on absolute errors where appropriate, etc.). Rows with `is_anomaly` are turned into synthetic `app.log` lines for offline baselines; the app does the same for live/deep predictors.

4. **Evaluation** — `tools/evaluate_anomaly_detection.py` scores those logs against manual ground-truth intervals (Feb 13 benchmark), producing metrics that feed **`comparison_table.csv`** / performance matrices.

So a row labelled **“median”** or **“gru_fresh”** means: **that forecaster for \(\|B\|\)** plus **the identical residual-based anomaly pipeline**, not two different detectors.

---

## Aligning with “historic load” (skip head)

The **improved** offline path supports **`--skip-initial-minutes`** (default **62** in that benchmark): all samples before `first_timestamp + N minutes` are dropped **per sensor** so baselines align with “after historic load” replay. Deep runs can mirror this via **`PREDICTOR_LEADING_TRAIN_MINUTES`** (fresh LSTM/GRU/Attention Bi-LSTM: train only on the first segment, one-step inference on the rest) or **`PREDICTOR_SKIP_INITIAL_MINUTES`** (pretrained / transformer: skip prediction on that head). See `_predictor_initial_split_env` in `run_suite_improved.py`.

---

## Offline baseline: EWMA

**Idea:** Exponentially weighted moving average (EWMA) of past \(\|B\|\); the value used as the **prediction at time \(t\)** is the EWMA level **as of \(t-1\)** (strictly causal).

**Implementation:**

```python
# Conceptually: level_t = EWMA(|B|); pred_t = level_{t-1}; back-fill first step
s = pd.Series(y, dtype=float)
lvl = s.ewm(alpha=0.35, adjust=False).mean()
pred = lvl.shift(1).bfill()
```

- **`alpha=0.35`** — Higher \(\alpha\) reacts faster to new samples.
- **`shift(1)`** — No peeking: prediction at \(t\) does not include \(y_t\) in the EWMA that defines it.
- **`bfill()`** — Fills the initial gap after shifting so every timestamp has a prediction for the detector.

**Tiny example (conceptual):** If \(\|B\|\) has been flat near 100 and then steps to 110, the EWMA level rises gradually; **`pred[t]`** lags by one step and smooths the jump compared to raw \(\|B\|\).

---

## Offline baseline: rolling **median**

**Idea:** At each time, predict \(\|B\|\) with the **median** of the last **W** magnitude samples **ending at \(t-1\)** (causal). Median is **robust** to short spikes inside the window: one wild point does not move the forecast as much as a mean would.

**Implementation (default `window=31`):**

```python
s = pd.Series(y, dtype=float)
pred = s.rolling(window=max(3, int(window)), min_periods=1).median().shift(1)
pred = pred.bfill()
```

- **`rolling(..., min_periods=1).median()`** — At index \(t\), median of the last `window` values **through \(t\)** (pandas convention before shift).
- **`shift(1)`** — Prediction at \(t\) uses information only through \(t-1\) in the sense above (causal one-step-ahead style baseline).
- **`bfill()`** — Same edge treatment as EWMA.

**Tiny example:** Window 5, last four values before \(t\) were \([100, 102, 101, 99, 200]\). Mean is inflated by 200; **median** of sorted \([99,100,101,102,200]\) is **101**. That median becomes part of the shifted forecast used for residuals at the next step boundary (see code for exact index alignment).

---

## Offline baseline: Savitzky–Golay (Savgol)

**Idea:** **Savitzky–Golay** smoothing fits a **low-order polynomial** locally in a **sliding odd-length window** and evaluates it at the **window centre**. Default parameters in code: **`window_length=31`**, **`polyorder=3`**, **`mode="interp"`** (SciPy) for edge handling.

**Implementation:**

```python
sm = savgol_filter(y, window_length=wl, polyorder=polyorder, mode="interp")
pred = np.roll(sm, 1)
pred[0] = sm[0]
```

- **`sm[t]`** — Smoothed magnitude at \(t\) from a **centred** local polynomial (SciPy’s filter uses past and future samples inside the window around \(t\)).
- **`np.roll(sm, 1)`** with **`pred[0] = sm[0]`** — Sets **`pred[t] = sm[t-1]`** for \(t \ge 1\): a **one-sample lag** on the smooth series so the baseline is not literally “use `sm[t]` as the forecast at \(t\)`”.

**Worked toy example** (smaller window for readability; production uses 31 and order 3):

Let \(y = [100, 101, 100, 105, 104, 103, 102, 101.5]\). With **`window_length=5`**, **`polyorder=2`**, **`mode="interp"`**, SciPy might produce something like:

```text
y:    [100.0, 101.0, 100.0, 105.0, 104.0, 103.0, 102.0, 101.5]
sm:   [ 99.89, 100.66, 101.71, 103.46, 104.51, 102.96, 102.13, 101.44]
pred: [ 99.89,  99.89, 100.66, 101.71, 103.46, 104.51, 102.96, 102.13]
```

At **\(t=4\)**: actual \(y_4=104\), **`pred_4 = sm_3 \approx 103.46`**, residual \(\approx 0.54\). The smooth curve **follows the bump** more than a median over only past data might at the same index (median stays flatter until more of the bump is inside its **past-only** window).

**Important caveat for papers:** SciPy’s Savitzky–Golay output at each \(t\) is **centred** on \(t\) (it uses samples **after** \(t\) inside half the window). The **`roll(1)`** adds a one-step delay but does **not** make the filter **strictly causal** in the same sense as **`rolling.median().shift(1)`**. When describing benchmarks, it is accurate to call Savgol a **lagged local-polynomial smooth baseline**, not an identical “past-only” construction to the median baseline.

---

## Deep-learning predictors (benchmark order and env)

The **full** Feb 13 suite (`src/benchmark_feb13_2026/run_suite.py`) runs headless **`app.py`** steps in this order (after offline EWMA → median → Savgol):

1. **Vanilla LSTM (fresh)** — `PREDICTOR_MODEL_FAMILY=lstm`, `PREDICTOR_MODEL_INIT=fresh`, window **W=15** (default there).
2. **Attention Bi-LSTM (fresh)** — `PREDICTOR_MODEL_FAMILY=attn_bilstm`, `PREDICTOR_MODEL_INIT=fresh`.
3. **Transformer (pretrained)** — `PREDICTOR_MODEL_FAMILY=transformer`, `PREDICTOR_MODEL_INIT=pretrained` (checkpoint-based forecaster; table label may appear as legacy stem in older runs).
4. **GRU (fresh)** — `PREDICTOR_MODEL_FAMILY=gru`, `PREDICTOR_MODEL_INIT=fresh`, **W=15**.
5. **GRU (pretrained)** — `PREDICTOR_MODEL_FAMILY=gru`, `PREDICTOR_MODEL_INIT=pretrained`.

The **improved** suite (`run_suite_improved.py`) keeps offline baselines (with skip-initial alignment) but only runs **deep LSTM fresh** and **deep GRU fresh**, both with **W=30** and **`MAGNAVIS_DEEP_RNN_BENCHMARK=1`** (8 recurrent + 4 dense layers in `predictor_ai.py` for that profile).

**At a high level:**

- **LSTM / GRU (fresh)** — Sequence models trained on session (or leading-train) data; predict next-step (or aligned) \(\|B\|\) from recent history.
- **Attention Bi-LSTM** — Attention over time → LayerNorm → Bi-LSTM → residual → GAP → head (see architecture doc).
- **Transformer (pretrained)** — Loads a fixed checkpoint; session fine-tuning behaviour is constrained (see `predictor_ai.py` and Transformer doc).

All of these still feed **`AnomalyDetector.calculate_differences`** in the app so comparison to baselines is on a **common footing** (same threshold machinery given the same `k` and detector parameters).

---

## Detector parameters in offline baselines

Offline `_one_sensor_anomaly_lines` constructs:

```python
AnomalyDetector(
    threshold_multiplier=float(k),
    min_samples_for_threshold=20,
    std_relative_floor=0.02,
)
```

Match these to your GUI / batch **`k`** when interpreting tables.

---

## Quick comparison table (forecasting style)

| Method | Forecast of \(\|B\|\) | Causal “past only” for the value used at \(t\)? |
|--------|-------------------------|--------------------------------------------------|
| EWMA | Exponentially smoothed level, lagged by 1 | Yes (after `shift(1)`) |
| Median | Rolling median of magnitudes, lagged by 1 | Yes (after `shift(1)`) |
| Savgol | Lagged centred polynomial smooth | **No** — centred SG + 1-step lag (see caveat above) |
| Deep nets | Learned mapping from recent window / architecture | Depends on training/inference policy (leading train vs full-series, etc.); see `predictor_ai.py` and benchmark env) |

---

## See also

- `src/benchmark_feb13_2026/README.md` — default CSV cap, sensors, `k`, historic minutes.
- `src/benchmark_feb13_2026_improved/README.md` — improved subset and skip alignment.
- `docs/scripts/src_predictor_ai.md` — generated/script-oriented notes for `predictor_ai.py` if present in your tree.
