# Attention Bi-LSTM architecture (magnetic forecaster)

This document describes the **Attention Bi-LSTM** sequence model used for magnetic-field prediction in `src/predictor_ai.py` when `PREDICTOR_MODEL_FAMILY=attn_bilstm` (or aliases such as `attention_bilstm`). The graph is **built in code** in `AttnBiLSTMPredictor.build_model` and **trained from scratch** on session data (the application sets this family to **fresh** init; there is no bundled pretrained checkpoint for this path).

---

## Role in the project

- Selected in `src/app.py` as **“Attention Bi-LSTM (fresh)”**; weights are learned during the predictor run, not loaded from a fixed checkpoint.
- Trained with **mean squared error** and the **Adam** optimiser (`learning_rate` from `_parse_learning_rate`, default **0.001** for non-GRU families).
- After training, the same **`forecast`** pipeline as other families produces predictions for anomaly detection (residuals vs actuals in `AnomalyDetector`).

---

## I/O summary

| Item | Value |
|------|--------|
| **Input shape** | `(batch, W, F)` — **`W`** = `window_size` from `PREDICTOR_GRU_WINDOW_SIZE` (clamped **5–3600**; benchmarks often use **15**). **`F`** = `feature_dim` from the built feature matrix (typically **5** when yearly cycle is on). |
| **Output shape** | `(batch, 1)` — one scalar regression output per window (absolute scaled magnitude target; inverted to physical units in `forecast`). |

### Typical five input channels (`use_yearly_cycle=True`)

Per timestep, the same encoding as other predictors in this project:

1. **Scaled total-field magnitude** (`StandardScaler` on the magnitude column)  
2. **sin** (time-of-day)  
3. **cos** (time-of-day)  
4. **sin** (day-of-year)  
5. **cos** (day-of-year)  

---

## Layer stack (functional Keras `Model`)

Implementation order in `build_model` for `MODEL_FAMILY_ATTN_BILSTM`:

| Step | Layer | Purpose |
|------|--------|---------|
| 1 | `Input(shape=(W, F))` | Sequence window. |
| 2 | `Attention()([inputs, inputs])` | **Self-attention**: query and value are the same tensor so each time step attends over the **time** axis of the window. |
| 3 | `LayerNormalization` | Normalise the attended sequence. |
| 4 | `Bidirectional(LSTM(16, return_sequences=True))` | **16 units per direction** → **32** channels per timestep, sequence kept for pooling. |
| 5 | `Dense(32)` on `norm_out` | Linear **projection** of the normalised attention output to width 32 for the residual. |
| 6 | `Add` + `Activation("tanh")` | **Residual**: Bi-LSTM output **plus** projected `norm_out`, then **tanh**. |
| 7 | `GlobalAveragePooling1D` | Collapse time → single vector of length **32**. |
| 8 | `Dense(16, activation="relu")` | Small MLP head. |
| 9 | `Dense(1)` | Linear regression output. |

In short: **self-attention → layer norm → bidirectional LSTM → residual skip from the norm stream → tanh → global average over time → dense head**.

---

## Hyperparameters (fixed in code)

| Hyperparameter | Value |
|----------------|--------|
| LSTM units (per direction) | **16** (Bi-LSTM merged width **32**) |
| Attention | Keras **`Attention`** layer, self-attention (`[x, x]`) |
| Post-attention FFN in head | **16** hidden units (ReLU) before scalar output |
| Residual projection width | **32** (matches Bi-LSTM concatenated width) |
| Pooling | **Global average** over time |
| Loss | **MSE** |
| Optimiser | **Adam** |

`window_size` and effective training horizon are **not** hard-coded here: they follow **`PREDICTOR_GRU_WINDOW_SIZE`**, **`TRAIN_WINDOW_MINUTES`**, and optional **`PREDICTOR_LEADING_TRAIN_MINUTES`** / **`PREDICTOR_SKIP_INITIAL_MINUTES`** policies in `forecast`, same as for LSTM/GRU.

---

## Code reference

The full graph is defined in:

`src/predictor_ai.py` — class `AttnBiLSTMPredictor`, method **`build_model`**, branch `elif self.model_family == MODEL_FAMILY_ATTN_BILSTM`.

---

## Relation to naming in benchmarks

Benchmark tables may label this row **“Attention Bi-LSTM (fresh)”** or **`attention_bi_lstm`** / **`attn_bilstm_fresh`** depending on the driver; all refer to this same `build_model` implementation.
