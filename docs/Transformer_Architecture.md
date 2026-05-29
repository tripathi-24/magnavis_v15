# Transformer architecture (magnetic forecaster)

This document describes the **pretrained magnetic-field Transformer** shipped as Keras checkpoints (`transformer_pretrained_*.keras`) and loaded by `src/predictor_ai.py`. The **graph is defined inside the checkpoint**, not rebuilt in code; the text below is derived from the saved `config.json` inside `models/transformer_pretrained_OBS2_1.keras` (other sensors use the same architecture with per-sensor weights).

---

## Role in the project

- **Training from scratch** for this Transformer is **not** implemented in `predictor_ai.py` (`build_model` raises for `MODEL_FAMILY_TRANSFORMER`).
- At runtime, **`keras.models.load_model`** loads the checkpoint plus optional `*_scaler.pkl` next to it.
- A small custom layer **`SinusoidalPositionEncoding`** is registered in `predictor_ai.py` so Keras can deserialize checkpoints that reference `magnavis>SinusoidalPositionEncoding`.

---

## Checkpoints

| Location | Pattern |
|----------|---------|
| Default | `models/transformer_pretrained_<SENSOR>.keras` |
| Scaler | `models/transformer_pretrained_<SENSOR>_scaler.pkl` (if present) |

Examples: `transformer_pretrained_OBS2_1.keras`, `transformer_pretrained_OBS1_3.keras`.

---

## I/O summary

| Item | Value |
|------|--------|
| **Input shape** | `(batch, 15, 5)` — **15** timesteps (sequence length **W = 15**), **5** features per step |
| **Output** | `(batch, 1)` from final `Dense(1)`; after **GlobalAveragePooling1D** the head maps pooled `64` → `1` |

### Five input channels (aligned with `predictor_ai` + yearly cycle)

When `use_yearly_cycle=True` (default in `predictor_ai` `__main__` / app usage), each timestep is:

1. **Scaled total-field magnitude** (StandardScaler from training bundle)  
2. **sin** (time-of-day encoding)  
3. **cos** (time-of-day encoding)  
4. **sin** (day-of-year encoding)  
5. **cos** (day-of-year encoding)  

So the model sees a **short window of past observations and clock context**, not raw 3‑component vectors unless you change upstream feature construction.

---

## Internal stack (encoder-style, regression)

Functional Keras model. High-level data flow:

1. **Embedding**: `Dense(64, linear)` applied per timestep → **d_model = 64**.
2. **Positional encoding**: `SinusoidalPositionEncoding(sequence_length=15, depth=64)` adds a **learned** `(15, 64)` table to embeddings (see `predictor_ai.SinusoidalPositionEncoding`).
3. **Three transformer-style blocks** (each block follows the same pattern):
   - **MultiHeadAttention**: **4 heads**, **key_dim = 16**, **value_dim = 16**, **dropout = 0.1**, **attention_axes = [1]** → **self-attention over time** (length 15).
   - **Add** + **LayerNorm** (residual + normalize after attention).
   - **FFN**: `Dense(128, relu)` → `Dense(64, linear)` (position-wise feed-forward).
   - **Add** + **LayerNorm** (residual + normalize after FFN).
4. **GlobalAveragePooling1D**: average over the time axis → vector of length **64**.
5. **Regression head**: `Dense(1, linear)` → scalar prediction per batch item.

In prose: **encoder-only** stack (no decoder, no cross-attention to a second sequence), **three** self-attention blocks, **d_model 64**, **4 heads × 16-dim** keys/values, **FFN inner width 128**, then **time pooling** and a **single linear** output.

---

## Hyperparameters (from checkpoint)

| Hyperparameter | Value |
|----------------|--------|
| Sequence length | **15** |
| Feature dim | **5** |
| Model width **d_model** | **64** (after first Dense) |
| Attention heads | **4** |
| Key / value dim per head | **16** (total key size 64) |
| Attention dropout | **0.1** |
| FFN hidden | **128** (ReLU) |
| FFN output back to | **64** (linear) |
| Attention axis | **1** (time) |
| Output pooling | **Global average over time** |
| Final activation | **linear** (regression) |

---

## Relation to `predictor_ai.forecast`

- The checkpoint expects tensors shaped like **`(1, window_size, n_features)`** with `window_size` read from the loaded model (`_sync_window_size_from_model`); for these checkpoints, **window_size = 15** and **n_features = 5** with the yearly cycle on.
- After load, **`_skip_finetune_on_session`** is **true** for Transformer: session CSV is **not** used to fine-tune weights in normal use.
- Optional policies (`PREDICTOR_SKIP_INITIAL_MINUTES`, `PREDICTOR_LEADING_TRAIN_MINUTES`) change **how windows are sliced** and whether predictions are one-step on real timestamps; they do **not** change the saved layer graph.

---

## References in code

| Topic | File |
|--------|------|
| Custom positional layer | `src/predictor_ai.py` — `SinusoidalPositionEncoding` |
| Load + no fresh `build_model` for Transformer | `src/predictor_ai.py` — `load_model`, `build_model`, `forecast` |
| Checkpoint discovery | `src/app.py` — `_resolve_pretrained_model_path` |

---

## Provenance note

Training data, loss schedule, and exact training script for these checkpoints are **not** recorded in this repository snapshot; this file documents only the **exported Keras architecture** as serialized in `transformer_pretrained_*.keras` (inspected via embedded `config.json`, weights in `model.weights.h5`).
