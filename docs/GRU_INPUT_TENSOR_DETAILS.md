# Input Tensor Details (`X: (B, 15, 5)`)

This note explains the standard sequence tensor shape used by the predictor pipeline.

Canonical docs:
- `docs/scripts/src_predictor_ai.md`
- `docs/scripts/src_application_temp.md`

## Meaning

- `B`: batch size (number of windows)
- `15`: timesteps per window (`window_size=15`)
- `5`: per-timestep features

Typical feature vector:
- scaled magnitude
- daily sin/cos cycle
- yearly sin/cos cycle (when enabled)

So each sample is a `(15, 5)` matrix; training batches are `(B, 15, 5)`.

## Example

- Training batch with 64 windows: `X.shape = (64, 15, 5)`
- Single latest-window inference: `X.shape = (1, 15, 5)`

For exact feature construction and environment controls, use the canonical docs.

