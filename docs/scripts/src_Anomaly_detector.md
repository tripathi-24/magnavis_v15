# Script Doc: `src/Anomaly_detector.py`

## Purpose

Detect anomalies from prediction residuals (`|actual - predicted|`) using adaptive EWMA-based thresholding.

## How to use

Imported as a module from app scripts.

Typical flow:
1. Create `AnomalyDetector(...)`.
2. Call `detect_anomalies(actual_time, actual_values, pred_time, pred_values)`.
3. Read anomaly flags/statistics.

## Runtime role

- Core anomaly logic for `src/app.py` (`ApplicationTemp`; launched via `src/application_temp.py` shim).
- Converts predictor output quality into operational anomaly markers.
- Maintains adaptive threshold memory so behavior is robust to non-stationary error levels.

## Inputs

- Actual timestamps and values.
- Predicted timestamps and values (can be on different grid; interpolation is handled).
- Detector hyperparameters (threshold multiplier, alpha, calibration settings).

## Outputs

- Anomaly labels per point (and related metrics/statistics).
- Internal EWMA stats available through helper methods.

## Key methods

- `calculate_differences(...)`:
  - aligns series and computes residuals
- `detect_anomalies(...)`:
  - computes adaptive threshold and labels anomaly points
- `get_statistics()`:
  - exposes summary diagnostics for monitoring/tuning

## Thresholding behavior

- Residual is `abs(actual - predicted)` after timeline alignment.
- EWMA tracks central tendency and dispersion.
- Adaptive threshold is derived from EWMA state and multiplier.
- Initial calibration period prevents unstable early flagging.

## Main functionality

1. Aligns prediction to actual timeline by interpolation.
2. Computes residual error series.
3. Updates EWMA mean/variance and robust threshold.
4. Flags points crossing adaptive threshold.

## Parameter tuning guidance

- Increase threshold multiplier for fewer false alarms.
- Decrease threshold multiplier for higher sensitivity.
- Increase EWMA memory (higher alpha) for slower adaptation.
- Decrease EWMA memory for faster response to changing baseline error.

## Caveats

- Needs enough initial non-anomalous points for stable calibration.
- Poor predictor quality inflates residual baseline and can reduce anomaly sensitivity.
- Timestamp overlap and ordering must be valid.

## Failure modes and troubleshooting

- Too many false positives:
  - increase threshold multiplier
  - verify predictor quality and data continuity
- Missed anomalies:
  - decrease threshold multiplier
  - confirm predictor horizon covers actual anomalies
- Unstable threshold:
  - increase calibration period
  - inspect residual spikes from bad input alignment

## Example usage

Used by `src/app.py` after `predict_out.csv` is generated (primary path: `calculate_differences` from `detect_anomalies_for_sensor`).

Example snippet:

```python
detector = AnomalyDetector(threshold_multiplier=2.5)
result = detector.detect_anomalies(actual_times, actual_values, pred_times, pred_values)
```

