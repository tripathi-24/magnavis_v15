# Script Doc: `tools/evaluate_anomaly_detection.py`

## Purpose

Evaluate anomaly detection quality by comparing anomaly timestamps from app logs against ground-truth event windows.

## How to run

```bash
python tools/evaluate_anomaly_detection.py --log-file src/sessions/<session_id>/app.log
```

## Runtime role

- Offline evaluation utility for detection quality assurance.
- Converts app logs + experiment schedule into reproducible metrics and plots.
- Used for model/protocol comparison and thesis/report evidence.

Common optional flags:

```bash
python tools/evaluate_anomaly_detection.py \
  --log-file src/sessions/<session_id>/app.log \
  --experiment-file Experiment_Data.csv \
  --sensor OBS2_1 \
  --base-date 2026-02-22 \
  --out-dir models/anomaly_eval \
  --prefix run_obs2_1
```

## Inputs

- App log file containing anomaly lines.
- Ground-truth schedule CSV (`Experiment_Data.csv` by default).
- Optional sensor/date/tolerance configuration flags.

## Output files

- `<prefix>_summary.json` (headline metrics)
- `<prefix>_point_metrics.csv` (timeline confusion metrics)
- `<prefix>_event_metrics.csv` (event overlap metrics)
- `<prefix>_metrics_plot.png` (quick visual summary)

## Main processing flow

1. Parse anomaly timestamps from log lines.
2. Build ground-truth intervals from schedule CSV.
3. Construct timeline grid and compute point-level confusion statistics.
4. Merge predicted anomaly points into events and compute event-level overlap metrics.
5. Write JSON/CSV outputs and render metrics figure.

## Important CLI flags (practical meaning)

- `--log-file`: required anomaly log source
- `--experiment-file`: GT schedule source
- `--sensor`: evaluate one sensor stream
- `--base-date`: explicit date anchor for HHMM GT values
- `--out-dir`: output directory
- `--prefix`: output filename prefix
- `--prediction-sensor-mode`:
  - **`union_all`** (default): a timeline second counts as **predicted positive** if **any** sensor logged an anomaly at that time (OR across sensors). Use this when the experiment is treated as one fused detection surface.
  - **`filter`**: restrict predicted anomalies to the sensor given by **`--sensor`** only (legacy single-stream behavior).

## Metric interpretation

- Point-level metrics answer: "How often is each second correctly labeled?"
- Event-level metrics answer: "How well do detected anomaly episodes overlap GT episodes?"
- Use both; point metrics alone can hide episode fragmentation errors.

## Canonical and alias note

- Canonical evaluator: `tools/evaluate_anomaly_detection.py`.
- Alias path: `tools/Anomaly_Detection_Evaluation/evaluate_anomaly_detection.py`.
- Use canonical script unless you explicitly override all input/output paths.

## Caveats

- If base date cannot be inferred robustly, pass `--base-date`.
- Log format must match parser regex patterns.
- When no predicted anomalies are found, some derived metrics may degenerate; inspect summary JSON details.
- Ensure log and GT period actually overlap before comparing scores.

## Troubleshooting

- Script runs but outputs empty metrics:
  - verify anomaly lines exist in log
  - verify correct sensor filter and base date
- Unexpectedly low recall:
  - confirm GT intervals match actual experiment session period
  - check tolerance and timestamp alignment assumptions
- Alias script writes to unexpected path:
  - use canonical script or pass explicit `--out-dir`

