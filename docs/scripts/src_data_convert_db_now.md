# Script Doc: `src/data_convert_db_now.py`

## Purpose

Provide all DB-backed ingestion utilities used by the DB/CSV runtime.

This script is the canonical source for:
- sensor discovery
- historic window fetch
- incremental fetch
- scalar and vector multi-sensor dataframe assembly

## How it is used

Imported by `src/app.py` (`ApplicationTemp`; `src/application_temp.py` is a thin launcher) for:
- latest sensor discovery
- historical window fetch
- incremental realtime fetch
- scalar magnetic series assembly (`time_H`, `mag_H_nT` from \(\|B\|\))

Vector fetch helpers (`fetch_vector_*`, `get_vector_data_*`) remain available for standalone analysis or future use; the main app currently ingests **scalar magnitude** only.

## Inputs

- DB configuration and live connection availability.
- Sensor IDs (single or list).
- Time bounds (`start_time`, `end_time`) and target window size hints.
- Row-limit / expansion controls in specific fetch paths.

## Outputs

- Dataframes for scalar series (usually `time_H`, `mag_H_nT`).
- Dataframes for vector series (`b_x`, `b_y`, `b_z`, optional orientation fields) when vector APIs are called.
- Empty but schema-consistent frames when no rows are found.

## Main functionality

1. Creates MySQL connections.
2. Executes optimized time-window and sensor-wise queries.
3. Converts DB rows to app-consumable dataframes.
4. Provides helpers for both scalar magnitude and full vector access paths.

## Key function groups

### Connection and guards

- connection creation / timeout setup
- mysql availability checks
- session timeout hardening

### Sensor discovery helpers

- latest sensor IDs / filtered IDs retrieval

### Scalar series helpers

- full window fetch for one/many sensors
- incremental fetch since timestamp
- bounded-between fetch for simulation slices

### Vector series helpers

- vector-equivalent variants of scalar helpers (multi-sensor `b_x`, `b_y`, `b_z`, optional `theta_*`)

## Data assumptions

- Table name and schema are fixed to expected project DB model (`qnav_magneticdatamodel`).
- Timestamp fields are parseable and sortable.
- Sensor IDs are stable strings used consistently across runtime sessions.
- Scalar `mag_H_nT` is total-field magnitude \(\sqrt{b_x^2 + b_y^2 + b_z^2}\).

## Query behavior notes

- Functions use caps and heuristics to avoid extremely large single pulls.
- Window and expansion logic is designed to recover enough rows for plotting/prediction without exhausting DB.
- Some functions may trade strict exactness for practical responsiveness in high-volume tables.

## Caveats

- DB config/credentials are embedded in code today; move to secure env management for production.
- Query performance depends on table indexes and row volume.
- Schema assumptions must match `qnav_magneticdatamodel`.

## Failure modes and troubleshooting

- Empty responses for known active sensors:
  - verify timestamp timezone handling
  - verify sensor ID exact values
- Slow fetch in multi-sensor mode:
  - reduce window size and limit rows
  - confirm DB indexes on `sensor_id`, `timestamp`, and ID path
- Intermittent DB connection failures:
  - validate timeout settings and network stability
  - use retry-safe caller logic in app layer

## Practical examples

### Example A: initial multi-sensor history

Caller requests a one-hour equivalent target window for selected sensors; function returns per-sensor scalar frames for blue baseline rendering.

### Example B: incremental realtime loop

Caller passes last-seen timestamp and receives only new rows to append into green realtime buffers.

### Example C: vector export (standalone)

Call `get_vector_data_multi` or `fetch_vector_between_multi` when you need full 3-axis components for offline analysis (not required by the default `app.py` UI path).

## Example usage

`src/app.py` calls multi-sensor **scalar** fetch functions repeatedly to fill:
- blue historical segment
- green realtime segment
