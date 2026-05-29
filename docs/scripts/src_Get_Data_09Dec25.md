# Script Doc: `src/Get_Data_09Dec25.py`

## Purpose

`Get_Data_09Dec25.py` is a Streamlit application for fetching large magnetic datasets from MySQL by time range and sensor filters, with export and progress-oriented UX.

## How to run

```bash
streamlit run src/Get_Data_09Dec25.py
```

## Primary use case

Use this tool when you need to extract historical raw magnetic rows for offline analysis, model training, or archival export.

## Inputs

### UI inputs

- Start datetime
- End datetime
- Optional sensor subset
- Column selection
- Optional downsampling factor

### Backend dependencies

- MySQL connectivity (`DB_CONFIG`)
- Table expected: `qnav_magneticdatamodel`

## Outputs

- Data table preview inside Streamlit
- Downloadable CSV/Excel-style exports (through UI actions)
- Runtime logs in `get_data.log`

## Main functionality flow

1. Initialize Streamlit page and session state.
2. Resolve available sensors from DB.
3. Compute approximate ID range for requested time period.
4. Fetch data using windowed/chunked queries with retry logic.
5. Optionally parallelize chunk fetching.
6. Merge/clean chunk results and return final dataframe.
7. Display summary metrics and provide export actions.

## Key functions (high-level)

- Connection helpers:
  - `get_connection_pool()`
  - `get_connection()`
- Metadata helpers:
  - `get_available_sensors()`
  - `get_data_range()`
- Range/chunk utilities:
  - `find_id_for_timestamp(...)`
  - `find_id_range_for_time_period(...)`
  - `fetch_single_window(...)`
  - `fetch_chunk(...)`
- Main fetch orchestrator:
  - `fetch_data(...)`
- Streamlit app entry:
  - `main()`

## Important constants and configuration

- `DB_CONFIG`, `POOL_CONFIG`
- `CHUNK_SIZE`
- `MAX_ROWS_PER_WINDOW`
- `ID_CHUNK_SIZE`
- `MAX_PARALLEL_WORKERS`
- `USE_PARALLEL_FETCHING`
- `LONG_NET_TIMEOUT`

## Failure modes and troubleshooting

- DB timeout or disconnect:
  - reduce time window
  - reduce chunk size / parallel workers
  - verify DB-side timeout and index performance
- Empty result set:
  - verify time interval and sensor filter
  - verify selected columns exist
- Slow fetch:
  - disable/enable parallel mode based on DB behavior
  - use narrower ID/time ranges and staged exports

## Security note

Current script contains inline DB credentials/config. For production, externalize secrets via environment variables or a secure secrets manager.

## Example workflows

### Example A: full observatory export for one hour

1. Launch Streamlit app.
2. Choose start/end over a 60-minute interval.
3. Leave sensor filter empty (all sensors).
4. Fetch and export dataset.

### Example B: focused OBS2 export for offline training

1. Select only OBS2 sensors.
2. Choose required columns (`sensor_id`, `timestamp`, `b_x`, `b_y`, `b_z`, orientation fields).
3. Fetch, validate row counts, export CSV for `train_gru_pretrained.py` / `train_lstm_pretrained.py` or benchmark replay.

