# Script Doc: `src/data_convert_now.py`

## Purpose

Fetch USGS magnetic data and convert it into a dataframe format expected by the base app.

## How to run

```bash
python src/data_convert_now.py
```

This runs a sample pull and writes a test output file.

## Runtime role

- Canonical ingestion adapter for the base `application.py` path.
- Encapsulates HTTP request, JSON persistence, and dataframe normalization.

## Inputs

- Fetch parameters: hours, start/end time, optional session ID.
- USGS endpoint availability and response format.

## Outputs

- Dataframes with timestamp and magnetic value columns.
- Downloaded JSON snapshots in session or local paths.

## Main functions

- `download_mag_data_file(...)`:
  - builds API request
  - executes robust HTTP call
  - writes JSON file to session path
- `get_timeseries_magnetic_data(...)`:
  - reads the downloaded payload
  - parses metadata + value rows
  - emits normalized dataframe used by app plotting logic

## Main functionality

1. Builds USGS query URL and performs HTTP requests.
2. Parses JSON payload.
3. Creates normalized dataframe columns for downstream plotting.
4. Returns data for app consumption.

## Data contract

- Output dataframe is expected to contain timestamp and scalar magnetic columns.
- Consumer code assumes sortable timestamps and numeric magnetic values.
- Column naming is derived from orientation metadata and normalized consistently for caller use.

## Caveats

- Requires network connectivity.
- External API downtime/rate limits can affect app behavior.
- Station/element assumptions are hardcoded unless script is modified.

## Failure modes and troubleshooting

- HTTP request failures:
  - verify internet access
  - retry with smaller windows
- Empty or malformed payload:
  - inspect saved JSON artifact
  - confirm station/element endpoint still returns expected schema
- Parsing issues:
  - verify metadata orientation key and value arrays
  - handle missing/null rows in payload

## Example usage

`application.py` calls `get_timeseries_magnetic_data(...)` to populate time-series tracks.

Direct call example:

```python
from data_convert_now import get_timeseries_magnetic_data
df = get_timeseries_magnetic_data(session_id="demo_session", hours=1)
```

