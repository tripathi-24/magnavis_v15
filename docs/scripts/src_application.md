# Script Doc: `src/application.py`

## Purpose

Main Qt desktop runtime for the original USGS-oriented Magnavis workflow.  
It owns the full heavy UI shell (plots + table + spatial view), session lifecycle, threaded fetch, and plot-refresh orchestration.

## How to run

```bash
python src/application.py
```

## Runtime position in the project

- Entry point for the base app path.
- Uses `src/data_convert_now.py` for USGS ingestion.
- Provides framework reused by temp variants.
- Not the primary DB production path (that is `src/app.py`).

## Inputs

- GUI interactions (menus, source selection, plot actions).
- Optional uploaded CSV files for visualization.
- USGS timeseries fetched through `get_timeseries_magnetic_data(...)`.

## Outputs

- Live GUI outputs:
  - time-series plots
  - map/spatial views
  - table preview
- Session artifacts in `src/sessions/<session_id>/`:
  - downloaded JSON snapshots
  - predictor input/output files (where enabled by active path)
  - subprocess logs

## Main functionality

1. Bootstraps `QApplication` and loads `.ui` windows.
2. Creates session context and data managers.
3. Starts thread-based fetch operations.
4. Receives/merges data and updates plot state.
5. Manages periodic timers for data and rendering refresh.
6. Handles file upload + spatial rendering path.

## Important classes/components

- `Application` (Qt app orchestration)
- `ApplicationWindow` (main window and widget wiring)
- session/data worker classes for threaded updates
- plot framework setup methods for static/dynamic visualization

## Operational assumptions

- PyQt and heavy plotting stack installed and working.
- Network access available for USGS calls.
- UI event loop remains responsive while worker threads fetch data.

## Key dependencies and caveats

- Heavy dependency surface: PyQt5, matplotlib, VTK, geospatial libs.
- USGS network failures directly impact fetch operations.
- For DB/CSV and model-family selection behavior, use `src/app.py`.

## Failure modes and troubleshooting

- App opens but no timeseries:
  - verify network and USGS endpoint
  - inspect fetch/log widgets
- UI freezes on heavy operations:
  - confirm threaded workers are active
  - reduce data window and plotting load
- Missing visualization modules:
  - install optional UI/geo dependencies from project requirements

## Example usage scenario

Use `src/application.py` when validating baseline UI behavior and USGS ingestion before moving to DB/CSV workflows.

