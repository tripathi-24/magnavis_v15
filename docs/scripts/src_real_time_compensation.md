# Script Doc: `src/real_time_compensation.py`

## Purpose

Acquire live serial magnetometer data, visualize it in real-time, and persist data to CSV (with optional API/DB integrations).

## How to run

```bash
python src/real_time_compensation.py
```

## Inputs

- Serial port + baudrate selected in GUI.
- Incoming sensor lines containing magnetic, GPS/location, and orientation values.

## Outputs

- Live pyqtgraph traces for magnetic channels.
- Session CSV file with parsed sensor records.
- Optional upstream API/DB data forwarding paths (depending on enabled runtime sections).

## Main functionality

1. Opens configured serial connection.
2. Parses incoming records into structured values.
3. Updates in-memory buffers and live plots.
4. Writes rows to CSV log.
5. Optionally posts to web API / DB writer pipeline.

## Caveats

- Includes hardcoded integration paths that may require local customization.
- Contains embedded DB/API config placeholders and environment-specific assumptions.
- Validate serial payload format before field deployment.

## Example usage

Use for lab-side live data capture and quick signal quality inspection before pushing streams into central DB workflows.

