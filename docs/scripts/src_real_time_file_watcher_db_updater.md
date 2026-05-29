# Script Doc: `src/real_time_file_watcher_db_updater.py`

## Purpose

Continuously monitor a growing sensor CSV and insert new rows into the MySQL magnetic data table.

## How to run

```bash
python src/real_time_file_watcher_db_updater.py
```

## Inputs

- A recent CSV file matching expected naming pattern.
- CSV columns including timestamp, magnetic components, orientation, location, and sensor ID.

## Outputs

- Inserted DB rows in magnetic data table.
- Runtime logs in `csv_db_updater.log`.

## Main functionality

1. Waits for/locates active sensor CSV file.
2. Reads only appended rows since last poll.
3. Parses and validates rows.
4. Batch inserts new records into DB.
5. Repeats continuously at polling interval.

## Caveats

- Embedded DB credentials should be externalized for secure deployment.
- Parsing assumes stable CSV structure.
- File-tail logic can be sensitive to partial writes.

## Example usage

Use alongside acquisition scripts that continuously append to a local CSV and need near-real-time DB mirroring.

