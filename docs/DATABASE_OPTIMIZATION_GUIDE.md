# Database Optimization Guide (Current-State Version)

This guide now documents **what to verify** in current code, instead of assuming optimizations are still pending.

Canonical script doc:
- `docs/scripts/src_data_convert_db_now.md`

## What to check in `src/data_convert_db_now.py`

1. Connection strategy and buffering behavior.
2. Batched sensor/timestamp query pattern (avoid per-sensor serial max queries where possible).
3. Query limit multipliers for initial window and incremental fetches.
4. ID-window and time-range boundaries used in SQL filters.
5. Index assumptions on DB table.

## Practical Validation Workflow

```bash
python -m py_compile src/data_convert_db_now.py
python src/app.py
```

Then validate:
- Initial historical load latency for multi-sensor selection.
- Incremental update latency per fetch cycle.
- SQL load profile on DB server.

## If runtime is still slow

- Add/verify composite index on `(sensor_id, timestamp)`.
- Reduce fetched row caps and window expansions where safe.
- Profile query times by function path used in app mode (real-time vs simulation vs CSV).
- Move DB credentials/configuration to environment-based config and tune connection settings per deployment.

For exact function-level behavior and call paths, use the canonical script doc above.

