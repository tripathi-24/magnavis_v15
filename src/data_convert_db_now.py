"""
Database-backed magnetic time-series ingestion for Magnavis.

This module is meant to be a drop-in replacement for `data_convert_now.py`:
- It exposes `get_timeseries_magnetic_data(...)` with a compatible signature.
- It returns a DataFrame with the same column names expected by `application.py`:
  - `time_H`
  - `mag_H_nT`

Data source:
- MySQL table: `qnav_magneticdatamodel`
- Credentials/host are taken from the reference script `Get_Data_09Dec25.py`.

Notes:
- The database schema (as seen in the provided CSV) includes vector components `b_x`, `b_y`, `b_z`.
- `application.py` expects a single scalar magnetic series. Here we compute **total field magnitude**
  (sqrt(b_x^2 + b_y^2 + b_z^2)) and expose it as `mag_H_nT` to keep the rest of the app unchanged.
  If you prefer a different scalar (e.g., `b_x` only or horizontal magnitude), we can switch it.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

try:
    import mysql.connector
except Exception as e:  # pragma: no cover
    mysql = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


APP_BASE = os.path.dirname(__file__)
TABLE_NAME = "qnav_magneticdatamodel"

# Copied from Get_Data_09Dec25.py (reference script provided by user)
DB_CONFIG = {
    "host": "50.63.129.30",
    "user": "devuser",
    "password": "devuser@221133",
    "database": "dbqnaviitk",
    "port": 3306,
    "connection_timeout": 60,
    "read_timeout": 7200,
    "write_timeout": 7200,
    "use_pure": True,
    "buffered": True,  # OPTIMIZATION: Enable buffered mode for faster bulk inserts/selects
    "autocommit": False,
    "pool_reset_session": True,
}

# Robustness settings (adapted from Get_Data_09Dec25.py)
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 2
MAX_BACKOFF_SECONDS = 30
LONG_NET_TIMEOUT_SECONDS = 7200
FETCH_CHUNK_SIZE = 2000  # fetchmany size


def _ensure_mysql_available() -> None:
    if _IMPORT_ERR is not None:
        raise ImportError(
            "mysql.connector is required for DB ingestion. "
            "Install mysql-connector-python. Original error: "
            f"{_IMPORT_ERR}"
        )


def _get_latest_sensor_id(conn) -> Optional[str]:
    """
    Try to pick the most recent sensor_id so we fetch a consistent stream.
    If the query fails, return None and fetch without sensor filter.
    """
    try:
        cur = conn.cursor()
        # Prefer ORDER BY id (primary key) for speed and index usage.
        cur.execute(f"SELECT sensor_id FROM {TABLE_NAME} ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        cur.close()
        if row and row[0]:
            return str(row[0])
    except Exception:
        return None
    return None


def _fill_missing_max_timestamps(sensor_ids: list[str], max_timestamps: dict[str, datetime]) -> None:
    """Batched MAX(timestamp) can omit a stream; ensure each requested sensor_id has an end bound."""
    for sid in sensor_ids:
        if max_timestamps.get(sid) is not None:
            continue
        mt = _get_max_timestamp(sensor_id=sid)
        if mt is not None:
            max_timestamps[str(sid)] = mt


def _get_max_timestamp(sensor_id: Optional[str] = None) -> Optional[datetime]:
    """Return the most recent timestamp present in the table (optionally filtered by sensor_id)."""
    _ensure_mysql_available()
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        _set_session_timeouts(conn)
        cur = conn.cursor()
        if sensor_id:
            # Table lacks composite timestamp indexing (105M rows). Use bounded ID window to stop catastrophic full scans!
            cur.execute(
                f"SELECT timestamp FROM {TABLE_NAME} WHERE sensor_id=%s AND id > (SELECT MAX(id) - 5000000 FROM {TABLE_NAME}) ORDER BY id DESC LIMIT 1",
                (sensor_id,),
            )
        else:
            cur.execute(f"SELECT timestamp FROM {TABLE_NAME} ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        cur.close()
        if row and row[0]:
            # mysql.connector may return datetime already; normalize to python datetime.
            ts = row[0]
            ts = pd.to_datetime(ts).to_pydatetime()
            # keep timezone-naive
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
            return ts
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass
    return None


def _empty_vector_df() -> pd.DataFrame:
    """Return an empty 1 Hz vector dataframe with the expected schema."""
    return pd.DataFrame(
        columns=["time_H", "b_x", "b_y", "b_z", "theta_x", "theta_y", "theta_z", "_ts_ns"]
    )


def get_latest_sensor_ids(limit: int = 6) -> list[str]:
    """
    Return up to `limit` sensor_ids, preferring the latest OBS streams.

    This is used by app.py (ApplicationTemp) to discover the 6 streams
    (2 observatories × 3 sensors each).

    Behavior:
    - Finds recent sensor streams using MAX(id) per sensor_id.
    - Prefers canonical OBS suffixes (OBS1_1..OBS1_3, OBS2_1..OBS2_3).
    - If sensor IDs contain day tokens (SYYYYMMDD...), prefers the newest day across
      canonical streams, then falls back to latest available IDs.
    """
    limit = max(int(limit), 1)
    canonical_suffixes = ("OBS1_1", "OBS1_2", "OBS1_3", "OBS2_1", "OBS2_2", "OBS2_3")

    def _obs_suffix(sensor_id: str) -> Optional[str]:
        m = re.search(r"(OBS\d+_\d+)$", str(sensor_id))
        return m.group(1) if m else None

    def _sensor_day_token(sensor_id: str) -> Optional[str]:
        m = re.search(r"S(\d{8})", str(sensor_id))
        return m.group(1) if m else None

    def _dedupe_keep_order(items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for sid in items:
            if sid in seen:
                continue
            seen.add(sid)
            out.append(sid)
        return out

    _ensure_mysql_available()
    for attempt in range(1, MAX_RETRIES + 1):
        conn = None
        cur = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            _set_session_timeouts(conn)
            cur = conn.cursor()
            # Use a bounded ID window and aggregate by MAX(id) so we truly get latest
            # stream variants per sensor_id (DISTINCT + ORDER BY id can return stale IDs).
            candidate_pool = max(limit * 24, 240)
            cur.execute(
                f"SELECT sensor_id, MAX(id) AS max_id "
                f"FROM {TABLE_NAME} "
                f"WHERE id > (SELECT MAX(id) - 5000000 FROM {TABLE_NAME}) "
                f"GROUP BY sensor_id "
                f"ORDER BY max_id DESC "
                f"LIMIT %s",
                (int(candidate_pool),),
            )
            rows = cur.fetchall()
            candidates: list[tuple[str, int]] = []
            for r in rows or []:
                if not r or not r[0]:
                    continue
                sid = str(r[0])
                try:
                    max_id = int(r[1]) if len(r) > 1 and r[1] is not None else 0
                except Exception:
                    max_id = 0
                candidates.append((sid, max_id))

            if not candidates:
                return []

            # Index the newest stream seen for each canonical OBS suffix.
            latest_any_by_suffix: dict[str, str] = {}
            for sid, _ in candidates:
                suffix = _obs_suffix(sid)
                if suffix in canonical_suffixes and suffix not in latest_any_by_suffix:
                    latest_any_by_suffix[suffix] = sid

            # Prefer a single newest day token across canonical streams when possible.
            canonical_days: list[str] = []
            for suffix in canonical_suffixes:
                sid = latest_any_by_suffix.get(suffix)
                if sid:
                    day = _sensor_day_token(sid)
                    if day:
                        canonical_days.append(day)
            target_day = max(canonical_days) if canonical_days else None

            chosen: list[str] = []
            for suffix in canonical_suffixes:
                sid_for_suffix: Optional[str] = None
                if target_day:
                    for sid, _ in candidates:
                        if _obs_suffix(sid) == suffix and _sensor_day_token(sid) == target_day:
                            sid_for_suffix = sid
                            break
                if sid_for_suffix is None:
                    sid_for_suffix = latest_any_by_suffix.get(suffix)
                if sid_for_suffix:
                    chosen.append(sid_for_suffix)

            # Fallback safety: if canonical extraction failed, return latest candidates.
            if not chosen:
                chosen = [sid for sid, _ in candidates]

            # Keep canonical-first ordering, then fill with remaining newest streams.
            merged = _dedupe_keep_order(chosen + [sid for sid, _ in candidates])
            return merged[:limit]
        except Exception:
            if attempt < MAX_RETRIES:
                delay = min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
                time.sleep(delay)
        finally:
            try:
                if cur is not None:
                    cur.close()
            except Exception:
                pass
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass

    # Discovery failure should not crash app startup; caller can fall back to manual/CSV selection.
    return []


def get_latest_sensor_id_like(pattern: str) -> Optional[str]:
    """
    Return the most recent sensor_id matching a SQL LIKE pattern, e.g. '%OBS1_1'.
    Uses ORDER BY id DESC to pick the latest stream variant.
    """
    _ensure_mysql_available()
    for attempt in range(1, MAX_RETRIES + 1):
        conn = None
        cur = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            _set_session_timeouts(conn)
            cur = conn.cursor()
            cur.execute(
                f"SELECT sensor_id FROM {TABLE_NAME} WHERE sensor_id LIKE %s ORDER BY id DESC LIMIT 1",
                (pattern,),
            )
            row = cur.fetchone()
            if row and row[0]:
                return str(row[0])
            return None
        except Exception:
            if attempt < MAX_RETRIES:
                delay = min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
                time.sleep(delay)
        finally:
            try:
                if cur is not None:
                    cur.close()
            except Exception:
                pass
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass

    # Discovery failure should not crash app startup.
    return None


def get_timeseries_magnetic_data_multi(
    sensor_ids: list[str],
    *,
    hours: float = 1.0,
    last_n_samples: int = 3600,  # 60 min @ 1 Hz
) -> dict[str, pd.DataFrame]:
    """
    Fetch the most recent `hours` of data for each sensor_id.
    
OPTIMIZED: Batched query for max timestamps (not sequential). Reduced multiplier from 3 to 2.

    Returns: dict sensor_id -> DataFrame with columns [time_H, mag_H_nT]
    """
    results: dict[str, pd.DataFrame] = {}
    
    if not sensor_ids:
        return results
    
    # OPTIMIZATION: Batch-fetch max timestamps for all sensors in one query
    _ensure_mysql_available()
    max_timestamps = {}
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        _set_session_timeouts(conn)
        cur = conn.cursor()
        
        # Single batched query: get latest timestamp per sensor
        cur.execute(
            f"SELECT sensor_id, MAX(timestamp) FROM {TABLE_NAME} "
            f"WHERE id > (SELECT MAX(id) - 10000000 FROM {TABLE_NAME}) "
            f"GROUP BY sensor_id"
        )
        for row in cur.fetchall():
            if row and len(row) >= 2:
                sid, ts = row[0], row[1]
                if ts:
                    ts = pd.to_datetime(ts).to_pydatetime()
                    if getattr(ts, "tzinfo", None) is not None:
                        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
                    max_timestamps[str(sid)] = ts
        cur.close()
    except Exception:
        # Fallback: individual sensor queries
        for sid in sensor_ids:
            max_ts = _get_max_timestamp(sensor_id=sid)
            if max_ts:
                max_timestamps[sid] = max_ts
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

    _fill_missing_max_timestamps(sensor_ids, max_timestamps)

    # OPTIMIZATION: Reduced multiplier from 3 to 2 for more efficient queries
    # Reasoning: modern databases are usually 1-2 samples per sec on average
    for sid in sensor_ids:
        try:
            end_ts = max_timestamps.get(sid)
            if end_ts is None:
                results[sid] = pd.DataFrame(columns=["time_H", "mag_H_nT"])
                continue
            start_ts = end_ts - timedelta(hours=float(hours))
            # OPTIMIZATION: Reduced multiplier from 3 to 2 based on typical sampling rate
            raw_limit_rows = max(int(last_n_samples) * 2, 8000)
            df_raw = _query_db(
                start_time=start_ts,
                end_time=end_ts,
                limit_rows=int(raw_limit_rows),
                sensor_id=sid,
                order_desc=True,
            )
            results[sid] = _raw_to_timeseries_df(df_raw, target_n_seconds=int(last_n_samples))
        except Exception:
            results[sid] = pd.DataFrame(columns=["time_H", "mag_H_nT"])
    return results


def get_timeseries_magnetic_data_since_multi(
    sensor_ids: list[str],
    *,
    since_times: dict[str, datetime],
    limit_rows: int = 5000,
) -> dict[str, pd.DataFrame]:
    """
    Incremental fetch: for each sensor, fetch rows from since_time to latest database time.
    
    OPTIMIZED: Batched max timestamp query. Critical for real-time updates every 20 sec.

    Returns dict sensor_id -> DataFrame [time_H, mag_H_nT]. Empty DF if no new data.
    """
    out: dict[str, pd.DataFrame] = {}
    
    if not sensor_ids:
        return out
    
    # OPTIMIZATION: Batch-fetch all max timestamps in single query
    _ensure_mysql_available()
    max_timestamps = {}
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        _set_session_timeouts(conn)
        cur = conn.cursor()
        
        # Single query for all sensors
        cur.execute(
            f"SELECT sensor_id, MAX(timestamp) FROM {TABLE_NAME} "
            f"WHERE id > (SELECT MAX(id) - 10000000 FROM {TABLE_NAME}) "
            f"GROUP BY sensor_id"
        )
        for row in cur.fetchall():
            if row and len(row) >= 2:
                sid, ts = row[0], row[1]
                if ts:
                    ts = pd.to_datetime(ts).to_pydatetime()
                    if getattr(ts, "tzinfo", None) is not None:
                        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
                    max_timestamps[str(sid)] = ts
        cur.close()
    except Exception:
        # Fallback to individual queries if batch fails
        for sid in sensor_ids:
            max_ts = _get_max_timestamp(sensor_id=sid)
            if max_ts:
                max_timestamps[sid] = max_ts
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

    _fill_missing_max_timestamps(sensor_ids, max_timestamps)

    for sid in sensor_ids:
        try:
            since = since_times.get(sid)
            if since is None:
                out[sid] = pd.DataFrame(columns=["time_H", "mag_H_nT"])
                continue
            
            end_ts = max_timestamps.get(sid)
            if end_ts is None or end_ts <= since:
                out[sid] = pd.DataFrame(columns=["time_H", "mag_H_nT"])
                continue
            
            # OPTIMIZATION: Reduced limit_rows from 5000 to 3000 for faster incremental fetches
            df_raw = _query_db(
                start_time=since,
                end_time=end_ts,
                limit_rows=min(int(limit_rows), 3000),
                sensor_id=sid,
                order_desc=False,
            )
            # Incremental: still average within each second; don't hard-cap here.
            out[sid] = _raw_to_timeseries_df(df_raw, target_n_seconds=None)
        except Exception:
            out[sid] = pd.DataFrame(columns=["time_H", "mag_H_nT"])
    return out


def get_vector_data_multi(
    sensor_ids: list[str],
    *,
    hours: float = 1.0,
    last_n_samples: int = 3600,  # 60 min @ 1 Hz
) -> dict[str, pd.DataFrame]:
    """
    Fetch recent vector data for each sensor_id and downsample to 1 Hz.
    
OPTIMIZED: Batched max timestamp query. Reduced multiplier from 3 to 2.

    Returns: dict sensor_id -> DataFrame [time_H, b_x, b_y, b_z, theta_x, theta_y, theta_z, _ts_ns]
    """
    results: dict[str, pd.DataFrame] = {}
    
    if not sensor_ids:
        return results
    
    # OPTIMIZATION: Batch-fetch all max timestamps
    _ensure_mysql_available()
    max_timestamps = {}
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        _set_session_timeouts(conn)
        cur = conn.cursor()
        
        cur.execute(
            f"SELECT sensor_id, MAX(timestamp) FROM {TABLE_NAME} "
            f"WHERE id > (SELECT MAX(id) - 10000000 FROM {TABLE_NAME}) "
            f"GROUP BY sensor_id"
        )
        for row in cur.fetchall():
            if row and len(row) >= 2:
                sid, ts = row[0], row[1]
                if ts:
                    ts = pd.to_datetime(ts).to_pydatetime()
                    if getattr(ts, "tzinfo", None) is not None:
                        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
                    max_timestamps[str(sid)] = ts
        cur.close()
    except Exception:
        for sid in sensor_ids:
            max_ts = _get_max_timestamp(sensor_id=sid)
            if max_ts:
                max_timestamps[sid] = max_ts
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

    _fill_missing_max_timestamps(sensor_ids, max_timestamps)

    for sid in sensor_ids:
        try:
            end_ts = max_timestamps.get(sid)
            if end_ts is None:
                results[sid] = _empty_vector_df()
                continue
            start_ts = end_ts - timedelta(hours=float(hours))
            # OPTIMIZATION: Reduced multiplier from 3 to 2 for vector data
            raw_limit_rows = max(int(last_n_samples) * 2, 8000)
            df_raw = _query_db_vector(
                start_time=start_ts,
                end_time=end_ts,
                limit_rows=int(raw_limit_rows),
                sensor_id=sid,
                order_desc=True,
            )
            results[sid] = _raw_to_vector_timeseries_df(
                df_raw, target_n_seconds=int(last_n_samples)
            )
        except Exception:
            results[sid] = _empty_vector_df()
    return results


def get_vector_data_since_multi(
    sensor_ids: list[str],
    *,
    since_times: dict[str, datetime],
    limit_rows: int = 5000,
) -> dict[str, pd.DataFrame]:
    """
    Incremental vector fetch for each sensor from since_time to latest database time.
    
    OPTIMIZED: Batched max timestamp query. Reduced limits from 5000 to 2500.
    """
    out: dict[str, pd.DataFrame] = {}
    
    if not sensor_ids:
        return out
    
    # OPTIMIZATION: Batch-fetch all max timestamps
    _ensure_mysql_available()
    max_timestamps = {}
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        _set_session_timeouts(conn)
        cur = conn.cursor()
        
        cur.execute(
            f"SELECT sensor_id, MAX(timestamp) FROM {TABLE_NAME} "
            f"WHERE id > (SELECT MAX(id) - 10000000 FROM {TABLE_NAME}) "
            f"GROUP BY sensor_id"
        )
        for row in cur.fetchall():
            if row and len(row) >= 2:
                sid, ts = row[0], row[1]
                if ts:
                    ts = pd.to_datetime(ts).to_pydatetime()
                    if getattr(ts, "tzinfo", None) is not None:
                        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
                    max_timestamps[str(sid)] = ts
        cur.close()
    except Exception:
        for sid in sensor_ids:
            max_ts = _get_max_timestamp(sensor_id=sid)
            if max_ts:
                max_timestamps[sid] = max_ts
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

    _fill_missing_max_timestamps(sensor_ids, max_timestamps)

    for sid in sensor_ids:
        try:
            since = since_times.get(sid)
            if since is None:
                out[sid] = _empty_vector_df()
                continue
            
            end_ts = max_timestamps.get(sid)
            if end_ts is None or end_ts <= since:
                out[sid] = _empty_vector_df()
                continue
            
            # OPTIMIZATION: Reduced limit_rows from 5000 to 2500 for vector incremental
            df_raw = _query_db_vector(
                start_time=since,
                end_time=end_ts,
                limit_rows=min(int(limit_rows), 2500),
                sensor_id=sid,
                order_desc=False,
            )
            out[sid] = _raw_to_vector_timeseries_df(df_raw, target_n_seconds=None)
        except Exception:
            out[sid] = _empty_vector_df()
    return out


def fetch_vector_window_multi(
    sensor_ids: list[str],
    *,
    start_time: datetime,
    end_time: datetime,
    target_n_seconds: int = 3600,  # 60 min @ 1 Hz
) -> dict[str, pd.DataFrame]:
    """
    Fetch vector data in [start_time, end_time] for each sensor_id at 1 Hz.
    
    OPTIMIZATION: Reduced initial multiplier from 50 to 10 (most sensors have 1-2 samples per sec).
    """
    out: dict[str, pd.DataFrame] = {}
    duration = end_time - start_time
    for sid in sensor_ids:
        try:
            # OPTIMIZATION: Reduced base multiplier from 50 to 10 (most sensors have 1-2 samples per sec)
            base_limit = int(target_n_seconds) * 10
            raw_limit_rows = min(max(base_limit, 8000), 500000)
            s2 = _get_min_timestamp_at_or_after(sid, start_time) or start_time
            e2 = s2 + duration
            df_vec = _empty_vector_df()
            max_expands = 5  # OPTIMIZATION: Reduced from 12 to 5 (less aggressive expansion)
            expands = 0
            while expands <= max_expands:
                # OPTIMIZATION: Reduced expansion rate from 0.25 to 0.15
                limit_this = min(int(raw_limit_rows * (1 + 0.15 * expands)), 600000)
                df_raw = _query_db_vector(
                    start_time=s2,
                    end_time=e2,
                    limit_rows=int(limit_this),
                    sensor_id=sid,
                    order_desc=False,
                )
                df_vec = _raw_to_vector_timeseries_df(df_raw, target_n_seconds=None)
                if df_vec is not None and len(df_vec) >= int(target_n_seconds):
                    df_vec = df_vec.tail(int(target_n_seconds)).reset_index(drop=True)
                    break
                e2 = e2 + duration
                expands += 1
            out[sid] = df_vec if df_vec is not None else _empty_vector_df()
        except Exception:
            out[sid] = _empty_vector_df()
    return out


def fetch_vector_between_multi(
    sensor_ids: list[str],
    *,
    start_time: datetime,
    end_time: datetime,
    limit_rows: int = 20000,
) -> dict[str, pd.DataFrame]:
    """
    Fetch vector data in a time window for each sensor_id at 1 Hz.
    """
    out: dict[str, pd.DataFrame] = {}
    for sid in sensor_ids:
        try:
            df_raw = _query_db_vector(
                start_time=start_time,
                end_time=end_time,
                limit_rows=int(limit_rows),
                sensor_id=sid,
                order_desc=False,
            )
            out[sid] = _raw_to_vector_timeseries_df(df_raw, target_n_seconds=None)
        except Exception:
            out[sid] = _empty_vector_df()
    return out


def _get_min_timestamp_at_or_after(sensor_id: str, start_time: datetime) -> Optional[datetime]:
    """Find the first available timestamp >= start_time for a given sensor_id."""
    _ensure_mysql_available()
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        _set_session_timeouts(conn)
        cur = conn.cursor()
        # Bounded scan first (table is huge); simulation replay may need older IDs — fallback below.
        cur.execute(
            f"SELECT timestamp FROM {TABLE_NAME} WHERE sensor_id=%s AND timestamp >= %s AND id > (SELECT MAX(id) - 10000000 FROM {TABLE_NAME}) ORDER BY id ASC LIMIT 1",
            (sensor_id, start_time),
        )
        row = cur.fetchone()
        if not row or not row[0]:
            cur.execute(
                f"SELECT timestamp FROM {TABLE_NAME} WHERE sensor_id=%s AND timestamp >= %s ORDER BY id ASC LIMIT 1",
                (sensor_id, start_time),
            )
            row = cur.fetchone()
        cur.close()
        if row and row[0]:
            ts = pd.to_datetime(row[0]).to_pydatetime()
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
            return ts
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass
    return None


def get_min_timestamp_at_or_after(sensor_id: str, start_time: datetime) -> Optional[datetime]:
    """Public wrapper around `_get_min_timestamp_at_or_after` (used by simulation mode)."""
    return _get_min_timestamp_at_or_after(sensor_id, start_time)


def fetch_timeseries_window_multi(
    sensor_ids: list[str],
    *,
    start_time: datetime,
    end_time: datetime,
    target_n_seconds: int = 3600,  # 60 min @ 1 Hz
) -> dict[str, pd.DataFrame]:
    """
    Fetch a fixed window [start_time, end_time] for each sensor_id and return 1 Hz series.
    If a sensor has no data in that window, shift start_time to first available point.
    
    OPTIMIZATION: Reduced multipliers from 50 to 10, expansion rate from 0.25 to 0.15.
    """
    out: dict[str, pd.DataFrame] = {}
    duration = end_time - start_time
    for sid in sensor_ids:
        try:
            # OPTIMIZATION: Reduced base multiplier from 50 to 10 (most sensors have ~1-2 samples per sec)
            base_limit = int(target_n_seconds) * 10
            raw_limit_rows = min(max(base_limit, 8000), 500000)

            # Align start per sensor to the first available point >= requested start_time.
            s2 = _get_min_timestamp_at_or_after(sid, start_time) or start_time
            e2 = s2 + duration

            # Expand forward until we accumulate enough 1 Hz seconds (or hit max expansions).
            df_ts = pd.DataFrame(columns=["time_H", "mag_H_nT"])
            max_expands = 5  # OPTIMIZATION: Reduced from 12 to 5 (less aggressive expansion)
            expands = 0
            while expands <= max_expands:
                # OPTIMIZATION: Reduced expansion rate from 0.25 to 0.15
                limit_this = min(int(raw_limit_rows * (1 + 0.15 * expands)), 600000)
                df_raw = _query_db(
                    start_time=s2,
                    end_time=e2,
                    limit_rows=int(limit_this),
                    sensor_id=sid,
                    order_desc=False,
                )
                df_ts = _raw_to_timeseries_df(df_raw, target_n_seconds=None)
                if df_ts is not None and len(df_ts) >= int(target_n_seconds):
                    df_ts = df_ts.tail(int(target_n_seconds)).reset_index(drop=True)
                    break
                # not enough data yet: extend end forward
                e2 = e2 + duration
                expands += 1

            out[sid] = df_ts if df_ts is not None else pd.DataFrame(columns=["time_H", "mag_H_nT"])
        except Exception:
            out[sid] = pd.DataFrame(columns=["time_H", "mag_H_nT"])
    return out


def fetch_timeseries_between_multi(
    sensor_ids: list[str],
    *,
    start_time: datetime,
    end_time: datetime,
    limit_rows: int = 20000,
) -> dict[str, pd.DataFrame]:
    """
    Fetch timeseries data in a time window for each sensor_id at 1 Hz.
    Incremental updates intended for simulated real-time mode.
    
    OPTIMIZATION: Reduced default limit_rows from 20000 to 10000 for faster fetches.
    """
    out: dict[str, pd.DataFrame] = {}
    for sid in sensor_ids:
        try:
            df_raw = _query_db(
                start_time=start_time,
                end_time=end_time,
                limit_rows=min(int(limit_rows), 10000),  # OPTIMIZATION: Cap at 10k even if higher requested
                sensor_id=sid,
                order_desc=False,
            )
            out[sid] = _raw_to_timeseries_df(df_raw, target_n_seconds=None)
        except Exception:
            out[sid] = pd.DataFrame(columns=["time_H", "mag_H_nT"])
    return out


def _set_session_timeouts(conn) -> None:
    """Best-effort session tuning to prevent long fetch disconnects."""
    try:
        cur = conn.cursor()
        cur.execute("SET SESSION wait_timeout=28800")
        cur.execute("SET SESSION interactive_timeout=28800")
        cur.execute(f"SET SESSION net_read_timeout={LONG_NET_TIMEOUT_SECONDS}")
        cur.execute(f"SET SESSION net_write_timeout={LONG_NET_TIMEOUT_SECONDS}")
        try:
            # 0 = no query timeout if supported (MySQL 5.7.8+)
            cur.execute("SET SESSION max_execution_time=0")
        except Exception:
            pass
        cur.close()
    except Exception:
        pass


def _fetch_rows_with_retries(
    *,
    query: str,
    params: list,
) -> list[dict]:
    """
    Execute a query with retries, streaming rows via fetchmany().

    This avoids pandas.read_sql() (which warns for mysql.connector) and follows
    the robust pattern used in Get_Data_09Dec25.py.
    """
    _ensure_mysql_available()
    last_err: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        conn = None
        cursor = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            _set_session_timeouts(conn)

            # Use a buffered cursor so we can safely fetch results without connection
            # state issues (and because our target payload is small: ~3400 rows).
            cursor = conn.cursor(dictionary=True, buffered=True)
            cursor.execute(query, params)

            rows: list[dict] = []
            fetched = 0
            while True:
                chunk = cursor.fetchmany(FETCH_CHUNK_SIZE)
                if not chunk:
                    break
                rows.extend(chunk)
                fetched += len(chunk)

            return rows
        except Exception as e:
            last_err = e
            # Backoff before retrying
            if attempt < MAX_RETRIES:
                delay = min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
                time.sleep(delay)
        finally:
            try:
                if cursor is not None:
                    cursor.close()
            except Exception:
                pass
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass

    # Exhausted retries
    if last_err is not None:
        raise last_err
    return []


def _query_db_with_cols(
    *,
    cols: list[str],
    start_time: datetime,
    end_time: datetime,
    limit_rows: int,
    sensor_id: Optional[str],
    order_desc: bool,
) -> pd.DataFrame:
    columns_str = ", ".join(cols)

    query = f"SELECT {columns_str} FROM {TABLE_NAME} WHERE timestamp >= %s AND timestamp <= %s"
    params: list = [start_time, end_time]
    if sensor_id:
        query += " AND sensor_id = %s"
        params.append(sensor_id)

    # OPTIMIZATION: Reduced ID boundary from 8000000 to 5000000 for faster query planning
    # This still covers ~5M recent rows (sufficient for multi-week history) while improving query speed
    query += f" AND id > (SELECT MAX(id) - 5000000 FROM {TABLE_NAME})"
    
    query += " ORDER BY id DESC" if order_desc else " ORDER BY id ASC"
    query += " LIMIT %s"
    params.append(int(limit_rows))

    rows = _fetch_rows_with_retries(query=query, params=params)
    return pd.DataFrame(rows)


def _query_db(
    *,
    start_time: datetime,
    end_time: datetime,
    limit_rows: int,
    sensor_id: Optional[str],
    order_desc: bool,
) -> pd.DataFrame:
    cols = ["id", "sensor_id", "timestamp", "b_x", "b_y", "b_z"]
    return _query_db_with_cols(
        cols=cols,
        start_time=start_time,
        end_time=end_time,
        limit_rows=limit_rows,
        sensor_id=sensor_id,
        order_desc=order_desc,
    )


def _query_db_vector(
    *,
    start_time: datetime,
    end_time: datetime,
    limit_rows: int,
    sensor_id: Optional[str],
    order_desc: bool,
) -> pd.DataFrame:
    """
    Query vector columns for direction modeling.

    Prefer theta columns when present; if DB schema lacks them, gracefully
    fall back to b_x/b_y/b_z only and downstream code will fill theta with 0.0.
    """
    cols_with_theta = [
        "id",
        "sensor_id",
        "timestamp",
        "b_x",
        "b_y",
        "b_z",
        "theta_x",
        "theta_y",
        "theta_z",
    ]
    cols_base = ["id", "sensor_id", "timestamp", "b_x", "b_y", "b_z"]
    try:
        return _query_db_with_cols(
            cols=cols_with_theta,
            start_time=start_time,
            end_time=end_time,
            limit_rows=limit_rows,
            sensor_id=sensor_id,
            order_desc=order_desc,
        )
    except Exception:
        return _query_db_with_cols(
            cols=cols_base,
            start_time=start_time,
            end_time=end_time,
            limit_rows=limit_rows,
            sensor_id=sensor_id,
            order_desc=order_desc,
        )


def _raw_to_timeseries_df(df_raw: pd.DataFrame, target_n_seconds: Optional[int] = None) -> pd.DataFrame:
    """
    Convert raw DB rows into a 1 Hz time series:
    - bucket timestamps by second (floor to seconds)
    - average all samples within the same second
    - return columns [time_H, mag_H_nT]

    If target_n_seconds is provided, keep only the latest `target_n_seconds` rows.
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["time_H", "mag_H_nT"])

    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=True).reset_index(drop=True)
    for c in ("b_x", "b_y", "b_z"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["mag_total_nT"] = (df["b_x"] ** 2 + df["b_y"] ** 2 + df["b_z"] ** 2) ** 0.5
    df = df.dropna(subset=["mag_total_nT"])

    # 1 Hz downsample by averaging within each second
    df["time_H"] = df["timestamp"].dt.floor("s")
    grouped = (
        df.groupby("time_H", as_index=False)["mag_total_nT"]
        .mean()
        .rename(columns={"mag_total_nT": "mag_H_nT"})
        .sort_values("time_H", ascending=True)
        .reset_index(drop=True)
    )

    if target_n_seconds is not None and target_n_seconds > 0 and len(grouped) > target_n_seconds:
        grouped = grouped.tail(int(target_n_seconds)).reset_index(drop=True)

    return grouped[["time_H", "mag_H_nT"]]


def _raw_to_vector_timeseries_df(
    df_raw: pd.DataFrame, target_n_seconds: Optional[int] = None
) -> pd.DataFrame:
    """
    Convert raw DB rows into a 1 Hz vector time series for direction modeling.

    Returns columns:
    [time_H, b_x, b_y, b_z, theta_x, theta_y, theta_z, _ts_ns]
    """
    if df_raw is None or df_raw.empty:
        return _empty_vector_df()

    df = df_raw.copy()
    if "timestamp" not in df.columns:
        return _empty_vector_df()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=True).reset_index(drop=True)

    for c in ("b_x", "b_y", "b_z"):
        if c not in df.columns:
            return _empty_vector_df()
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ("theta_x", "theta_y", "theta_z"):
        if c not in df.columns:
            # Allow direction model inference even when DB schema omits theta.
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["b_x", "b_y", "b_z"])
    if df.empty:
        return _empty_vector_df()

    df["time_H"] = df["timestamp"].dt.floor("s")
    grouped = (
        df.groupby("time_H", as_index=False)[
            ["b_x", "b_y", "b_z", "theta_x", "theta_y", "theta_z"]
        ]
        .mean()
        .sort_values("time_H", ascending=True)
        .reset_index(drop=True)
    )

    if target_n_seconds is not None and target_n_seconds > 0 and len(grouped) > target_n_seconds:
        grouped = grouped.tail(int(target_n_seconds)).reset_index(drop=True)

    grouped["_ts_ns"] = pd.to_datetime(grouped["time_H"]).astype("int64")
    return grouped[["time_H", "b_x", "b_y", "b_z", "theta_x", "theta_y", "theta_z", "_ts_ns"]]


def get_timeseries_magnetic_data(
    session_id: Optional[str] = None,
    last_n_samples: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    hours: Optional[float] = None,
):
    # Docstring temporarily removed to debug syntax errors
    # TODO: Restore proper docstring
    
    # Determine time bounds.
    #
    # Important: MySQL timestamps are commonly stored/handled as timezone-naive values.
    # To avoid driver issues with tz-aware datetimes, we use naive UTC for query params.
    if end_time is None:
        end_time = datetime.utcnow()
    else:
        end_time = pd.to_datetime(end_time).to_pydatetime()
        if getattr(end_time, "tzinfo", None) is not None:
            end_time = end_time.astimezone(timezone.utc).replace(tzinfo=None)

    if start_time is None:
        if hours is None:
            hours = 1.0
        start_time = end_time - timedelta(hours=float(hours))
    else:
        start_time = pd.to_datetime(start_time).to_pydatetime()
        if getattr(start_time, "tzinfo", None) is not None:
            start_time = start_time.astimezone(timezone.utc).replace(tzinfo=None)

    # Desired output size:
    # We return ONE averaged sample per second, so last_n_samples means "last N seconds".
    target_seconds: Optional[int]
    if last_n_samples is not None:
        target_seconds = int(last_n_samples)
    else:
        target_seconds = 3400

    # Decide whether to fetch most-recent rows (DESC) or forward-in-time rows (ASC).
    # - Initial "last 60 minutes" load: use DESC LIMIT N then sort ASC for plotting.
    # - Incremental fetch (start_time provided by caller): use ASC so we append correctly.
    is_incremental = start_time is not None and hours is None
    order_desc = not is_incremental

    # If incremental and caller did not specify last_n_samples, do not hard-cap by seconds.
    # (Caller usually wants "all new points since start_time".)
    if is_incremental and last_n_samples is None:
        target_seconds = None

    # Raw row limit: multiple samples per second, so fetch more than the target seconds.
    if target_seconds is None:
        limit_rows = 5000
    else:
        limit_rows = max(int(target_seconds) * 3, 12000)

    # Query
    _ensure_mysql_available()
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        sensor_id = _get_latest_sensor_id(conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    df_raw = _query_db(
        start_time=start_time,
        end_time=end_time,
        limit_rows=limit_rows,
        sensor_id=sensor_id,
        order_desc=order_desc,
    )

    # If we got no rows, it may be because the DB timestamps are in a different timezone
    # than the local machine clock. In that case, use the DB's latest timestamp as "now"
    # and fetch the last hour before that.
    if (df_raw is None or df_raw.empty) and hours is not None and start_time is not None and end_time is not None:
        try:
            db_max = _get_max_timestamp(sensor_id=sensor_id)
            if db_max is not None:
                end_time_db = db_max
                start_time_db = end_time_db - timedelta(hours=float(hours))
                df_raw = _query_db(
                    start_time=start_time_db,
                    end_time=end_time_db,
                    limit_rows=limit_rows,
                    sensor_id=sensor_id,
                    order_desc=order_desc,
                )
        except Exception:
            # Keep empty result; caller will handle.
            pass

    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["time_H", "mag_H_nT"])

    # Convert to 1 Hz series (average within each second)
    out = _raw_to_timeseries_df(df_raw, target_n_seconds=target_seconds)

    # Optional: persist fetched raw data for the session (debug/repro)
    if session_id:
        try:
            folder = os.path.join(APP_BASE, "sessions", session_id)
            os.makedirs(folder, exist_ok=True)
            out.to_csv(os.path.join(folder, "download_mag_db.csv"), index=False)
        except Exception:
            # Non-fatal: app should continue even if debug file can't be written.
            pass

    return out


