
from __future__ import annotations

import os
import glob
import re
import sys
import uuid
import subprocess
import math
from collections import deque
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
#
# IMPORTANT: `application.py` imports `data_convert_now.py` (USGS/requests).
# For the DB-based temp app we do not need USGS at all, and on some locked-down
# environments importing requests/cert bundles can fail. We therefore pre-seed
# `sys.modules['data_convert_now']` with a tiny stub so `application.py` can import,
# while `ApplicationWindowTemp.startThreads()` provides the real DB fetch path.
#
import types as _types

if "data_convert_now" not in sys.modules:
    _stub = _types.ModuleType("data_convert_now")

    def _stub_get_timeseries_magnetic_data(*args, **kwargs):  # pragma: no cover
        # Fallback signature; should not be used by ApplicationTemp.
        from data_convert_db_now import get_timeseries_magnetic_data

        return get_timeseries_magnetic_data(*args, **kwargs)

    _stub.get_timeseries_magnetic_data = _stub_get_timeseries_magnetic_data
    sys.modules["data_convert_now"] = _stub

# Avoid slow GUI startup and repeated cache rebuilds by forcing Matplotlib/font cache
# into writable project directories.
_APP_BASE_DIR = os.path.dirname(__file__)
_LOCAL_CACHE = os.path.join(_APP_BASE_DIR, ".cache")
_MPL_CACHE = os.path.join(_LOCAL_CACHE, "mpl")
_XDG_CACHE = os.path.join(_LOCAL_CACHE, "xdg")
try:
    os.makedirs(_MPL_CACHE, exist_ok=True)
    os.makedirs(_XDG_CACHE, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE)
    os.environ.setdefault("XDG_CACHE_HOME", _XDG_CACHE)
except Exception:
    pass

import application as base_app

from PyQt5 import Qt, QtCore
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from Anomaly_detector import AnomalyDetector
from data_convert_db_now import (
    get_latest_sensor_ids,
    get_latest_sensor_id_like,
    get_min_timestamp_at_or_after,
    fetch_timeseries_window_multi,
    fetch_timeseries_between_multi,
    get_timeseries_magnetic_data_multi,
    get_timeseries_magnetic_data_since_multi,
)


def _sensor_sort_key(sensor_id: str) -> Tuple[int, int, str]:
    """
    Prefer ordering as OBS1_1..3 then OBS2_1..3 if the sensor_id matches that pattern.
    Fallback: keep stable ordering by sensor_id.
    """
    m = re.search(r"(OBS(\d+))_(\d+)$", sensor_id)
    if not m:
        return (99, 99, sensor_id)
    obs_num = int(m.group(2))
    sensor_num = int(m.group(3))
    return (obs_num, sensor_num, sensor_id)


def _sensor_display_name(sensor_id: str) -> str:
    m = re.search(r"(OBS\d+_\d+)$", sensor_id)
    return m.group(1) if m else sensor_id


def _is_obs1_ui_sensor_label(name: str) -> bool:
    """
    True when ``name`` is an Observatory-1 stream label (OBS1_1 … OBS1_3).

    Used so introduced-anomaly GT overlays appear only on OBS1 time-series panels,
    never on OBS2 panels.
    """
    if not name:
        return False
    u = str(name).strip().upper()
    return u.startswith("OBS1_")


# Default historic data window (minutes); user is prompted at startup to choose (see _configure_startup_mode).
# If available data is less than the chosen minutes, the app loads all of it (see on_db_data_updated).
# Display caps at this many points; training uses the full historic (and later historic+realtime) series.
HISTORIC_MINUTES = 60
HISTORIC_POINTS_1HZ = 60 * 60  # 3600 points at 1 Hz (default)
MODEL_FAMILY_GRU = "gru"
MODEL_FAMILY_TRANSFORMER = "transformer"
MODEL_FAMILY_ATTN_BILSTM = "attn_bilstm"
MODEL_FAMILY_LSTM = "lstm"
MODEL_INIT_PRETRAINED = "pretrained"
MODEL_INIT_FRESH = "fresh"
# GRU sequence length (timesteps); fresh-GRU startup prompt and PREDICTOR_GRU_WINDOW_SIZE for predictor_ai.
DEFAULT_PREDICTOR_GRU_WINDOW = 15
PREDICTOR_GRU_WINDOW_MIN = 5
PREDICTOR_GRU_WINDOW_MAX = 3600
# Fresh GRU: quick picks in the startup dialog (1 Hz samples per training / prediction step).
FRESH_GRU_WINDOW_PRESETS: List[int] = [15, 30, 45, 60, 90, 120, 180, 240]
TIMESERIES_FONT_FAMILIES = ["Arial", "DejaVu Sans", "sans-serif"]
TIMESERIES_FIG_DPI = 170
TIMESERIES_COLOR_BACKGROUND = "#FFFFFF"
TIMESERIES_COLOR_TEXT = "#111111"
TIMESERIES_COLOR_GRID = "#C7CCD4"
TIMESERIES_COLOR_BASELINE = "#0057FF"
TIMESERIES_COLOR_REALTIME = "#00A63E"
TIMESERIES_COLOR_PREDICTION = "#6F2CFF"
TIMESERIES_COLOR_ANOMALY = "#FFB3B3"
# Single UI color for introduced-anomaly overlays (legend + bottom strips); replaces separate yellow/brown.
TIMESERIES_COLOR_ANOMALY_INTRODUCED = "#AD1457"  # dark pink
TIMESERIES_COLOR_GROUND_TRUTH = TIMESERIES_COLOR_ANOMALY_INTRODUCED
TIMESERIES_COLOR_TRIMMER = TIMESERIES_COLOR_ANOMALY_INTRODUCED

# CSV basename -> (local calendar day for HHMM/HHMMSS cells, magnet GT ranges, optional trimmer-only ranges).
# Apr 2026 multi-window exports: every listed window is magnet-based; the trimmer tuple is empty (no separate trimmer GT).
_MANUAL_EXPERIMENT_CSV_GT: Dict[str, Tuple[datetime, List[Tuple[str, str]], List[Tuple[str, str]]]] = {
    "magnetic_data_20260206_151500_to_20260206_161500.csv": (
        datetime(2026, 2, 6),
        [
            ("1536", "1537"),
            ("1543", "1543.5"),
            ("1545", "1546"),
            ("1555", "1556"),
            ("1601", "1603"),
        ],
        [("1546", "1550"), ("1552", "1554"), ("1557", "1559")],
    ),
    "magnetic_data_20260210_110000_to_20260210_124500.csv": (
        datetime(2026, 2, 10),
        [
            ("1226", "1227"),
            ("1227", "1228"),
            ("1229", "1231"),
            ("1232", "1233"),
            ("1234", "1235"),
            ("1236", "1237"),
        ],
        [],
    ),
    "magnetic_data_20260213_150000_to_20260213_163000.csv": (
        datetime(2026, 2, 13),
        [
            ("160623", "160747"),
            ("160910", "161010"),
            ("161315", "161415"),
        ],
        [],
    ),
    # Synthetic Feb-13 export: Apr-27 GT waveforms grafted at these wall times (see Datafiles README / thesis notes).
    "magnetic_data_20260213_150000_to_20260213_180000_synthetic_no_anomaly.csv": (
        datetime(2026, 2, 13),
        [
            ("1515", "1545"),
            ("1610", "1640"),
            ("1700", "1730"),
        ],
        [],
    ),
    # 26–27 Apr 2026 export: GT bands align to **2026-04-27** wall times (three magnet windows only).
    "magnetic_data_20260426_060000_to_20260427_090000.csv": (
        datetime(2026, 4, 27),
        [
            ("062520", "065400"),
            ("072500", "075503"),
            ("082604", "085600"),
        ],
        [],
    ),
    # 1 Hz resample of the same campaign; same three windows as the non-1 Hz basename above.
    "magnetic_data_20260426_060000_to_20260427_090000_1hz.csv": (
        datetime(2026, 4, 27),
        [
            ("062520", "065400"),
            ("072500", "075503"),
            ("082604", "085600"),
        ],
        [],
    ),
}

# When non-empty: magnet intervals draw on OBS1_* only and trimmer intervals on OBS2_* only.
# Apr 2026 exports are magnet-only for all sensors — keep this empty so every stream uses the same magnet GT bands.
_CSV_GT_MAGNET_OBS1_TRIMMER_OBS2: FrozenSet[str] = frozenset()

TIMESERIES_LINEWIDTH = 2.4
TIMESERIES_ALPHA_ANOMALY = 0.3
TIMESERIES_ZORDER_ANOMALY = 1
# Draw order (low z = drawn first): historic baseline, then prediction, then actual on top.
TIMESERIES_ZORDER_BASELINE = 3
TIMESERIES_ZORDER_PREDICTION = 5
TIMESERIES_ZORDER_REALTIME = 7
TIMESERIES_XTICK_LABELSIZE = 5
TIMESERIES_YTICK_LABELSIZE = TIMESERIES_XTICK_LABELSIZE
# Matplotlib slows down badly with 50k+ points per line; keep display under this cap.
MAX_CANVAS_LINE_POINTS = 12000


def _parse_export_fig_dpi() -> int:
    """Matplotlib ``savefig`` DPI for HD exports (env ``MAGNAVIS_EXPORT_FIG_DPI``, default 300)."""
    try:
        v = int(str(os.environ.get("MAGNAVIS_EXPORT_FIG_DPI", "300")).strip())
        return max(72, min(v, 1200))
    except Exception:
        return 300


def _parse_ui_snapshot_scale() -> float:
    """Logical scale for full-window raster export (env ``MAGNAVIS_UI_SNAPSHOT_SCALE``, default 2)."""
    try:
        s = float(str(os.environ.get("MAGNAVIS_UI_SNAPSHOT_SCALE", "2")).strip())
        return max(1.0, min(s, 4.0))
    except Exception:
        return 2.0


def _parse_headless_snapshot_before_ts() -> Optional[pd.Timestamp]:
    """Upper bound of simulated time for a one-shot headless window capture (inclusive is handled by caller)."""
    raw = str(os.environ.get("MAGNAVIS_HEADLESS_SNAPSHOT_BEFORE", "")).strip()
    if not raw:
        return None
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _parse_headless_snapshot_window_seconds() -> float:
    """Capture when latest sample time lies in ``[T - window, T)`` (seconds)."""
    try:
        return max(5.0, float(os.environ.get("MAGNAVIS_HEADLESS_SNAPSHOT_WINDOW_SEC", "60")))
    except Exception:
        return 60.0


def _parse_headless_snapshot_png_path() -> Optional[str]:
    raw = str(os.environ.get("MAGNAVIS_HEADLESS_SNAPSHOT_PNG", "")).strip()
    return raw or None


def _headless_batch_enabled() -> bool:
    return str(os.environ.get("MAGNAVIS_HEADLESS_BATCH", "")).strip().lower() in ("1", "true", "yes", "on")


def _fast_csv_playback_requested() -> bool:
    """
    When ``csv_enabled`` (checked by the caller), ingest the magnetic CSV in large simulated time
    steps with a short Qt timer instead of real-time pacing (about 20 s wall per 20 s sim without
    fast mode). Skips periodic matplotlib redraws until playback finishes, then refreshes once.

    **Default is on** (same behaviour as ``run_suite_improved.py --fast-csv-playback``): unset
    ``MAGNAVIS_FAST_CSV_PLAYBACK`` enables fast mode. Set ``MAGNAVIS_FAST_CSV_PLAYBACK=0`` (or
    ``false`` / ``off`` / ``no``) for wall-clock-paced CSV replay.
    """
    raw = str(os.environ.get("MAGNAVIS_FAST_CSV_PLAYBACK", "")).strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    # Unset: fast playback (interactive GUI and headless CSV sessions).
    return True


def _parse_fast_csv_data_interval_ms() -> int:
    """Wall-clock interval between CSV fetch ticks in fast mode (default 1 ms)."""
    try:
        v = int(float(str(os.environ.get("MAGNAVIS_CSV_FAST_DATA_INTERVAL_MS", "1")).strip()))
        return max(0, min(v, 60_000))
    except Exception:
        return 1


def _parse_fast_csv_sim_step_seconds() -> int:
    """
    Simulated time span per CSV incremental fetch in fast mode (default 1 day).
    Lower if a single step would load an impractically large number of rows.
    """
    try:
        v = int(float(str(os.environ.get("MAGNAVIS_CSV_FAST_SIM_STEP_SECONDS", "86400")).strip()))
        return max(20, min(v, 86400 * 120))
    except Exception:
        return 86400


def _fast_db_simulation_requested() -> bool:
    """
    When ``sim_enabled`` and not ``csv_enabled`` (checked by the caller), use a short Qt data timer
    and a larger ``sim_step_seconds`` per DB fetch instead of 20 wall seconds per 20 simulated seconds.

    **Default is on.** Set ``MAGNAVIS_FAST_SIM_PLAYBACK=0`` (or ``false`` / ``off`` / ``no``) for the
    legacy 20 s / 20 s pacing.
    """
    raw = str(os.environ.get("MAGNAVIS_FAST_SIM_PLAYBACK", "")).strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return True


def _parse_fast_sim_data_interval_ms() -> int:
    """Wall-clock interval between DB simulation fetch ticks when fast sim is on (default 50 ms)."""
    try:
        v = int(float(str(os.environ.get("MAGNAVIS_SIM_FAST_DATA_INTERVAL_MS", "50")).strip()))
        return max(10, min(v, 60_000))
    except Exception:
        return 50


def _parse_fast_sim_step_seconds() -> int:
    """Simulated time per DB ``new=True`` fetch when fast sim is on (default 600 s = 10 min)."""
    try:
        v = int(float(str(os.environ.get("MAGNAVIS_SIM_FAST_STEP_SECONDS", "600")).strip()))
        return max(20, min(v, 86400 * 31))
    except Exception:
        return 600


def _parse_batch_csv_end_timestamp() -> Optional[pd.Timestamp]:
    """
    Optional inclusive upper time bound for magnetic CSV rows.

    Set ``MAGNAVIS_BATCH_CSV_END`` to a full timestamp, e.g. ``2026-02-13 16:18:30``, or set
    ``MAGNAVIS_BATCH_CSV_END_DATE=2026-02-13`` with ``MAGNAVIS_BATCH_CSV_END=16:18:30`` (time-only).
    """
    raw = str(os.environ.get("MAGNAVIS_BATCH_CSV_END", "")).strip()
    if not raw:
        return None
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        date_part = str(os.environ.get("MAGNAVIS_BATCH_CSV_END_DATE", "")).strip()
        if not date_part:
            return None
        ts = pd.to_datetime(f"{date_part} {raw}", errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _parse_batch_csv_start_timestamp() -> Optional[pd.Timestamp]:
    """
    Optional inclusive lower time bound for magnetic CSV rows (headless batch / replay trim).

    Set ``MAGNAVIS_BATCH_CSV_START`` to a full timestamp, or ``MAGNAVIS_BATCH_CSV_START_DATE`` with
    time-only in ``MAGNAVIS_BATCH_CSV_START`` (same pattern as ``_parse_batch_csv_end_timestamp``).
    """
    raw = str(os.environ.get("MAGNAVIS_BATCH_CSV_START", "")).strip()
    if not raw:
        return None
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        date_part = str(os.environ.get("MAGNAVIS_BATCH_CSV_START_DATE", "")).strip()
        if not date_part:
            return None
        ts = pd.to_datetime(f"{date_part} {raw}", errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _normalize_model_family(value: Optional[str]) -> str:
    """Normalize PREDICTOR_MODEL_FAMILY for the UI and predictor_ai subprocess."""
    raw = str(value or "").strip().lower()
    if raw in {"gru", "gru_rnn", "gated_recurrent_unit"}:
        return MODEL_FAMILY_GRU
    if raw in {"transformer", "tfm", "trf"}:
        return MODEL_FAMILY_TRANSFORMER
    if raw in {"lstm", "vanilla_lstm", "plain_lstm", "stacked_lstm"}:
        return MODEL_FAMILY_LSTM
    if raw in {"attn_bilstm", "attn-bilstm", "attention_bilstm", "attention-bilstm", "attn", "bilstm"}:
        return MODEL_FAMILY_ATTN_BILSTM
    return MODEL_FAMILY_GRU


def _normalize_model_init(value: Optional[str]) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"fresh", "scratch", "new", "train_new", "train-fresh"}:
        return MODEL_INIT_FRESH
    return MODEL_INIT_PRETRAINED


def _clamp_predictor_gru_window(value: int) -> int:
    try:
        v = int(value)
    except Exception:
        v = DEFAULT_PREDICTOR_GRU_WINDOW
    return max(PREDICTOR_GRU_WINDOW_MIN, min(v, PREDICTOR_GRU_WINDOW_MAX))


def _predictor_gru_window_from_env() -> int:
    raw = os.environ.get("PREDICTOR_GRU_WINDOW_SIZE", "").strip()
    if not raw:
        return DEFAULT_PREDICTOR_GRU_WINDOW
    try:
        return _clamp_predictor_gru_window(int(float(raw)))
    except Exception:
        return DEFAULT_PREDICTOR_GRU_WINDOW


def _initial_threshold_k_from_env() -> float:
    raw = (
        os.environ.get("MAGNAVIS_INITIAL_THRESHOLD_K", "").strip()
        or os.environ.get("INITIAL_ANOMALY_THRESHOLD_K", "").strip()
    )
    if not raw:
        return 2.5
    try:
        return float(max(0.1, min(10.0, float(raw))))
    except Exception:
        return 2.5


def _initial_train_window_minutes_from_env() -> Optional[int]:
    """
    Predictor training-window cap from env (matches GUI spinbox semantics).

    - unset / negative / -1 → None (train on all loaded historic + realtime)
    - 0 → predict-only first run when a pretrained/runtime checkpoint exists (no initial fit)
    - N≥1 → train only on the last N minutes before each training run
    """
    raw = os.environ.get("MAGNAVIS_INITIAL_TRAIN_WINDOW_MINUTES", "").strip()
    if not raw:
        return None
    try:
        v = int(float(raw))
        if v < 0:
            return None
        if v == 0:
            return 0
        return max(1, min(v, 1_000_000))
    except Exception:
        return None


def _ground_truth_visualization_enabled() -> bool:
    """Set MAGNAVIS_DISABLE_GROUND_TRUTH=1 to run without any GT overlays (session, manual CSV, azimuth schedule)."""
    v = os.environ.get("MAGNAVIS_DISABLE_GROUND_TRUTH", "").strip().lower()
    return v not in ("1", "true", "yes", "on")


@dataclass
class SensorContext:
    sensor_id: str
    display_name: str

    # Data buffers
    # Historic snapshot (blue) loaded at startup (up to 60 min @ 1 Hz = 3600 points)
    base_x_t: List[datetime] = field(default_factory=list)
    base_y_mag_t: List[float] = field(default_factory=list)
    # Unsmoothed magnitudes (same indices as base_/rt_); used for predictor CSV when MAGNAVIS_PREDICTOR_RAW_MAG=1.
    base_y_mag_raw_t: List[float] = field(default_factory=list)
    plot_baseline_nT: Optional[float] = None  # used only for visualization (ΔB = B - baseline)

    # Realtime stream (green) accumulated from incremental fetches
    rt_x_t: List[datetime] = field(default_factory=list)
    rt_y_mag_t: List[float] = field(default_factory=list)
    rt_y_mag_raw_t: List[float] = field(default_factory=list)

    # Latest incremental chunk (for anomaly comparison and for "just arrived" UI updates)
    new_x_t: List[datetime] = field(default_factory=list)
    new_y_mag_t: List[float] = field(default_factory=list)

    has_seen_realtime: bool = False  # becomes True once we receive any incremental (green) data
    needs_update_lims: bool = False

    # Prediction buffers
    predict_x_t: List[datetime] = field(default_factory=list)
    predict_y_t: List[float] = field(default_factory=list)
    predictor_input_file: Optional[str] = None
    prediction_process: Optional[subprocess.Popen] = None
    predict_app_started: bool = False

    # Anomaly detection buffers
    anomaly_detector: AnomalyDetector = field(
        default_factory=lambda: AnomalyDetector(
            threshold_multiplier=2.5,
            min_samples_for_threshold=20,
            std_relative_floor=0.02,
        )
    )
    anomaly_times: List[datetime] = field(default_factory=list)
    anomaly_values: List[float] = field(default_factory=list)
    anomaly_vertical_lines: list = field(default_factory=list)
    anomaly_vertical_lines_static: list = field(default_factory=list)
    ground_truth_vertical_bands: list = field(default_factory=list)
    # Incremental anomaly-processing cursor:
    # only realtime points after this timestamp are sent to the detector next time.
    # This also acts as training cut-off for "safe" non-anomalous data usage.
    last_anomaly_checked_time: Optional[datetime] = None

    # UI refs (per-tab)
    static_canvas: Optional[FigureCanvas] = None
    dynamic_canvas: Optional[FigureCanvas] = None
    static_ax = None
    dynamic_ax = None
    static_line = None
    dynamic_line = None
    dynamic_new_line = None
    predictions_line = None

    # Settings per sensor — predictor training window (minutes):
    #   None = no time trim (train on all rows in predict_input when training runs).
    #   0 = skip the initial full-history fit on first run when a checkpoint exists (predict-only first);
    #       if no checkpoint yet (fresh model), first run still trains.
    #   >=1 = trim to the last N minutes before training.
    train_window_minutes: Optional[int] = None
    retrain_interval_minutes: int = 60

    # Low-pass filter state (per-sensor)
    last_filtered_value: Optional[float] = None

    # Performance throttles
    last_saved_points: int = 0
    last_redraw_points: int = 0
    last_pred_poll_ts: float = 0.0
    last_pred_start_ts: float = 0.0
    last_pred_complete_ts: float = 0.0
    last_model_train_data_time: Optional[datetime] = None
    current_run_training: bool = False
    current_run_train_ref_time: Optional[datetime] = None
    runtime_model_path: Optional[str] = None
    # When sequential catch-up caps predict_input at latest_pred, subprocess gets
    # PREDICTOR_COVER_UNTIL=latest_actual so one-step+skip mode can autoreg past the CSV end.
    predict_cover_until: Optional[datetime] = None


class MultiFetchWorker(QObject):
    finished = pyqtSignal()
    updated = pyqtSignal(dict, bool)  # (sensor_id -> mag df), new_flag

    def __init__(self, app=None):
        super().__init__()
        self._app = app

    def fetch_initial_sim(self, sensor_ids: List[str], start_time: datetime, end_time: datetime, last_n: int):
        try:
            dfs = fetch_timeseries_window_multi(sensor_ids, start_time=start_time, end_time=end_time, target_n_seconds=last_n)
        except Exception:
            dfs = {sid: pd.DataFrame(columns=["time_H", "mag_H_nT"]) for sid in sensor_ids}
        self.updated.emit(dfs, False)
        self.finished.emit()

    def fetch_incremental_sim(self, sensor_ids: List[str], start_time: datetime, end_time: datetime):
        try:
            dfs = fetch_timeseries_between_multi(sensor_ids, start_time=start_time, end_time=end_time, limit_rows=20000)
        except Exception:
            dfs = {sid: pd.DataFrame(columns=["time_H", "mag_H_nT"]) for sid in sensor_ids}
        self.updated.emit(dfs, True)
        self.finished.emit()

    def fetch_initial_realtime(self, sensor_ids: List[str], hours: float, last_n: int):
        try:
            dfs = get_timeseries_magnetic_data_multi(
                sensor_ids, hours=float(hours), last_n_samples=int(last_n)
            )
        except Exception:
            dfs = {sid: pd.DataFrame(columns=["time_H", "mag_H_nT"]) for sid in sensor_ids}
        self.updated.emit(dfs, False)
        self.finished.emit()

    def fetch_incremental_realtime(self, sensor_ids: List[str], since_times: Dict[str, datetime]):
        try:
            dfs = get_timeseries_magnetic_data_since_multi(
                sensor_ids, since_times=since_times, limit_rows=5000
            )
        except Exception:
            dfs = {sid: pd.DataFrame(columns=["time_H", "mag_H_nT"]) for sid in sensor_ids}
        self.updated.emit(dfs, True)
        self.finished.emit()

    def fetch_initial_csv(
        self,
        sensor_ids: List[str],
        start_time: datetime,
        end_time: datetime,
        last_n: Optional[int],
    ):
        try:
            if self._app is None:
                raise RuntimeError("CSV worker missing app reference")
            # ``last_n`` is an optional max row-count tail cap. Pass None so the full [start_time, end_time]
            # window is kept (multi-Hz CSVs would otherwise keep only last ``minutes*60`` rows, not minutes of wall time).
            dfs = self._app._fetch_csv_window_multi(
                sensor_ids,
                start_time=start_time,
                end_time=end_time,
                target_n_seconds=last_n,
                incremental=False,
            )
        except Exception:
            dfs = {sid: pd.DataFrame(columns=["time_H", "mag_H_nT"]) for sid in sensor_ids}
        self.updated.emit(dfs, False)
        self.finished.emit()

    def fetch_incremental_csv(self, sensor_ids: List[str], start_time: datetime, end_time: datetime):
        try:
            if self._app is None:
                raise RuntimeError("CSV worker missing app reference")
            dfs = self._app._fetch_csv_window_multi(
                sensor_ids, start_time=start_time, end_time=end_time, target_n_seconds=None, incremental=True
            )
        except Exception:
            dfs = {sid: pd.DataFrame(columns=["time_H", "mag_H_nT"]) for sid in sensor_ids}
        self.updated.emit(dfs, True)
        self.finished.emit()


class SensorMagTimeSeriesWidget(base_app.MagTimeSeriesWidget):
    """
    Reuse the existing UI widget, but route threshold/train-window changes
    to ALL sensor contexts (shared settings across all sensors).
    """

    def __init__(self, app, sensor_id: str, parent=None):
        self.sensor_id = sensor_id
        super().__init__(app, parent=parent)
        self._add_error_smoothing_control()
        self._add_retrain_interval_control()
        # Sync spinbox values from first sensor's context (all sensors share same values)
        self._sync_controls_from_context()
        # Register this widget so we can sync all widgets when one changes
        if not hasattr(app, '_sensor_control_widgets'):
            app._sensor_control_widgets = []
        app._sensor_control_widgets.append(self)

    def _add_error_smoothing_control(self):
        """Add EWMA alpha control for adaptive-threshold memory."""
        ctx = self._ctx()
        try:
            alpha_layout = QHBoxLayout()
            alpha_label = QLabel("Error Memory (EWMA alpha):")
            alpha_layout.addWidget(alpha_label)

            self.error_smoothing_spinbox = QDoubleSpinBox()
            self.error_smoothing_spinbox.setMinimum(0.9000)
            self.error_smoothing_spinbox.setMaximum(0.9999)
            self.error_smoothing_spinbox.setSingleStep(0.0010)
            self.error_smoothing_spinbox.setDecimals(4)
            alpha_val = float(getattr(ctx.anomaly_detector, "error_smoothing_alpha", 0.995))
            self.error_smoothing_spinbox.setValue(alpha_val)
            self.error_smoothing_spinbox.setToolTip(
                "EWMA alpha for error memory.\n"
                "Higher alpha = longer memory of older errors.\n"
                "Update: m_t = alpha*m_(t-1) + (1-alpha)*e_t"
            )
            alpha_layout.addWidget(self.error_smoothing_spinbox)
            self.error_smoothing_spinbox.valueChanged.connect(self.on_error_smoothing_alpha_changed)

            # Place below the training window status label
            self.gridLayout.addLayout(alpha_layout, 8, 0, 1, 5)
        except Exception:
            pass

    def _add_retrain_interval_control(self):
        """Add retrain-interval control (minutes) for predictor cadence."""
        ctx = self._ctx()
        try:
            retrain_layout = QHBoxLayout()
            retrain_label = QLabel("Model Retrain Interval (minutes):")
            retrain_layout.addWidget(retrain_label)

            self.retrain_interval_spinbox = QDoubleSpinBox()
            self.retrain_interval_spinbox.setMinimum(1)
            self.retrain_interval_spinbox.setMaximum(1440)
            self.retrain_interval_spinbox.setSingleStep(5)
            self.retrain_interval_spinbox.setDecimals(0)
            retrain_val = int(max(1, getattr(ctx, "retrain_interval_minutes", 60)))
            self.retrain_interval_spinbox.setValue(float(retrain_val))
            self.retrain_interval_spinbox.setToolTip(
                "Train only once every T minutes (data-time basis).\n"
                "Between retrains, predictor runs inference only."
            )
            retrain_layout.addWidget(self.retrain_interval_spinbox)
            self.retrain_interval_spinbox.valueChanged.connect(self.on_retrain_interval_changed)

            # Place below EWMA alpha control
            self.gridLayout.addLayout(retrain_layout, 9, 0, 1, 5)
        except Exception:
            pass

    def _ctx(self) -> SensorContext:
        return self.app.sensor_ctx[self.sensor_id]

    def _sync_controls_from_context(self):
        """Initialize spinbox values from the first sensor's context (all sensors share same values)."""
        # Use the first sensor's context as the source of truth for shared settings
        if not self.app.sensor_ctx:
            return
        first_ctx = next(iter(self.app.sensor_ctx.values()))
        try:
            # Sync threshold multiplier
            if hasattr(self, 'threshold_spinbox'):
                self.threshold_spinbox.setValue(first_ctx.anomaly_detector.threshold_multiplier)
            # Sync training window
            if hasattr(self, 'train_window_spinbox'):
                tw = first_ctx.train_window_minutes
                if tw is None:
                    self.train_window_spinbox.setValue(-1)
                elif tw == 0:
                    self.train_window_spinbox.setValue(0)
                else:
                    self.train_window_spinbox.setValue(int(tw))
            # Sync EWMA alpha
            if hasattr(self, 'error_smoothing_spinbox'):
                alpha_val = float(getattr(first_ctx.anomaly_detector, "error_smoothing_alpha", 0.995))
                self.error_smoothing_spinbox.setValue(alpha_val)
            # Sync retrain interval
            if hasattr(self, 'retrain_interval_spinbox'):
                retrain_val = int(max(1, getattr(first_ctx, "retrain_interval_minutes", 60)))
                self.retrain_interval_spinbox.setValue(float(retrain_val))
            # Update status label
            self.update_train_window_status()
        except Exception:
            pass  # If controls don't exist yet, ignore

    def on_threshold_changed(self, value):
        """Apply threshold multiplier change to ALL sensors (shared setting)."""
        old_values = {}
        for sid, ctx in self.app.sensor_ctx.items():
            old_values[sid] = ctx.anomaly_detector.threshold_multiplier
            ctx.anomaly_detector.threshold_multiplier = value
            self.app.reset_anomaly_state_for_sensor(sid)
            # Re-run anomaly detection for this sensor
            self.app.detect_anomalies_for_sensor(sid)
            QTimer.singleShot(200, lambda s=sid: self.app.update_canvas_for_sensor(s))
        
        # Sync all other control widgets to show the new value (without triggering their callbacks)
        if hasattr(self.app, '_sensor_control_widgets'):
            for widget in self.app._sensor_control_widgets:
                if widget is not self and hasattr(widget, 'threshold_spinbox'):
                    # Temporarily block signals to avoid recursive updates
                    widget.threshold_spinbox.blockSignals(True)
                    widget.threshold_spinbox.setValue(value)
                    widget.threshold_spinbox.blockSignals(False)
        
        # Log change for all sensors
        sensor_names = ", ".join([ctx.display_name for ctx in self.app.sensor_ctx.values()])
        old_avg = sum(old_values.values()) / len(old_values) if old_values else value
        self.app.log(
            f'[All Sensors: {sensor_names}] Anomaly threshold multiplier changed {old_avg:.2f} -> {value:.2f} (shared setting)',
            level="Info"
        )

    def on_train_window_changed(self, value):
        """Apply training window change to ALL sensors (shared setting)."""
        if value < 0:
            train_minutes = None
        elif value == 0:
            train_minutes = 0
        else:
            train_minutes = int(value)
        for sid, ctx in self.app.sensor_ctx.items():
            ctx.train_window_minutes = train_minutes
        
        # Sync all other control widgets to show the new value (without triggering their callbacks)
        if hasattr(self.app, '_sensor_control_widgets'):
            for widget in self.app._sensor_control_widgets:
                if widget is not self and hasattr(widget, 'train_window_spinbox'):
                    widget.train_window_spinbox.blockSignals(True)
                    widget.train_window_spinbox.setValue(value)
                    widget.train_window_spinbox.blockSignals(False)
                widget.update_train_window_status()  # Update status label for all widgets
        
        # Log change
        sensor_names = ", ".join([ctx.display_name for ctx in self.app.sensor_ctx.values()])
        if train_minutes is None:
            self.app.log(
                f'[All Sensors: {sensor_names}] Training window: all loaded data (shared setting)',
                level="Info",
            )
        elif train_minutes == 0:
            self.app.log(
                f'[All Sensors: {sensor_names}] Training window: 0 min → first predictor run is predict-only '
                f"when a checkpoint/pretrained model exists (shared setting)",
                level="Info",
            )
        else:
            self.app.log(
                f'[All Sensors: {sensor_names}] Training window: last {train_minutes} minutes (shared setting)',
                level="Info",
            )
        
        # Update status label for this widget
        self.update_train_window_status()

    def on_error_smoothing_alpha_changed(self, value):
        """Apply EWMA alpha change to ALL sensors (shared setting)."""
        alpha = float(value)
        old_values = {}
        for sid, ctx in self.app.sensor_ctx.items():
            old_values[sid] = float(getattr(ctx.anomaly_detector, "error_smoothing_alpha", 0.995))
            ctx.anomaly_detector.error_smoothing_alpha = alpha

        # Sync all other control widgets to show the new value
        if hasattr(self.app, '_sensor_control_widgets'):
            for widget in self.app._sensor_control_widgets:
                if widget is not self and hasattr(widget, 'error_smoothing_spinbox'):
                    widget.error_smoothing_spinbox.blockSignals(True)
                    widget.error_smoothing_spinbox.setValue(alpha)
                    widget.error_smoothing_spinbox.blockSignals(False)
                widget.update_train_window_status()

        sensor_names = ", ".join([ctx.display_name for ctx in self.app.sensor_ctx.values()])
        old_avg = sum(old_values.values()) / len(old_values) if old_values else alpha
        self.app.log(
            f'[All Sensors: {sensor_names}] EWMA alpha changed {old_avg:.4f} -> {alpha:.4f} (shared setting)',
            level="Info",
        )
        self.app.log(
            f'[All Sensors: {sensor_names}] EWMA memory update: m_t = alpha*m_(t-1) + (1-alpha)*e_t',
            level="Debug",
        )
        self.update_train_window_status()

    def on_retrain_interval_changed(self, value):
        """Apply model retrain interval change to ALL sensors (shared setting)."""
        interval_minutes = int(max(1, value))
        old_values = {}
        for sid, ctx in self.app.sensor_ctx.items():
            old_values[sid] = int(max(1, getattr(ctx, "retrain_interval_minutes", 60)))
            ctx.retrain_interval_minutes = interval_minutes
        self.app._default_retrain_interval_minutes = interval_minutes
        os.environ["PREDICTOR_RETRAIN_INTERVAL_MINUTES"] = str(interval_minutes)

        # Sync all other control widgets to show the new value
        if hasattr(self.app, '_sensor_control_widgets'):
            for widget in self.app._sensor_control_widgets:
                if widget is not self and hasattr(widget, 'retrain_interval_spinbox'):
                    widget.retrain_interval_spinbox.blockSignals(True)
                    widget.retrain_interval_spinbox.setValue(float(interval_minutes))
                    widget.retrain_interval_spinbox.blockSignals(False)
                widget.update_train_window_status()

        sensor_names = ", ".join([ctx.display_name for ctx in self.app.sensor_ctx.values()])
        old_avg = sum(old_values.values()) / len(old_values) if old_values else interval_minutes
        self.app.log(
            f"[All Sensors: {sensor_names}] Model retrain interval changed {old_avg:.0f} -> {interval_minutes:.0f} min (shared setting)",
            level="Info",
        )
        self.update_train_window_status()

    def update_train_window_status(self):
        """Update status label using first sensor's context (all sensors share same values)."""
        if not self.app.sensor_ctx:
            return
        first_ctx = next(iter(self.app.sensor_ctx.values()))
        minutes = first_ctx.train_window_minutes
        if minutes is None:
            window_text = "Training window: all loaded data"
        elif minutes == 0:
            window_text = "Training window: 0 min (predict-only first if checkpoint)"
        else:
            window_text = f"Training window: last {int(minutes)} min"
        alpha = float(getattr(first_ctx.anomaly_detector, "error_smoothing_alpha", 0.995))
        retrain_minutes = int(max(1, getattr(first_ctx, "retrain_interval_minutes", 60)))
        self.train_window_status_label.setText(
            f"{window_text} | Retrain: {retrain_minutes} min | EWMA alpha: {alpha:.4f} | "
            f"Time-of-day features: on (daily sin/cos) | <i>(shared across all sensors)</i>"
        )




class ApplicationWindowTemp(base_app.ApplicationWindow):
    def __init__(self, app, parent=None):
        super().__init__(app, parent=parent)
        self.framework_2_loaded = False
        self._fetch_thread: Optional[QThread] = None
        self._fetch_worker: Optional[MultiFetchWorker] = None
        self._temp_timeseries_container: Optional[QWidget] = None
        self._simplify_ui_for_multisensor()

    def _simplify_ui_for_multisensor(self):
        """
        Two-column layout:
        - LEFT half: sensor stream panels (3 panels, each scrollable L/R/U/D) — filled by load_plot_framework_2.
        - RIGHT half: selection parameters and logs.
        """
        try:
            self.setMinimumSize(1100, 650)

            try:
                self.menuBar().hide()
            except Exception:
                pass
            try:
                self.statusbar.hide()
            except Exception:
                pass

            log_widget = self.textEditLog
            try:
                log_widget.setParent(None)
            except Exception:
                pass

            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(0)

            # Top-left: one-click HD capture of the whole main window (same pipeline as Ctrl+Shift+S).
            if not _headless_batch_enabled():
                top_bar = QWidget()
                top_row = QHBoxLayout(top_bar)
                top_row.setContentsMargins(0, 0, 0, 2)
                top_row.setSpacing(8)
                snap_btn = QPushButton("HD snapshot (full window)")
                snap_btn.setObjectName("magnavisHdFullWindowSnapshotButton")
                snap_btn.setToolTip(
                    "Saves the main content (everything below this bar) as a high-resolution PNG — the snapshot button row is excluded. "
                    "Resolution multiplier: MAGNAVIS_UI_SNAPSHOT_SCALE (default 2). Same as Ctrl+Shift+S; path is logged after export."
                )
                try:
                    snap_btn.clicked.connect(self._app._export_hd_window_snapshot)
                except Exception:
                    pass
                top_row.addWidget(snap_btn, 0, QtCore.Qt.AlignLeft)
                top_row.addStretch(1)
                layout.addWidget(top_bar, 0)

            # Horizontal splitter: left = sensor streams, right = parameters + log
            h_splitter = QSplitter(QtCore.Qt.Horizontal)
            layout.addWidget(h_splitter)
            # HD snapshot renders this widget so the top snapshot bar is not included in the PNG.
            self._hd_snapshot_capture_widget = h_splitter

            # ---- LEFT: placeholder for sensor stream panels (each scrollable)
            left_half = QWidget()
            left_layout = QVBoxLayout(left_half)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(6)
            left_half.setMinimumWidth(400)
            left_half.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            h_splitter.addWidget(left_half)
            self._temp_timeseries_container = left_half

            # ---- RIGHT: parameters placeholder + log
            right_half = QWidget()
            right_layout = QVBoxLayout(right_half)
            right_layout.setContentsMargins(6, 6, 6, 6)
            right_layout.setSpacing(8)
            right_half.setMinimumWidth(320)
            right_half.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._right_half = right_half
            self._right_layout = right_layout

            # 1) Slot for shared parameters (load_plot_framework_2 will insert at index 0)
            # Keep a ref so we can insert the params panel; no placeholder widget to avoid blank area
            self._right_parameters_placeholder = None  # unused; we insert into _right_layout at 0

            # 2) Log — scrollable
            try:
                log_widget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
                log_widget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
                log_widget.setMinimumHeight(120)
                log_widget.setMaximumHeight(400)
            except Exception:
                pass
            right_layout.addWidget(log_widget, 1)

            h_splitter.addWidget(right_half)
            h_splitter.setStretchFactor(0, 1)  # left
            h_splitter.setStretchFactor(1, 1)  # right
            # Initial sizes: left 50%, right 50% (in pixels, approximate)
            h_splitter.setSizes([550, 550])
        except Exception:
            self._temp_timeseries_container = None
            self._right_layout = getattr(self, "_right_layout", None)

    def startThreads(self, hours, start_time, new):
        # Replace USGS fetch threads with DB multi-sensor fetch.
        try:
            self._app._discover_sensors()
        except Exception as e:
            # Keep the app alive on transient DB discovery failures.
            self._app.sensor_ids = []
            self._app.sensor_ctx = {}
            self._app.log(f"Sensor discovery failed at startup: {e}", level="Error")
        if not getattr(self._app, "_startup_configured", True):
            return
        if not self._app.sensor_ids:
            self._app.log(
                "No sensor streams to fetch (CSV not loaded, DB empty, or sensor selection missing). "
                "Check CSV columns and sensor selection.",
                level="Error",
            )
            return
        # Wait until the last CSV incremental batch is applied on the GUI thread (worker may finish first).
        if new and getattr(self._app, "csv_enabled", False):
            if getattr(self._app, "_csv_incremental_gui_busy", False):
                return
        # Qt may delete the underlying C++ QThread after `deleteLater()`. If we keep
        # a Python reference, calling methods like isRunning() can raise:
        # "RuntimeError: wrapped C/C++ object of type QThread has been deleted".
        if self._fetch_thread is not None:
            try:
                if self._fetch_thread.isRunning():
                    return
            except RuntimeError:
                # Stale reference to a deleted thread; clear and continue.
                self._fetch_thread = None

        self._app.log(f"Fetching data for sensors: {', '.join(self._app.sensor_ids)}", level="Info")

        # Keep strong references to avoid PyQt GC destroying objects while running.
        thread = QThread(self)
        worker = MultiFetchWorker(app=self._app)
        worker.moveToThread(thread)

        if not new:
            if self._app.csv_enabled:
                # CSV: initial historic window (up to 60 min; or all data if less available).
                thread.started.connect(
                    lambda: worker.fetch_initial_csv(
                        self._app.sensor_ids,
                        self._app.sim_start_time,
                        self._app.sim_hist_end_time,
                        None,
                    )
                )
            elif self._app.sim_enabled:
                # Simulation: initial historic window (up to 60 min; or all if less).
                thread.started.connect(
                    lambda: worker.fetch_initial_sim(
                        self._app.sensor_ids,
                        self._app.sim_start_time,
                        self._app.sim_hist_end_time,
                        self._app.historic_points_1hz,
                    )
                )
            else:
                # Real-time: most recent historic window from DB (or all available if less).
                thread.started.connect(
                    lambda: worker.fetch_initial_realtime(
                        self._app.sensor_ids, hours=self._app.historic_minutes / 60.0, last_n=self._app.historic_points_1hz
                    )
                )
        else:
            if self._app.csv_enabled:
                # CSV: fetch next slice for simulated realtime playback.
                thread.started.connect(
                    lambda: worker.fetch_incremental_csv(
                        self._app.sensor_ids,
                        self._app.sim_rt_start_time,
                        self._app.sim_rt_end_time,
                    )
                )
            elif self._app.sim_enabled:
                # Simulation: fetch the next slice after the current simulated time.
                thread.started.connect(
                    lambda: worker.fetch_incremental_sim(
                        self._app.sensor_ids,
                        self._app.sim_rt_start_time,
                        self._app.sim_rt_end_time,
                    )
                )
            else:
                # Real-time: fetch new points since last known timestamps.
                thread.started.connect(
                    lambda: worker.fetch_incremental_realtime(
                        self._app.sensor_ids,
                        self._app.get_since_times(),
                    )
                )

        # QueuedConnection: worker runs on a QThread; handler touches Qt widgets / model state on the GUI thread.
        worker.updated.connect(self._app.on_db_data_updated, QtCore.Qt.QueuedConnection)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: setattr(self, "_fetch_thread", None))
        thread.finished.connect(lambda: setattr(self, "_fetch_worker", None))

        self._fetch_thread = thread
        self._fetch_worker = worker
        if new and getattr(self._app, "csv_enabled", False):
            self._app._csv_incremental_gui_busy = True
        thread.start()

    def updateData(self):
        # DB updates come via on_db_data_updated; keep compatibility.
        if not self.framework_2_loaded:
            if self._app.sensor_ctx and any(len(ctx.base_x_t) > 0 for ctx in self._app.sensor_ctx.values()):
                self._app.load_plot_framework_2()
                self.framework_2_loaded = True
        else:
            # Defer redraw so we don't stack heavy Matplotlib work in the same event-loop slice as data ingest.
            # Fast CSV pauses the draw timer, so we must still refresh here or predictions/anomalies never paint.
            QTimer.singleShot(0, self._app.update_all_canvases)


class ApplicationTemp(base_app.Application):
    def __init__(self, arg):
        # Pre-init fields used by overridden startThreads/updateData
        self.sensor_ids: List[str] = []
        self.sensor_ctx: Dict[str, SensorContext] = {}
        self._time_series_tabs: Optional[QTabWidget] = None
        self._multi_data_timer: Optional[QTimer] = None
        self._multi_draw_timer: Optional[QTimer] = None
        self._multi_pred_timer: Optional[QTimer] = None
        self._slow_redraw_tick: int = 0
        self._predict_queue: deque[str] = deque()
        self._predict_active: set[str] = set()
        # Allow up to 3 predictor processes in parallel (one per sensor when 3 sensors are selected).
        # This keeps sensors independent while still avoiding unbounded TensorFlow concurrency.
        self._predict_max_concurrent: int = 3
        self._predict_sched_timer: Optional[QTimer] = None
        self._predict_cooldown_seconds: int = 20  # per-sensor minimum gap between predictor runs
        # Debounce writing large predict_input.csv files (CSV playback was blocking the GUI thread every fetch).
        self._predict_input_pending: Dict[str, Dict[str, Any]] = {}
        self._predict_input_save_timer = QTimer()
        self._predict_input_save_timer.setSingleShot(True)
        self._predict_input_save_timer.timeout.connect(self._flush_predict_input_writes)
        try:
            _retrain_env = int(float(os.environ.get("PREDICTOR_RETRAIN_INTERVAL_MINUTES", "60")))
        except Exception:
            _retrain_env = 60
        self._default_retrain_interval_minutes: int = max(1, _retrain_env)
        # Plotting semantics to mirror application.py:
        # - blue = historic snapshot (up to HISTORIC_MINUTES; less if that much not available)
        # - green = realtime accumulated
        self._rolling_window_points: int = HISTORIC_POINTS_1HZ
        self._predict_start_grace_seconds: int = 25  # if no realtime arrives, still start predictor after this delay

        # Simulation clock: start from 2026-01-05 and advance in fixed steps to simulate realtime.
        self.sim_enabled: bool = True
        self.sim_start_time: datetime = datetime(2026, 1, 5, 0, 0, 0)
        # Historic window length: HISTORIC_MINUTES. Downsample to 1 Hz, keep up to HISTORIC_POINTS_1HZ (or all if less).
        self.sim_hist_end_time: datetime = self.sim_start_time + timedelta(minutes=HISTORIC_MINUTES)
        self.sim_step_seconds: int = 20  # aligns with fetch timer cadence (every 20 sec)
        self.sim_rt_start_time: datetime = self.sim_hist_end_time
        self.sim_rt_end_time: datetime = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)
        self._startup_configured: bool = False
        self._initial_fetch_retry: bool = False
        self.csv_enabled: bool = False
        self.csv_path: Optional[str] = None
        self._csv_timeseries_by_sensor: Dict[str, pd.DataFrame] = {}
        self._csv_time_min: Optional[datetime] = None
        self._csv_time_max: Optional[datetime] = None
        self._csv_playback_complete: bool = False  # True when we've passed end of CSV and stopped fetch timer
        # When fast CSV mode is on (default for csv_enabled; see _fast_csv_playback_requested): skip heavy plot refresh during ingest.
        self._csv_fast_playback: bool = False
        self._predict_cooldown_seconds_before_fast: int = 20
        # Serialize CSV incremental fetches: worker can finish before the GUI slot runs; without this,
        # the next timer tick starts another fetch while the clock/data is still stale → races and a frozen plot.
        self._csv_incremental_gui_busy: bool = False
        # After fast CSV ingest: drive predict-only catch-up + full anomaly pass for snapshot-ready UI.
        self._csv_catchup_predict_only: bool = False
        self._csv_catchup_timer: Optional[QTimer] = None
        self._csv_catchup_ticks: int = 0
        self._predict_max_concurrent_before_csv_catchup: int = 3
        self._headless_snapshot_taken: bool = False
        self.csv_hist_minutes: int = HISTORIC_MINUTES
        self.csv_hist_points: int = HISTORIC_POINTS_1HZ
        self.historic_minutes: int = HISTORIC_MINUTES  # user-configured initial load (minutes)
        self.predictor_model_family: str = _normalize_model_family(os.environ.get("PREDICTOR_MODEL_FAMILY"))
        self.predictor_model_init: str = _normalize_model_init(os.environ.get("PREDICTOR_MODEL_INIT"))
        self.predictor_gru_window_size: int = _predictor_gru_window_from_env()
        # Startup prompts (_configure_startup_mode) refine these; used when creating each SensorContext.
        self._initial_threshold_k: float = _initial_threshold_k_from_env()
        self._initial_predictor_train_window_minutes: Optional[int] = _initial_train_window_minutes_from_env()
        self.historic_points_1hz: int = HISTORIC_POINTS_1HZ  # historic_minutes * 60
        self._selected_sensor_ids: Optional[List[str]] = None
        self._ground_truth_anomaly_times: List[datetime] = []
        self._ground_truth_intervals: List[Tuple[datetime, datetime]] = []
        self._ground_truth_intervals_by_sensor: Dict[str, List[Tuple[datetime, datetime]]] = {}
        self._ground_truth_magnet_intervals: List[Tuple[datetime, datetime]] = []
        self._ground_truth_trimmer_intervals: List[Tuple[datetime, datetime]] = []
        self._ground_truth_magnet_intervals_by_sensor: Dict[str, List[Tuple[datetime, datetime]]] = {}
        self._ground_truth_trimmer_intervals_by_sensor: Dict[str, List[Tuple[datetime, datetime]]] = {}

        # Low-pass filter configuration (simple exponential moving average)
        # alpha close to 0 => heavier smoothing, close to 1 => lighter smoothing.
        # Default slightly higher than legacy 0.2 so the GRU sees more high-frequency structure (env: MAGNAVIS_LOWPASS_ALPHA).
        try:
            self._lowpass_alpha = float(os.environ.get("MAGNAVIS_LOWPASS_ALPHA", "0.45"))
            self._lowpass_alpha = max(0.01, min(self._lowpass_alpha, 1.0))
        except Exception:
            self._lowpass_alpha = 0.45
        self._predictor_use_raw_mag: bool = str(os.environ.get("MAGNAVIS_PREDICTOR_RAW_MAG", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        try:
            _nfc = str(os.environ.get("PREDICTOR_N_FUTURE_CAP", "")).strip()
            self._predictor_n_future_cap: Optional[int] = max(100, int(_nfc)) if _nfc else None
        except Exception:
            self._predictor_n_future_cap = None

        # Use base init (VTK, map, menus, etc.)
        super().__init__(arg)
        self._configure_high_contrast_timeseries_theme()
        self._configure_startup_mode()
        self._startup_configured = True
        # Defer GT load so the Qt event loop can paint and stay responsive (GT may touch disk).
        if _ground_truth_visualization_enabled():
            QTimer.singleShot(0, self._load_ground_truth_from_reference_session)
        else:
            self.log(
                "Ground truth overlays disabled (MAGNAVIS_DISABLE_GROUND_TRUTH).",
                level="Info",
            )
        # Base __init__ triggers an initial startThreads() call. Ensure we actually
        # fetch initial data for the discovered sensors (in case the first call
        # happened before discovery or before startup selection).
        try:
            self.appWin.startThreads(hours=1, start_time=None, new=False)
        except Exception:
            pass
        if _headless_batch_enabled():
            self._install_headless_batch_quit_watcher()
        else:
            QTimer.singleShot(800, self._install_ui_capture_shortcuts)

    def _snapshots_export_root(self) -> str:
        """Directory for HD captures: ``<project_root>/snapshots`` (created on demand)."""
        project_root = os.path.dirname(base_app.APP_BASE)
        root = os.path.join(project_root, "snapshots")
        os.makedirs(root, exist_ok=True)
        return root

    def _install_ui_capture_shortcuts(self) -> None:
        """
        Keyboard shortcuts for publication-quality captures (menubar is hidden in the temp UI).

        - **Ctrl+Shift+P** — export all Matplotlib figures (per-sensor time series)
          using ``Figure.savefig(..., dpi=MAGNAVIS_EXPORT_FIG_DPI)`` (default 300).
        - **Ctrl+Shift+S** — rasterise the whole main window at ``MAGNAVIS_UI_SNAPSHOT_SCALE`` (default 2×)
          logical resolution via ``QPainter`` + ``QWidget.render`` (PNG, not a Matplotlib figure).

        Output: ``snapshots/export_<timestamp>/`` under the project root (see log line after export).
        """
        if _headless_batch_enabled():
            return
        win = getattr(self, "appWin", None)
        if win is None:
            return
        try:
            sc_plots = QShortcut(QKeySequence("Ctrl+Shift+P"), win)
            sc_plots.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            sc_plots.activated.connect(self._export_hd_matplotlib_figures)
            sc_win = QShortcut(QKeySequence("Ctrl+Shift+S"), win)
            sc_win.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            sc_win.activated.connect(self._export_hd_window_snapshot)
        except Exception as e:
            self.log(f"Could not register HD capture shortcuts: {e}", level="Warning")
            return
        self.log(
            "HD capture: top-left “HD snapshot (full window)” or Ctrl+Shift+S → PNG of main content "
            "below the snapshot bar (scale MAGNAVIS_UI_SNAPSHOT_SCALE, default 2). Ctrl+Shift+P → matplotlib figures "
            "(DPI from MAGNAVIS_EXPORT_FIG_DPI, default 300). Saves under snapshots/.",
            level="Info",
        )

    def _export_hd_matplotlib_figures(self) -> None:
        """Write high-DPI PNGs for each sensor timeseries figure, if present."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self._snapshots_export_root(), f"export_{stamp}")
        os.makedirs(out_dir, exist_ok=True)
        dpi = _parse_export_fig_dpi()
        n = 0
        try:
            for sid in getattr(self, "sensor_ids", []) or []:
                ctx = self.sensor_ctx.get(sid)
                if ctx is None or getattr(ctx, "dynamic_canvas", None) is None:
                    continue
                fig = ctx.dynamic_canvas.figure
                try:
                    ctx.dynamic_canvas.draw()
                except Exception:
                    pass
                safe = re.sub(r"[^\w.\-]+", "_", str(sid))
                path = os.path.join(out_dir, f"magnetic_timeseries_{safe}.png")
                fig.savefig(
                    path,
                    dpi=dpi,
                    bbox_inches="tight",
                    facecolor=TIMESERIES_COLOR_BACKGROUND,
                    edgecolor="none",
                    pad_inches=0.05,
                )
                n += 1
        except Exception as e:
            self.log(f"HD matplotlib export failed: {e}", level="Warning")
            return
        self.log(f"HD matplotlib export: {n} file(s) → {out_dir} (dpi={dpi})", level="Info")

    def _export_hd_window_snapshot(self) -> None:
        """Rasterise the main content widget (or full window) at integer scale for a sharp PNG (not Matplotlib)."""
        win = getattr(self, "appWin", None)
        if win is None:
            return
        capture = getattr(win, "_hd_snapshot_capture_widget", None)
        target = (
            capture
            if (
                capture is not None
                and isinstance(capture, QWidget)
                and capture.isVisible()
                and capture.width() > 0
                and capture.height() > 0
            )
            else win
        )
        scale = _parse_ui_snapshot_scale()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self._snapshots_export_root(), f"export_{stamp}")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "magnavis_full_window.png")
        try:
            target.repaint()
            QApplication.processEvents()
            w, h = target.width(), target.height()
            if w <= 0 or h <= 0:
                self.log("Window snapshot skipped: invalid target size.", level="Warning")
                return
            pw = max(1, int(round(w * scale)))
            ph = max(1, int(round(h * scale)))
            pm = QPixmap(pw, ph)
            pm.fill(QtCore.Qt.white)
            painter = QPainter(pm)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.scale(scale, scale)
            target.render(painter)
            painter.end()
            if not pm.save(path, "PNG"):
                self.log(f"Window snapshot failed to save: {path}", level="Warning")
                return
        except Exception as e:
            self.log(f"Window snapshot failed: {e}", level="Warning")
            return
        region = "main content (below snapshot bar)" if target is not win else "full window"
        self.log(
            f"HD window snapshot ({region}): {path} ({pw}×{ph} px, scale={scale}× logical {w}×{h})",
            level="Info",
        )

    def _export_hd_window_snapshot_to(self, out_path: str) -> bool:
        """Rasterise main content (or full window) to ``out_path`` (PNG). Same pipeline as Ctrl+Shift+S."""
        win = getattr(self, "appWin", None)
        if win is None:
            return False
        capture = getattr(win, "_hd_snapshot_capture_widget", None)
        target = (
            capture
            if (
                capture is not None
                and isinstance(capture, QWidget)
                and capture.isVisible()
                and capture.width() > 0
                and capture.height() > 0
            )
            else win
        )
        scale = _parse_ui_snapshot_scale()
        _snap_dir = os.path.dirname(os.path.abspath(out_path))
        if _snap_dir:
            try:
                os.makedirs(_snap_dir, exist_ok=True)
            except Exception:
                pass
        try:
            target.repaint()
            QApplication.processEvents()
            w, h = target.width(), target.height()
            if w <= 0 or h <= 0:
                self.log("Headless snapshot skipped: invalid target size.", level="Warning")
                return False
            pw = max(1, int(round(w * scale)))
            ph = max(1, int(round(h * scale)))
            pm = QPixmap(pw, ph)
            pm.fill(QtCore.Qt.white)
            painter = QPainter(pm)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.scale(scale, scale)
            target.render(painter)
            painter.end()
            if not pm.save(out_path, "PNG"):
                self.log(f"Headless snapshot failed to save: {out_path}", level="Warning")
                return False
        except Exception as e:
            self.log(f"Headless snapshot failed: {e}", level="Warning")
            return False
        region = "main content" if target is not win else "full window"
        self.log(
            f"Headless HD snapshot ({region}): {out_path} ({pw}×{ph} px, scale={scale}× logical {w}×{h})",
            level="Info",
        )
        return True

    def _glob_latest_sample_time(self) -> Optional[pd.Timestamp]:
        """Latest timestamp across sensors (historic + realtime buffers)."""
        tbest: Optional[pd.Timestamp] = None
        for sid in getattr(self, "sensor_ids", []) or []:
            ctx = self.sensor_ctx.get(sid)
            if ctx is None:
                continue
            for seq in (ctx.rt_x_t, ctx.base_x_t):
                if not seq:
                    continue
                try:
                    t = pd.Timestamp(seq[-1])
                except Exception:
                    continue
                if tbest is None or t > tbest:
                    tbest = t
        return tbest

    def _maybe_headless_snapshot_before_csv_end(self) -> None:
        """
        One-shot full-window PNG during headless CSV replay, when simulated data time is
        just before ``MAGNAVIS_HEADLESS_SNAPSHOT_BEFORE`` (benchmarks set this to CSV end).

        Requires ``MAGNAVIS_HEADLESS_SNAPSHOT_PNG`` (output path). Intended for GRU pretrained
        runs: set ``PREDICTOR_MODEL_FAMILY``/``INIT`` accordingly; capture uses Qt rasterisation
        (same as interactive ``MAGNAVIS_UI_SNAPSHOT_SCALE``), not Plotly.
        """
        if not _headless_batch_enabled() or self._headless_snapshot_taken:
            return
        out_path = _parse_headless_snapshot_png_path()
        if not out_path:
            return
        if self.predictor_model_family != MODEL_FAMILY_GRU or self.predictor_model_init != MODEL_INIT_PRETRAINED:
            return
        t_cut = _parse_headless_snapshot_before_ts()
        if t_cut is None or pd.isna(t_cut):
            return
        win_sec = _parse_headless_snapshot_window_seconds()
        lo = t_cut - pd.Timedelta(seconds=win_sec)
        hi = t_cut
        glob_t = self._glob_latest_sample_time()
        if glob_t is None or pd.isna(glob_t):
            return
        if not (lo <= glob_t < hi):
            return
        if self._export_hd_window_snapshot_to(out_path):
            self._headless_snapshot_taken = True

    def _configure_predictor_model_family_headless(self) -> None:
        """Apply predictor env without modal dialogs (MAGNAVIS_HEADLESS_BATCH)."""
        self.predictor_model_family = _normalize_model_family(os.environ.get("PREDICTOR_MODEL_FAMILY"))
        self.predictor_model_init = _normalize_model_init(os.environ.get("PREDICTOR_MODEL_INIT"))
        if self.predictor_model_family == MODEL_FAMILY_GRU and self.predictor_model_init == MODEL_INIT_FRESH:
            self.predictor_gru_window_size = _clamp_predictor_gru_window(int(self.predictor_gru_window_size))
            os.environ["PREDICTOR_GRU_WINDOW_SIZE"] = str(self.predictor_gru_window_size)
        elif self.predictor_model_family == MODEL_FAMILY_LSTM and self.predictor_model_init == MODEL_INIT_FRESH:
            self.predictor_gru_window_size = _clamp_predictor_gru_window(int(self.predictor_gru_window_size))
            os.environ["PREDICTOR_GRU_WINDOW_SIZE"] = str(self.predictor_gru_window_size)
        else:
            os.environ.pop("PREDICTOR_GRU_WINDOW_SIZE", None)
        os.environ["PREDICTOR_MODEL_FAMILY"] = self.predictor_model_family
        os.environ["PREDICTOR_MODEL_INIT"] = self.predictor_model_init
        self.log(
            f"Headless batch predictor: family={self.predictor_model_family} init={self.predictor_model_init}",
            level="Info",
        )
        # Optional single knob: map to predictor subprocess env (benchmarks set LEADING/SKIP explicitly).
        split_raw = os.environ.get("MAGNAVIS_PREDICTOR_INITIAL_SPLIT_MINUTES", "").strip()
        if split_raw and not os.environ.get("PREDICTOR_LEADING_TRAIN_MINUTES") and not os.environ.get(
            "PREDICTOR_SKIP_INITIAL_MINUTES"
        ):
            try:
                sm = float(split_raw)
            except Exception:
                sm = 0.0
            if sm > 0:
                if self.predictor_model_init == MODEL_INIT_FRESH and self.predictor_model_family in (
                    MODEL_FAMILY_GRU,
                    MODEL_FAMILY_LSTM,
                    MODEL_FAMILY_ATTN_BILSTM,
                ):
                    os.environ["PREDICTOR_LEADING_TRAIN_MINUTES"] = str(sm)
                elif self.predictor_model_init == MODEL_INIT_PRETRAINED or self.predictor_model_family == MODEL_FAMILY_TRANSFORMER:
                    os.environ["PREDICTOR_SKIP_INITIAL_MINUTES"] = str(sm)

    def _configure_startup_mode_headless(self) -> None:
        """Non-interactive startup: CSV path, sensors, GT, and predictor all from environment."""
        csv_path = os.environ.get("MAGNAVIS_BATCH_CSV", "").strip()
        if not csv_path:
            raise SystemExit("MAGNAVIS_HEADLESS_BATCH=1 requires MAGNAVIS_BATCH_CSV (path to CSV).")
        if not os.path.isabs(csv_path):
            _root = os.path.dirname(base_app.APP_BASE)
            csv_path = os.path.normpath(os.path.join(_root, csv_path))
        try:
            hm = int(os.environ.get("MAGNAVIS_BATCH_HISTORIC_MINUTES", "90"))
        except Exception:
            hm = 90
        hm = max(0, min(hm, 10080))
        self.historic_minutes = hm
        self.historic_points_1hz = self.historic_minutes * 60
        self.csv_hist_minutes = self.historic_minutes
        self.csv_hist_points = self.historic_points_1hz
        self.sim_hist_end_time = self.sim_start_time + timedelta(minutes=self.historic_minutes)
        self.log(f"Headless batch: historic window {self.historic_minutes} min", level="Info")

        self._configure_predictor_model_family_headless()
        self._prompt_initial_detector_and_training_settings()

        if not self._load_csv_source(csv_path):
            raise SystemExit(f"Headless batch: failed to load CSV {csv_path!r}")
        self.csv_enabled = True
        self.sim_enabled = True
        self._configure_sensor_selection()
        self._discover_sensors()
        if _ground_truth_visualization_enabled():
            self._apply_manual_ground_truth_for_known_csv_experiment()
        self.log(f"Headless batch: CSV loaded ({os.path.basename(csv_path)}), sensors={self.sensor_ids}", level="Info")

    def _install_headless_batch_quit_watcher(self) -> None:
        """After CSV playback and predictor subprocesses finish, exit the Qt loop."""
        self._headless_quit_phase = "csv"
        self._headless_idle_ticks = 0
        self._headless_total_ticks = 0

        def _tick() -> None:
            self._headless_total_ticks += 1
            if self._headless_total_ticks > 3600:  # ~3 h at 3 s/tick
                self.log("Headless batch: watchdog timeout — forcing quit.", level="Warning")
                try:
                    self._headless_quit_timer.stop()
                except Exception:
                    pass
                QApplication.quit()
                return
            if self._headless_quit_phase == "csv":
                if getattr(self, "_csv_playback_complete", False):
                    self._headless_quit_phase = "predict"
                    self.log("Headless batch: CSV playback complete; waiting for predictor subprocesses…", level="Info")
                return
            busy = False
            for ctx in self.sensor_ctx.values():
                proc = getattr(ctx, "prediction_process", None)
                if proc is not None and proc.poll() is None:
                    busy = True
                    break
            if busy:
                self._headless_idle_ticks = 0
                return
            self._headless_idle_ticks += 1
            if self._headless_idle_ticks >= 4:
                self.log("Headless batch: predictors idle — quitting.", level="Info")
                try:
                    self._headless_quit_timer.stop()
                except Exception:
                    pass
                QApplication.quit()

        self._headless_quit_timer = QTimer(self)
        self._headless_quit_timer.timeout.connect(_tick)
        self._headless_quit_timer.start(3000)

    def _prompt_historic_minutes_interactive(self, extra_hint: str = "") -> None:
        """After data mode is chosen: how many minutes of historic data to load (real-time / simulation)."""
        hint = (extra_hint.strip() + "\n\n") if extra_hint.strip() else ""
        minutes, ok = QInputDialog.getInt(
            self.appWin,
            "Initial Historic Data",
            hint
            + "How many minutes of historic data should be loaded initially?\n"
            + "(Real-time: rolling DB window. Simulation: length of the initial window from your start date.)",
            value=int(max(1, min(self.historic_minutes, 10080))),
            min=1,
            max=10080,
        )
        if ok and minutes >= 1:
            self.historic_minutes = int(minutes)
            self.historic_points_1hz = self.historic_minutes * 60
            self._rolling_window_points = self.historic_points_1hz
            self.csv_hist_minutes = self.historic_minutes
            self.csv_hist_points = self.historic_points_1hz
            self.log(f"Initial historic window: {self.historic_minutes} minutes.", level="Info")

    def _prompt_csv_historic_time_range_interactive(self) -> None:
        """
        After a CSV is loaded interactively, let the user define the historic (blue) segment
        as inclusive start and end timestamps within the file span.
        """
        if self._csv_time_min is None or self._csv_time_max is None:
            return
        t_file_lo = pd.Timestamp(self._csv_time_min).floor("s")
        t_file_hi = pd.Timestamp(self._csv_time_max).ceil("s")
        def_start = pd.Timestamp(self.sim_start_time).floor("s")
        def_end = pd.Timestamp(self.sim_hist_end_time).ceil("s")
        if def_start < t_file_lo or def_start > t_file_hi:
            def_start = t_file_lo
        if def_end < def_start or def_end > t_file_hi:
            def_end = min(def_start + pd.Timedelta(minutes=max(1, int(self.csv_hist_minutes))), t_file_hi)
        span_hint = (
            f"Timestamps must fall within the loaded file:\n  {t_file_lo}  —  {t_file_hi}\n\n"
            f"Accepted examples: 2026-02-13 15:00:00, 2026-02-13T15:00:00\n\n"
            f"Defaults (from file start + current historic length) are pre-filled; press Cancel on a field to keep defaults."
        )
        s_text, ok_s = QInputDialog.getText(
            self.appWin,
            "CSV historic window — start",
            "Historic segment START (inclusive):\n\n" + span_hint,
            text=str(def_start),
        )
        if ok_s and str(s_text).strip():
            ts = pd.to_datetime(str(s_text).strip(), errors="coerce")
            if pd.isna(ts):
                QMessageBox.warning(self.appWin, "Invalid start time", "Could not parse start time; using default.")
            else:
                def_start = pd.Timestamp(ts).floor("s")
        e_text, ok_e = QInputDialog.getText(
            self.appWin,
            "CSV historic window — end",
            "Historic segment END (inclusive):\n\n" + span_hint,
            text=str(def_end),
        )
        if ok_e and str(e_text).strip():
            ts = pd.to_datetime(str(e_text).strip(), errors="coerce")
            if pd.isna(ts):
                QMessageBox.warning(self.appWin, "Invalid end time", "Could not parse end time; using default.")
            else:
                def_end = pd.Timestamp(ts).ceil("s")

        def_start = max(t_file_lo, min(def_start, t_file_hi))
        def_end = max(def_start, min(def_end, t_file_hi))
        if def_end <= def_start:
            QMessageBox.warning(
                self.appWin,
                "Invalid range",
                "End was not after start; extending end by one minute within the file span.",
            )
            def_end = min(def_start + pd.Timedelta(minutes=1), t_file_hi)
            if def_end <= def_start:
                def_end = t_file_hi

        delta_sec = float((def_end - def_start).total_seconds())
        delta_min = max(1, int(math.ceil(delta_sec / 60.0)))

        self.sim_start_time = def_start.to_pydatetime()
        self.sim_hist_end_time = def_end.to_pydatetime()
        self.sim_rt_start_time = self.sim_hist_end_time
        self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)
        self.csv_hist_minutes = delta_min
        self.csv_hist_points = delta_min * 60
        self.historic_minutes = delta_min
        self.historic_points_1hz = self.csv_hist_points
        self._rolling_window_points = max(1, int(self.csv_hist_points))
        self.log(
            f"CSV historic segment: [{self.sim_start_time} .. {self.sim_hist_end_time}] "
            f"(~{delta_min} min, {self.historic_points_1hz} points @ 1 Hz cap).",
            level="Info",
        )

    def _configure_startup_mode(self):
        # Startup prompts run during __init__ while base Application still shows the splash and
        # defers showMaximized for 3s — modal dialogs can sit behind the splash and be missed.
        try:
            splash = getattr(self, "splash", None)
            if splash is not None and splash.isVisible():
                splash.finish(self.appWin)
        except Exception:
            try:
                if getattr(self, "splash", None) is not None:
                    self.splash.close()
            except Exception:
                pass
        try:
            self.appWin.show()
            self.appWin.raise_()
            self.appWin.activateWindow()
            QApplication.processEvents()
        except Exception:
            pass

        if _headless_batch_enabled():
            self._configure_startup_mode_headless()
            return

        self._configure_predictor_model_family()

        msg = (
            "Choose data source first:\n\n"
            "Real-time: load the latest historic window from the database\n"
            "Simulation: choose a calendar start date, then how much historic data to simulate\n"
            "CSV file: pick a file, then define the historic (blue) time range within the file"
        )
        box = QMessageBox(self.appWin)
        box.setWindowTitle("Data Source")
        box.setText(msg)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Yes)
        try:
            box.button(QMessageBox.Yes).setText("Real-time")
            box.button(QMessageBox.No).setText("Simulation")
            box.button(QMessageBox.Cancel).setText("CSV File")
        except Exception:
            pass
        choice = box.exec()

        if choice == QMessageBox.Cancel:
            path = self._prompt_for_csv_path()
            if path and self._load_csv_source(path):
                self.csv_enabled = True
                self.sim_enabled = True
                self._prompt_csv_historic_time_range_interactive()
                self._configure_sensor_selection()
                # Prompt for k / training window before sensor discovery so ``_prime_new_sensor_context`` sees
                # the user's values (real-time/simulation discover later; CSV used to discover first and left stale ctx + spinboxes).
                self._prompt_initial_detector_and_training_settings()
                self._discover_sensors()
                if _ground_truth_visualization_enabled():
                    self._apply_manual_ground_truth_for_known_csv_experiment()
                self.log(f"Data mode selected: CSV file ({os.path.basename(path)}).", level="Info")
            else:
                if path:
                    QMessageBox.warning(self.appWin, "CSV Load Failed", "Falling back to real-time mode.")
                self.csv_enabled = False
                self.sim_enabled = False
                self._prompt_historic_minutes_interactive(
                    "You did not load a CSV (or load failed). Configure historic data for real-time mode."
                )
                self.log("Data mode selected: real-time (after CSV cancel or failure).", level="Info")
                self._configure_sensor_selection()
                self._prompt_initial_detector_and_training_settings()
                return
            return

        if choice == QMessageBox.Yes:
            self.csv_enabled = False
            self.sim_enabled = False
            self._prompt_historic_minutes_interactive()
            self.log("Data mode selected: real-time (last {} minutes).".format(self.historic_minutes), level="Info")
            self._configure_sensor_selection()
            self._prompt_initial_detector_and_training_settings()
            return

        while True:
            default_str = self.sim_start_time.strftime("%Y-%m-%d")
            text, ok = QInputDialog.getText(
                self.appWin,
                "Simulation Start Date",
                "Enter start date (e.g., 2025-10-10 or 10 Oct 2025):",
                text=default_str,
            )
            if not ok:
                self.csv_enabled = False
                self.sim_enabled = False
                self._prompt_historic_minutes_interactive("Simulation was cancelled; configure historic data for real-time.")
                self.log("Data mode selected: real-time (simulation cancelled).", level="Info")
                self._configure_sensor_selection()
                self._prompt_initial_detector_and_training_settings()
                return

            raw = text.strip()
            if not raw:
                QMessageBox.warning(self.appWin, "Invalid Date", "Please enter a valid date.")
                continue

            dt = None
            for fmt in ("%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
                try:
                    dt = datetime.strptime(raw, fmt)
                    break
                except Exception:
                    continue
            if dt is None:
                QMessageBox.warning(self.appWin, "Invalid Date", "Use YYYY-MM-DD or e.g., 10 Oct 2025.")
                continue

            self.csv_enabled = False
            self.sim_enabled = True
            self.sim_start_time = dt
            self._prompt_historic_minutes_interactive(
                f"Simulation calendar start: {self.sim_start_time.date()}.\nHow long should the initial historic window be?"
            )
            self._init_sim_times()
            self.log(f"Data mode selected: simulation from {self.sim_start_time}.", level="Info")
            self._configure_sensor_selection()
            self._prompt_initial_detector_and_training_settings()
            return

    def _prompt_fresh_gru_sequence_window(self) -> None:
        """
        Fresh GRU or vanilla LSTM: choose W = past 1 Hz samples per window (passed to
        predictor_ai as PREDICTOR_GRU_WINDOW_SIZE).
        """
        if _headless_batch_enabled():
            self.predictor_gru_window_size = _clamp_predictor_gru_window(int(self.predictor_gru_window_size))
            os.environ["PREDICTOR_GRU_WINDOW_SIZE"] = str(self.predictor_gru_window_size)
            return
        current = _clamp_predictor_gru_window(int(self.predictor_gru_window_size))
        self.predictor_gru_window_size = current
        labels: List[str] = [f"W = {w}  ({w} past samples per step @ 1 Hz)" for w in FRESH_GRU_WINDOW_PRESETS]
        labels.append("Custom… (type any allowed W)")
        default_index = len(FRESH_GRU_WINDOW_PRESETS)
        if current in FRESH_GRU_WINDOW_PRESETS:
            default_index = FRESH_GRU_WINDOW_PRESETS.index(current)
        fam = _normalize_model_family(self.predictor_model_family)
        is_lstm = fam == MODEL_FAMILY_LSTM
        dlg_title = "LSTM sequence window (W)" if is_lstm else "GRU sequence window (W)"
        arch = "LSTM" if is_lstm else "GRU"
        sel, ok = QInputDialog.getItem(
            self.appWin,
            dlg_title,
            f"The {arch} sees a fixed-length history of magnetic samples at each step.\n\n"
            "Same W is used when building training batches (supervised windows) and when "
            "rolling the window during prediction. Typical: 15 or 30; must be less than the "
            f"number of points available to train on.\n\n"
            f"Allowed range: {PREDICTOR_GRU_WINDOW_MIN}–{PREDICTOR_GRU_WINDOW_MAX}.",
            labels,
            default_index,
            False,
        )
        if ok and sel:
            try:
                idx = labels.index(sel)
            except ValueError:
                idx = -1
            if 0 <= idx < len(FRESH_GRU_WINDOW_PRESETS):
                self.predictor_gru_window_size = FRESH_GRU_WINDOW_PRESETS[idx]
            elif idx == len(FRESH_GRU_WINDOW_PRESETS):
                w, ok2 = QInputDialog.getInt(
                    self.appWin,
                    f"Custom {arch} window W",
                    f"Integer W — past 1 Hz samples included in each {arch} input (train + predict):",
                    value=current,
                    min=PREDICTOR_GRU_WINDOW_MIN,
                    max=PREDICTOR_GRU_WINDOW_MAX,
                    step=1,
                )
                if ok2:
                    self.predictor_gru_window_size = _clamp_predictor_gru_window(int(w))
        os.environ["PREDICTOR_GRU_WINDOW_SIZE"] = str(self.predictor_gru_window_size)

    def _configure_predictor_model_family(self):
        if _headless_batch_enabled():
            self._configure_predictor_model_family_headless()
            return
        options = [
            "Pretrained GRU",
            "Pretrained LSTM",
            "Fresh GRU",
            "Vanilla LSTM (fresh)",
            "Attention Bi-LSTM (fresh)",
            "Pretrained Transformer",
        ]
        current = (_normalize_model_family(self.predictor_model_family), _normalize_model_init(self.predictor_model_init))
        option_map = {
            options[0]: (MODEL_FAMILY_GRU, MODEL_INIT_PRETRAINED),
            options[1]: (MODEL_FAMILY_LSTM, MODEL_INIT_PRETRAINED),
            options[2]: (MODEL_FAMILY_GRU, MODEL_INIT_FRESH),
            options[3]: (MODEL_FAMILY_LSTM, MODEL_INIT_FRESH),
            options[4]: (MODEL_FAMILY_ATTN_BILSTM, MODEL_INIT_FRESH),
            options[5]: (MODEL_FAMILY_TRANSFORMER, MODEL_INIT_PRETRAINED),
        }
        default_index = 0
        for idx, opt in enumerate(options):
            if option_map[opt] == current:
                default_index = idx
                break
        selected_text, ok = QInputDialog.getItem(
            self.appWin,
            "Prediction model",
            "Choose model to use before startup:\n\n"
            "Note: If you pick \"Fresh GRU\" or \"Vanilla LSTM (fresh)\", the next dialog sets sequence length W "
            f"(e.g. 15 or 30 past samples at 1 Hz). Pretrained GRU/LSTM/Transformer use W from the checkpoint. "
            "\"Attention Bi-LSTM\" trains fresh from session data (no bundled checkpoint).",
            options,
            default_index,
            False,
        )
        if ok and selected_text:
            self.predictor_model_family, self.predictor_model_init = option_map.get(
                selected_text,
                (MODEL_FAMILY_GRU, MODEL_INIT_PRETRAINED),
            )
        self.predictor_model_family = _normalize_model_family(self.predictor_model_family)
        self.predictor_model_init = _normalize_model_init(self.predictor_model_init)
        os.environ["PREDICTOR_MODEL_FAMILY"] = self.predictor_model_family
        os.environ["PREDICTOR_MODEL_INIT"] = self.predictor_model_init

        if self.predictor_model_family == MODEL_FAMILY_GRU and self.predictor_model_init == MODEL_INIT_FRESH:
            self._prompt_fresh_gru_sequence_window()
        elif self.predictor_model_family == MODEL_FAMILY_LSTM and self.predictor_model_init == MODEL_INIT_FRESH:
            self._prompt_fresh_gru_sequence_window()
        else:
            os.environ.pop("PREDICTOR_GRU_WINDOW_SIZE", None)

        if self.predictor_model_family == MODEL_FAMILY_TRANSFORMER:
            self.log(
                "Prediction model selected: Transformer (pre-trained checkpoints transformer_pretrained_*.keras in models/). "
                "Architecture/W is fixed by the checkpoint (no startup prompt).",
                level="Info",
            )
        elif self.predictor_model_family == MODEL_FAMILY_ATTN_BILSTM:
            self.log(
                "Prediction model selected: Attention Bi-LSTM (fresh training from session data; predictor_ai attn path).",
                level="Info",
            )
        elif self.predictor_model_family == MODEL_FAMILY_LSTM and self.predictor_model_init == MODEL_INIT_FRESH:
            self.log(
                f"Prediction model selected: Vanilla LSTM (fresh training), window W={self.predictor_gru_window_size}",
                level="Info",
            )
        elif self.predictor_model_family == MODEL_FAMILY_LSTM and self.predictor_model_init == MODEL_INIT_PRETRAINED:
            self.log(
                "Prediction model selected: LSTM (pre-trained checkpoints lstm_pretrained_*.keras in models/). "
                "Sequence length W is taken from each checkpoint (no startup prompt).",
                level="Info",
            )
        elif self.predictor_model_init == MODEL_INIT_FRESH:
            self.log(
                f"Prediction model selected: GRU (fresh training), window W={self.predictor_gru_window_size}",
                level="Info",
            )
        else:
            self.log(
                "Prediction model selected: GRU (pre-trained checkpoints gru_pretrained_*.keras in models/). "
                "Sequence length W is taken from each checkpoint (no startup prompt).",
                level="Info",
            )

    def _prompt_initial_detector_and_training_settings(self) -> None:
        """
        Ask once at startup for anomaly threshold k and predictor training-window minutes.
        Values seed each new SensorContext; the same controls remain editable during the run.
        """
        if _headless_batch_enabled():
            _tw = self._initial_predictor_train_window_minutes
            if _tw is None:
                tw_log = "all available data"
            elif _tw == 0:
                tw_log = "0 min (predict-only first run when checkpoint exists)"
            else:
                tw_log = f"last {int(_tw)} min"
            self.log(
                f"Headless batch: k={self._initial_threshold_k:.2f}; predictor training window: {tw_log}.",
                level="Info",
            )
            try:
                self.anomaly_detector.threshold_multiplier = float(self._initial_threshold_k)
                self.train_window_minutes = self._initial_predictor_train_window_minutes
            except Exception:
                pass
            self._push_initial_detector_settings_to_all_sensor_contexts()
            self._sync_all_sensor_control_widgets_from_context()
            return
        k_val, ok_k = QInputDialog.getDouble(
            self.appWin,
            "Anomaly threshold (k)",
            "Multiplier k in: threshold = EWMA_mean(|error|) + k × σ(|error|).\n"
            "Larger k → fewer detections (stricter). Typical ~1.5–4.0 (matches the per-sensor spinbox).",
            float(self._initial_threshold_k),
            0.1,
            10.0,
            2,
        )
        if ok_k:
            self._initial_threshold_k = float(max(0.1, min(10.0, float(k_val))))

        _itw = self._initial_predictor_train_window_minutes
        tw_disp = -1 if _itw is None else int(_itw)
        tw, ok_tw = QInputDialog.getInt(
            self.appWin,
            "Predictor training data",
            "Minutes of data used to cap the training set (matches the training-window spinbox):\n"
            "  • -1 = train on all loaded historic + realtime rows (no time cap)\n"
            "  •  0 = skip initial full-history fit: first run is predict-only when a pretrained/runtime checkpoint exists\n"
            "  •  N≥1 = train only on the last N minutes before each training run",
            tw_disp,
            -1,
            1_000_000,
            1,
        )
        if ok_tw:
            tv = int(tw)
            if tv < 0:
                self._initial_predictor_train_window_minutes = None
            elif tv == 0:
                self._initial_predictor_train_window_minutes = 0
            else:
                self._initial_predictor_train_window_minutes = max(1, min(tv, 1_000_000))

        if self._initial_predictor_train_window_minutes is None:
            tw_log = "all loaded data"
        elif self._initial_predictor_train_window_minutes == 0:
            tw_log = "0 min (predict-only first run when checkpoint exists)"
        else:
            tw_log = f"last {int(self._initial_predictor_train_window_minutes)} min"
        self.log(
            f"Initial detector threshold k={self._initial_threshold_k:.2f}; predictor training window: {tw_log}.",
            level="Info",
        )
        try:
            self.anomaly_detector.threshold_multiplier = float(self._initial_threshold_k)
            self.train_window_minutes = self._initial_predictor_train_window_minutes
        except Exception:
            pass
        self._push_initial_detector_settings_to_all_sensor_contexts()
        self._sync_all_sensor_control_widgets_from_context()

    def _push_initial_detector_settings_to_all_sensor_contexts(self) -> None:
        """Copy startup k / training-window choices into every SensorContext (shared settings)."""
        for ctx in self.sensor_ctx.values():
            try:
                ctx.anomaly_detector.threshold_multiplier = float(
                    max(0.1, min(10.0, float(self._initial_threshold_k)))
                )
            except Exception:
                pass
            ctx.train_window_minutes = self._initial_predictor_train_window_minutes

    def _sync_all_sensor_control_widgets_from_context(self) -> None:
        """Refresh per-sensor control panels if they were created before the startup prompts finished."""
        for w in getattr(self, "_sensor_control_widgets", None) or []:
            if hasattr(w, "_sync_controls_from_context"):
                try:
                    w._sync_controls_from_context()
                except Exception:
                    pass

    def _prime_new_sensor_context(self, ctx: SensorContext) -> None:
        """Apply startup k and training-window defaults to a newly created sensor context."""
        try:
            ctx.anomaly_detector.threshold_multiplier = float(max(0.1, min(10.0, float(self._initial_threshold_k))))
        except Exception:
            ctx.anomaly_detector.threshold_multiplier = 2.5
        ctx.train_window_minutes = self._initial_predictor_train_window_minutes

    def _find_candidate_app_logs(self) -> List[str]:
        """
        Find possible prior-session app.log files, newest first.
        Prefer standard session logs; fall back to any app.log in repository tree.
        """
        logs: List[str] = []
        try:
            session_glob = os.path.join(base_app.APP_BASE, "sessions", "**", "app.log")
            logs.extend(glob.glob(session_glob, recursive=True))
        except Exception:
            pass

        if not logs:
            try:
                repo_root = os.path.dirname(base_app.APP_BASE)
                logs.extend(glob.glob(os.path.join(repo_root, "**", "app.log"), recursive=True))
            except Exception:
                pass

        uniq: List[str] = []
        seen = set()
        for p in logs:
            if p in seen:
                continue
            seen.add(p)
            if os.path.isfile(p):
                uniq.append(p)
        uniq.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return uniq

    @staticmethod
    def _extract_anomaly_times_from_log(log_path: str) -> List[datetime]:
        """
        Parse timestamps from lines like:
        'Anomaly detected | time=YYYY-MM-DD HH:MM:SS'
        """
        pattern = re.compile(
            r"Anomaly detected\s*\|\s*time=([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})"
        )
        times: List[datetime] = []
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = pattern.search(line)
                    if not m:
                        continue
                    try:
                        times.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
                    except Exception:
                        continue
        except Exception:
            return []
        return times

    @staticmethod
    def _extract_sensor_anomaly_times_from_log(log_path: str) -> Dict[str, List[datetime]]:
        """
        Parse per-sensor anomaly timestamps from lines like:
        [OBS2_1] Anomaly detected | time=YYYY-MM-DD HH:MM:SS
        """
        pat = re.compile(
            r"\[(OBS\d+_\d+)\]\s+Anomaly detected\s*\|\s*time=([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})"
        )
        out: Dict[str, List[datetime]] = {}
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = pat.search(line)
                    if not m:
                        continue
                    sensor, ts_raw = m.group(1), m.group(2)
                    try:
                        ts = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        continue
                    out.setdefault(sensor, []).append(ts)
        except Exception:
            return {}
        return out

    @staticmethod
    def _extract_sensor_anomaly_times_from_terminal_log(log_path: str, session_id: str) -> Dict[str, List[datetime]]:
        """
        Parse per-sensor anomalies from terminal transcript for a specific session id.
        Only lines after seeing the target session id path are considered.
        """
        pat_session = re.compile(r"/sessions/([0-9a-fA-F\-]{36})/")
        pat_anom = re.compile(
            r"\[(OBS\d+_\d+)\]\s+Anomaly detected\s*\|\s*time=([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})"
        )
        out: Dict[str, List[datetime]] = {}
        active = False
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    sm = pat_session.search(line)
                    if sm:
                        sid = sm.group(1)
                        if sid == session_id:
                            active = True
                        elif active and sid != session_id:
                            # Another session has started; stop using this file.
                            break
                    if not active:
                        continue
                    am = pat_anom.search(line)
                    if not am:
                        continue
                    sensor, ts_raw = am.group(1), am.group(2)
                    try:
                        ts = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        continue
                    out.setdefault(sensor, []).append(ts)
        except Exception:
            return {}
        return out

    @staticmethod
    def _build_ground_truth_intervals(
        times: List[datetime],
        max_gap_seconds: float = 1.5,
        pad_seconds: float = 0.35,
    ) -> List[Tuple[datetime, datetime]]:
        """
        Merge near-contiguous anomaly timestamps into wider intervals for plotting.
        This keeps UI responsive and makes background bands readable.
        """
        if not times:
            return []
        ts = sorted(times)
        intervals: List[Tuple[datetime, datetime]] = []
        cur_start = ts[0]
        cur_end = ts[0]
        max_gap = timedelta(seconds=max_gap_seconds)
        for t in ts[1:]:
            if (t - cur_end) <= max_gap:
                cur_end = t
            else:
                intervals.append((cur_start, cur_end))
                cur_start = t
                cur_end = t
        intervals.append((cur_start, cur_end))

        pad = timedelta(seconds=pad_seconds)
        return [(s - pad, e + pad) for s, e in intervals]

    def _load_ground_truth_from_reference_session(self) -> None:
        """
        Load GT anomalies per sensor from a specific reference session.
        Priority:
        1) src/sessions/<session_id>/app.log
        2) Optional: terminal transcripts (off by default; see GT_FALLBACK_TERMINAL_LOGS)
        """
        if not _ground_truth_visualization_enabled():
            return
        self._ground_truth_anomaly_times = []
        self._ground_truth_intervals = []
        self._ground_truth_intervals_by_sensor = {}
        self._ground_truth_magnet_intervals = []
        self._ground_truth_trimmer_intervals = []
        self._ground_truth_magnet_intervals_by_sensor = {}
        self._ground_truth_trimmer_intervals_by_sensor = {}

        ref_session_id = os.environ.get("GT_REFERENCE_SESSION_ID", "09d7e268-bc9e-481f-a3bd-36c61a52659a").strip()
        if not ref_session_id:
            self.log("Ground truth overlay: GT_REFERENCE_SESSION_ID is empty.", level="Warning")
            self._apply_manual_ground_truth_for_known_csv_experiment()
            return

        sensor_times: Dict[str, List[datetime]] = {}
        source_desc = ""

        # 1) Preferred: session app.log
        session_log = os.path.join(base_app.APP_BASE, "sessions", ref_session_id, "app.log")
        if os.path.isfile(session_log):
            sensor_times = self._extract_sensor_anomaly_times_from_log(session_log)
            source_desc = f"session log ({os.path.relpath(session_log, os.path.dirname(base_app.APP_BASE))})"

        # 2) Optional fallback: terminal transcripts (off by default; a naive recursive
        # glob under ~/.cursor/projects can match thousands of files and freeze the GUI).
        if not sensor_times:
            scan_env = os.environ.get("GT_FALLBACK_TERMINAL_LOGS", "").strip().lower()
            if scan_env in ("1", "true", "yes", "on"):
                term_glob = os.path.expanduser("~/.cursor/projects/**/terminals/*.txt")
                term_files = glob.glob(term_glob, recursive=True)
                term_files = sorted(
                    (p for p in term_files if os.path.isfile(p)),
                    key=lambda p: os.path.getmtime(p),
                    reverse=True,
                )[:50]
                for tf in term_files:
                    try:
                        with open(tf, "r", encoding="utf-8", errors="ignore") as f:
                            txt = f.read()
                        if ref_session_id not in txt:
                            continue
                    except Exception:
                        continue
                    parsed = self._extract_sensor_anomaly_times_from_terminal_log(tf, ref_session_id)
                    if parsed:
                        sensor_times = parsed
                        source_desc = f"terminal log ({tf})"
                        break

        if not sensor_times:
            self.log(
                "Ground truth overlay: no GT loaded for reference session "
                f"{ref_session_id}. "
                "Place `app.log` under `src/sessions/<session_id>/`, or set "
                "GT_FALLBACK_TERMINAL_LOGS=1 to search up to 50 recent Cursor terminal files.",
                level="Warning",
            )
            self._apply_manual_ground_truth_for_known_csv_experiment()
            return

        # Build per-sensor GT intervals and a global union (for compatibility/debug).
        all_times: List[datetime] = []
        total_points = 0
        total_intervals = 0
        for sensor, ts_list in sensor_times.items():
            # Session logs may list OBS1 and OBS2; UI shows introduced-anomaly GT on OBS1 only.
            if not _is_obs1_ui_sensor_label(sensor):
                continue
            uniq = sorted(set(ts_list))
            if not uniq:
                continue
            intervals = self._build_ground_truth_intervals(uniq)
            self._ground_truth_intervals_by_sensor[sensor] = intervals
            all_times.extend(uniq)
            total_points += len(uniq)
            total_intervals += len(intervals)

        self._ground_truth_anomaly_times = sorted(set(all_times))
        self._ground_truth_intervals = self._build_ground_truth_intervals(self._ground_truth_anomaly_times)
        # By default, keep reference-session GT as magnet GT.
        self._ground_truth_magnet_intervals = list(self._ground_truth_intervals)
        self._ground_truth_magnet_intervals_by_sensor = {
            k: list(v)
            for k, v in self._ground_truth_intervals_by_sensor.items()
            if _is_obs1_ui_sensor_label(k)
        }

        self.log(
            f"Ground truth overlay: loaded session {ref_session_id} from {source_desc} | "
            f"sensors={len(self._ground_truth_intervals_by_sensor)} | "
            f"anomaly_timestamps={total_points} | intervals={total_intervals}.",
            level="Info",
        )
        # Deferred session load runs after CSV may already be open; re-apply scripted CSV windows.
        self._apply_manual_ground_truth_for_known_csv_experiment()

    def _clear_manual_csv_gt_overlay_only(self) -> None:
        """Drop magnet/trimmer dict overrides so plots fall back to session GT intervals."""
        self._ground_truth_trimmer_intervals = []
        self._ground_truth_trimmer_intervals_by_sensor = {}
        self._ground_truth_magnet_intervals_by_sensor = {}
        self._ground_truth_magnet_intervals = list(self._ground_truth_intervals)

    @staticmethod
    def _build_manual_hhmm_intervals_for_date(
        day: datetime,
        ranges_hhmm: List[Tuple[str, str]],
    ) -> List[Tuple[datetime, datetime]]:
        def _parse_hhmm_or_half_min(token: str) -> Optional[Tuple[int, int, int]]:
            t = str(token).strip()
            if not t:
                return None
            # HHMMSS (e.g. 160635 -> 16:06:35) for second-accurate experiment logs
            if len(t) == 6 and t.isdigit():
                hh = int(t[:2])
                mm = int(t[2:4])
                ss = int(t[4:6])
                if 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59:
                    return hh, mm, ss
                return None
            half = False
            if t.endswith(".5"):
                half = True
                t = t[:-2]
            if len(t) != 4 or not t.isdigit():
                return None
            hh = int(t[:2])
            mm = int(t[2:])
            if hh < 0 or hh > 23 or mm < 0 or mm > 59:
                return None
            ss = 30 if half else 0
            return hh, mm, ss

        intervals: List[Tuple[datetime, datetime]] = []
        for start_hhmm, end_hhmm in ranges_hhmm:
            try:
                start_parts = _parse_hhmm_or_half_min(start_hhmm)
                end_parts = _parse_hhmm_or_half_min(end_hhmm)
                if not start_parts or not end_parts:
                    continue
                sh, sm, ss = start_parts
                eh, em, es = end_parts
                start_dt = day.replace(hour=sh, minute=sm, second=ss, microsecond=0)
                # Use exact parsed end timestamp; do not auto-extend to minute+59s.
                end_dt = day.replace(hour=eh, minute=em, second=es, microsecond=0)
                if end_dt < start_dt:
                    continue
                intervals.append((start_dt, end_dt))
            except Exception:
                continue
        return intervals

    def _apply_manual_ground_truth_for_known_csv_experiment(self) -> None:
        """
        Apply explicit GT intervals for known experiment CSV runs (magnet windows and optional
        separate trimmer-only windows when the trimmer list in ``_MANUAL_EXPERIMENT_CSV_GT`` is non-empty).
        """
        if not _ground_truth_visualization_enabled():
            return
        if not self.csv_path:
            return
        csv_name = os.path.basename(self.csv_path)
        cfg = _MANUAL_EXPERIMENT_CSV_GT.get(csv_name)
        if cfg is None:
            self._clear_manual_csv_gt_overlay_only()
            return

        base_day, magnet_ranges, trimmer_ranges = cfg
        magnet_intervals = self._build_manual_hhmm_intervals_for_date(base_day, magnet_ranges)
        trimmer_intervals = self._build_manual_hhmm_intervals_for_date(base_day, trimmer_ranges)
        if not magnet_intervals and not trimmer_intervals:
            return

        # Key bands by OBS*_ labels that match plot redraw (must not use stale DB sensor_ids
        # before CSV discovery). Mirror _discover_sensors CSV logic.
        sensor_names: List[str] = []
        if self._csv_timeseries_by_sensor:
            csv_sensors = sorted(self._csv_timeseries_by_sensor.keys(), key=_sensor_sort_key)
            if self._selected_sensor_ids:
                picked = [sid for sid in self._selected_sensor_ids if sid in csv_sensors]
                use_ids = picked if picked else csv_sensors[:3]
            else:
                use_ids = csv_sensors[:3]
            sensor_names = [_sensor_display_name(sid) for sid in use_ids]
        else:
            for sid in self.sensor_ids:
                ctx = self.sensor_ctx.get(sid)
                sensor_names.append(_sensor_display_name(ctx.display_name if ctx else sid))
            if not sensor_names:
                return

        self._ground_truth_magnet_intervals = list(magnet_intervals)
        self._ground_truth_trimmer_intervals = list(trimmer_intervals)
        split_obs1_obs2 = csv_name in _CSV_GT_MAGNET_OBS1_TRIMMER_OBS2
        self._ground_truth_magnet_intervals_by_sensor = {}
        self._ground_truth_trimmer_intervals_by_sensor = {}
        for s in sensor_names:
            # Never attach introduced-anomaly GT strips to OBS2 streams (paper / OBS1-only UI).
            if not _is_obs1_ui_sensor_label(s):
                self._ground_truth_magnet_intervals_by_sensor[s] = []
                self._ground_truth_trimmer_intervals_by_sensor[s] = []
                continue
            if split_obs1_obs2:
                if s.startswith("OBS1_"):
                    self._ground_truth_magnet_intervals_by_sensor[s] = list(magnet_intervals)
                    self._ground_truth_trimmer_intervals_by_sensor[s] = []
                elif s.startswith("OBS2_"):
                    self._ground_truth_magnet_intervals_by_sensor[s] = []
                    self._ground_truth_trimmer_intervals_by_sensor[s] = list(trimmer_intervals)
                else:
                    self._ground_truth_magnet_intervals_by_sensor[s] = list(magnet_intervals)
                    self._ground_truth_trimmer_intervals_by_sensor[s] = list(trimmer_intervals)
            else:
                self._ground_truth_magnet_intervals_by_sensor[s] = list(magnet_intervals)
                self._ground_truth_trimmer_intervals_by_sensor[s] = list(trimmer_intervals)
        gt_kind = (
            "magnet and trimmer windows"
            if trimmer_intervals
            else "magnet-only windows"
        )
        self.log(
            f"Ground truth overlay: using manual {gt_kind} for {csv_name} on OBS1 sensors only "
            f"(panels: {', '.join(n for n in sensor_names if _is_obs1_ui_sensor_label(n)) or '—'})"
            f"{' (OBS1=magnet bands, OBS2=trimmer bands when split mode CSV is listed)' if split_obs1_obs2 else ''}.",
            level="Info",
        )

    def _resolve_pretrained_model_path(self, sensor_id: str, model_dir: str, model_family: str) -> Optional[str]:
        safe_sensor_id = sensor_id.replace("/", "_").replace("\\", "_")
        m = re.search(r"(OBS\d+_\d+)", sensor_id)
        obs_part = m.group(1) if m else None

        if model_family == MODEL_FAMILY_TRANSFORMER:
            prefix = "transformer_pretrained_"
            loose_glob = [f"*transformer*{obs_part}*.keras", f"*transformer*{obs_part}*.h5"] if obs_part else []
        elif model_family == MODEL_FAMILY_LSTM:
            prefix = "lstm_pretrained_"
            loose_glob = (
                [f"*lstm*{obs_part}*.keras", f"*lstm*{obs_part}*.h5"]
                if obs_part
                else ["*lstm*.keras", "*lstm*.h5"]
            )
        else:
            prefix = "gru_pretrained_"
            loose_glob = (
                [f"*gru*{obs_part}*.keras", f"*gru*{obs_part}*.h5"]
                if obs_part
                else ["*gru*.keras", "*gru*.h5"]
            )

        exact_candidates: List[str] = []
        glob_patterns: List[str] = []
        exact_candidates.extend(
            [
                os.path.join(model_dir, f"{prefix}{safe_sensor_id}.keras"),
                os.path.join(model_dir, f"{prefix}{safe_sensor_id}.h5"),
            ]
        )
        if obs_part:
            exact_candidates.extend(
                [
                    os.path.join(model_dir, f"{prefix}{obs_part}.keras"),
                    os.path.join(model_dir, f"{prefix}{obs_part}.h5"),
                ]
            )
            glob_patterns.extend(
                [
                    f"{prefix}*{obs_part}*.keras",
                    f"{prefix}*{obs_part}*.h5",
                ]
                + loose_glob
            )
        else:
            glob_patterns.extend([f"{prefix}*.keras", f"{prefix}*.h5"] + loose_glob)

        for p in exact_candidates:
            if os.path.isfile(p):
                return p

        matches: List[str] = []
        for pattern in glob_patterns:
            matches.extend(glob.glob(os.path.join(model_dir, pattern)))
            matches.extend(glob.glob(os.path.join(model_dir, "**", pattern), recursive=True))
        matches = [p for p in set(matches) if os.path.isfile(p)]
        if not matches:
            return None
        try:
            return max(matches, key=lambda p: (os.path.getmtime(p), p))
        except Exception:
            return sorted(matches)[-1]

    def _configure_sensor_selection(self):
        if self.csv_enabled:
            csv_sensors = sorted(self._csv_timeseries_by_sensor.keys(), key=_sensor_sort_key)
            if not csv_sensors:
                return
            if _headless_batch_enabled():
                raw = os.environ.get("MAGNAVIS_BATCH_SENSORS", "").strip()
                if raw:
                    want = [s.strip() for s in raw.split(",") if s.strip()]
                    selected: List[str] = []
                    for w in want:
                        hit: Optional[str] = None
                        if w in csv_sensors:
                            hit = w
                        else:
                            for sid in csv_sensors:
                                if _sensor_display_name(sid) == w:
                                    hit = sid
                                    break
                        if hit and hit not in selected:
                            selected.append(hit)
                    if not selected:
                        selected = csv_sensors[:3]
                else:
                    preferred = ["OBS2_1", "OBS2_2", "OBS2_3"]
                    selected = [s for s in preferred if s in csv_sensors]
                    for s in csv_sensors:
                        if s not in selected:
                            selected.append(s)
                        if len(selected) >= 3:
                            break
                if len(selected) > 3:
                    selected = selected[:3]
                self._selected_sensor_ids = selected
                self.sensor_ids = []
                self.log(f"Headless batch sensor pick: {', '.join(selected)}", level="Info")
                return
            selected = self._prompt_for_sensor_selection(csv_sensors)
            # Limit to at most 3 sensors.
            if selected:
                if len(selected) > 3:
                    self.log(
                        f"More than 3 sensors selected; using first 3: {', '.join(selected[:3])}",
                        level="Warning",
                    )
                    selected = selected[:3]
            else:
                selected = csv_sensors[:3]
            self._selected_sensor_ids = selected
            # Clear existing sensor_ids to force rediscovery with new selection
            self.sensor_ids = []
            return
        try:
            default_ids = get_latest_sensor_ids(limit=6)
        except Exception:
            default_ids = []
        if default_ids:
            selected = self._prompt_for_sensor_selection(default_ids)
            if selected:
                if len(selected) > 3:
                    self.log(
                        f"More than 3 sensors selected; using first 3: {', '.join(selected[:3])}",
                        level="Warning",
                    )
                    selected = selected[:3]
            else:
                selected = default_ids[:3]
            self._selected_sensor_ids = selected
            # Clear existing sensor_ids to force rediscovery with new selection
            self.sensor_ids = []
            return
        default_text = "OBS1_1, OBS1_2, OBS1_3, OBS2_1, OBS2_2, OBS2_3"
        text, ok = QInputDialog.getText(
            self.appWin,
            "Select Sensors",
            "Enter sensor IDs to plot (comma-separated):",
            text=default_text,
        )
        if not ok:
            self._selected_sensor_ids = None
            self.sensor_ids = []
            return
        raw = text.strip()
        if not raw:
            self._selected_sensor_ids = None
            self.sensor_ids = []
            return
        ids = [s.strip() for s in raw.split(",") if s.strip()]
        if ids and len(ids) > 3:
            self.log(
                f"More than 3 sensors entered; using first 3: {', '.join(ids[:3])}",
                level="Warning",
            )
            ids = ids[:3]
        self._selected_sensor_ids = ids if ids else None
        # Clear existing sensor_ids to force rediscovery with new selection
        self.sensor_ids = []

    def _prompt_for_sensor_selection(self, sensor_ids: List[str]) -> List[str]:
        dlg = QDialog(self.appWin)
        dlg.setWindowTitle("Select Sensors")
        layout = QVBoxLayout()
        dlg.setLayout(layout)

        label = QLabel("Select up to 3 sensors to plot:")
        layout.addWidget(label)

        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.MultiSelection)
        for sid in sensor_ids:
            # Show canonical OBS labels in UI; keep the full sensor_id as hidden payload.
            item = QListWidgetItem(_sensor_display_name(sid))
            item.setData(QtCore.Qt.UserRole, sid)
            item.setToolTip(sid)
            item.setSelected(True)
            list_widget.addItem(item)
        layout.addWidget(list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return []
        selected = [item.data(QtCore.Qt.UserRole) or item.text() for item in list_widget.selectedItems()]
        return selected

    def _prompt_for_csv_path(self) -> Optional[str]:
        try:
            path, _ = QFileDialog.getOpenFileName(
                self.appWin,
                "Select Magnetic CSV File",
                os.path.dirname(base_app.APP_BASE),
                "CSV Files (*.csv);;All Files (*)",
            )
            return path if path else None
        except Exception:
            return None

    def _derive_base_date_from_filename(self, path: str) -> Optional[datetime]:
        name = os.path.basename(path)
        m = re.search(r"magnetic_data_(\d{8})_\d{6}_to_(\d{8})_\d{6}", name)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1), "%Y%m%d")
        except Exception:
            return None

    def _parse_csv_timestamps(self, series: pd.Series, path: str) -> pd.Series:
        ts = pd.to_datetime(series, errors="coerce")
        valid_ratio = float(ts.notna().mean()) if len(ts) else 0.0
        if valid_ratio >= 0.5:
            return ts

        base_date = self._derive_base_date_from_filename(path)
        if base_date is None:
            base_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        td = pd.to_timedelta(series, errors="coerce")
        if td.notna().any():
            return pd.to_datetime(base_date) + td

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            return pd.to_datetime(base_date) + pd.to_timedelta(numeric, unit="s", errors="coerce")

        return ts

    def _csv_raw_to_timeseries_df(self, df_raw: pd.DataFrame, path: str) -> pd.DataFrame:
        if df_raw is None or df_raw.empty:
            return pd.DataFrame(columns=["time_H", "mag_H_nT"])

        df = df_raw.copy()
        if "timestamp" not in df.columns:
            return pd.DataFrame(columns=["time_H", "mag_H_nT"])

        df["timestamp"] = self._parse_csv_timestamps(df["timestamp"], path)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=True).reset_index(drop=True)

        for c in ("b_x", "b_y", "b_z"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if not all(c in df.columns for c in ("b_x", "b_y", "b_z")):
            return pd.DataFrame(columns=["time_H", "mag_H_nT"])

        df["mag_total_nT"] = (df["b_x"] ** 2 + df["b_y"] ** 2 + df["b_z"] ** 2) ** 0.5
        df = df.dropna(subset=["mag_total_nT"])
        df["time_H"] = df["timestamp"].dt.floor("s")
        grouped = (
            df.groupby("time_H", as_index=False)["mag_total_nT"]
            .mean()
            .rename(columns={"mag_total_nT": "mag_H_nT"})
            .sort_values("time_H", ascending=True)
            .reset_index(drop=True)
        )
        return grouped[["time_H", "mag_H_nT"]]

    @staticmethod
    def _obs_sensor_suffix(sensor_id: str) -> Optional[str]:
        """Return canonical OBS suffix like OBS2_1 from full sensor_id."""
        m = re.search(r"(OBS\d+_\d+)$", str(sensor_id))
        return m.group(1) if m else None

    def _load_csv_source(self, path: str) -> bool:
        try:
            usecols = ["sensor_id", "timestamp", "b_x", "b_y", "b_z", "theta_x", "theta_y", "theta_z"]
            df = pd.read_csv(path, usecols=usecols)
        except Exception:
            try:
                df = pd.read_csv(path)
            except Exception:
                return False

        if df is None or df.empty or "sensor_id" not in df.columns:
            return False

        tclip = pd.to_datetime(df["timestamp"], errors="coerce") if "timestamp" in df.columns else None
        end_cap = _parse_batch_csv_end_timestamp()
        start_cap = _parse_batch_csv_start_timestamp()
        if tclip is not None and (end_cap is not None or start_cap is not None):
            n0 = int(len(df))
            mask = pd.Series(True, index=df.index)
            # Match ``evaluate_anomaly_detection._allowed_seconds_from_magnetic_csv``: compare on
            # **floor-second** timestamps so rows in e.g. 06:35:00.099 remain when end_cap is 06:35:00.
            tsec = tclip.dt.floor("s")
            if end_cap is not None:
                mask &= tsec <= end_cap
            if start_cap is not None:
                mask &= tsec >= start_cap
            df = df.loc[mask].copy()
            n1 = int(len(df))
            self.log(
                f"CSV time trim: kept {n1}/{n0} rows (start_cap={start_cap}, end_cap={end_cap}).",
                level="Info",
            )
            if df.empty:
                self.log(
                    "CSV load failed: no rows remain after MAGNAVIS_BATCH_CSV_START / MAGNAVIS_BATCH_CSV_END truncation.",
                    level="Error",
                )
                return False

        self._csv_timeseries_by_sensor = {}
        for sid, df_sensor in df.groupby("sensor_id"):
            ts_df = self._csv_raw_to_timeseries_df(df_sensor, path)
            self._csv_timeseries_by_sensor[str(sid)] = ts_df

        nonempty = [
            str(sid)
            for sid, df_ts in self._csv_timeseries_by_sensor.items()
            if df_ts is not None and not df_ts.empty and "time_H" in df_ts.columns
        ]
        if not nonempty:
            self.log(
                "CSV load failed: no usable magnetic rows per sensor (need timestamp + b_x,b_y,b_z).",
                level="Error",
            )
            self._csv_timeseries_by_sensor = {}
            self._csv_time_min = None
            self._csv_time_max = None
            return False

        all_times = []
        for df_ts in self._csv_timeseries_by_sensor.values():
            if df_ts is not None and not df_ts.empty and "time_H" in df_ts.columns:
                all_times.append(df_ts["time_H"].min())
                all_times.append(df_ts["time_H"].max())
        self._csv_time_min = min(all_times)
        self._csv_time_max = max(all_times)
        self._sync_csv_sim_clock_after_load()

        self.csv_path = path
        self._csv_playback_complete = False  # Reset so new CSV playback can run
        # Rolling buffer size for UI / smoothing (avoid 0 when headless historic minutes = 0).
        self._rolling_window_points = max(1, int(self.csv_hist_points))
        return True

    def _fetch_csv_window_multi(
        self,
        sensor_ids: List[str],
        *,
        start_time: datetime,
        end_time: datetime,
        target_n_seconds: Optional[int],
        incremental: bool,
    ) -> Dict[str, pd.DataFrame]:
        st = pd.Timestamp(start_time)
        et = pd.Timestamp(end_time)
        out: Dict[str, pd.DataFrame] = {}
        for sid in sensor_ids:
            df_ts = self._csv_timeseries_by_sensor.get(sid)
            if df_ts is None or df_ts.empty:
                out[sid] = pd.DataFrame(columns=["time_H", "mag_H_nT"])
                continue
            tcol = pd.to_datetime(df_ts["time_H"], errors="coerce")
            df_ts = df_ts.copy()
            df_ts["time_H"] = tcol
            df_ts = df_ts.dropna(subset=["time_H"])
            if incremental:
                mask = (df_ts["time_H"] > st) & (df_ts["time_H"] <= et)
            else:
                mask = (df_ts["time_H"] >= st) & (df_ts["time_H"] <= et)
            df_win = df_ts.loc[mask].copy()
            # If the requested sim window does not overlap file data (stale sim clock, TZ mixups),
            # fall back to the first historic segment of the file.
            if not incremental and df_win.empty and len(df_ts) > 0:
                tmin = df_ts["time_H"].min()
                tmax = df_ts["time_H"].max()
                if et < tmin or st > tmax:
                    self.log(
                        f"CSV fetch: requested [{st}, {et}] does not overlap data [{tmin}, {tmax}]; "
                        f"clamping to file start with {int(self.csv_hist_minutes)} min window.",
                        level="Warning",
                    )
                span = pd.Timedelta(minutes=int(self.csv_hist_minutes))
                st2 = tmin
                et2 = min(tmin + span, tmax)
                mask = (df_ts["time_H"] >= st2) & (df_ts["time_H"] <= et2)
                df_win = df_ts.loc[mask].copy()
            if target_n_seconds is not None and target_n_seconds > 0 and len(df_win) > int(target_n_seconds):
                df_win = df_win.tail(int(target_n_seconds)).reset_index(drop=True)
            out[sid] = df_win.reset_index(drop=True)
        return out

    def _init_sim_times(self):
        # Ensure sim time values are consistent and pure datetimes (no Qt objects).
        self.sim_hist_end_time = self.sim_start_time + timedelta(minutes=self.historic_minutes)
        self.sim_rt_start_time = self.sim_hist_end_time
        self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)

    def _sync_csv_sim_clock_after_load(self) -> None:
        """Align CSV playback window to file time span (call after _csv_time_min/_csv_time_max are set)."""
        if self._csv_time_min is None:
            return
        self.sim_start_time = pd.Timestamp(self._csv_time_min).to_pydatetime()
        self._init_csv_times()
        self.log(
            f"CSV playback window: sim_start={self.sim_start_time} sim_hist_end={self.sim_hist_end_time} "
            f"(historic_minutes={self.csv_hist_minutes}) | file range [{self._csv_time_min} .. {self._csv_time_max}]",
            level="Info",
        )

    def _init_csv_times(self):
        # CSV mode: historic window length = csv_hist_minutes (from startup prompt).
        self.sim_hist_end_time = self.sim_start_time + timedelta(minutes=int(self.csv_hist_minutes))
        self.sim_rt_start_time = self.sim_hist_end_time
        self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)

    def initViews(self):
        wnd = ApplicationWindowTemp(self)
        return wnd

    # ---- Disable unused frameworks from base application.py (keep only TimeSeries + log) ----
    def load_visualization_framework(self):
        # Skip VTK/spatial initialization for the minimal GUI.
        return

    def load_plot_framework(self):
        # Skip Map/contour plotting for the minimal GUI.
        return

    def _discover_sensors(self):
        # Only skip if we already have sensors AND they match the selected sensors
        if self.sensor_ids and self._selected_sensor_ids and set(self.sensor_ids) == set(self._selected_sensor_ids):
            return
        
        if self.csv_enabled:
            csv_sensors = sorted(self._csv_timeseries_by_sensor.keys(), key=_sensor_sort_key)
            if self._selected_sensor_ids:
                self.sensor_ids = [sid for sid in self._selected_sensor_ids if sid in csv_sensors]
                if not self.sensor_ids:
                    # Selected sensors not found in CSV, fall back to available sensors
                    self.log(f"Selected sensors {self._selected_sensor_ids} not found in CSV. Using available sensors.", level="Warning")
                    self.sensor_ids = csv_sensors[:3]
            else:
                self.sensor_ids = csv_sensors[:3]
            # Restrict sensor_ctx to only selected sensors (drop any from earlier discovery)
            self.sensor_ctx = {sid: self.sensor_ctx[sid] for sid in self.sensor_ids if sid in self.sensor_ctx}
            for sid in self.sensor_ids:
                if sid not in self.sensor_ctx:
                    self.sensor_ctx[sid] = SensorContext(sensor_id=sid, display_name=_sensor_display_name(sid))
                    self._prime_new_sensor_context(self.sensor_ctx[sid])
                self.sensor_ctx[sid].retrain_interval_minutes = int(
                    max(1, getattr(self.sensor_ctx[sid], "retrain_interval_minutes", self._default_retrain_interval_minutes))
                )
            # Do not reset sim_start_time / sim_hist_end_time here. ``_load_csv_source`` already ran
            # ``_sync_csv_sim_clock_after_load`` (defaults from file min + historic length); interactive CSV
            # then sets the real window in ``_prompt_csv_historic_time_range_interactive``. Re-applying
            # file start here would overwrite the user's [start, end] with the first N minutes of the file.
            self.log(f"Using CSV sensor streams: {', '.join(self.sensor_ids)}", level="Info")
            return
        
        if self._selected_sensor_ids:
            # Use selected sensors only. Drop any sensor not in the user's selection (e.g. OBS1_1
            # that was added by an earlier startThreads() call before the user had chosen sensors).
            # _configure_sensor_selection already enforces a maximum of 3 sensors.
            self.sensor_ids = self._selected_sensor_ids.copy()
            # Restrict sensor_ctx to only selected sensors so UI/title show only what user chose
            self.sensor_ctx = {sid: self.sensor_ctx[sid] for sid in self.sensor_ids if sid in self.sensor_ctx}
            for sid in self.sensor_ids:
                if sid not in self.sensor_ctx:
                    self.sensor_ctx[sid] = SensorContext(sensor_id=sid, display_name=_sensor_display_name(sid))
                    self._prime_new_sensor_context(self.sensor_ctx[sid])
                self.sensor_ctx[sid].retrain_interval_minutes = int(
                    max(1, getattr(self.sensor_ctx[sid], "retrain_interval_minutes", self._default_retrain_interval_minutes))
                )
            self._init_sim_times()
            self.log(f"Using selected sensors: {', '.join(self.sensor_ids)}", level="Info")
            return
        
        # Fallback: only OBS1 sensor-1 stream (single-sensor bring-up).
        sid = None
        try:
            sid = get_latest_sensor_id_like("%OBS1_1")
        except Exception as e:
            self.log(f"Sensor discovery query failed for %OBS1_1: {e}", level="Warning")

        # Secondary fallback: use latest available stream if OBS1_1 pattern is unavailable.
        if not sid:
            try:
                latest_ids = get_latest_sensor_ids(limit=1)
                sid = latest_ids[0] if latest_ids else None
            except Exception as e:
                self.log(f"Fallback sensor discovery query failed: {e}", level="Warning")

        if not sid:
            # Hard fallback: keep app alive with empty list
            self.sensor_ids = []
            self.sensor_ctx = {}
            self.log("No sensor_id found during DB discovery (OBS1_1 or latest stream).", level="Error")
            return
        self.sensor_ids = [sid]
        for sid in self.sensor_ids:
            if sid not in self.sensor_ctx:
                self.sensor_ctx[sid] = SensorContext(sensor_id=sid, display_name=_sensor_display_name(sid))
                self._prime_new_sensor_context(self.sensor_ctx[sid])
            self.sensor_ctx[sid].retrain_interval_minutes = int(
                max(1, getattr(self.sensor_ctx[sid], "retrain_interval_minutes", self._default_retrain_interval_minutes))
            )
        self._init_sim_times()
        self.log(f"Using single sensor stream: {self.sensor_ctx[self.sensor_ids[0]].display_name}", level="Info")

    def _lowpass_series(self, ctx: SensorContext, values: List[float], reset: bool = False) -> List[float]:
        """
        Apply a simple first-order low-pass filter (exponential moving average) to `values`.
        This is used as a lightweight denoising step per sensor.
        """
        if not values:
            return []
        alpha = self._lowpass_alpha
        filtered: List[float] = []
        # Initialize state
        if reset or ctx.last_filtered_value is None:
            prev = float(values[0])
        else:
            prev = float(ctx.last_filtered_value)
        for v in values:
            v_f = float(v)
            prev = alpha * v_f + (1.0 - alpha) * prev
            filtered.append(prev)
        ctx.last_filtered_value = prev
        return filtered

    def on_db_data_updated(self, dfs: dict, is_new: bool):
        _csv_incremental_batch = is_new and self.csv_enabled
        got_points_this_tick = False
        try:
            self._discover_sensors()
            for sid, df in dfs.items():
                ctx = self.sensor_ctx.get(sid)
                if ctx is None:
                    continue
                if df is None or df.empty:
                    # If initial load is empty in simulation, try once to jump to the first
                    # available timestamp within the selected date (midnight..midnight+1d).
                    if not is_new and self.sim_enabled and not self._initial_fetch_retry and not self.csv_enabled:
                        self._initial_fetch_retry = True
                        try:
                            nxt = get_min_timestamp_at_or_after(sid, self.sim_start_time)
                        except Exception:
                            nxt = None
                        if nxt is not None and nxt < (self.sim_start_time + timedelta(days=1)):
                            self.sim_start_time = nxt
                            self._init_sim_times()
                            self.log(f"No data at midnight; retrying from {self.sim_start_time}.", level="Warning")
                            QTimer.singleShot(200, lambda: self.appWin.startThreads(hours=1, start_time=None, new=False))
                        else:
                            msg = "No data found on the selected date. Please choose another date."
                            try:
                                QMessageBox.warning(self.appWin, "No Data", msg)
                            except Exception:
                                pass
                            self.log("No data found on selected date.", level="Warning")
                    continue
                times = pd.to_datetime(df["time_H"]).tolist()
                vals = df["mag_H_nT"].astype(float).tolist()
                if not is_new:
                    # Initial (historic) snapshot: DB/sim cap by rolling window; CSV keeps full [sim_start, sim_hist_end] fetch.
                    if self.csv_enabled:
                        n = len(times)
                    else:
                        n = min(len(times), self._rolling_window_points)
                    ctx.last_filtered_value = None
                    if n:
                        # Low-pass filter the historic series once and keep only the most recent window.
                        filtered_all = self._lowpass_series(ctx, vals, reset=True)
                        ctx.base_x_t = times[-n:]
                        ctx.base_y_mag_t = filtered_all[-n:]
                        ctx.base_y_mag_raw_t = [float(v) for v in vals[-n:]]
                    else:
                        ctx.base_x_t = []
                        ctx.base_y_mag_t = []
                        ctx.base_y_mag_raw_t = []
                    # Baseline (median of filtered values) retained for any auxiliary use.
                    ctx.plot_baseline_nT = float(np.median(ctx.base_y_mag_t)) if ctx.base_y_mag_t else None
                    ctx.rt_x_t = []
                    ctx.rt_y_mag_t = []
                    ctx.rt_y_mag_raw_t = []
                    ctx.new_x_t = []
                    ctx.new_y_mag_t = []
                    ctx.has_seen_realtime = False
                    ctx.needs_update_lims = True
                    # DB-backed simulation: align the realtime cursor to the last loaded historic sample so the
                    # next incremental fetch returns rows after that time. CSV playback must NOT do this: the
                    # cursor is already ``sim_hist_end_time`` (user-chosen end of the blue window); resetting from
                    # ``base_x_t[-1]`` breaks multi-Hz files (truncated last time != historic end) and defers green to the wrong clock.
                    if self.sim_enabled and ctx.base_x_t and not self.csv_enabled:
                        self.sim_rt_start_time = ctx.base_x_t[-1]
                        self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)
                else:
                    # Append only points with strictly increasing timestamps
                    # Keep `new_*` only for the latest chunk (for anomaly comparison),
                    # while the realtime series (green) accumulates.
                    last_time = None
                    if ctx.rt_x_t:
                        last_time = ctx.rt_x_t[-1]
                    elif ctx.base_x_t:
                        last_time = ctx.base_x_t[-1]
                    new_chunk_t: List[datetime] = []
                    new_chunk_v: List[float] = []
                    for t, v in zip(times, vals):
                        if last_time is None or t > last_time:
                            new_chunk_t.append(t)
                            new_chunk_v.append(v)
                            last_time = t

                    # Append to realtime accumulated (green) series
                    if new_chunk_t:
                        # Low-pass filter only the truly new values, continuing from previous state.
                        filtered_chunk = self._lowpass_series(ctx, new_chunk_v, reset=False)
                        ctx.rt_x_t.extend(new_chunk_t)
                        ctx.rt_y_mag_t.extend(filtered_chunk)
                        ctx.rt_y_mag_raw_t.extend(float(v) for v in new_chunk_v)
                        # Keep a cumulative realtime buffer for anomaly detection
                        ctx.new_x_t.extend(new_chunk_t)
                        ctx.new_y_mag_t.extend(filtered_chunk)
                        ctx.has_seen_realtime = True
                        ctx.needs_update_lims = True
                        got_points_this_tick = True

                # Save/train only when we have new information (avoid rewriting CSVs every draw tick)
                total_points = len(ctx.base_x_t) + len(ctx.rt_x_t)
                if total_points > 0 and total_points != ctx.last_saved_points:
                    # Training uses the full historic + realtime series (no length cap); model trains on entire available data.
                    x_all = ctx.base_x_t + ctx.rt_x_t
                    y_all = ctx.base_y_mag_t + ctx.rt_y_mag_t
                    # Always allow predictor scheduling when new data arrives. This avoids
                    # delaying first forecast until after realtime plotting starts.
                    self.save_data_for_sensor(sid, x_all, y_all, start_predictor=True)
                    ctx.last_saved_points = total_points

                # Run anomaly detection once we have both predictions and new realtime data
                if ctx.rt_x_t and ctx.predict_x_t:
                    self.detect_anomalies_for_sensor(sid)

            # Apply debounced predictor CSV writes before polling subprocess output so predict_input
            # on disk matches the buffers we just saved (otherwise update_predictions can miss a run).
            self._flush_predict_input_writes()
            # Pick up predict_out.csv as soon as the subprocess may have finished — do not wait for the
            # slow pred timer — then anomaly detection runs immediately when both streams are merged.
            for sid in self.sensor_ids:
                self.update_predictions_for_sensor(sid)

            # Trigger initial plot framework creation if not loaded yet
            self.appWin.updateData()
            self._maybe_headless_snapshot_before_csv_end()

            # Advance simulation clock after we successfully fetched an incremental slice.
            if is_new and self.sim_enabled:
                # Use points received in *this* batch only (new_x_t is cumulative and would always be truthy).
                if got_points_this_tick:
                    self.sim_rt_start_time = self.sim_rt_end_time
                    self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)
                else:
                    if self.csv_enabled:
                        # CSV mode: no new data this tick. If we've passed end of CSV, stop the fetch
                        # timer and run a final anomaly-detection pass so the full range is processed.
                        past_end = False
                        if not self._csv_playback_complete and self._csv_time_max is not None:
                            try:
                                # Normalise to pandas Timestamp for reliable comparison (datetime vs pd.Timestamp)
                                t_max = pd.Timestamp(self._csv_time_max)
                                t_start = pd.Timestamp(self.sim_rt_start_time)
                                t_end = pd.Timestamp(self.sim_rt_end_time)
                                past_end = t_start >= t_max or t_end > t_max
                            except Exception:
                                past_end = self.sim_rt_start_time >= self._csv_time_max
                        if past_end:
                            self._csv_playback_complete = True
                            try:
                                tmr = getattr(self, "_multi_data_timer", None)
                                if tmr is not None and tmr.isActive():
                                    tmr.stop()
                                    self.log(
                                        "CSV playback complete: end of data reached; fetch timer stopped.",
                                        level="Info",
                                    )
                            except Exception:
                                pass
                            # Final pass: run anomaly detection and force predictor catch-up so
                            # the full arrived range is covered even after fetch timer stops.
                            for sid in self.sensor_ids:
                                ctx = self.sensor_ctx.get(sid)
                                if ctx is None or not ctx.rt_x_t:
                                    continue
                                if ctx.predict_x_t:
                                    self.detect_anomalies_for_sensor(sid)
                                self._ensure_prediction_covers_actual(sid, force=True)
                            self.log(
                                "CSV playback complete: final prediction/anomaly catch-up scheduled for full range.",
                                level="Info",
                            )
                            self._finalize_csv_fast_playback_plots()
                        elif not self._csv_playback_complete:
                            # Not yet at end; advance by step so next fetch uses new window.
                            self.sim_rt_start_time = self.sim_rt_end_time
                            self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)
                    else:
                        sid0 = self.sensor_ids[0] if self.sensor_ids else None
                        if sid0:
                            try:
                                nxt = get_min_timestamp_at_or_after(sid0, self.sim_rt_end_time)
                                if nxt is not None:
                                    self.sim_rt_start_time = nxt
                                    self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)
                                else:
                                    # Bounded DB lookup can miss historical replay rows; still advance the sim clock.
                                    self.sim_rt_start_time = self.sim_rt_end_time
                                    self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)
                            except Exception:
                                # DB unavailable or connection lost; advance by step to keep UI alive
                                self.sim_rt_start_time = self.sim_rt_end_time
                                self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)
        finally:
            if _csv_incremental_batch:
                self._csv_incremental_gui_busy = False

    def _detrend_for_plot(self, ctx: SensorContext, ys: List[float]) -> List[float]:
        """
        Prepare values for plotting.

        For the multi-sensor DB workflow we now plot the (optionally low-pass filtered)
        resultant magnetic field directly, without subtracting a baseline.
        """
        return ys

    @staticmethod
    def _downsample_for_canvas(xs: List, ys: List[float]) -> Tuple[List, List[float]]:
        """Evenly subsample so Matplotlib is responsive on long CSV/DB windows."""
        n = len(xs)
        if n <= MAX_CANVAS_LINE_POINTS or n != len(ys):
            return xs, ys
        idx = np.unique(np.linspace(0, n - 1, num=MAX_CANVAS_LINE_POINTS, dtype=int))
        return [xs[i] for i in idx], [float(ys[i]) for i in idx]

    def _configure_high_contrast_timeseries_theme(self) -> None:
        """
        Force readable typography/colors so exported/screenshot plots stay clear.
        """
        try:
            import matplotlib as mpl

            mpl.rcParams["font.family"] = TIMESERIES_FONT_FAMILIES
            mpl.rcParams["font.weight"] = "bold"
            mpl.rcParams["axes.labelweight"] = "bold"
            mpl.rcParams["axes.titleweight"] = "bold"
            mpl.rcParams["axes.edgecolor"] = TIMESERIES_COLOR_TEXT
            mpl.rcParams["axes.labelcolor"] = TIMESERIES_COLOR_TEXT
            mpl.rcParams["xtick.color"] = TIMESERIES_COLOR_TEXT
            mpl.rcParams["ytick.color"] = TIMESERIES_COLOR_TEXT
        except Exception:
            pass

    def _apply_timeseries_axis_style(self, ax) -> None:
        if ax is None:
            return
        try:
            fig = ax.figure
            if fig is not None:
                fig.set_facecolor(TIMESERIES_COLOR_BACKGROUND)
            ax.set_facecolor(TIMESERIES_COLOR_BACKGROUND)
            ax.grid(True, color=TIMESERIES_COLOR_GRID, linewidth=1.0, alpha=0.8)
            for spine in ax.spines.values():
                spine.set_color(TIMESERIES_COLOR_TEXT)
                spine.set_linewidth(1.2)
            ax.tick_params(axis="x", which="major", labelsize=TIMESERIES_XTICK_LABELSIZE, colors=TIMESERIES_COLOR_TEXT, width=1.2)
            ax.tick_params(axis="y", which="major", labelsize=TIMESERIES_YTICK_LABELSIZE, colors=TIMESERIES_COLOR_TEXT, width=1.2)
            if hasattr(ax, "xaxis") and getattr(ax.xaxis, "label", None) is not None:
                ax.xaxis.label.set_fontfamily(TIMESERIES_FONT_FAMILIES)
                ax.xaxis.label.set_fontweight("bold")
                ax.xaxis.label.set_color(TIMESERIES_COLOR_TEXT)
            if hasattr(ax, "yaxis") and getattr(ax.yaxis, "label", None) is not None:
                ax.yaxis.label.set_fontfamily(TIMESERIES_FONT_FAMILIES)
                ax.yaxis.label.set_fontweight("bold")
                ax.yaxis.label.set_color(TIMESERIES_COLOR_TEXT)
            for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                lbl.set_fontfamily(TIMESERIES_FONT_FAMILIES)
                lbl.set_fontweight("bold")
                lbl.set_color(TIMESERIES_COLOR_TEXT)
        except Exception:
            pass

    def get_since_times(self) -> Dict[str, datetime]:
        since: Dict[str, datetime] = {}
        for sid in self.sensor_ids:
            ctx = self.sensor_ctx.get(sid)
            if ctx is None:
                continue
            # Incremental fetch should start after the latest known timestamp in the plotted stream.
            if ctx.rt_x_t:
                since[sid] = ctx.rt_x_t[-1]
            elif ctx.base_x_t:
                since[sid] = ctx.base_x_t[-1]
        return since

    def load_plot_framework_2(self):
        """
        Create a single multi-sensor TimeSeries view.

        Up to 3 selected sensors are shown simultaneously in one window with:
        - ONE shared control panel (parameters apply to all sensors)
        - Separate time-series plots for each sensor (low-pass filtered resultant magnetic field)
        """
        self._discover_sensors()
        window = self.appWin

        # Attach our multi-sensor TimeSeries UI to the minimal container created by ApplicationWindowTemp.
        host = getattr(window, "_temp_timeseries_container", None)
        if host is None:
            # Fallback: if minimal UI isn't available for some reason, use the original tab_2.
            host = window.tab_2

        # Clear any existing layout contents (best-effort)
        if host.layout() is None:
            host.setLayout(Qt.QVBoxLayout())
        outer = host.layout()
        while outer.count():
            item = outer.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        # We no longer use per-sensor QTabWidget.
        self._time_series_tabs = None
        
        # Initialize list to track control widgets for synchronization
        self._sensor_control_widgets = []

        # ===== RIGHT HALF: shared parameters — insert at top of right column =====
        right_layout = getattr(window, "_right_layout", None)
        first_sensor_id = self.sensor_ids[0] if self.sensor_ids else None
        if first_sensor_id and right_layout is not None:
            params_container = QWidget()
            params_container.setMinimumHeight(260)
            params_container.setMaximumHeight(320)
            params_container.setMinimumWidth(220)
            params_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            params_layout = Qt.QVBoxLayout(params_container)
            params_layout.setContentsMargins(0, 0, 0, 0)
            params_layout.setSpacing(4)

            controls_header = QLabel("<b>Shared Parameters (apply to all sensors)</b>")
            controls_header.setStyleSheet("font-size: 11px; padding: 3px 6px; background-color: #e8f4f8; border: 1px solid #4a90e2; font-weight: bold;")
            params_layout.addWidget(controls_header)

            shared_controls = SensorMagTimeSeriesWidget(self, sensor_id=first_sensor_id, parent=window)
            shared_controls.setMinimumHeight(240)
            shared_controls.setMaximumHeight(300)
            shared_controls.setMinimumWidth(200)
            try:
                shared_controls.comboBox.setItemText(0, "IITK Observatory")
            except Exception:
                pass

            def hide_plot_area():
                try:
                    shared_controls.scrollArea.setWidgetResizable(False)
                    scroll_widget = shared_controls.scrollArea.widget()
                    if scroll_widget:
                        scroll_widget.setFixedHeight(10)
                        scroll_widget.setFixedWidth(max(shared_controls.scrollArea.width(), 400))
                        plot_layout = getattr(shared_controls, "verticalLayout_3", None)
                        if plot_layout is not None:
                            while plot_layout.count():
                                item = plot_layout.takeAt(0)
                                if item:
                                    w = item.widget()
                                    if w:
                                        if isinstance(w, FigureCanvas):
                                            try:
                                                w.figure.set_size_inches(1.0, 1.0, forward=False)
                                                w.setMinimumSize(1, 1)
                                                w.setMaximumSize(1, 1)
                                            except Exception:
                                                pass
                                        w.hide()
                                        w.setParent(None)
                                    else:
                                        plot_layout.removeItem(item)
                        scroll_widget.hide()
                    shared_controls.scrollArea.setFixedHeight(0)
                    shared_controls.scrollArea.hide()
                    shared_controls.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                    shared_controls.setMinimumHeight(240)
                    shared_controls.setMaximumHeight(300)
                except Exception as e:
                    self.log(f"Warning: Could not fully configure controls widget layout: {e}", level="Warning")

            QTimer.singleShot(300, hide_plot_area)
            params_layout.addWidget(shared_controls)
            shared_controls.show()

            right_layout.insertWidget(0, params_container, 1)
            params_container.show()
            if getattr(window, "_right_half", None) is not None:
                window._right_half.updateGeometry()

        # ===== LEFT HALF: one scrollable panel per sensor (each scrollable L/R/U/D) =====
        for sid in self.sensor_ids:
            ctx = self.sensor_ctx.get(sid)
            if ctx is None:
                continue

            # Inner panel for this sensor (label + toolbar + canvas)
            plot_panel = QWidget()
            plot_panel_layout = Qt.QVBoxLayout(plot_panel)
            plot_panel_layout.setContentsMargins(4, 4, 4, 4)
            plot_panel_layout.setSpacing(4)
            plot_panel.setMinimumSize(440, 320)  # Keep panel compact enough for full-window visibility
            plot_panel.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

            sensor_label = QLabel(f"<b>Sensor: {ctx.display_name}</b>")
            sensor_label.setStyleSheet("font-size: 12px; padding: 4px; background-color: #f0f0f0; border: 1px solid #ccc;")
            plot_panel_layout.addWidget(sensor_label)

            fig_width, fig_height = 6.0, 3.0
            dynamic_canvas = FigureCanvas(Figure(figsize=(fig_width, fig_height), dpi=TIMESERIES_FIG_DPI))
            dynamic_canvas.setMinimumSize(400, 240)
            dynamic_canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
            plot_panel_layout.addWidget(NavigationToolbar(dynamic_canvas, window))
            plot_panel_layout.addWidget(dynamic_canvas)

            ctx.static_canvas = None
            ctx.static_ax = None
            ctx.static_line = None
            ctx.dynamic_canvas = dynamic_canvas
            ctx.dynamic_ax = dynamic_canvas.figure.subplots()
            self._apply_timeseries_axis_style(ctx.dynamic_ax)
            try:
                # Label top-to-bottom sensor plots as B1, B2, B3 on Y-axis.
                sensor_position = self.sensor_ids.index(sid) + 1
                ctx.dynamic_ax.set_ylabel(
                    rf"$B_{{{sensor_position}}}$ (nT)",
                    fontsize=12,
                    fontweight="bold",
                    fontfamily=TIMESERIES_FONT_FAMILIES,
                    color=TIMESERIES_COLOR_TEXT,
                )
            except Exception:
                pass
            ctx.dynamic_line = None

            if ctx.base_x_t and ctx.base_y_mag_t:
                # Plot the historic (filtered) snapshot as the initial line.
                y0 = self._detrend_for_plot(ctx, ctx.base_y_mag_t)
                ctx.dynamic_line, = ctx.dynamic_ax.plot(
                    ctx.base_x_t,
                    y0,
                    color=TIMESERIES_COLOR_BASELINE,
                    linewidth=TIMESERIES_LINEWIDTH,
                    zorder=TIMESERIES_ZORDER_BASELINE,
                )
                # Save once on initial load and start predictor immediately so
                # forecast is available before/alongside new realtime points.
                self.save_data_for_sensor(sid, ctx.base_x_t, ctx.base_y_mag_t, start_predictor=True)
                ctx.last_saved_points = len(ctx.base_x_t)

            # Wrap each sensor panel in its own scroll area (scrollable left, right, up, down)
            sensor_scroll = QScrollArea()
            sensor_scroll.setWidget(plot_panel)
            sensor_scroll.setWidgetResizable(True)
            sensor_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            sensor_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            sensor_scroll.setMinimumHeight(220)
            sensor_scroll.setFrameShape(QFrame.StyledPanel)
            outer.addWidget(sensor_scroll, 1)  # stretch 1 so all three panels share left half equally

        # Timers for periodic fetch + drawing
        self._multi_data_timer = QTimer()
        self._multi_data_timer.timeout.connect(lambda: self.appWin.startThreads(hours=None, start_time=None, new=True))
        self._multi_data_timer.start(1000 * 20)

        self._multi_draw_timer = QTimer()
        # Redraw all visible sensor plots on each tick (up to 3 sensors).
        self._multi_draw_timer.timeout.connect(self.update_all_canvases)
        # 800 ms keeps Matplotlib + anomaly overlays off the hot path (400 ms was heavy with long series).
        self._multi_draw_timer.start(800)

        # Poll predictor outputs at a lower rate to keep UI responsive
        self._multi_pred_timer = QTimer()
        self._multi_pred_timer.timeout.connect(self.poll_predictions_all_sensors)
        # Short interval so completed predictor runs are merged quickly when not driven by on_db_data_updated.
        self._multi_pred_timer.start(400)

        # Predictor scheduler: start at most N predictor subprocesses at a time.
        self._predict_sched_timer = QTimer()
        self._predict_sched_timer.timeout.connect(self._drain_predict_queue)
        self._predict_sched_timer.start(500)

        # Fallback: if the DB does not advance (no "green" data arrives),
        # start predictors after a short grace period so predictions still appear.
        QTimer.singleShot(self._predict_start_grace_seconds * 1000, self._start_predictors_if_idle)

        self._maybe_apply_csv_fast_playback_mode()
        self._maybe_apply_fast_db_simulation_mode()

        # Window title: show selected sensor name(s) instead of hardcoded text below "Magnavis"
        try:
            sensor_label = ", ".join(ctx.display_name for ctx in self.sensor_ctx.values())
            if sensor_label:
                window.setWindowTitle(f"Magnavis – {sensor_label}")
            else:
                window.setWindowTitle("Magnavis")
        except Exception:
            window.setWindowTitle("Magnavis")

        self.log(f"Multi-sensor TimeSeries loaded: left={len(self.sensor_ids)} sensor streams (each scrollable), right=parameters + log.", level="Info")

    def _maybe_apply_csv_fast_playback_mode(self) -> None:
        """
        After normal CSV startup (all dialogs done), enable fast ingest by default (see
        ``_fast_csv_playback_requested``). Re-tunes timers so CSV ingest is not wall-clock-paced
        to realtime; matplotlib refresh is paused until ``_finalize_csv_fast_playback_plots``
        runs at end-of-file.
        """
        if not getattr(self, "csv_enabled", False):
            return
        if not _fast_csv_playback_requested():
            return
        self._csv_fast_playback = True
        self._predict_cooldown_seconds_before_fast = int(getattr(self, "_predict_cooldown_seconds", 20))
        self._predict_cooldown_seconds = 0

        step = _parse_fast_csv_sim_step_seconds()
        self.sim_step_seconds = step
        try:
            self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=int(step))
        except Exception:
            pass

        ms = _parse_fast_csv_data_interval_ms()
        data_iv = max(1, ms)  # Qt: 0 ms is not portable; use ≥1 ms

        try:
            tmr = getattr(self, "_multi_data_timer", None)
            if tmr is not None:
                tmr.stop()
                tmr.start(int(data_iv))
        except Exception:
            pass

        try:
            draw_tmr = getattr(self, "_multi_draw_timer", None)
            if draw_tmr is not None and draw_tmr.isActive():
                draw_tmr.stop()
        except Exception:
            pass

        try:
            pred_tmr = getattr(self, "_multi_pred_timer", None)
            if pred_tmr is not None:
                pred_tmr.stop()
                pred_tmr.start(50)
        except Exception:
            pass

        try:
            sched = getattr(self, "_predict_sched_timer", None)
            if sched is not None:
                sched.stop()
                sched.start(50)
        except Exception:
            pass

        self.log(
            f"Fast CSV playback enabled: data timer={data_iv} ms, sim_step={step}s, "
            f"matplotlib redraw paused until end of file (set MAGNAVIS_FAST_CSV_PLAYBACK=0 for wall-clock CSV pacing).",
            level="Info",
        )

    def _maybe_apply_fast_db_simulation_mode(self) -> None:
        """
        DB-backed **simulation** (no CSV file): shorten wall-clock wait between fetches and advance
        the simulation clock by more than 20 s per tick so replays do not track real-time 1:1.
        Does not pause matplotlib (unlike fast CSV): there is no single EOF to restore timers.
        """
        if not getattr(self, "sim_enabled", False) or getattr(self, "csv_enabled", False):
            return
        if not _fast_db_simulation_requested():
            return
        self._predict_cooldown_seconds_before_fast = int(getattr(self, "_predict_cooldown_seconds", 20))
        self._predict_cooldown_seconds = 0

        step = _parse_fast_sim_step_seconds()
        self.sim_step_seconds = step
        try:
            self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=int(step))
        except Exception:
            pass

        ms = _parse_fast_sim_data_interval_ms()
        data_iv = max(10, ms)

        try:
            tmr = getattr(self, "_multi_data_timer", None)
            if tmr is not None:
                tmr.stop()
                tmr.start(int(data_iv))
        except Exception:
            pass

        try:
            pred_tmr = getattr(self, "_multi_pred_timer", None)
            if pred_tmr is not None:
                pred_tmr.stop()
                pred_tmr.start(50)
        except Exception:
            pass

        try:
            sched = getattr(self, "_predict_sched_timer", None)
            if sched is not None:
                sched.stop()
                sched.start(50)
        except Exception:
            pass

        self.log(
            f"Fast DB simulation: data timer={data_iv} ms, sim_step={step}s "
            f"(set MAGNAVIS_FAST_SIM_PLAYBACK=0 for 20 s / 20 s pacing; tune with "
            f"MAGNAVIS_SIM_FAST_STEP_SECONDS, MAGNAVIS_SIM_FAST_DATA_INTERVAL_MS).",
            level="Info",
        )

    def _finalize_csv_fast_playback_plots(self) -> None:
        """Re-enable periodic plot refresh and draw the final series after fast CSV ingest completes."""
        if not getattr(self, "_csv_fast_playback", False):
            return
        self._csv_fast_playback = False
        try:
            self._predict_cooldown_seconds = int(getattr(self, "_predict_cooldown_seconds_before_fast", 20))
        except Exception:
            self._predict_cooldown_seconds = 20

        try:
            draw_tmr = getattr(self, "_multi_draw_timer", None)
            if draw_tmr is not None and not draw_tmr.isActive():
                draw_tmr.start(800)
        except Exception:
            pass

        try:
            pred_tmr = getattr(self, "_multi_pred_timer", None)
            if pred_tmr is not None:
                pred_tmr.stop()
                pred_tmr.start(400)
        except Exception:
            pass

        try:
            sched = getattr(self, "_predict_sched_timer", None)
            if sched is not None:
                sched.stop()
                sched.start(500)
        except Exception:
            pass

        self.log("Fast CSV playback finished: restoring chart timers; running prediction catch-up for snapshot.", level="Info")
        self._flush_predict_input_writes()
        self._csv_catchup_predict_only = True
        self._predict_max_concurrent_before_csv_catchup = int(getattr(self, "_predict_max_concurrent", 3))
        self._predict_max_concurrent = 1
        self._csv_catchup_ticks = 0
        if self._csv_catchup_timer is None:
            self._csv_catchup_timer = QTimer(self)
            self._csv_catchup_timer.timeout.connect(self._csv_fast_playback_catchup_tick)
        else:
            self._csv_catchup_timer.stop()
        self._csv_catchup_timer.start(150)

    def _csv_stop_catchup_timer(self) -> None:
        try:
            if self._csv_catchup_timer is not None:
                self._csv_catchup_timer.stop()
        except Exception:
            pass
        self._csv_catchup_predict_only = False
        try:
            self._predict_max_concurrent = int(getattr(self, "_predict_max_concurrent_before_csv_catchup", 3))
        except Exception:
            self._predict_max_concurrent = 3

    def _csv_fast_playback_catchup_tick(self) -> None:
        """
        After fast CSV EOF: poll subprocesses, extend forecasts until ``latest_pred >= latest_actual``,
        then one full anomaly pass + redraw so HD snapshot shows purple + detected anomalies.
        """
        if not getattr(self, "csv_enabled", False):
            self._csv_stop_catchup_timer()
            return
        self._csv_catchup_ticks += 1
        max_ticks = 900  # ~135 s at 150 ms
        try:
            self._flush_predict_input_writes()
            for sid in self.sensor_ids:
                self.update_predictions_for_sensor(sid)

            any_running = False
            for sid in self.sensor_ids:
                ctx = self.sensor_ctx.get(sid)
                if ctx is not None and ctx.prediction_process is not None and ctx.prediction_process.poll() is None:
                    any_running = True
                    break
            self.update_all_canvases()
            if any_running:
                if self._csv_catchup_ticks >= max_ticks:
                    self.log(
                        "CSV catch-up: predictors still running after extended wait — snapshot may lack full purple/anomalies.",
                        level="Warning",
                    )
                    self._csv_stop_catchup_timer()
                return

            any_gap = False
            for sid in self.sensor_ids:
                ctx = self.sensor_ctx.get(sid)
                if ctx is None or not ctx.rt_x_t:
                    continue
                latest_actual = pd.to_datetime(ctx.rt_x_t[-1])
                latest_pred = pd.to_datetime(ctx.predict_x_t[-1]) if ctx.predict_x_t else None
                if latest_pred is None or latest_pred < latest_actual:
                    any_gap = True
                    self._ensure_prediction_covers_actual(sid, force=True)

            if not any_gap:
                for sid in self.sensor_ids:
                    ctx = self.sensor_ctx.get(sid)
                    if ctx is None or not ctx.rt_x_t or not ctx.predict_x_t:
                        continue
                    self.reset_anomaly_state_for_sensor(sid)
                    self.detect_anomalies_for_sensor(sid)
                    self._redraw_anomalies(sid)
                self.update_all_canvases()
                self.log(
                    "CSV fast playback: predictions cover full actual — purple + anomaly overlays ready. "
                    "Use HD snapshot (toolbar or Ctrl+Shift+S) for the final figure.",
                    level="Info",
                )
                self._csv_stop_catchup_timer()
            elif self._csv_catchup_ticks >= max_ticks:
                self.log(
                    "CSV catch-up: timed out before forecasts reached latest actual; check predict_stderr.log per sensor.",
                    level="Warning",
                )
                self._csv_stop_catchup_timer()
        except Exception:
            self._csv_stop_catchup_timer()

    def update_all_canvases(self):
        """
        Redraw all sensor canvases.

        With the stacked multi-sensor layout (no tabs), all up-to-3 sensors are visible
        simultaneously, so we simply refresh every sensor on each timer tick.
        """
        # During fast CSV the multi-draw timer is stopped; this runs from updateData after each ingest
        # batch so the purple prediction trace and anomaly overlays stay in sync with ``predict_out``.
        for sid in self.sensor_ids:
            self.update_canvas_for_sensor(sid, poll_predictions=True)

    def update_canvas_for_sensor(self, sensor_id: str, poll_predictions: bool = True):
        ctx = self.sensor_ctx.get(sensor_id)
        if ctx is None or ctx.dynamic_ax is None:
            return

        # Update/plot prediction first so it stays beneath actual traces.
        if poll_predictions:
            self.update_predictions_for_sensor(sensor_id)
        if ctx.predict_x_t and ctx.predict_y_t:
            px, py = self._downsample_for_canvas(list(ctx.predict_x_t), list(ctx.predict_y_t))
            py_plot = self._detrend_for_plot(ctx, py)
            if ctx.predictions_line is None:
                ctx.predictions_line, = ctx.dynamic_ax.plot(
                    px,
                    py_plot,
                    color=TIMESERIES_COLOR_PREDICTION,
                    linewidth=TIMESERIES_LINEWIDTH,
                    zorder=TIMESERIES_ZORDER_PREDICTION,
                )
            else:
                ctx.predictions_line.set_data(px, py_plot)
                ctx.predictions_line.set_color(TIMESERIES_COLOR_PREDICTION)
                ctx.predictions_line.set_linewidth(TIMESERIES_LINEWIDTH)
                ctx.predictions_line.set_zorder(TIMESERIES_ZORDER_PREDICTION)
        elif ctx.predictions_line is not None:
            try:
                ctx.predictions_line.set_data([], [])
            except Exception:
                pass

        # Ensure the historic (blue) line stays on the original snapshot (do not overwrite).
        if ctx.dynamic_line is not None and ctx.base_x_t and ctx.base_y_mag_t:
            try:
                bx, by = self._downsample_for_canvas(list(ctx.base_x_t), list(ctx.base_y_mag_t))
                ctx.dynamic_line.set_data(bx, self._detrend_for_plot(ctx, by))
                ctx.dynamic_line.set_color(TIMESERIES_COLOR_BASELINE)
                ctx.dynamic_line.set_linewidth(TIMESERIES_LINEWIDTH)
                ctx.dynamic_line.set_zorder(TIMESERIES_ZORDER_BASELINE)
            except Exception:
                pass

        # Real-time (green): prefer actuals only up to the last predicted time so the purple line
        # (lower z than green) stays visible for the overlap; green is drawn on top. If there are no
        # predictions yet, or the filter drops everything (timing mismatch), plot all actuals.
        rt_plot_x: List[datetime] = []
        rt_plot_y: List[float] = []
        if ctx.rt_x_t and ctx.rt_y_mag_t:
            if ctx.predict_x_t and ctx.predict_y_t:
                latest_pred_t = pd.to_datetime(ctx.predict_x_t[-1])
                for t, v in zip(ctx.rt_x_t, ctx.rt_y_mag_t):
                    if pd.to_datetime(t) <= latest_pred_t:
                        rt_plot_x.append(t)
                        rt_plot_y.append(v)
                if not rt_plot_x:
                    rt_plot_x = list(ctx.rt_x_t)
                    rt_plot_y = list(ctx.rt_y_mag_t)
            else:
                rt_plot_x = list(ctx.rt_x_t)
                rt_plot_y = list(ctx.rt_y_mag_t)

        if rt_plot_x and rt_plot_y:
            rx, ry = self._downsample_for_canvas(rt_plot_x, rt_plot_y)
            ry_plot = self._detrend_for_plot(ctx, ry)
            if ctx.dynamic_new_line is None:
                ctx.dynamic_new_line, = ctx.dynamic_ax.plot(
                    rx,
                    ry_plot,
                    color=TIMESERIES_COLOR_REALTIME,
                    linewidth=TIMESERIES_LINEWIDTH,
                    zorder=TIMESERIES_ZORDER_REALTIME,
                )
            else:
                ctx.dynamic_new_line.set_data(rx, ry_plot)
                ctx.dynamic_new_line.set_color(TIMESERIES_COLOR_REALTIME)
                ctx.dynamic_new_line.set_linewidth(TIMESERIES_LINEWIDTH)
                ctx.dynamic_new_line.set_zorder(TIMESERIES_ZORDER_REALTIME)
        elif ctx.dynamic_new_line is not None:
            try:
                ctx.dynamic_new_line.set_data([], [])
            except Exception:
                pass

        # Update anomaly vertical lines (dynamic + static)
        self._redraw_anomalies(sensor_id)

        restyle_axes = False
        # Update limits similar to application.py when new data arrives
        if ctx.needs_update_lims and ctx.base_x_t:
            try:
                x0 = ctx.base_x_t[0]
                x1_candidates = [ctx.base_x_t[-1]]
                if rt_plot_x:
                    x1_candidates.append(rt_plot_x[-1])
                if ctx.predict_x_t:
                    x1_candidates.append(ctx.predict_x_t[-1])
                x1 = max(pd.to_datetime(t) for t in x1_candidates)
                y_candidates = ctx.base_y_mag_t + (rt_plot_y if rt_plot_y else [])
                if ctx.predict_y_t:
                    y_candidates += ctx.predict_y_t
                yr = self._detrend_for_plot(ctx, y_candidates)
                if yr:
                    ymax = max(yr)
                    ymin = min(yr)
                    _xrange = (x1 - x0) if hasattr(x1, "__sub__") else 1
                    _yrange = ymax - ymin if ymax != ymin else 1.0
                    ctx.dynamic_ax.set_xlim(x0, x1)
                    ctx.dynamic_ax.set_ylim(ymin - 0.05 * _yrange, ymax + 0.05 * _yrange)
                    if ctx.static_ax is not None:
                        ctx.static_ax.set_xlim(x0, x1)
                        ctx.static_ax.set_ylim(ymin - 0.05 * _yrange, ymax + 0.05 * _yrange)
                    restyle_axes = True
                ctx.needs_update_lims = False
            except Exception:
                ctx.needs_update_lims = False

        # Draw (avoid re-styling tick fonts / rebuilding legend every timer tick — that was freezing the UI).
        try:
            if restyle_axes:
                self._apply_timeseries_axis_style(ctx.dynamic_ax)
            if restyle_axes or not getattr(self, "_timeseries_legend_ready", False):
                self._update_first_sensor_legend(sensor_id, ctx)
                if self.sensor_ids and sensor_id == self.sensor_ids[0]:
                    self._timeseries_legend_ready = True
            if ctx.dynamic_canvas:
                ctx.dynamic_canvas.draw_idle()
            if ctx.static_canvas:
                ctx.static_canvas.draw_idle()
        except Exception:
            pass

    def _update_first_sensor_legend(self, sensor_id: str, ctx: SensorContext) -> None:
        """Show a compact color legend only on the first sensor plot."""
        if ctx.dynamic_ax is None or not self.sensor_ids:
            return

        first_sensor_id = self.sensor_ids[0]
        if sensor_id != first_sensor_id:
            # Keep legend exclusive to the first sensor panel.
            old_leg = ctx.dynamic_ax.get_legend()
            if old_leg is not None:
                try:
                    old_leg.remove()
                except Exception:
                    pass
            return

        handles = [
            Line2D([0], [0], color=TIMESERIES_COLOR_BASELINE, linewidth=TIMESERIES_LINEWIDTH, label="Historic data"),
            Line2D([0], [0], color=TIMESERIES_COLOR_REALTIME, linewidth=TIMESERIES_LINEWIDTH, label="Actual data"),
            Line2D([0], [0], color=TIMESERIES_COLOR_PREDICTION, linewidth=TIMESERIES_LINEWIDTH, label="Predicted data"),
            Line2D([0], [0], color=TIMESERIES_COLOR_ANOMALY, linewidth=3.0, alpha=TIMESERIES_ALPHA_ANOMALY, label="Detected anomaly"),
            Patch(
                facecolor=TIMESERIES_COLOR_ANOMALY_INTRODUCED,
                edgecolor="none",
                alpha=0.75,
                label="Anomaly introduced",
            ),
        ]
        ctx.dynamic_ax.legend(
            handles=handles,
            loc="upper left",
            frameon=True,
            fontsize=7,
        )

    def poll_predictions_all_sensors(self):
        """Low-frequency polling of predictor outputs to avoid blocking UI redraws."""
        for sid in self.sensor_ids:
            self.update_predictions_for_sensor(sid)

    def _redraw_anomalies(self, sensor_id: str):
        ctx = self.sensor_ctx.get(sensor_id)
        if ctx is None or ctx.dynamic_ax is None:
            return

        # Remove old
        for v in ctx.anomaly_vertical_lines:
            try:
                v.remove()
            except Exception:
                pass
        for v in ctx.anomaly_vertical_lines_static:
            try:
                v.remove()
            except Exception:
                pass
        for b in ctx.ground_truth_vertical_bands:
            try:
                b.remove()
            except Exception:
                pass
        ctx.anomaly_vertical_lines = []
        ctx.anomaly_vertical_lines_static = []
        ctx.ground_truth_vertical_bands = []

        # Draw ground-truth overlays as bottom strips (in front),
        # so they remain visible even when anomaly red lines overlap.
        # Introduced-anomaly GT strips: OBS1 time-series only (never OBS2), including odd DB ids.
        sensor_key = _sensor_display_name(ctx.display_name)
        show_gt_bands = _is_obs1_ui_sensor_label(sensor_key) or _is_obs1_ui_sensor_label(sensor_id) or _is_obs1_ui_sensor_label(
            ctx.display_name or ""
        )
        gt_magnet_intervals = (
            self._ground_truth_magnet_intervals_by_sensor.get(sensor_key)
            or self._ground_truth_intervals_by_sensor.get(sensor_key, [])
        )
        if show_gt_bands and gt_magnet_intervals:
            for s, e in gt_magnet_intervals:
                try:
                    band = ctx.dynamic_ax.axvspan(
                        s,
                        e,
                        ymin=0.00,
                        ymax=0.10,  # thicker bottom strip for visibility
                        color=TIMESERIES_COLOR_ANOMALY_INTRODUCED,
                        alpha=0.85,
                        ec="none",
                        lw=0.0,
                        zorder=12.0,  # keep in very front
                    )
                    ctx.ground_truth_vertical_bands.append(band)
                except Exception:
                    pass
        gt_trimmer_intervals = self._ground_truth_trimmer_intervals_by_sensor.get(sensor_key, [])
        if show_gt_bands and gt_trimmer_intervals:
            for s, e in gt_trimmer_intervals:
                try:
                    band = ctx.dynamic_ax.axvspan(
                        s,
                        e,
                        ymin=0.10,
                        ymax=0.20,
                        color=TIMESERIES_COLOR_ANOMALY_INTRODUCED,
                        alpha=0.85,
                        ec="none",
                        lw=0.0,
                        zorder=12.1,
                    )
                    ctx.ground_truth_vertical_bands.append(band)
                except Exception:
                    pass

        if not ctx.anomaly_times:
            return

        for t in ctx.anomaly_times:
            try:
                ctx.anomaly_vertical_lines.append(
                    ctx.dynamic_ax.axvline(
                        x=t,
                        color=TIMESERIES_COLOR_ANOMALY,
                        linestyle="-",
                        linewidth=2.8,
                        alpha=TIMESERIES_ALPHA_ANOMALY,
                        zorder=TIMESERIES_ZORDER_ANOMALY,
                    )
                )
            except Exception:
                pass

    def _apply_predict_input_payload(self, sensor_id: str, payload: Dict[str, Any]) -> None:
        """Write one pending predictor_input.csv and optionally enqueue a predictor run."""
        ctx = self.sensor_ctx.get(sensor_id)
        if ctx is None:
            return
        inp_file = payload["inp_file"]
        filtered_x = payload["filtered_x"]
        filtered_y = payload["filtered_y"]
        start_predictor = bool(payload.get("start_predictor", True))

        pd.DataFrame({"x": filtered_x, "y": filtered_y}).to_csv(inp_file, index=False)
        ctx.predictor_input_file = inp_file

        # First kickoff only: extensions are driven by update_predictions_for_sensor →
        # _ensure_prediction_covers_actual (avoids spawning a predictor on every debounced CSV write).
        if start_predictor and not ctx.predict_app_started and not ctx.predict_x_t:
            now = time.time()
            if ctx.last_pred_start_ts and (now - ctx.last_pred_start_ts) < self._predict_cooldown_seconds:
                return
            ctx.predict_app_started = True
            ctx.last_pred_start_ts = now
            self._enqueue_prediction(sensor_id)

    def _flush_predict_input_writes(self) -> None:
        try:
            self._predict_input_save_timer.stop()
        except Exception:
            pass
        batch = self._predict_input_pending
        self._predict_input_pending = {}
        for sid, payload in batch.items():
            self._apply_predict_input_payload(sid, payload)
        self._drain_predict_queue()

    def save_data_for_sensor(
        self,
        sensor_id: str,
        x_t: List[datetime],
        y_t: List[float],
        start_predictor: bool = True,
        up_to_time: Optional[datetime] = None,
    ):
        """
        Write time series to predictor input CSV for model training/inference.

        IMPORTANT:
        - Default behavior includes full arrived series (historic + realtime).
        - For sequential catch-up runs, caller may cap input to `up_to_time` so
          predictor extends from the last covered timestamp without skipping.
        - Drop timestamps that were flagged as anomalies from training input.
        - After the first on-disk file exists, writes are debounced (~400ms) so CSV playback
          does not block the GUI with full-file rewrites on every fetch tick.
        """
        ctx = self.sensor_ctx.get(sensor_id)
        if ctx is None:
            return

        # Session subfolder per sensor
        folder = os.path.join(base_app.APP_BASE, "sessions", self.session_id, sensor_id)
        os.makedirs(folder, exist_ok=True)
        inp_file = os.path.join(folder, "predict_input.csv")

        # Filter anomalies by timestamp. We intentionally keep all arrived points
        # (except anomalies) to avoid predictor-input stalls when overlap lags.
        cutoff = pd.to_datetime(up_to_time) if up_to_time is not None else None
        anomaly_times_set = set(pd.to_datetime(ctx.anomaly_times)) if ctx.anomaly_times else set()
        source_y = list(y_t)
        if getattr(self, "_predictor_use_raw_mag", False):
            raw = list(ctx.base_y_mag_raw_t) + list(ctx.rt_y_mag_raw_t)
            if len(raw) == len(x_t):
                source_y = raw
        filtered_x = []
        filtered_y = []
        for t, v in zip(x_t, source_y):
            t_dt = pd.to_datetime(t)
            if cutoff is not None and t_dt > cutoff:
                continue
            if t_dt not in anomaly_times_set:
                filtered_x.append(t_dt)
                filtered_y.append(v)

        payload: Dict[str, Any] = {
            "inp_file": inp_file,
            "filtered_x": filtered_x,
            "filtered_y": filtered_y,
            "start_predictor": start_predictor,
        }
        self._predict_input_pending[sensor_id] = payload

        # First write must be synchronous so subprocesses always see a real file path.
        if not os.path.isfile(inp_file):
            # A leftover predict_out.csv (e.g. crashed run) leaves predict_x_t-looking state and blocks
            # the ``not ctx.predict_x_t`` first-start gate in ``_apply_predict_input_payload``.
            out_path = os.path.join(folder, "predict_out.csv")
            if os.path.isfile(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            ctx.predict_x_t = []
            ctx.predict_y_t = []
            self._apply_predict_input_payload(sensor_id, payload)
            self._predict_input_pending.pop(sensor_id, None)
            return

        self._predict_input_save_timer.stop()
        self._predict_input_save_timer.start(400)

    def _start_predictors_if_idle(self):
        """
        If we haven't started predictors yet (no realtime updates), start them anyway
        so the app still produces predictions on purely "historic" data.
        """
        for sid, ctx in self.sensor_ctx.items():
            if ctx.predictor_input_file and not ctx.predict_app_started and ctx.prediction_process is None:
                ctx.predict_app_started = True
                ctx.last_pred_start_ts = time.time()
                self._enqueue_prediction(sid)

    def reset_anomaly_state_for_sensor(self, sensor_id: str) -> None:
        """
        Reset anomaly detector history for one sensor while preserving its settings.
        """
        ctx = self.sensor_ctx.get(sensor_id)
        if ctx is None:
            return
        det = ctx.anomaly_detector
        try:
            ctx.anomaly_detector = AnomalyDetector(
                threshold_multiplier=float(getattr(det, "threshold_multiplier", 2.5)),
                min_samples_for_threshold=int(getattr(det, "min_samples_for_threshold", 20)),
                error_smoothing_alpha=float(getattr(det, "error_smoothing_alpha", 0.995)),
                recent_error_buffer_size=int(getattr(det, "recent_error_buffer_size", 1000)),
                std_relative_floor=float(getattr(det, "std_relative_floor", 0.02)),
            )
        except Exception:
            ctx.anomaly_detector = AnomalyDetector()
            ctx.anomaly_detector.threshold_multiplier = float(getattr(det, "threshold_multiplier", 2.5))
            ctx.anomaly_detector.error_smoothing_alpha = float(getattr(det, "error_smoothing_alpha", 0.995))
            ctx.anomaly_detector.min_samples_for_threshold = int(getattr(det, "min_samples_for_threshold", 20))
            ctx.anomaly_detector.std_relative_floor = float(getattr(det, "std_relative_floor", 0.02))
        ctx.anomaly_times = []
        ctx.anomaly_values = []
        ctx.anomaly_vertical_lines = []
        ctx.anomaly_vertical_lines_static = []
        ctx.ground_truth_vertical_bands = []
        ctx.last_anomaly_checked_time = None

    def _enqueue_prediction(self, sensor_id: str):
        if sensor_id in self._predict_active:
            self._drain_predict_queue()
            return
        if sensor_id in self._predict_queue:
            self._drain_predict_queue()
            return
        self._predict_queue.append(sensor_id)
        self._drain_predict_queue()

    def _drain_predict_queue(self):
        # Start predictors until we hit concurrency limit.
        while len(self._predict_active) < self._predict_max_concurrent and self._predict_queue:
            sid = self._predict_queue.popleft()
            ctx = self.sensor_ctx.get(sid)
            if ctx is None or not ctx.predictor_input_file:
                continue
            # If process already exists/running, treat as active.
            if ctx.prediction_process is not None and ctx.prediction_process.poll() is None:
                self._predict_active.add(sid)
                continue
            self._predict_active.add(sid)
            self.start_prediction_process_for_sensor(sid)

    def _ensure_prediction_covers_actual(self, sensor_id: str, force: bool = False) -> None:
        """
        Ensure prediction horizon reaches the latest arrived actual timestamp.

        If actual data extends beyond the current prediction range, refresh predictor
        input and enqueue a new predictor run.

        Sequential catch-up rule:
        - if predictions already exist, rebuild predictor input only up to the last
          predicted timestamp so the next run forecasts the immediate next segment
          instead of jumping ahead to the newest actual point.
        """
        ctx = self.sensor_ctx.get(sensor_id)
        if ctx is None:
            return

        # If a predictor is already running, let it finish first.
        if ctx.prediction_process is not None and ctx.prediction_process.poll() is None:
            return

        if not ctx.base_x_t and not ctx.rt_x_t:
            return

        latest_actual = pd.to_datetime(ctx.rt_x_t[-1] if ctx.rt_x_t else ctx.base_x_t[-1])
        latest_pred = pd.to_datetime(ctx.predict_x_t[-1]) if ctx.predict_x_t else None
        if latest_pred is not None and latest_pred >= latest_actual:
            ctx.predict_cover_until = None
            return

        # Keep predictor input aligned with currently arrived series.
        # If we already have predictions, cap training input at latest_pred so
        # subsequent runs extend predictions sequentially with no timestamp gaps.
        x_all = ctx.base_x_t + ctx.rt_x_t
        y_all = ctx.base_y_mag_t + ctx.rt_y_mag_t
        if not x_all:
            return
        input_cutoff = latest_pred if latest_pred is not None else None
        if input_cutoff is not None:
            ctx.predict_cover_until = pd.Timestamp(latest_actual).to_pydatetime()
        else:
            ctx.predict_cover_until = None
        if input_cutoff is not None:
            self.log(
                f"[{ctx.display_name}] Sequential catch-up: latest_pred={pd.to_datetime(latest_pred)} "
                f"latest_actual={pd.to_datetime(latest_actual)}",
                level="Debug",
            )
        self.save_data_for_sensor(
            sensor_id,
            x_all,
            y_all,
            start_predictor=False,
            up_to_time=input_cutoff,
        )

        now = time.time()
        if (not force) and ctx.last_pred_start_ts and (now - ctx.last_pred_start_ts) < self._predict_cooldown_seconds:
            return
        if sensor_id in self._predict_queue:
            return
        # ``_predict_active`` can go stale relative to a finished subprocess; do not let it block EOF catch-up.
        if sensor_id in self._predict_active:
            self._predict_active.discard(sensor_id)

        ctx.predict_app_started = True
        ctx.last_pred_start_ts = now
        self._enqueue_prediction(sensor_id)

    def _latest_arrived_data_time(self, ctx: SensorContext) -> Optional[datetime]:
        """Latest timestamp currently available for this sensor."""
        try:
            if ctx.rt_x_t:
                return pd.to_datetime(ctx.rt_x_t[-1]).to_pydatetime()
            if ctx.base_x_t:
                return pd.to_datetime(ctx.base_x_t[-1]).to_pydatetime()
        except Exception:
            return None
        return None

    def _should_train_sensor_now(
        self, ctx: SensorContext, resolved_model_path: Optional[str]
    ) -> Tuple[bool, Optional[datetime], str]:
        """
        Decide if this predictor run should train.
        Training cadence is controlled by `ctx.retrain_interval_minutes` (default 60).

        When ``ctx.train_window_minutes == 0`` and a loadable model path exists, the **first**
        session run is predict-only (no initial ``fit`` on the full CSV). Fresh models with no
        checkpoint still require an initial training pass.
        """
        latest_data_time = self._latest_arrived_data_time(ctx)
        interval_minutes = int(max(1, getattr(ctx, "retrain_interval_minutes", self._default_retrain_interval_minutes)))

        if getattr(self, "_csv_catchup_predict_only", False) and getattr(self, "csv_enabled", False):
            return False, latest_data_time, "CSV fast-playback catch-up (predict-only)"

        if ctx.last_model_train_data_time is None:
            tw0 = getattr(ctx, "train_window_minutes", None)
            if tw0 == 0 and resolved_model_path:
                return False, latest_data_time, "predict-only first run (0 min window; checkpoint present)"
            return True, latest_data_time, "initial training"

        if latest_data_time is None:
            # If no data-time reference is available, keep previous model and predict-only.
            return False, latest_data_time, "no new data timestamp"

        elapsed_minutes = (latest_data_time - ctx.last_model_train_data_time).total_seconds() / 60.0
        if elapsed_minutes >= interval_minutes:
            return True, latest_data_time, f"elapsed {elapsed_minutes:.1f} min >= {interval_minutes} min"
        return False, latest_data_time, f"elapsed {elapsed_minutes:.1f} min < {interval_minutes} min"

    def _prediction_timestamp_key(self, t) -> pd.Timestamp:
        """Stable dict key for prediction times (avoids duplicate-key splits from tz/type mismatch)."""
        return pd.Timestamp(t).floor("us")

    def _compute_n_future_for_predictor(self, ctx: SensorContext) -> int:
        """
        Forecast steps so one run can reach the latest actual time from the last input row.
        Matches predictor_ai.forecast step sizing (median positive delta).

        Uses in-memory series only — never reads the full predict_input.csv (can be huge).
        """
        min_h = 100
        max_h = 200000
        margin = 30
        x_all = ctx.base_x_t + ctx.rt_x_t
        if len(x_all) < 1:
            return min_h
        try:
            ts = pd.to_datetime(pd.Series(list(x_all)), errors="coerce").dropna()
            if ts.empty:
                return min_h
            ts = ts.sort_values()
            dt_last = ts.iloc[-1]
            if len(ts) > 1:
                diffs = ts.diff().dropna()
                diffs = diffs[diffs > pd.Timedelta(0)]
                step = diffs.median() if len(diffs) else pd.Timedelta(seconds=1)
            else:
                step = pd.Timedelta(seconds=1)
            sec = max(step.total_seconds(), 1e-6)
        except Exception:
            return min_h

        latest_actual = pd.to_datetime(x_all[-1])
        # Sequential catch-up: subprocess input ends at last merged pred; horizon must span to predict_cover_until.
        pcu = getattr(ctx, "predict_cover_until", None)
        if pcu is not None and ctx.predict_x_t:
            try:
                last_pred = pd.to_datetime(ctx.predict_x_t[-1])
                delta_sec = max(0.0, (pd.Timestamp(pcu) - last_pred).total_seconds())
                if delta_sec > 0:
                    n = int(math.ceil(delta_sec / sec)) + margin
                    out = max(min_h, min(n, max_h))
                    cap = getattr(self, "_predictor_n_future_cap", None)
                    if cap is not None:
                        out = min(out, int(cap))
                    return out
            except Exception:
                pass
        # Cover new actuals since the last completed training (not (latest − latest) which was always 0).
        train_t = getattr(ctx, "last_model_train_data_time", None)
        if train_t is not None:
            anchor = pd.to_datetime(train_t)
            delta_sec = max(0.0, (latest_actual - anchor).total_seconds())
        else:
            # First predictor run in this session: input already ends at latest_actual; short horizon is enough.
            delta_sec = 0.0
        if delta_sec <= 0:
            return min_h
        n = int(math.ceil(delta_sec / sec)) + margin
        out = max(min_h, min(n, max_h))
        cap = getattr(self, "_predictor_n_future_cap", None)
        if cap is not None:
            out = min(out, int(cap))
        return out

    def start_prediction_process_for_sensor(self, sensor_id: str):
        ctx = self.sensor_ctx.get(sensor_id)
        if ctx is None or not ctx.predictor_input_file:
            return

        python_exe = sys.executable
        predictor_script = os.path.join(base_app.APP_BASE, "predictor_ai.py")
        input_file = ctx.predictor_input_file
        work_dir = os.path.dirname(input_file)

        env = os.environ.copy()
        # Reduce CPU contention so the GUI remains responsive.
        # (Does not change model semantics; only controls parallelism.)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
        env.setdefault("TF_NUM_INTEROP_THREADS", "1")
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        # predictor_ai historically called fit() on every autoregressive step (extremely slow). Off by default there.
        env.setdefault("PREDICTOR_ONLINE_FIT_EACH_STEP", "0")
        env.setdefault("PREDICTOR_EPOCHS_PER_UPDATE", "40")
        twm = getattr(ctx, "train_window_minutes", None)
        if twm is not None and twm > 0:
            env["TRAIN_WINDOW_MINUTES"] = str(twm)
        else:
            env.pop("TRAIN_WINDOW_MINUTES", None)
        model_family = _normalize_model_family(self.predictor_model_family)
        model_init = _normalize_model_init(self.predictor_model_init)
        if model_family == MODEL_FAMILY_TRANSFORMER and model_init == MODEL_INIT_FRESH:
            model_init = MODEL_INIT_PRETRAINED
        if model_family == MODEL_FAMILY_ATTN_BILSTM:
            model_init = MODEL_INIT_FRESH
        env["PREDICTOR_MODEL_FAMILY"] = model_family
        env["PREDICTOR_MODEL_INIT"] = model_init
        if (model_family == MODEL_FAMILY_GRU and model_init == MODEL_INIT_FRESH) or (
            model_family == MODEL_FAMILY_LSTM and model_init == MODEL_INIT_FRESH
        ):
            env["PREDICTOR_GRU_WINDOW_SIZE"] = str(_clamp_predictor_gru_window(int(self.predictor_gru_window_size)))
        else:
            env.pop("PREDICTOR_GRU_WINDOW_SIZE", None)
        if model_family == MODEL_FAMILY_TRANSFORMER:
            model_label = "Transformer"
        elif model_family == MODEL_FAMILY_ATTN_BILSTM:
            model_label = "Attention-BiLSTM"
        elif model_family == MODEL_FAMILY_LSTM:
            model_label = "Vanilla-LSTM"
        else:
            model_label = "GRU"
        # Persist a runtime model checkpoint per sensor so inference-only runs can reload it.
        runtime_model_path = os.path.join(work_dir, f"predictor_runtime_{model_family}.keras")
        ctx.runtime_model_path = runtime_model_path
        env["PREDICTOR_CHECKPOINT_PATH"] = runtime_model_path

        # Candidate model path to load before prediction:
        # 1) runtime checkpoint (latest trained model in this session), else
        # 2) configured pre-trained model (when mode allows).
        resolved_model_path = runtime_model_path if os.path.exists(runtime_model_path) else None
        if resolved_model_path:
            env["PRETRAINED_MODEL_PATH"] = resolved_model_path
        else:
            env.pop("PRETRAINED_MODEL_PATH", None)
            if model_init != MODEL_INIT_FRESH:
                # Set pre-trained model path per sensor (if available)
                # Default: project root "models" folder (works when run from any directory).
                pretrained_model_dir = os.environ.get("PRETRAINED_GRU_MODEL_DIR", None)
                if not pretrained_model_dir:
                    pretrained_model_dir = os.environ.get("PRETRAINED_MODEL_DIR", None)
                if model_family == MODEL_FAMILY_TRANSFORMER and not pretrained_model_dir:
                    pretrained_model_dir = os.environ.get("PRETRAINED_TRANSFORMER_MODEL_DIR", None)
                if model_family == MODEL_FAMILY_LSTM and not pretrained_model_dir:
                    pretrained_model_dir = os.environ.get("PRETRAINED_LSTM_MODEL_DIR", None)
                if not pretrained_model_dir:
                    _project_root = os.path.dirname(base_app.APP_BASE)
                    _default_models = os.path.join(_project_root, "models")
                    if os.path.isdir(_default_models):
                        pretrained_model_dir = _default_models
                if pretrained_model_dir:
                    model_path = self._resolve_pretrained_model_path(
                        sensor_id=sensor_id, model_dir=pretrained_model_dir, model_family=model_family
                    )
                    if model_path:
                        resolved_model_path = model_path
                        env["PRETRAINED_MODEL_PATH"] = model_path
                        self.log(f"[{ctx.display_name}] Using {model_label} pre-trained model: {model_path}", level="Info")
                    else:
                        self.log(
                            f"[{ctx.display_name}] No {model_label} pre-trained model found for sensor_id={sensor_id} in {pretrained_model_dir}",
                            level="Warning",
                        )
            else:
                self.log(
                    f"[{ctx.display_name}] Using {model_label} fresh mode (no external pre-trained model).",
                    level="Info",
                )

        # Decide whether this run should train or inference-only.
        should_train, train_ref_time, train_reason = self._should_train_sensor_now(ctx, resolved_model_path)
        if str(env.get("PREDICTOR_UPDATE_TRAINING", "")).strip() in ("0", "false", "no", "off"):
            should_train = False
            train_reason = "predict-only (PREDICTOR_UPDATE_TRAINING=0)"
        elif str(env.get("PREDICTOR_SKIP_FINETUNE_ON_SESSION", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            should_train = False
            train_reason = "predict-only (PREDICTOR_SKIP_FINETUNE_ON_SESSION)"
        if (not should_train) and (not resolved_model_path):
            # Safety: cannot run inference-only with no model available.
            should_train = True
            train_reason = "forced training (no model available for inference-only)"

        ctx.current_run_training = should_train
        ctx.current_run_train_ref_time = train_ref_time
        env["PREDICTOR_UPDATE_TRAINING"] = "1" if should_train else "0"
        # UI mode: keep predictions aligned to actual input timestamps so full-series overlays
        # and anomaly checks have consistent time keys.
        env.setdefault("PREDICTOR_ON_INPUT_TIMESTAMPS", "1")
        interval_minutes = int(max(1, getattr(ctx, "retrain_interval_minutes", self._default_retrain_interval_minutes)))
        env["PREDICTOR_RETRAIN_INTERVAL_MINUTES"] = str(interval_minutes)
        env["PREDICTOR_N_FUTURE"] = str(self._compute_n_future_for_predictor(ctx))
        pcu = getattr(ctx, "predict_cover_until", None)
        if pcu is not None:
            env["PREDICTOR_COVER_UNTIL"] = pd.Timestamp(pcu).isoformat()
        else:
            env.pop("PREDICTOR_COVER_UNTIL", None)
        for _k in ("PREDICTOR_LEADING_TRAIN_MINUTES", "PREDICTOR_SKIP_INITIAL_MINUTES"):
            _v = os.environ.get(_k, "").strip()
            if _v:
                env[_k] = _v
            else:
                env.pop(_k, None)

        stdout_f = None
        stderr_f = None
        try:
            stdout_f = open(os.path.join(work_dir, "predict_stdout.log"), "w")
            stderr_f = open(os.path.join(work_dir, "predict_stderr.log"), "w")
            cmd = [python_exe, predictor_script, input_file]
            # Lower OS scheduling priority on mac/linux to keep UI smooth.
            if os.name != "nt":
                cmd = ["nice", "-n", "10"] + cmd
            mode_text = "train+predict" if should_train else "predict-only"
            self.log(
                f"[{ctx.display_name}] Starting predictor ({model_family}, {mode_text}, interval={interval_minutes} min; {train_reason}): {' '.join(cmd)}",
                level="Info",
            )
            ctx.prediction_process = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, cwd=work_dir, env=env)
        except Exception as e:
            for fh in (stdout_f, stderr_f):
                if fh is not None:
                    try:
                        fh.close()
                    except Exception:
                        pass
            ctx.prediction_process = None
            ctx.predict_app_started = False
            self._predict_active.discard(sensor_id)
            self.log(f"[{ctx.display_name}] Failed to start predictor subprocess: {e}", level="Error")
            self._drain_predict_queue()
            return

    def update_predictions_for_sensor(self, sensor_id: str):
        ctx = self.sensor_ctx.get(sensor_id)
        if ctx is None or not ctx.predictor_input_file:
            return

        out_file = os.path.join(os.path.dirname(ctx.predictor_input_file), "predict_out.csv")
        proc = ctx.prediction_process
        if proc is not None:
            rc = proc.poll()
            if rc is None:
                # New actuals can arrive while a run is still training; use existing predict_* immediately.
                if ctx.rt_x_t and ctx.predict_x_t:
                    self.detect_anomalies_for_sensor(sensor_id)
                return  # still running
            if rc != 0:
                # failed
                err_path = os.path.join(os.path.dirname(ctx.predictor_input_file), "predict_stderr.log")
                err_tail = ""
                try:
                    if os.path.isfile(err_path):
                        with open(err_path, "r", encoding="utf-8", errors="replace") as ef:
                            lines = ef.readlines()[-25:]
                        err_tail = "".join(lines).strip()
                except Exception:
                    pass
                if err_tail:
                    self.log(
                        f"[{ctx.display_name}] predictor_ai exited with code {rc}. Last stderr lines:\n{err_tail}",
                        level="Error",
                    )
                else:
                    self.log(
                        f"[{ctx.display_name}] predictor_ai exited with code {rc}. See {err_path}",
                        level="Error",
                    )
                ctx.prediction_process = None
                ctx.predict_app_started = False
                self._predict_active.discard(sensor_id)
                ctx.current_run_training = False
                ctx.current_run_train_ref_time = None
                self._ensure_prediction_covers_actual(sensor_id, force=False)
                return
            ctx.prediction_process = None
            ctx.predict_app_started = False
            self._predict_active.discard(sensor_id)
            ctx.last_pred_complete_ts = time.time()
            # Retrain cadence must use the latest **actual** time when this run finished, not the
            # run-start train ref. CSV fast playback loads the full green series in wall-clock time
            # while the first predict-only job was queued with an early train_ref (~historic end);
            # anchoring there makes (latest_actual - last_model_train) look like ~hours of "elapsed"
            # data and forces an immediate train+predict at EOF before predict_out is ever merged.
            anchor = self._latest_arrived_data_time(ctx)
            if anchor is not None:
                ctx.last_model_train_data_time = pd.to_datetime(anchor).to_pydatetime()
            ctx.current_run_training = False
            ctx.current_run_train_ref_time = None

        if not os.path.exists(out_file):
            self._ensure_prediction_covers_actual(sensor_id, force=False)
            if ctx.rt_x_t and ctx.predict_x_t:
                self.detect_anomalies_for_sensor(sensor_id)
            return

        try:
            pred = pd.read_csv(out_file)
            pred["x"] = pd.to_datetime(pred["x"])
            new_x = pred["x"].tolist()
            new_y = pred["y"].astype(float).tolist()
            if new_x:
                merged = {}
                for t, v in zip(ctx.predict_x_t, ctx.predict_y_t):
                    merged[self._prediction_timestamp_key(t)] = v
                for t, v in zip(new_x, new_y):
                    merged[self._prediction_timestamp_key(t)] = v
                merged_times = sorted(merged.keys())
                ctx.predict_x_t = [pd.Timestamp(t) for t in merged_times]
                ctx.predict_y_t = [merged[t] for t in merged_times]
            if ctx.predict_x_t:
                try:
                    pd.DataFrame({"x": ctx.predict_x_t, "y": ctx.predict_y_t}).to_csv(out_file, index=False)
                except Exception:
                    pass
            if ctx.rt_x_t and ctx.predict_x_t:
                self.detect_anomalies_for_sensor(sensor_id)
        except Exception:
            if ctx.rt_x_t and ctx.predict_x_t:
                self.detect_anomalies_for_sensor(sensor_id)

        # Keep extending prediction until latest actual is covered.
        self._ensure_prediction_covers_actual(sensor_id, force=False)

    def detect_anomalies_for_sensor(self, sensor_id: str):
        ctx = self.sensor_ctx.get(sensor_id)
        if ctx is None:
            return
        if not ctx.rt_x_t or not ctx.predict_x_t:
            return

        # Process only newly arrived actual points that have not yet been compared.
        # Reprocessing the full history on every poll can repeatedly feed the same
        # errors into EWMA statistics and artificially inflate thresholds.
        actual_times = ctx.rt_x_t
        actual_values = ctx.rt_y_mag_t
        if ctx.last_anomaly_checked_time is not None:
            cutoff = pd.to_datetime(ctx.last_anomaly_checked_time)
            new_pairs = [
                (t, v)
                for t, v in zip(ctx.rt_x_t, ctx.rt_y_mag_t)
                if pd.to_datetime(t) > cutoff
            ]
            if not new_pairs:
                return
            actual_times = [t for t, _ in new_pairs]
            actual_values = [v for _, v in new_pairs]

        differences_df = ctx.anomaly_detector.calculate_differences(
            actual_times=actual_times,
            actual_values=actual_values,
            predicted_times=ctx.predict_x_t,
            predicted_values=ctx.predict_y_t,
        )
        if differences_df is None or differences_df.empty:
            return

        # Update "safe for training" cutoff: all timestamps in differences_df have now been
        # compared with predictions and either classified as normal or anomalous.
        try:
            latest_checked = pd.to_datetime(differences_df["time"]).max()
        except Exception:
            latest_checked = None
        if latest_checked is not None:
            if ctx.last_anomaly_checked_time is None or latest_checked > ctx.last_anomaly_checked_time:
                ctx.last_anomaly_checked_time = latest_checked

        threshold = ctx.anomaly_detector.anomaly_threshold
        anomalies_df = differences_df[differences_df["is_anomaly"]].copy()
        new_times = pd.to_datetime(anomalies_df["time"]).tolist()
        new_vals = anomalies_df["actual"].astype(float).tolist()

        existing = set(pd.to_datetime(ctx.anomaly_times)) if ctx.anomaly_times else set()
        newly_added = []
        for t, v in zip(new_times, new_vals):
            if t not in existing:
                ctx.anomaly_times.append(t)
                ctx.anomaly_values.append(v)
                existing.add(t)
                newly_added.append((t, v))

        if newly_added:
            for t, v in newly_added:
                self.log(
                    f"[{ctx.display_name}] Anomaly detected | time={t} | magnitude={v:.1f} nT",
                    level="Info",
                )

        thr_log = f"{threshold:.2f}" if threshold is not None else "n/a (EWMA not ready)"
        self.log(
            f"[{ctx.display_name}] matched_pairs={len(differences_df)} total_anomalies={len(ctx.anomaly_times)} "
            f"threshold={thr_log} nT",
            level="Info",
        )


if __name__ == "__main__":
    # Use the temp app class (multi-sensor TimeSeries tabs)
    app = ApplicationTemp([])
    sys.exit(app.exec())


