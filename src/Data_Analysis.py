"""
Magnetic Sensor Data Visualizer - Cleaned and Simplified Version
Maintains all functionality while improving code organization and readability
"""

import copy
import io
import os
import re
from typing import List, Tuple, Optional, Dict, Union, Literal
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import pearsonr, spearmanr
from scipy.signal import spectrogram, savgol_filter, butter, filtfilt, detrend, medfilt, welch
from scipy import signal as scipy_signal
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================================
# DATA LOADING & UTILITIES
# ============================================================================

_SENSOR_ID_ALIASES = ("sensor_id", "sensor", "device_id", "sensor_name", "SensorID")


def _normalize_sensor_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a single ``sensor_id`` column; strip labels so N streams stay distinct."""
    out = df.copy()
    src = None
    for c in _SENSOR_ID_ALIASES:
        if c in out.columns:
            src = c
            break
    if src is None:
        return out
    if src != "sensor_id":
        out["sensor_id"] = out[src]
    out["sensor_id"] = out["sensor_id"].map(lambda x: str(x).strip() if pd.notna(x) else np.nan)
    out.loc[out["sensor_id"].isin(("", "nan", "None", "NaT")), "sensor_id"] = np.nan
    return out


@st.cache_data(show_spinner=False)
def read_single_csv(file_path_or_buffer) -> pd.DataFrame:
    """Load and validate a single CSV file"""
    df = pd.read_csv(file_path_or_buffer)
    required_mag = ["id", "b_x", "b_y", "b_z", "timestamp"]
    optional_rest = ["lat", "lon", "alt", "theta_x", "theta_y", "theta_z"]
    missing = [c for c in required_mag if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not any(a in df.columns for a in _SENSOR_ID_ALIASES):
        raise ValueError(
            f"Missing sensor column (need one of: {', '.join(_SENSOR_ID_ALIASES)})"
        )
    for c in optional_rest:
        if c not in df.columns:
            df[c] = np.nan
    df = _normalize_sensor_id_column(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["b_x", "b_y", "b_z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def read_multiple_csvs(files: List) -> pd.DataFrame:
    """Load multiple CSV files and combine"""
    frames = []
    for f in files:
        try:
            frames.append(read_single_csv(f))
        except Exception as e:
            st.warning(f"Failed to read {getattr(f, 'name', str(f))}: {e}")
    if not frames:
        return pd.DataFrame(
            columns=[
                "id",
                "b_x",
                "b_y",
                "b_z",
                "timestamp",
                "lat",
                "lon",
                "alt",
                "theta_x",
                "theta_y",
                "theta_z",
                "sensor_id",
            ]
        )
    df = pd.concat(frames, ignore_index=True)
    df = _normalize_sensor_id_column(df)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    return df


def compute_resultant(df: pd.DataFrame) -> pd.Series:
    """Calculate resultant magnetic field magnitude: √(b_x² + b_y² + b_z²)"""
    bx, by, bz = df["b_x"].astype(float), df["b_y"].astype(float), df["b_z"].astype(float)
    return np.sqrt(bx * bx + by * by + bz * bz)


def estimate_sampling_rate(timestamps: pd.Series) -> float:
    """Estimate sampling rate from timestamp differences"""
    ts = timestamps.dropna().astype("int64").values.astype(float) / 1e9
    if ts.size < 2:
        return 0.0
    dt = np.diff(ts)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 0.0
    median_dt = float(np.median(dt))
    return 1.0 / median_dt if median_dt > 0 else 0.0


SensorFilter = Union[Literal["All"], List[str]]


def filter_df(
    df: pd.DataFrame,
    sensor_selection: SensorFilter,
    time_range: Tuple[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    """Filter dataframe by sensor(s) and time range. ``sensor_selection`` is ``\"All\"`` or a non-empty list of ids."""
    if df.empty:
        return df
    if sensor_selection == "All" or sensor_selection is None:
        filtered = df
    elif isinstance(sensor_selection, list):
        if len(sensor_selection) == 0:
            filtered = df
        else:
            filtered = df[df["sensor_id"].isin(sensor_selection)]
    else:
        filtered = df[df["sensor_id"] == sensor_selection]
    t0, t1 = time_range
    if "timestamp" in filtered.columns and pd.notna(t0) and pd.notna(t1):
        mask = (filtered["timestamp"] >= t0) & (filtered["timestamp"] <= t1)
        filtered = filtered.loc[mask]
    return filtered


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def preprocess_signal(signal: np.ndarray, enable_enhancement: bool = True, 
                     enhancement_factor: float = 50.0, enable_smoothing: bool = True) -> np.ndarray:
    """Preprocess signal: remove DC, apply smoothing, and optional enhancement"""
    signal_processed = signal - np.mean(signal)
    
    if enable_smoothing and len(signal_processed) > 21:
        try:
            window_length = min(21, len(signal_processed) // 10)
            if window_length % 2 == 0:
                window_length += 1
            if window_length >= 5:
                signal_processed = savgol_filter(signal_processed, window_length, 3)
        except Exception:
            pass
    
    if enable_enhancement:
        signal_std = np.std(signal_processed)
        if signal_std > 0:
            amplification = min(1000, max(10, 0.01 / signal_std))
            if np.max(np.abs(signal_processed)) * amplification < 1e10:
                signal_processed = signal_processed * amplification
            else:
                signal_processed = signal_processed * min(amplification, 100)
    
    return signal_processed


def compute_spectrogram(signal: np.ndarray, sampling_hz: float, window_seconds: float,
                        overlap_ratio: float, mode: str = "psd", window: str = "hann") -> Optional[Tuple]:
    """Compute spectrogram using simple, working approach from spectrogram_app.py"""
    if sampling_hz <= 0 or len(signal) < 32:
        return None
    
    # Remove DC component
    signal_processed = signal - np.mean(signal)
    
    # Apply Savitzky-Golay smoothing if possible
    if len(signal_processed) > 21:
        try:
            window_length = min(21, len(signal_processed) // 10)
            if window_length % 2 == 0:
                window_length += 1
            if window_length >= 5:
                signal_processed = savgol_filter(signal_processed, window_length, 3)
        except Exception:
            pass
    
    # Calculate window size
    nperseg = max(8, int(window_seconds * sampling_hz))
    nperseg = min(nperseg, len(signal_processed) // 2)
    noverlap = int(overlap_ratio * nperseg)
    
    try:
        if mode == "psd":
            f, t, sxx = spectrogram(
                signal_processed,
                fs=sampling_hz,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="density",
                mode="psd",
                detrend='linear',
                return_onesided=True,
                window="hann",
            )
        else:
            f, t, sxx = spectrogram(
                signal_processed,
                fs=sampling_hz,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="spectrum",
                mode="magnitude",
                detrend='linear',
                return_onesided=True,
                window="hann",
            )
        
        # Apply log scaling for better visualization
        if mode == "psd":
            sxx = 10 * np.log10(sxx + 1e-15)
        else:
            sxx = np.log10(sxx + 1e-15)
        
        return f, t, sxx
    except Exception as e:
        st.error(f"Spectrogram computation failed: {e}")
        return None


def detect_anomalies(sxx: np.ndarray, f: np.ndarray, t: np.ndarray, threshold: float = 2.0) -> Tuple[List, List]:
    """Detect anomalies in spectrogram using z-score analysis"""
    if sxx.size == 0:
        return [], []
    sxx_mean = np.mean(sxx, axis=1, keepdims=True)
    sxx_std = np.std(sxx, axis=1, keepdims=True) + 1e-12
    z_scores = (sxx - sxx_mean) / sxx_std
    anomaly_mask = np.abs(z_scores) > threshold
    anomaly_freqs = [f[i] for i in range(sxx.shape[0]) for j in range(sxx.shape[1]) if anomaly_mask[i, j]]
    anomaly_times = [t[j] for i in range(sxx.shape[0]) for j in range(sxx.shape[1]) if anomaly_mask[i, j]]
    return anomaly_freqs, anomaly_times


def detect_time_series_anomalies(signal_values: np.ndarray, timestamps: pd.Series, 
                                 threshold: float = 2.0) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Enhanced anomaly detection using multiple methods with conservative approach:
    1. Global Z-score (primary method)
    2. IQR method (robust outlier detection)
    3. Rolling window z-score (local anomalies, stricter threshold)
    4. Change point detection (only very significant shifts)
    
    Uses weighted voting: requires strong evidence from multiple methods
    """
    if len(signal_values) < 10:
        return []
    
    # Convert to pandas Series for easier manipulation
    signal_series = pd.Series(signal_values)
    
    # Use weighted voting instead of simple OR
    anomaly_scores = np.zeros(len(signal_values))
    
    # Method 1: Global Z-score (primary method, weight=2)
    signal_mean = np.mean(signal_values)
    signal_std = np.std(signal_values) + 1e-12
    z_scores = np.abs((signal_values - signal_mean) / signal_std)
    anomaly_scores += (z_scores > threshold) * 2.0  # Primary method gets higher weight
    
    # Method 2: IQR (Interquartile Range) - more robust, weight=1.5
    Q1 = np.percentile(signal_values, 25)
    Q3 = np.percentile(signal_values, 75)
    IQR = Q3 - Q1
    if IQR > 0:
        # Use stricter IQR bounds (1.5x IQR is standard, we use threshold-adjusted)
        iqr_multiplier = max(1.5, threshold * 0.75)  # Stricter than threshold
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        iqr_mask = (signal_values < lower_bound) | (signal_values > upper_bound)
        anomaly_scores += iqr_mask * 1.5
    
    # Method 3: Rolling window z-score (local anomalies, stricter threshold, weight=1.0)
    window_size = min(100, len(signal_values) // 10)  # Adaptive window size
    if window_size >= 10:
        rolling_mean = signal_series.rolling(window=window_size, center=True).mean()
        rolling_std = signal_series.rolling(window=window_size, center=True).std() + 1e-12
        local_z_scores = np.abs((signal_values - rolling_mean.values) / rolling_std.values)
        # Use stricter threshold for local anomalies (threshold * 1.2)
        anomaly_scores += (local_z_scores > threshold * 1.2) * 1.0
    
    # Method 4: Change point detection (only very significant shifts, weight=1.0)
    if len(signal_values) > 20:
        # Calculate first derivative (rate of change)
        diff = np.abs(np.diff(signal_values))
        diff_mean = np.mean(diff)
        diff_std = np.std(diff) + 1e-12
        # Only mark very significant changes (threshold * 1.5)
        change_threshold = diff_mean + threshold * 1.5 * diff_std
        change_points = diff > change_threshold
        # Mark both the point before and after the change
        change_mask = np.zeros(len(signal_values), dtype=bool)
        change_mask[:-1] |= change_points
        change_mask[1:] |= change_points
        anomaly_scores += change_mask * 1.0
    
    # Require strong evidence: at least 2.0 total score (equivalent to primary method alone, 
    # or combination of other methods)
    # This prevents false positives from single weak detections
    anomaly_mask = anomaly_scores >= 2.0
    
    # Filter out very small regions (less than 3 consecutive points)
    if not np.any(anomaly_mask):
        return []
    
    # Group consecutive anomalies into regions
    anomaly_regions = []
    in_anomaly = False
    start_idx = None
    
    for i, is_anomaly in enumerate(anomaly_mask):
        if is_anomaly and not in_anomaly:
            start_idx = i
            in_anomaly = True
        elif not is_anomaly and in_anomaly:
            if start_idx is not None and i < len(timestamps):
                # Only add region if it has at least 3 points
                if (i - start_idx) >= 3:
                    anomaly_regions.append((timestamps.iloc[start_idx], timestamps.iloc[i-1]))
            in_anomaly = False
            start_idx = None
    
    # Handle case where anomaly extends to end
    if in_anomaly and start_idx is not None:
        if (len(timestamps) - start_idx) >= 3:
            anomaly_regions.append((timestamps.iloc[start_idx], timestamps.iloc[-1]))
    
    return anomaly_regions


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def compute_correlation_analysis(df: pd.DataFrame, sensor1: str, sensor2: str,
                                 field1: str, field2: str) -> Dict:
    """Compute correlation between two sensors"""
    sensor1_data = df[df["sensor_id"] == sensor1].sort_values("timestamp")
    sensor2_data = df[df["sensor_id"] == sensor2].sort_values("timestamp")
    
    if sensor1_data.empty or sensor2_data.empty:
        return {"error": "One or both sensors have no data"}
    
    # Get field values
    get_field_values = lambda data, field: (compute_resultant(data).values if field == "resultant"
                                           else pd.to_numeric(data[field], errors="coerce").dropna().values)
    
    sensor1_values = get_field_values(sensor1_data, field1)
    sensor2_values = get_field_values(sensor2_data, field2)
    
    if len(sensor1_values) == 0 or len(sensor2_values) == 0:
        return {"error": f"No valid data for sensors (fields: {field1}, {field2})"}
    
    # Align data by timestamp
    sensor1_series = pd.Series(sensor1_values, index=sensor1_data["timestamp"])
    sensor2_series = pd.Series(sensor2_values, index=sensor2_data["timestamp"])
    
    if not sensor1_series.index.is_unique:
        sensor1_series = sensor1_series.groupby(level=0).mean()
    if not sensor2_series.index.is_unique:
        sensor2_series = sensor2_series.groupby(level=0).mean()
    
    common_start = max(sensor1_series.index.min(), sensor2_series.index.min())
    common_end = min(sensor1_series.index.max(), sensor2_series.index.max())
    
    if common_start >= common_end:
        return {"error": "No overlapping time range between sensors"}
    
    # Resample to common grid
    time_index = pd.date_range(start=common_start, end=common_end, freq=pd.Timedelta(seconds=1))
    sensor1_aligned = sensor1_series.reindex(time_index).interpolate(method="time").dropna()
    sensor2_aligned = sensor2_series.reindex(time_index).interpolate(method="time").dropna()
    common_idx = sensor1_aligned.index.intersection(sensor2_aligned.index)
    
    if len(common_idx) < 10:
        return {"error": "Insufficient overlapping data points"}
    
    x_values = sensor1_aligned.loc[common_idx].values
    y_values = sensor2_aligned.loc[common_idx].values
    
    try:
        pearson_corr, pearson_p = pearsonr(x_values, y_values)
        spearman_corr, spearman_p = spearmanr(x_values, y_values)
    except Exception as e:
        return {"error": f"Correlation computation failed: {str(e)}"}
    
    return {
        "sensor1": sensor1, "sensor2": sensor2, "field1": field1, "field2": field2,
        "n_points": len(common_idx), "time_range": (common_start, common_end),
        "pearson_correlation": pearson_corr, "pearson_p_value": pearson_p,
        "spearman_correlation": spearman_corr, "spearman_p_value": spearman_p,
        "r_squared": pearson_corr ** 2,
        "sensor1_mean": np.mean(x_values), "sensor2_mean": np.mean(y_values),
        "sensor1_std": np.std(x_values), "sensor2_std": np.std(y_values),
        "x_values": x_values, "y_values": y_values, "timestamps": common_idx
    }


def plot_correlation_analysis(corr_data: Dict):
    """Plot correlation analysis with simplified visualization options"""
    if "error" in corr_data:
        st.error(f"Correlation Analysis Error: {corr_data['error']}")
        return
    
    st.subheader(f"🔗 Correlation: {corr_data['sensor1']} [{corr_data['field1']}] vs "
                 f"{corr_data['sensor2']} [{corr_data['field2']}]")
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pearson r", f"{corr_data['pearson_correlation']:.4f}")
    with col2:
        st.metric("Spearman r", f"{corr_data['spearman_correlation']:.4f}")
    with col3:
        st.metric("R²", f"{corr_data['r_squared']:.4f}")
    with col4:
        st.metric("P-value", f"{corr_data['pearson_p_value']:.2e}")
    
    # Interpretation
    r = corr_data['pearson_correlation']
    strength = "Very Strong" if abs(r) > 0.8 else "Strong" if abs(r) > 0.6 else "Moderate" if abs(r) > 0.4 else "Weak"
    direction = "positive" if r > 0 else "negative"
    st.info(f"📊 **{strength} {direction} correlation** (r = {r:.3f})")
    
    # Visualization method
    viz_method = st.selectbox(
        "Visualization Method:",
        ["Comprehensive Dashboard", "Scatter Plot", "Time Series Overlay", "Density Heatmap"],
        key="corr_viz_method"
    )
    
    x_vals, y_vals = corr_data['x_values'], corr_data['y_values']
    timestamps = corr_data['timestamps']
    
    if viz_method == "Comprehensive Dashboard":
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Scatter Plot", "Time Series", "Residuals", "Correlation Matrix"],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Scatter with regression
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name='Data',
                                 marker=dict(color='blue', opacity=0.6, size=6)), row=1, col=1)
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        fig.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines', name='Regression',
                                line=dict(color='red', width=3)), row=1, col=1)
        
        # Time series
        fig.add_trace(go.Scatter(x=timestamps, y=x_vals, mode='lines', name=corr_data['sensor1'],
                                line=dict(color='blue')), row=1, col=2)
        fig.add_trace(go.Scatter(x=timestamps, y=y_vals, mode='lines', name=corr_data['sensor2'],
                                line=dict(color='red')), row=1, col=2)
        
        # Residuals
        y_pred = p(x_vals)
        residuals = y_vals - y_pred
        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals',
                                marker=dict(color='green', opacity=0.6)), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        # Correlation matrix
        corr_matrix = np.array([[1.0, r], [r, 1.0]])
        fig.add_trace(go.Heatmap(z=corr_matrix, x=[corr_data['sensor1'], corr_data['sensor2']],
                                 y=[corr_data['sensor1'], corr_data['sensor2']], colorscale='RdBu',
                                 zmid=0, text=np.round(corr_matrix, 3), texttemplate="%{text}",
                                 showscale=True), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True, title_text=f"Correlation Analysis: "
                         f"{corr_data['sensor1']} vs {corr_data['sensor2']}")
        fig.update_xaxes(title_text=f"{corr_data['sensor1']} ({corr_data['field1']})", row=1, col=1)
        fig.update_yaxes(title_text=f"{corr_data['sensor2']} ({corr_data['field2']})", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_xaxes(title_text="Predicted", row=2, col=1)
        fig.update_yaxes(title_text="Residuals", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_method == "Scatter Plot":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name='Data',
                                marker=dict(color='blue', opacity=0.6, size=8)))
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        fig.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines', name=f'Regression (r={r:.3f})',
                                line=dict(color='red', width=3)))
        fig.update_layout(
            title=f"Scatter: {corr_data['sensor1']} vs {corr_data['sensor2']}",
            xaxis_title=f"{corr_data['sensor1']} ({corr_data['field1']}) [nT]",
            yaxis_title=f"{corr_data['sensor2']} ({corr_data['field2']}) [nT]",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_method == "Time Series Overlay":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timestamps, y=x_vals, mode='lines', name=corr_data['sensor1'],
                                line=dict(color='blue', width=2), yaxis='y'))
        fig.add_trace(go.Scatter(x=timestamps, y=y_vals, mode='lines', name=corr_data['sensor2'],
                                line=dict(color='red', width=2), yaxis='y2'))
        fig.update_layout(
            title=f"Time Series Comparison",
            xaxis_title="Time",
            yaxis=dict(title=f"{corr_data['sensor1']} ({corr_data['field1']}) [nT]", side='left'),
            yaxis2=dict(title=f"{corr_data['sensor2']} ({corr_data['field2']}) [nT]", 
                       side='right', overlaying='y'),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_method == "Density Heatmap":
        fig = go.Figure()
        fig.add_trace(go.Histogram2d(x=x_vals, y=y_vals, colorscale='Blues', nbinsx=30, nbinsy=30,
                                     showscale=True, colorbar=dict(title="Density")))
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        fig.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines', name='Regression',
                                line=dict(color='red', width=3)))
        fig.update_layout(
            title=f"Density Heatmap: {corr_data['sensor1']} vs {corr_data['sensor2']}",
            xaxis_title=f"{corr_data['sensor1']} ({corr_data['field1']}) [nT]",
            yaxis_title=f"{corr_data['sensor2']} ({corr_data['field2']}) [nT]",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)


def correlation_study_interface(df: pd.DataFrame):
    """Correlation study interface"""
    st.subheader("🔗 Correlation Study")
    
    if df.empty:
        st.warning("No data available for correlation analysis.")
        return
    
    available_sensors = sorted(df["sensor_id"].dropna().unique().tolist())
    if len(available_sensors) < 2:
        st.warning("At least 2 sensors are required for correlation analysis.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        sensor1 = st.selectbox("Select First Sensor", options=available_sensors, key="corr_sensor1")
    with col2:
        sensor2 = st.selectbox("Select Second Sensor", 
                              options=[s for s in available_sensors if s != sensor1], key="corr_sensor2")
    
    field_options = ["b_x", "b_y", "b_z", "resultant"]
    field_labels = {"b_x": "Bx", "b_y": "By", "b_z": "Bz", "resultant": "Resultant"}
    
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        field1 = st.selectbox(f"Field for {sensor1}", options=field_options,
                             format_func=lambda x: field_labels[x], key="corr_field1")
    with fcol2:
        field2 = st.selectbox(f"Field for {sensor2}", options=field_options,
                             format_func=lambda x: field_labels[x], key="corr_field2")
    
    # Time range
    if df["timestamp"].notna().any():
        tmin, tmax = df["timestamp"].min(), df["timestamp"].max()
    else:
        tmin, tmax = pd.Timestamp.now() - pd.Timedelta(hours=1), pd.Timestamp.now()
    
    time_range = st.slider("Time Range", min_value=tmin.to_pydatetime(), max_value=tmax.to_pydatetime(),
                           value=(tmin.to_pydatetime(), tmax.to_pydatetime()),
                           step=pd.Timedelta(seconds=1), format="YYYY-MM-DD HH:mm:ss", key="corr_time_range")
    
    df_filtered = filter_df(df, "All", (pd.Timestamp(time_range[0]), pd.Timestamp(time_range[1])))
    
    if st.button("🔍 Run Correlation Analysis", type="primary"):
        with st.spinner("Computing correlation..."):
            corr_data = compute_correlation_analysis(df_filtered, sensor1, sensor2, field1, field2)
            st.session_state.corr_data = corr_data
    
    if 'corr_data' in st.session_state:
        plot_correlation_analysis(st.session_state.corr_data)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

# Distinct line colors for many sensors (overlay or per-facet trace).
_TS_LINE_COLORS: Tuple[str, ...] = (
    "#DC2626",
    "#2563EB",
    "#16A34A",
    "#CA8A04",
    "#9333EA",
    "#DB2777",
    "#0891B2",
    "#EA580C",
    "#4F46E5",
    "#059669",
)


def _compact_sensor_title(sensor_id: object, max_len: int = 36) -> str:
    """Short label for subplot titles (avoids overlap when many sensors)."""
    s = str(sensor_id).strip()
    if not s:
        return "?"
    parts = s.split("_")
    if len(parts) >= 2:
        tail = "_".join(parts[-2:])
        if len(tail) <= max_len:
            return tail
    return (s[: max_len - 1] + "…") if len(s) > max_len else s


def _sensor_legend_subscript_label(sensor_id: object, order_index: int) -> str:
    """Legend text S with trailing digits as Unicode subscripts (e.g. OBS2_1 -> S₁)."""
    s = str(sensor_id).strip()
    i = len(s) - 1
    digits = ""
    while i >= 0 and s[i].isdigit():
        digits = s[i] + digits
        i -= 1
    if digits:
        trans = str.maketrans("0123456789", "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")
        return "S" + digits.translate(trans)
    subs = "\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
    if 1 <= order_index <= len(subs):
        return "S" + subs[order_index - 1]
    return f"S ({s})"


def _scale_plotly_font_sizes(obj, factor: float) -> None:
    """Multiply every Plotly `font.size` (and common *_font_size keys) by factor for static export."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k == "font" and isinstance(v, dict) and v.get("size") is not None:
                try:
                    v["size"] = max(1, int(round(float(v["size"]) * factor)))
                except (TypeError, ValueError):
                    pass
            elif k in ("annotation_font_size",) and isinstance(v, (int, float)):
                try:
                    obj[k] = max(1, int(round(float(v) * factor)))
                except (TypeError, ValueError):
                    pass
            else:
                _scale_plotly_font_sizes(v, factor)
    elif isinstance(obj, list):
        for it in obj:
            _scale_plotly_font_sizes(it, factor)


def _scale_trace_line_widths(data_list, factor: float) -> None:
    cap = min(float(factor), 2.8)
    if not isinstance(data_list, list):
        return
    for tr in data_list:
        if not isinstance(tr, dict):
            continue
        ln = tr.get("line")
        if isinstance(ln, dict) and ln.get("width") is not None:
            try:
                ln["width"] = min(10.0, max(1.0, float(ln["width"]) * cap))
            except (TypeError, ValueError):
                pass


def _scale_layout_margins(layout_dict: dict, factor: float) -> None:
    cap = min(float(factor), 2.6)
    mg = layout_dict.get("margin")
    if not isinstance(mg, dict):
        return
    for k in ("b", "l", "t", "r", "pad"):
        if k not in mg or mg[k] is None:
            continue
        try:
            mg[k] = int(round(float(mg[k]) * cap))
        except (TypeError, ValueError):
            pass


def _figure_for_static_export(fig: go.Figure, render_width_px: int, ref_display_width_px: float = 1050.0) -> go.Figure:
    """
    Clone figure and scale typography so text occupies a similar fraction of the canvas
    as on a ~ref_display_width_px Streamlit view (fixes tiny text in wide PNG/PDF exports).
    """
    factor = max(1.0, float(render_width_px) / float(ref_display_width_px))
    d = copy.deepcopy(fig.to_dict())
    layout = d.get("layout")
    if isinstance(layout, dict):
        _scale_plotly_font_sizes(layout, factor)
        _scale_layout_margins(layout, factor)
    _scale_plotly_font_sizes(d.get("data", []), factor)
    _scale_trace_line_widths(d.get("data", []), factor)
    return go.Figure(d)


def _export_plotly_figure_bytes(
    fig: go.Figure,
    file_format: str,
    width: int,
    height: int,
    scale: float = 1.0,
    ref_display_width: float = 1050.0,
) -> Optional[bytes]:
    """Export figure to bytes (PNG/PDF/SVG). Requires Kaleido: pip install kaleido"""
    try:
        fig_out = _figure_for_static_export(fig, render_width_px=width, ref_display_width_px=ref_display_width)
        buf = io.BytesIO()
        pio.write_image(fig_out, buf, format=file_format, width=width, height=height, scale=scale)
        return buf.getvalue()
    except Exception:
        return None


def plot_time_series(df: pd.DataFrame, series: str, enable_anomaly_detection: bool = True, 
                     anomaly_threshold: float = 2.0):
    """Plot time series with anomaly highlighting. Multiple sensors use one subplot each (shared time axis)."""
    _c_ann = "#DC2626"

    if df.empty:
        st.info("No data to plot.")
        return
    
    plot_df = df.copy()
    if series == "resultant":
        plot_df["resultant"] = compute_resultant(plot_df)

    sensors_sorted = sorted(plot_df["sensor_id"].dropna().unique(), key=lambda x: str(x))
    n_s = len(sensors_sorted)
    if n_s == 0:
        st.info("No rows with a valid sensor_id — check the sensor column in your CSV.")
        return

    use_facets = n_s >= 2
    y_title = "Resultant (nT)" if series == "resultant" else f"{series} (nT)"
    y_title_short = "nT"
    # Tighter typography when many stacked panels (reduces title/tick overlap).
    if use_facets:
        _fs_mul = 1.15 if n_s >= 6 else (1.35 if n_s >= 4 else 1.6)
        _axis_title_fs = max(11, int(13 * _fs_mul))
        _tick_fs = max(9, int(10 * _fs_mul))
        _legend_fs = max(9, int(10 * _fs_mul))
        _ann_fs = max(8, int(8 * _fs_mul))
    else:
        _fs_mul = 2.0
        _axis_title_fs = int(14 * _fs_mul)
        _tick_fs = int(12 * _fs_mul)
        _legend_fs = int(12 * _fs_mul)
        _ann_fs = int(9 * _fs_mul)
    _b = dict(family="Arial", weight=700)

    if use_facets:
        st.caption(
            f"Showing **{n_s} sensors** on **{n_s} stacked panels** (shared time axis). "
            f"Y-axis quantity: **{y_title}**. Hover shows full sensor id. "
            "Each panel has its own vertical scale."
        )
        titles = [_compact_sensor_title(s, max_len=34) for s in sensors_sorted]
        # Extra vertical gap between domains so subplot titles do not collide with traces below.
        vspace = max(0.055, min(0.11, 0.62 / max(n_s, 1)))
        fig = make_subplots(
            rows=n_s,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=vspace,
            subplot_titles=titles,
        )
    else:
        fig = go.Figure()

    for order_idx, sensor_id in enumerate(sensors_sorted, start=1):
        sensor_data = plot_df[plot_df["sensor_id"] == sensor_id].sort_values("timestamp")

        if series == "resultant":
            y_values = compute_resultant(sensor_data).values
        else:
            y_values = pd.to_numeric(sensor_data[series], errors="coerce").values

        timestamps = sensor_data["timestamp"].values

        legend_name = _sensor_legend_subscript_label(sensor_id, order_idx)
        line_color = _TS_LINE_COLORS[(order_idx - 1) % len(_TS_LINE_COLORS)]
        trace = go.Scatter(
            x=timestamps,
            y=y_values,
            mode="lines",
            name=f"{legend_name} ({sensor_id})",
            line=dict(width=2.2 if use_facets else 2.5, color=line_color),
            showlegend=not use_facets,
            hoverlabel=dict(
                bgcolor="white",
                bordercolor=line_color,
                font={**_b, "size": _tick_fs, "color": "black"},
            ),
            hovertemplate=(
                f"<b>{str(sensor_id)}</b><br>{legend_name}<br>"
                f"Time: %{{x|%Y-%m-%d %H:%M:%S}}<br>"
                f"{series}: %{{y:.2f}} nT<extra></extra>"
            ),
        )
        if use_facets:
            fig.add_trace(trace, row=order_idx, col=1)
        else:
            fig.add_trace(trace)

        if enable_anomaly_detection and len(y_values) > 10:
            anomaly_regions = detect_time_series_anomalies(
                y_values,
                pd.Series(timestamps),
                threshold=anomaly_threshold,
            )
            for start_time, end_time in anomaly_regions:
                vrect_kw = dict(
                    x0=start_time,
                    x1=end_time,
                    fillcolor="lightcoral",
                    opacity=0.35,
                    layer="below",
                    line_width=1,
                    line_color="crimson",
                    line_dash="solid",
                )
                if not use_facets:
                    vrect_kw.update(
                        annotation_text="Anomaly",
                        annotation_position="top left",
                        annotation_font_size=_ann_fs,
                        annotation_font_color=_c_ann,
                        annotation_font_family="Arial Black",
                    )
                if use_facets:
                    fig.add_vrect(**vrect_kw, row=order_idx, col=1)
                else:
                    fig.add_vrect(**vrect_kw)

    if use_facets:
        row_h = 240 if n_s >= 6 else (220 if n_s >= 4 else 200)
        plot_h = int(min(3200, max(560, 80 + n_s * row_h)))
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            title=dict(
                text=f"{series.replace('_', ' ')} — {n_s} sensors",
                font={**_b, "size": int(_axis_title_fs * 1.05), "color": "black"},
                x=0.5,
                xanchor="center",
                yanchor="top",
                y=0.995,
                pad=dict(t=4, b=8),
            ),
            font={**_b, "size": _tick_fs},
            hoverlabel=dict(bgcolor="white", font={**_b, "size": _tick_fs, "color": "black"}),
            hovermode="x unified",
            height=plot_h,
            showlegend=False,
            margin=dict(b=100, l=72, t=56, r=28),
        )
        # Subplot title annotations: smaller font so titles do not collide with neighboring panels.
        ann_font = max(9, int(_tick_fs * 0.92))
        fig.for_each_annotation(
            lambda a: a.update(font=dict(family="Arial", size=ann_font, color="#111827"))
        )
        x_tick_fmt = "%m-%d %H:%M" if n_s >= 4 else "%Y-%m-%d %H:%M:%S"
        x_nticks = 7 if n_s >= 6 else (9 if n_s >= 4 else 12)
        for i in range(1, n_s):
            fig.update_xaxes(
                showticklabels=False,
                showgrid=True,
                gridcolor="rgba(0,0,0,0.06)",
                zeroline=False,
                row=i,
                col=1,
            )
        fig.update_xaxes(
            title=dict(text="Time", font={**_b, "size": _axis_title_fs, "color": "black"}),
            tickfont=dict(family="Arial", size=_tick_fs, color="black"),
            tickangle=-35,
            tickformat=x_tick_fmt,
            nticks=x_nticks,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            row=n_s,
            col=1,
        )
        for i in range(1, n_s + 1):
            fig.update_yaxes(
                title=dict(text=y_title_short, font={**_b, "size": max(10, int(_axis_title_fs * 0.75)), "color": "black"}),
                tickfont=dict(family="Arial", size=_tick_fs, color="black"),
                nticks=6,
                showgrid=True,
                gridcolor="rgba(0,0,0,0.08)",
                zeroline=False,
                row=i,
                col=1,
            )
    else:
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={**_b, "size": _tick_fs},
            xaxis=dict(
                title=dict(
                    text="Date and time",
                    font={**_b, "size": _axis_title_fs, "color": "black"},
                ),
                tickfont={**_b, "size": _tick_fs, "color": "black"},
                tickangle=-45,
                tickformat="%Y-%m-%d %H:%M:%S",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.08)",
                zeroline=False,
            ),
            yaxis=dict(
                title=dict(
                    text=y_title,
                    font={**_b, "size": _axis_title_fs, "color": "black"},
                ),
                tickfont={**_b, "size": _tick_fs, "color": "black"},
                showgrid=True,
                gridcolor="rgba(0,0,0,0.08)",
                zeroline=False,
            ),
            legend=dict(font={**_b, "size": _legend_fs, "color": "black"}),
            hoverlabel=dict(bgcolor="white", font={**_b, "size": _tick_fs, "color": "black"}),
            hovermode="x unified",
            height=640,
            margin=dict(b=180, l=110, t=48, r=44),
        )
    
    if enable_anomaly_detection:
        st.caption(f"🔍 Enhanced anomaly detection enabled (multi-method: z-score, IQR, rolling window, change points, percentiles | threshold: {anomaly_threshold:.1f})")

    # LaTeX-friendly export: screen snapshots are low-DPI; use downloads or Plotly's camera icon.
    _safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(series))[:48] or "series"
    # Match ~on-screen layout width so text is not tiny; use scale for pixel count (LaTeX PNG).
    _plotly_dl_config = {
        "displayModeBar": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": f"time_series_{_safe}",
            "width": 1100,
            "height": 660,
            "scale": 3,
        },
    }
    st.plotly_chart(fig, use_container_width=True, config=_plotly_dl_config)

    # Vector / high-res raster for \includegraphics / \includesvg in LaTeX (needs kaleido).
    exp_w, exp_h, exp_scale = 3200, 1920, 2.0
    c1, c2, c3 = st.columns(3)
    png_bytes = _export_plotly_figure_bytes(fig, "png", exp_w, exp_h, scale=exp_scale)
    pdf_bytes = _export_plotly_figure_bytes(fig, "pdf", 1600, 960, scale=1.0)
    svg_bytes = _export_plotly_figure_bytes(fig, "svg", 1600, 960, scale=1.0)
    with c1:
        if png_bytes:
            st.download_button(
                "LaTeX: high-res PNG (~6400 px wide)",
                data=png_bytes,
                file_name=f"time_series_{_safe}.png",
                mime="image/png",
                key=f"ts_exp_png_{_safe}",
                help="High-res raster; fonts scaled to match on-screen proportions. \\includegraphics[width=\\linewidth]{...}",
            )
    with c2:
        if pdf_bytes:
            st.download_button(
                "LaTeX: PDF (vector)",
                data=pdf_bytes,
                file_name=f"time_series_{_safe}.pdf",
                mime="application/pdf",
                key=f"ts_exp_pdf_{_safe}",
                help="Vector PDF; sharp at any size in LaTeX.",
            )
    with c3:
        if svg_bytes:
            st.download_button(
                "LaTeX: SVG (vector)",
                data=svg_bytes,
                file_name=f"time_series_{_safe}.svg",
                mime="image/svg+xml",
                key=f"ts_exp_svg_{_safe}",
                help="Use with svg package, e.g. \\includesvg{width=\\linewidth}{...}",
            )
    if not (png_bytes or pdf_bytes or svg_bytes):
        st.info(
            "For print-quality figures in LaTeX, install **kaleido** once: `pip install kaleido`, "
            "then use the download buttons above (or the plot **camera** icon — already set to a higher resolution)."
        )


def plot_spectrogram(df: pd.DataFrame, series: str, window_seconds: float, overlap_ratio: float):
    """Plot spectrogram using exact logic from spectrogram_app.py"""
    if df.empty:
        st.info("No data for spectrogram.")
        return
    
    work = df.copy()
    if series == "resultant":
        work["resultant"] = compute_resultant(work)
    y_col = series
    
    # Process each sensor
    for sid in work["sensor_id"].dropna().unique():
        sub = work[work["sensor_id"] == sid].sort_values("timestamp")
        if sub.empty:
            continue
        
        # Get signal values directly (like spectrogram_app.py)
        if series == "resultant":
            signal_values = compute_resultant(sub).values
        else:
            signal_values = pd.to_numeric(sub[series], errors="coerce").dropna().values
        
        if len(signal_values) < 10:
            st.warning(f"Insufficient data points ({len(signal_values)}) for spectrogram: {sid}")
            continue
        
        # Estimate sampling rate
        fs = estimate_sampling_rate(sub["timestamp"])
        if fs <= 0:
            st.warning(f"Cannot estimate sampling rate for {sid}")
            continue
        
        # Spectrogram settings
        with st.sidebar.expander("🔧 Spectrogram Settings", expanded=False):
            min_freq = st.number_input("Min frequency (Hz)", min_value=0.0, 
                                      value=st.session_state.get('min_freq', 0.0), step=0.001, key=f"min_freq_{sid}")
            max_freq = st.number_input("Max frequency (Hz, 0=auto)", min_value=0.0,
                                       value=st.session_state.get('max_freq', 0.0), step=0.001, key=f"max_freq_{sid}")
            enable_anomaly = st.session_state.get('enable_anomaly_spec', True)
            anomaly_threshold = st.session_state.get('anomaly_threshold', 2.0)
            colormap = st.selectbox("Colormap", ["Viridis", "Plasma", "Inferno", "Turbo", "Cividis"], 0, key=f"colormap_{sid}")
            log_freq = st.checkbox("Log Frequency Axis", value=True, key=f"log_freq_{sid}")
            spec_mode = st.selectbox("Mode", ["psd", "magnitude"], 0, key=f"mode_{sid}")
        
        # Compute spectrogram using the working function
        with st.spinner(f"Computing spectrogram for {sid}..."):
            result = compute_spectrogram(signal_values, fs, window_seconds, overlap_ratio, mode=spec_mode)
        
        if result is None:
            st.error(f"Failed to compute spectrogram for {sid}")
            continue
        
        f, t, sxx = result
        
        # Apply frequency range filtering
        nyquist_freq = fs / 2.0
        freq_mask = np.ones_like(f, dtype=bool)
        use_custom_range = (min_freq > 0) or (max_freq > 0)
        
        if use_custom_range:
            if min_freq > 0:
                freq_mask &= f >= min_freq
            if max_freq > 0:
                effective_max = min(max_freq, nyquist_freq)
                freq_mask &= f <= effective_max
            else:
                freq_mask &= f <= nyquist_freq
            
            if np.any(freq_mask):
                f_display = f[freq_mask]
                sxx_display = sxx[freq_mask, :]
                freq_range_info = f" (Filtered: {f_display[0]:.6f} - {f_display[-1]:.6f} Hz)"
            else:
                st.warning(f"No frequencies in selected range for {sid}")
                f_display = f
                sxx_display = sxx
                freq_range_info = ""
        else:
            f_display = f
            sxx_display = sxx
            freq_range_info = ""
        
        # Convert time bins to actual timestamps
        start_time = sub["timestamp"].min()
        window_center_times = [start_time + pd.Timedelta(seconds=float(t_bin)) for t_bin in t]
        
        # Anomaly detection
        anomaly_freqs, anomaly_times = [], []
        anomaly_regions = []
        if enable_anomaly:
            anomaly_freqs, anomaly_times = detect_anomalies(sxx_display, f_display, t, anomaly_threshold)
            
            # Group anomalies into time regions
            if anomaly_times:
                anomaly_times_sorted = sorted(set(anomaly_times))
                if len(anomaly_times_sorted) > 0:
                    anomaly_timestamps = [start_time + pd.Timedelta(seconds=float(t_anom)) 
                                         for t_anom in anomaly_times_sorted]
                    if len(anomaly_timestamps) > 1:
                        current_start = anomaly_timestamps[0]
                        for i in range(1, len(anomaly_timestamps)):
                            time_diff = (anomaly_timestamps[i] - anomaly_timestamps[i-1]).total_seconds()
                            if time_diff > window_seconds * 2:
                                anomaly_regions.append((current_start, anomaly_timestamps[i-1]))
                                current_start = anomaly_timestamps[i]
                        anomaly_regions.append((current_start, anomaly_timestamps[-1]))
                    else:
                        if anomaly_timestamps:
                            anomaly_regions.append((anomaly_timestamps[0], anomaly_timestamps[0]))
        
        # Create spectrogram plot (use sxx_display directly, no extra processing)
        heatmap = go.Heatmap(
            x=window_center_times,
            y=f_display,
            z=sxx_display,
            colorscale=colormap.lower(),
            colorbar=dict(
                title="Power (dB)" if spec_mode == "psd" else "Magnitude (log)"
            ),
            zsmooth="best",
        )
        
        fig = go.Figure(data=heatmap)
        
        # Add shaded regions for anomalies
        if enable_anomaly and anomaly_regions:
            for region_start, region_end in anomaly_regions:
                fig.add_vrect(
                    x0=region_start,
                    x1=region_end,
                    fillcolor="red",
                    opacity=0.15,
                    layer="above",
                    line_width=0,
                )
        
        # Add anomaly markers
        if enable_anomaly and anomaly_freqs:
            anomaly_x = [start_time + pd.Timedelta(seconds=float(t_anom)) for t_anom in anomaly_times]
            fig.add_trace(go.Scattergl(
                x=anomaly_x,
                y=anomaly_freqs,
                mode="markers",
                marker=dict(color="red", size=8, symbol="x", line=dict(width=2, color="darkred")),
                name=f"Anomalies (z>{anomaly_threshold:.1f})",
                hovertemplate="Time: %{x}<br>Freq: %{y:.6f} Hz<extra>Anomaly</extra>"
            ))
        
        fig.update_layout(
            title=f"Spectrogram: {y_col} - {sid}{freq_range_info}",
            xaxis_title="Time (Window Center)",
            yaxis_title="Frequency (Hz)",
            yaxis_type='log' if (log_freq and len(f_display) > 0 and np.min(f_display) > 0) else 'linear',
            height=600,
        )
        
        # Set y-axis range if custom range is used
        if use_custom_range and len(f_display) > 0:
            fig.update_yaxes(range=[f_display[0], f_display[-1]])
        
        st.plotly_chart(fig, use_container_width=True)
        
        if anomaly_freqs:
            st.success(f"🔍 Found {len(anomaly_freqs)} anomaly points in {len(anomaly_regions)} region(s) for {sid}")


def plot_3d_vector_field(df: pd.DataFrame):
    """Plot 3D vector field"""
    if df.empty:
        st.info("No data to plot.")
        return
    
    max_points = 1000
    df_sample = df.sample(n=max_points).sort_values("timestamp") if len(df) > max_points else df.sort_values("timestamp")
    if len(df) > max_points:
        st.caption(f"📊 Showing {max_points} sampled points from {len(df)} total")
    
    fig = go.Figure()
    for sensor_id in df_sample["sensor_id"].dropna().unique():
        sensor_data = df_sample[df_sample["sensor_id"] == sensor_id]
        x, y, z = sensor_data["b_x"].values, sensor_data["b_y"].values, sensor_data["b_z"].values
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', name=f"Sensor {sensor_id}",
                                  marker=dict(size=4, color=magnitude, colorscale='Viridis',
                                             colorbar=dict(title="Magnitude (nT)"), opacity=0.7),
                                  hovertemplate="Bx: %{x:.2f}<br>By: %{y:.2f}<br>Bz: %{z:.2f}<br>Mag: %{marker.color:.2f}<extra></extra>"))
    
    fig.update_layout(title="3D Magnetic Field Vector Space",
                     scene=dict(xaxis_title="Bx (nT)", yaxis_title="By (nT)", zaxis_title="Bz (nT)",
                               aspectmode='cube'), height=700)
    st.plotly_chart(fig, use_container_width=True)


def plot_distribution_histograms(df: pd.DataFrame):
    """Plot distribution histograms"""
    if df.empty:
        st.info("No data to plot.")
        return
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Bx", "By", "Bz"), horizontal_spacing=0.1)
    components, colors = ["b_x", "b_y", "b_z"], ["blue", "green", "red"]
    
    for idx, (comp, color) in enumerate(zip(components, colors), 1):
        values = pd.to_numeric(df[comp], errors="coerce").dropna()
        fig.add_trace(go.Histogram(x=values, nbinsx=50, name=comp, marker_color=color, opacity=0.7), row=1, col=idx)
        fig.add_vline(x=np.mean(values), line_dash="dash", line_color="black",
                     annotation_text=f"Mean: {np.mean(values):.2f}", row=1, col=idx)
    
    fig.update_xaxes(title_text="Magnetic Field (nT)", row=1, col=1)
    fig.update_xaxes(title_text="Magnetic Field (nT)", row=1, col=2)
    fig.update_xaxes(title_text="Magnetic Field (nT)", row=1, col=3)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_layout(title="Component Distributions", height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    for idx, (comp, col) in enumerate(zip(components, [col1, col2, col3])):
        values = pd.to_numeric(df[comp], errors="coerce").dropna()
        with col:
            st.metric(f"{comp} Mean", f"{np.mean(values):.2f} nT")
            st.metric(f"{comp} Std", f"{np.std(values):.2f} nT")


def plot_box_plots(df: pd.DataFrame):
    """Plot box plots comparing sensors"""
    if df.empty:
        st.info("No data to plot.")
        return
    
    sensors = df["sensor_id"].dropna().unique()
    if len(sensors) == 0:
        return
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Bx", "By", "Bz"), horizontal_spacing=0.1)
    components = ["b_x", "b_y", "b_z"]
    
    for idx, comp in enumerate(components, 1):
        for sensor in sensors:
            sensor_data = df[df["sensor_id"] == sensor]
            values = pd.to_numeric(sensor_data[comp], errors="coerce").dropna()
            if len(values) > 0:
                fig.add_trace(go.Box(y=values.values, name=sensor, boxpoints='outliers', showlegend=(idx==1)),
                             row=1, col=idx)
    
    fig.update_yaxes(title_text="Value (nT)", row=1, col=1)
    fig.update_layout(title="Component Distributions by Sensor", height=500)
    st.plotly_chart(fig, use_container_width=True)


def plot_statistical_dashboard(df: pd.DataFrame):
    """Statistical summary dashboard"""
    if df.empty:
        st.info("No data to plot.")
        return
    
    st.subheader("📊 Statistical Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Points", f"{len(df):,}")
    with col2:
        st.metric("Unique Sensors", f"{df['sensor_id'].nunique()}")
    with col3:
        if df["timestamp"].notna().any():
            duration = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600
            st.metric("Duration", f"{duration:.2f} hours")
    with col4:
        if df["timestamp"].notna().any():
            fs = estimate_sampling_rate(df["timestamp"])
            st.metric("Sampling Rate", f"{fs:.3f} Hz")
    
    st.subheader("📈 Component Statistics")
    components = ["b_x", "b_y", "b_z"]
    stats_data = []
    for comp in components:
        values = pd.to_numeric(df[comp], errors="coerce").dropna()
        if len(values) > 0:
            stats_data.append({
                "Component": comp, "Mean (nT)": np.mean(values), "Std (nT)": np.std(values),
                "Min (nT)": np.min(values), "Max (nT)": np.max(values),
                "Range (nT)": np.max(values) - np.min(values), "Median (nT)": np.median(values)
            })
    
    if stats_data:
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    
    resultant = compute_resultant(df)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Resultant", f"{np.mean(resultant):.2f} nT")
    with col2:
        st.metric("Std Resultant", f"{np.std(resultant):.2f} nT")
    with col3:
        st.metric("Min Resultant", f"{np.min(resultant):.2f} nT")
    with col4:
        st.metric("Max Resultant", f"{np.max(resultant):.2f} nT")


def plot_frequency_domain_analysis(df: pd.DataFrame, series: str):
    """Plot FFT and PSD"""
    if df.empty:
        st.info("No data to plot.")
        return
    
    sensors = df["sensor_id"].dropna().unique()
    for sensor_id in sensors:
        sensor_data = df[df["sensor_id"] == sensor_id].sort_values("timestamp")
        if sensor_data.empty:
            continue
        
        signal_values = (compute_resultant(sensor_data).values if series == "resultant"
                        else pd.to_numeric(sensor_data[series], errors="coerce").dropna().values)
        
        if len(signal_values) < 32:
            continue
        
        fs = estimate_sampling_rate(sensor_data['timestamp'])
        if fs <= 0:
            continue
        
        signal_values = signal_values - np.mean(signal_values)
        
        # FFT
        n = len(signal_values)
        fft_vals = np.fft.fft(signal_values)
        fft_freq = np.fft.fftfreq(n, 1/fs)
        positive_mask = fft_freq > 0
        fft_freq_positive = fft_freq[positive_mask]
        fft_magnitude = np.abs(fft_vals[positive_mask])
        
        # PSD
        nperseg = min(256, len(signal_values) // 4, max(32, len(signal_values) // 4))
        try:
            psd_freq, psd = welch(signal_values, fs=fs, nperseg=nperseg, scaling='density')
        except Exception:
            psd_freq, psd = fft_freq_positive, (fft_magnitude ** 2) / fs
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=(f"FFT Magnitude - {sensor_id}",
                                                           f"PSD - {sensor_id}"), vertical_spacing=0.15)
        fig.add_trace(go.Scatter(x=fft_freq_positive, y=fft_magnitude, mode='lines',
                                name='FFT', line=dict(color='blue', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=psd_freq, y=10*np.log10(psd + 1e-15), mode='lines',
                                name='PSD', line=dict(color='red', width=2)), row=2, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", type='log', row=1, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", type='log', row=2, col=1)
        fig.update_yaxes(title_text="Magnitude", row=1, col=1)
        fig.update_yaxes(title_text="PSD (dB)", row=2, col=1)
        fig.update_layout(title=f"Frequency Analysis: {series} - {sensor_id}", height=700)
        st.plotly_chart(fig, use_container_width=True)


def plot_multi_component_comparison(df: pd.DataFrame):
    """Plot Bx, By, Bz in a dedicated 3-row stack for each sensor (avoids mixing six sensors on three axes)."""
    if df.empty:
        st.info("No data to plot.")
        return

    sensors_sorted = sorted(df["sensor_id"].dropna().unique(), key=lambda x: str(x))
    if not sensors_sorted:
        st.info("No rows with a valid sensor_id.")
        return

    components = ["b_x", "b_y", "b_z"]
    comp_colors = ("#2563EB", "#16A34A", "#DC2626")

    for order_idx, sensor_id in enumerate(sensors_sorted, start=1):
        sensor_data = df[df["sensor_id"] == sensor_id].sort_values("timestamp")
        if sensor_data.empty:
            continue
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("Bx", "By", "Bz"),
            vertical_spacing=0.08,
            shared_xaxes=True,
        )
        for idx, (comp, color) in enumerate(zip(components, comp_colors), 1):
            values = pd.to_numeric(sensor_data[comp], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=sensor_data["timestamp"],
                    y=values,
                    mode="lines",
                    name=f"{comp}",
                    line=dict(color=color, width=1.8),
                    showlegend=True,
                ),
                row=idx,
                col=1,
            )
        sid_short = str(sensor_id)[:64] + ("…" if len(str(sensor_id)) > 64 else "")
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Bx (nT)", row=1, col=1)
        fig.update_yaxes(title_text="By (nT)", row=2, col=1)
        fig.update_yaxes(title_text="Bz (nT)", row=3, col=1)
        fig.update_layout(
            title=f"Multi-component — {sid_short}",
            height=780,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)


def plot_component_heatmap(df: pd.DataFrame):
    """Plot component heatmap over time"""
    if df.empty:
        st.info("No data to plot.")
        return
    
    for sensor_id in df["sensor_id"].dropna().unique():
        sensor_data = df[df["sensor_id"] == sensor_id].sort_values("timestamp")
        if sensor_data.empty:
            continue
        
        time_range = (sensor_data["timestamp"].max() - sensor_data["timestamp"].min()).total_seconds()
        n_bins = min(100, max(20, int(time_range / 60)))
        sensor_data = sensor_data.copy()
        sensor_data["time_bin"] = pd.cut(
            (sensor_data["timestamp"] - sensor_data["timestamp"].min()).dt.total_seconds(),
            bins=n_bins, labels=False
        )
        
        heatmap_data, time_labels = [], []
        for bin_idx in range(n_bins):
            bin_data = sensor_data[sensor_data["time_bin"] == bin_idx]
            if not bin_data.empty:
                time_labels.append(bin_data["timestamp"].iloc[0])
                heatmap_data.append([
                    pd.to_numeric(bin_data["b_x"], errors="coerce").mean(),
                    pd.to_numeric(bin_data["b_y"], errors="coerce").mean(),
                    pd.to_numeric(bin_data["b_z"], errors="coerce").mean()
                ])
        
        if not heatmap_data:
            continue
        
        heatmap_array = np.array(heatmap_data).T
        fig = go.Figure(data=go.Heatmap(z=heatmap_array, x=time_labels, y=["b_x", "b_y", "b_z"],
                                        colorscale='RdBu', colorbar=dict(title="Magnetic Field (nT)")))
        fig.update_layout(title=f"Component Heatmap - {sensor_id}", xaxis_title="Time",
                         yaxis_title="Component", height=400)
        st.plotly_chart(fig, use_container_width=True)


def plot_polar_direction(df: pd.DataFrame):
    """Plot polar direction plot"""
    if df.empty:
        st.info("No data to plot.")
        return
    
    max_points = 500
    df_sample = df.sample(n=max_points) if len(df) > max_points else df
    if len(df) > max_points:
        st.caption(f"📊 Showing {max_points} sampled points")
    
    b_x = pd.to_numeric(df_sample["b_x"], errors="coerce").values
    b_y = pd.to_numeric(df_sample["b_y"], errors="coerce").values
    b_z = pd.to_numeric(df_sample["b_z"], errors="coerce").values
    azimuth = np.arctan2(b_y, b_x) * 180 / np.pi
    magnitude = np.sqrt(b_x**2 + b_y**2 + b_z**2)
    
    fig = go.Figure()
    sensors = df_sample["sensor_id"].dropna().unique()
    colors = list(_TS_LINE_COLORS) + ["#65A30D", "#0D9488"]
    
    for idx, sensor_id in enumerate(sensors):
        sensor_mask = df_sample["sensor_id"] == sensor_id
        if not sensor_mask.any():
            continue
        fig.add_trace(go.Scatterpolar(r=magnitude[sensor_mask], theta=azimuth[sensor_mask],
                                     mode='markers', name=f"Sensor {sensor_id}",
                                     marker=dict(size=6, color=colors[idx % len(colors)], opacity=0.7,
                                                line=dict(width=1, color='black')),
                                     hovertemplate="Magnitude: %{r:.2f} nT<br>Azimuth: %{theta:.1f}°<extra></extra>"))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(title="Magnitude (nT)", range=[0, np.max(magnitude) * 1.1]),
                  angularaxis=dict(thetaunit="degrees", rotation=90, direction="counterclockwise")),
        title="Magnetic Field Direction (Polar Plot)", height=600
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SIDEBAR & MAIN
# ============================================================================

def layout_sidebar(
    df: pd.DataFrame,
) -> Tuple[SensorFilter, str, Tuple[pd.Timestamp, pd.Timestamp], float, float]:
    """Create sidebar controls. Sensor filter is ``\"All\"`` or a list of selected ``sensor_id`` values."""
    st.sidebar.header("Controls")

    sensor_options = sorted(df["sensor_id"].dropna().unique().tolist()) if not df.empty else []
    st.sidebar.markdown("**Sensors**")
    sensor_scope = st.sidebar.radio(
        "Scope",
        ("All sensors", "Choose multiple"),
        index=0,
        horizontal=True,
        help="Pick several IDs (e.g. three) to plot only those streams; facets stack one row per sensor.",
    )
    sensor_selection: SensorFilter = "All"
    if sensor_scope == "Choose multiple" and sensor_options:
        n_opt = len(sensor_options)
        max_pick = int(
            st.sidebar.number_input(
                "Max sensors to select at once",
                min_value=1,
                max_value=max(1, n_opt),
                value=min(3, n_opt),
                step=1,
                help="Upper bound on how many IDs you can pick in the list below (e.g. 3 for three-way comparison).",
            )
        )
        default_pick = sensor_options[: min(max_pick, n_opt)]
        multiselect_kwargs = dict(
            label="Sensor IDs (multi-select)",
            options=sensor_options,
            default=default_pick,
            help="Cmd/Ctrl-click to change selection. Defaults to the first few IDs up to your max.",
        )
        try:
            picked = st.sidebar.multiselect(**multiselect_kwargs, max_selections=max_pick)
        except TypeError:
            picked = st.sidebar.multiselect(**multiselect_kwargs)
            if len(picked) > max_pick:
                st.sidebar.warning(f"Keeping only the first **{max_pick}** selected sensors (upgrade Streamlit for hard cap).")
                picked = picked[:max_pick]
        if picked:
            sensor_selection = picked
        else:
            st.sidebar.info("None selected — using **all** sensors.")
            sensor_selection = "All"
    selected_series = st.sidebar.selectbox("Series", options=["b_x", "b_y", "b_z", "resultant"], index=0)
    
    if not df.empty and df["timestamp"].notna().any():
        tmin, tmax = df["timestamp"].min(), df["timestamp"].max()
    else:
        tmin, tmax = pd.Timestamp.now() - pd.Timedelta(hours=1), pd.Timestamp.now()
    
    time_range = st.sidebar.slider("Time range", min_value=tmin.to_pydatetime(),
                                   max_value=tmax.to_pydatetime(),
                                   value=(tmin.to_pydatetime(), tmax.to_pydatetime()),
                                   step=pd.Timedelta(seconds=1), format="YYYY-MM-DD HH:mm:ss")
    
    bin_width_seconds = st.sidebar.number_input("Spectrogram window (s)", min_value=0.05,
                                               max_value=12000.0, value=1.0, step=0.05)
    overlap_ratio = st.sidebar.slider("Overlap", min_value=0.0, max_value=0.99, value=0.5, step=0.05)
    
    return sensor_selection, selected_series, (pd.Timestamp(time_range[0]), pd.Timestamp(time_range[1])), bin_width_seconds, overlap_ratio


def main():
    """Main application"""
    st.set_page_config(page_title="MagSpectrogram", layout="wide")
    st.title("🧲 Magnetic Sensor Data Visualizer")
    st.markdown("Upload CSV files with magnetic field data. Resultant = √(b_x² + b_y² + b_z²) in **nT**")
    
    # File upload
    with st.expander("📁 Import Data", expanded=True):
        uploaded = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
        dataset_dir = os.path.join(os.getcwd(), "Dataset")
        quick_load = st.checkbox("Load all CSVs from ./Dataset", value=False)
        files_from_dir = []
        if quick_load and os.path.isdir(dataset_dir):
            files_from_dir = [os.path.join(dataset_dir, name) for name in os.listdir(dataset_dir)
                             if name.lower().endswith(".csv")]
    
    df = read_multiple_csvs((uploaded or []) + files_from_dir)
    st.caption(f"✅ Loaded {len(df)} rows from {len((uploaded or [])) + len(files_from_dir)} files")
    
    if df.empty:
        st.stop()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Visualization", "🔗 Correlation", "ℹ️ About"])
    
    with tab1:
        sensor_selection, selected_series, time_range, bin_width_seconds, overlap_ratio = layout_sidebar(df)
        view_df = filter_df(df, sensor_selection, time_range)
        if sensor_selection == "All":
            st.caption(f"🔍 Filtered: **{len(view_df)}** rows (from {len(df)} total) — **all** sensors")
        else:
            sel = list(sensor_selection)
            st.caption(
                f"🔍 Filtered: **{len(view_df)}** rows (from {len(df)} total) — "
                f"**{len(sel)}** sensor(s): {', '.join(sel)}"
            )
        
        viz_mode = st.selectbox("Visualization Type:", [
            "Time Series & Spectrogram", "3D Vector Field", "Distribution Histograms",
            "Box Plots", "Statistical Dashboard", "Frequency Analysis (FFT/PSD)",
            "Multi-Component Comparison", "Component Heatmap", "Polar Direction Plot"
        ], key="viz_mode")
        
        if viz_mode == "Time Series & Spectrogram":
            # Anomaly detection settings
            with st.sidebar.expander("🔍 Anomaly Detection", expanded=False):
                enable_anomaly_ts = st.checkbox("Enable Time Series Anomaly Detection", value=True)
                enable_anomaly_spec = st.checkbox("Enable Spectrogram Anomaly Detection", value=True)
                anomaly_threshold = st.slider("Anomaly Threshold (z-score)", 1.0, 5.0, 
                                             value=st.session_state.get('anomaly_threshold', 2.0), step=0.1)
                st.session_state.anomaly_threshold = anomaly_threshold
                st.session_state.enable_anomaly_ts = enable_anomaly_ts
                st.session_state.enable_anomaly_spec = enable_anomaly_spec
            
            st.subheader("Time Series")
            plot_time_series(view_df, selected_series, enable_anomaly_ts, anomaly_threshold)
            st.subheader("Spectrogram")
            plot_spectrogram(view_df, selected_series, bin_width_seconds, overlap_ratio)
        elif viz_mode == "3D Vector Field":
            plot_3d_vector_field(view_df)
        elif viz_mode == "Distribution Histograms":
            plot_distribution_histograms(view_df)
        elif viz_mode == "Box Plots":
            plot_box_plots(view_df)
        elif viz_mode == "Statistical Dashboard":
            plot_statistical_dashboard(view_df)
        elif viz_mode == "Frequency Analysis (FFT/PSD)":
            plot_frequency_domain_analysis(view_df, selected_series)
        elif viz_mode == "Multi-Component Comparison":
            plot_multi_component_comparison(view_df)
        elif viz_mode == "Component Heatmap":
            plot_component_heatmap(view_df)
        elif viz_mode == "Polar Direction Plot":
            plot_polar_direction(view_df)
    
    with tab2:
        correlation_study_interface(df)
    
    with tab3:
        st.subheader("ℹ️ About")
        st.markdown("""
        **Units:** Magnetic field components (b_x, b_y, b_z) and resultant in **nT (nanotesla)**
        
        **Features:**
        - Robust spectrogram computation optimized for magnetic field data
        - Signal enhancement for weak magnetic variations
        - Anomaly detection using statistical z-score analysis
        - Adaptive window sizing based on signal characteristics
        - Advanced preprocessing with DC removal and filtering
        - Correlation analysis between sensors and field components
        - Multiple visualization modes for comprehensive analysis
        
        **Correlation Interpretation:**
        - |r| > 0.8: Very strong | |r| > 0.6: Strong | |r| > 0.4: Moderate | |r| > 0.2: Weak
        
        **Statistical Significance:** p < 0.05: Significant | p < 0.01: Very significant | p < 0.001: Highly significant
        """)


if __name__ == "__main__":
    main()

