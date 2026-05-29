'''
Anomaly Detection Module for Magnetic Field Data
Compares LSTM predictions with actual real-time data to detect anomalies

This module implements a statistical anomaly detection system that:
1. Compares LSTM model predictions with actual magnetic field measurements
2. Calculates the difference (error) between predicted and actual values
3. Uses statistical methods to determine a dynamic threshold
4. Flags data points where the error exceeds the threshold as anomalies
'''

from typing import Optional

import numpy as np  # For numerical operations (mean, std, etc.)
import pandas as pd  # For data manipulation and time-series operations


class AnomalyDetector:
    """
    Detects anomalies by comparing predicted magnetic field values with actual measurements.
    An anomaly is flagged when the difference exceeds a dynamically calculated threshold.
    
    How it works:
    - The detector learns from historical prediction errors to set an adaptive threshold
    - Uses statistical methods (mean + multiple of standard deviation) to identify outliers
    - Adapts to changing conditions using exponentially weighted error statistics
      so older errors are retained with gradual decay instead of hard truncation
    """
    
    def __init__(
        self,
        threshold_multiplier=2.5,
        min_samples_for_threshold=20,
        error_smoothing_alpha=0.995,
        recent_error_buffer_size=1000,
        std_relative_floor=0.02,
    ):
        """
        Initialize the anomaly detector with configuration parameters.
        
        Parameters:
        -----------
        threshold_multiplier : float
            Multiplier for standard deviation to set threshold (default: 2.5)
            This means anomalies are detected when error > mean_error + 2.5 * std_error
            Higher values = fewer anomalies detected (more strict)
            Lower values = more anomalies detected (more sensitive)
        min_samples_for_threshold : int
            How many absolute-error samples must be ingested into EWMA before anomaly
            flags are emitted. The first (min_samples_for_threshold) points only establish
            the EWMA baseline (mean + k×sqrt(var) from EWMA, no fixed nT default).
        error_smoothing_alpha : float
            EWMA smoothing factor in (0, 1). Higher value => longer memory.
            Update rule:
              ewma_t = alpha * ewma_(t-1) + (1-alpha) * error_t
            Threshold uses EWMA mean and EWMA std, so old errors still contribute
            with exponential decay.
        recent_error_buffer_size : int
            Size of recent raw-error buffer kept for diagnostics/logging only.
            Threshold itself uses EWMA statistics (not this hard window).
        std_relative_floor : float
            Minimum std used in threshold = mean + k*std, as a fraction of EWMA mean
            absolute error. Prevents threshold ~= mean when variance is still tiny (which
            flags almost every slightly-above-mean point). Scale is data-driven, not a
            fixed nT constant.
        """
        # Store the multiplier used for threshold calculation
        # This determines how many standard deviations away from mean is considered anomalous
        self.threshold_multiplier = threshold_multiplier
        
        # Minimum number of error samples needed before we can calculate statistics
        # Too few samples would give unreliable statistics
        self.min_samples_for_threshold = min_samples_for_threshold

        # EWMA configuration and state
        if not (0 < float(error_smoothing_alpha) < 1):
            error_smoothing_alpha = 0.995
        self.error_smoothing_alpha = float(error_smoothing_alpha)
        self.recent_error_buffer_size = int(max(10, recent_error_buffer_size))
        self.std_relative_floor = float(max(0.0, std_relative_floor))
        self.ewma_error = None
        self.ewma_var = 0.0
        self.total_error_samples = 0
        
        # Recent raw error buffer for diagnostics/legacy logging (EWMA updates only).
        self.prediction_errors = []
        # Recent non-anomalous |actual−pred| values for robust scale (MAD).
        # Keep this aligned with EWMA updates so anomalous points do not influence
        # adaptive-threshold statistics.
        self._residual_scale_ring: list = []
        self._residual_scale_ring_max: int = 512
        
        # The calculated threshold value (in nanoTesla, nT)
        # Any prediction error above this value is considered an anomaly
        # Initially None, will be calculated after enough samples are collected
        self.anomaly_threshold = None

    def _robust_sigma_from_recent_buffer(self) -> Optional[float]:
        """
        Scale estimate from recent non-anomalous absolute errors (MAD → σ). Used when
        EWMA variance is tiny for one axis but residuals still have spread.
        """
        buf = self._residual_scale_ring if len(self._residual_scale_ring) >= 8 else self.prediction_errors
        if len(buf) < 8:
            return None
        arr = np.asarray(buf[-min(500, len(buf)) :], dtype=float)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        if mad <= 0:
            return None
        return 1.4826 * mad

    def _append_residual_ring(self, abs_diff: float) -> None:
        self._residual_scale_ring.append(float(abs_diff))
        if len(self._residual_scale_ring) > self._residual_scale_ring_max:
            self._residual_scale_ring = self._residual_scale_ring[-self._residual_scale_ring_max :]

    def _update_error_statistics(self, abs_errors):
        """
        Update EWMA mean/variance from new absolute-error samples.
        Uses a numerically stable exponentially weighted variance update.
        """
        a = self.error_smoothing_alpha
        for e in abs_errors:
            x = float(e)
            self.total_error_samples += 1

            if self.ewma_error is None:
                self.ewma_error = x
                self.ewma_var = 0.0
            else:
                prev_mean = self.ewma_error
                # EWMA mean: m_t = a*m_(t-1) + (1-a)*x_t
                self.ewma_error = a * prev_mean + (1.0 - a) * x
                # EWMA variance (innovation form): v_t = a*(v_(t-1) + (1-a)*(x_t-m_(t-1))^2)
                delta = x - prev_mean
                self.ewma_var = a * (self.ewma_var + (1.0 - a) * (delta ** 2))

            # Keep recent raw errors for UI/debug compatibility.
            self.prediction_errors.append(x)

        if len(self.prediction_errors) > self.recent_error_buffer_size:
            self.prediction_errors = self.prediction_errors[-self.recent_error_buffer_size :]

    def _refresh_threshold_from_ewma(self) -> None:
        """
        Set anomaly_threshold = EWMA_mean + multiplier × sqrt(EWMA_variance).

        No fixed nanoTesla default: the scale always comes from errors already processed
        through the EWMA updates.
        """
        if self.total_error_samples <= 0 or self.ewma_error is None:
            self.anomaly_threshold = None
            return
        mean_error = float(self.ewma_error)
        raw_std = float(np.sqrt(max(self.ewma_var, 0.0)))
        # Some axes show ~0 EWMA variance (smooth |error|) while the residual buffer still
        # has spread — threshold collapses and almost every point exceeds k·σ. Blend in a
        # robust σ from recent errors so behavior matches noisier channels (e.g. OBS1_3).
        robust = self._robust_sigma_from_recent_buffer()
        if robust is not None:
            raw_std = max(raw_std, robust)
        # Relative floor: scale by max(EWMA mean, median of recent |errors|) so a low
        # smoothed mean does not shrink the floor when the batch spread is larger.
        scale = mean_error
        med_src = self._residual_scale_ring if len(self._residual_scale_ring) >= 8 else self.prediction_errors
        if len(med_src) >= 8:
            scale = max(scale, float(np.median(med_src)))
        rel_floor = self.std_relative_floor * max(scale, 1e-12)
        std_error = max(raw_std, rel_floor)
        self.anomaly_threshold = mean_error + self.threshold_multiplier * std_error
        
    def calculate_differences(self, actual_times, actual_values, predicted_times, predicted_values):
        """
        Calculate differences between actual and predicted values for overlapping time periods.
        
        This is the core method that:
        1. Interpolates predicted values at exact actual timestamps (more accurate than nearest-neighbor)
        2. Calculates the error (difference) between actual and interpolated predicted values
        3. Updates the error history for threshold calculation
        4. Calculates/updates the anomaly threshold
        5. Marks which points are anomalies
        
        Timestamp Matching Strategy:
        ----------------------------
        Uses linear interpolation to find predicted values at exact actual timestamps.
        This approach is more accurate than nearest-neighbor matching because:
        - Eliminates timing errors (exact time alignment)
        - Uses information from neighboring prediction points
        - Handles different sampling rates gracefully
        - Provides smoother, more accurate comparisons
        
        Only interpolates within the actual prediction time range (no edge tolerance).
        Points outside this range or where interpolation isn't possible are excluded from comparison.
        This prevents "fake" predictions from being created using nearest neighbor fallback.
        
        Parameters:
        -----------
        actual_times : list
            List of datetime objects for actual measurements (from sensors or API)
        actual_values : list
            List of actual magnetic field values (in nanoTesla, nT)
        predicted_times : list
            List of datetime objects for predictions (from LSTM model)
        predicted_values : list
            List of predicted magnetic field values (in nanoTesla, nT)
            
        Returns:
        --------
        differences_df : pandas.DataFrame
            DataFrame with columns: 'time', 'actual', 'predicted', 'difference', 'is_anomaly'
            Each row represents a matched pair of actual and interpolated predicted values
            'predicted' column contains interpolated values at exact actual timestamps
        """
        # Step 1: Check if we have data to compare
        # If either list is empty, return an empty DataFrame
        if not actual_times or not predicted_times:
            return pd.DataFrame(columns=['time', 'actual', 'predicted', 'difference', 'is_anomaly'])
        
        # Step 2: Convert lists to pandas DataFrames for easier manipulation
        # This allows us to use pandas' powerful time-series operations
        actual_df = pd.DataFrame({
            'time': pd.to_datetime(actual_times),  # Ensure times are datetime objects
            'actual': actual_values  # Actual magnetic field measurements
        })
        
        predicted_df = pd.DataFrame({
            'time': pd.to_datetime(predicted_times),  # Ensure times are datetime objects
            'predicted': predicted_values  # Predicted magnetic field values from LSTM
        })
        
        # Step 3: Match actual and predicted data by timestamp using interpolation
        # This approach interpolates predicted values at exact actual timestamps,
        # providing more accurate comparisons than nearest-neighbor matching.
        # 
        # Advantages of interpolation:
        # - Exact time alignment (no timing errors)
        # - Uses information from neighboring prediction points
        # - Handles different sampling rates gracefully
        # - More accurate for anomaly detection
        
        # Sort both dataframes by time (required for interpolation)
        actual_sorted = actual_df.sort_values('time').reset_index(drop=True)
        predicted_sorted = predicted_df.sort_values('time').reset_index(drop=True)

        pred_min_time = predicted_sorted['time'].min()
        pred_max_time = predicted_sorted['time'].max()
        last_pred_y = float(predicted_sorted['predicted'].iloc[-1])

        # (A) Actual samples strictly after the last prediction timestamp: compare to the
        # last predicted value (constant hold-out). Without this, those points were dropped
        # entirely — large real deviations produced no row and were never flagged.
        actual_after_pred = actual_sorted[actual_sorted['time'] > pred_max_time].copy()

        # (B) Actual samples within [pred_min, pred_max]: time-interpolate predicted value
        actual_within_range = actual_sorted[
            (actual_sorted['time'] >= pred_min_time) & (actual_sorted['time'] <= pred_max_time)
        ].copy()

        merged_parts: list = []

        if len(actual_within_range) > 0:
            predicted_indexed = predicted_sorted.set_index('time')
            actual_indexed = actual_within_range.set_index('time')
            predicted_interpolated = predicted_indexed.reindex(actual_indexed.index).interpolate(
                method='time'
            )
            merged_mid = pd.DataFrame(
                {
                    'time': actual_indexed.index,
                    'actual': actual_indexed['actual'].values,
                    'predicted': predicted_interpolated['predicted'].values,
                }
            ).reset_index(drop=True)
            merged_mid = merged_mid.dropna(subset=['predicted'])
            if len(merged_mid) > 0:
                merged_parts.append(merged_mid)

        if len(actual_after_pred) > 0:
            merged_tail = pd.DataFrame(
                {
                    'time': actual_after_pred['time'].values,
                    'actual': actual_after_pred['actual'].values,
                    'predicted': last_pred_y,
                }
            )
            merged_parts.append(merged_tail)

        if not merged_parts:
            return pd.DataFrame(columns=['time', 'actual', 'predicted', 'difference', 'is_anomaly'])

        merged = pd.concat(merged_parts, ignore_index=True)
        merged = merged.sort_values('time').reset_index(drop=True)
        merged = merged.drop_duplicates(subset=['time'], keep='last')
        
        # Ensure time column is datetime (should already be, but safety check)
        if len(merged) > 0:
            merged['time'] = pd.to_datetime(merged['time'])
        
        # Step 4: Calculate the error (difference) between actual and predicted values
        # Positive difference means actual > predicted (model underestimated)
        # Negative difference means actual < predicted (model overestimated)
        merged['difference'] = merged['actual'] - merged['predicted']
        
        # We care about the magnitude of error, not the direction
        # So we take the absolute value
        merged['abs_difference'] = abs(merged['difference'])
        
        # Steps 5–7: Per-sample thresholding and EWMA updates.
        #
        # Bootstrap: ingest the first min_samples_for_threshold absolute errors into EWMA
        # without flagging, so mean and variance reflect real residuals (no fixed nT floor).
        # Then: threshold = EWMA_mean + k×sqrt(EWMA_var); update EWMA only from non-anomalous errors.
        is_anomaly_flags = []
        for i in range(len(merged)):
            abs_diff = float(merged.iloc[i]["abs_difference"])

            if self.total_error_samples < self.min_samples_for_threshold:
                # Learn baseline distribution; do not flag during bootstrap.
                is_anomaly_flags.append(False)
                self._update_error_statistics([abs_diff])
                self._refresh_threshold_from_ewma()
                self._append_residual_ring(abs_diff)
                continue

            self._refresh_threshold_from_ewma()
            thr = self.anomaly_threshold
            if thr is None:
                is_anomaly_flags.append(False)
                continue

            is_anom = abs_diff > thr
            is_anomaly_flags.append(is_anom)
            if not is_anom:
                self._update_error_statistics([abs_diff])
                self._append_residual_ring(abs_diff)

        merged["is_anomaly"] = is_anomaly_flags

        # Return only the columns we need, excluding 'abs_difference' (internal use only)
        return merged[['time', 'actual', 'predicted', 'difference', 'is_anomaly']]
    
    def detect_anomalies(self, actual_times, actual_values, predicted_times, predicted_values):
        """
        Main method to detect anomalies - this is the primary interface for the detector.
        
        This method:
        1. Calls calculate_differences to compare actual vs predicted data
        2. Filters to keep only the points marked as anomalies
        3. Returns the anomalies along with the threshold used
        
        Parameters:
        -----------
        actual_times : list
            List of datetime objects for actual measurements
        actual_values : list
            List of actual magnetic field values
        predicted_times : list
            List of datetime objects for predictions
        predicted_values : list
            List of predicted magnetic field values
            
        Returns:
        --------
        anomalies_df : pandas.DataFrame
            DataFrame containing ONLY the anomaly points (filtered from all comparisons)
            Columns: 'time', 'actual', 'predicted', 'difference'
            Each row is a detected anomaly
        threshold : float
            The threshold value (in nT) that was used for detection
            Useful for logging and understanding detection sensitivity
        """
        # Step 1: Calculate differences and identify anomalies
        # This does all the heavy lifting: matching, error calculation, threshold calculation
        differences_df = self.calculate_differences(
            actual_times, actual_values, predicted_times, predicted_values
        )
        
        # Step 2: Filter to keep only the anomalies
        # The 'is_anomaly' column is True for anomalies, False for normal points
        # We use boolean indexing to filter: differences_df[differences_df['is_anomaly']]
        # .copy() creates a new DataFrame so we don't modify the original
        anomalies_df = differences_df[differences_df['is_anomaly']].copy()
        
        # Return both the anomalies and the threshold used
        # The threshold is useful for logging and understanding detection sensitivity
        return anomalies_df, self.anomaly_threshold
    
    def get_statistics(self):
        """
        Get statistics about prediction errors and the current threshold.
        
        This is useful for:
        - Monitoring the detector's performance
        - Understanding how the threshold is being calculated
        - Debugging and tuning the detector
        
        Returns:
        --------
        dict : Dictionary with statistics containing:
            - 'mean_error': Average prediction error (in nT)
            - 'std_error': Standard deviation of prediction errors (in nT)
            - 'threshold': Current anomaly threshold (in nT)
            - 'total_samples': Number of error samples used for statistics
        """
        # If we haven't collected any errors yet, return zeros
        if self.total_error_samples == 0:
            return {
                'mean_error': 0,  # No errors yet, so mean is 0
                'std_error': 0,  # No errors yet, so std is 0
                'threshold': self.anomaly_threshold if self.anomaly_threshold is not None else 0,
                'total_samples': 0,  # Total seen by EWMA
                'recent_buffer_samples': 0,  # Recent raw error buffer size
                'error_smoothing_alpha': self.error_smoothing_alpha,
            }
        
        # Return statistics from exponentially weighted error history
        mean_error = float(self.ewma_error) if self.ewma_error is not None else 0.0
        std_error = float(np.sqrt(max(self.ewma_var, 0.0)))
        return {
            'mean_error': mean_error,  # EWMA mean error
            'std_error': std_error,  # EWMA std error
            'threshold': self.anomaly_threshold if self.anomaly_threshold is not None else 0,
            'total_samples': self.total_error_samples,  # Total errors seen (lifetime)
            'recent_buffer_samples': len(self.prediction_errors),  # Raw recent buffer size
            'error_smoothing_alpha': self.error_smoothing_alpha,
        }
