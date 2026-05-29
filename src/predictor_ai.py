import json
import math
import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Bidirectional,
    GRU,
    Dropout,
    LayerNormalization,
    Add,
    Activation,
    Attention,
    GlobalAveragePooling1D,
)

from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

MODEL_FAMILY_ATTN_BILSTM = "attn_bilstm"
MODEL_FAMILY_GRU = "gru"
MODEL_FAMILY_TRANSFORMER = "transformer"
MODEL_FAMILY_LSTM = "lstm"


@keras.saving.register_keras_serializable(package="magnavis", name="SinusoidalPositionEncoding")
class SinusoidalPositionEncoding(tf.keras.layers.Layer):
    """
    Positional encoding for the magnetic transformer checkpoints in models/transformer_pretrained_*.keras.

    Weights are a (sequence_length, depth) table (one weight matrix), matching saved checkpoints.
    """

    def __init__(self, sequence_length=15, depth=64, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)
        self.depth = int(depth)

    def get_config(self):
        c = super().get_config()
        c.update({"sequence_length": self.sequence_length, "depth": self.depth})
        return c

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            name="positional_encoding",
            shape=(self.sequence_length, self.depth),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_emb[:seq_len, :]


def _normalize_model_family(value):
    raw = str(value or "").strip().lower()
    if raw in {"gru", "gru_rnn", "gated_recurrent_unit"}:
        return MODEL_FAMILY_GRU
    if raw in {"transformer", "tfm", "trf"}:
        return MODEL_FAMILY_TRANSFORMER
    if raw in {"lstm", "vanilla_lstm", "plain_lstm", "stacked_lstm"}:
        return MODEL_FAMILY_LSTM
    if raw in {"attn_bilstm", "attn-bilstm", "attention_bilstm", "attention-bilstm", "attn", "bilstm"}:
        return MODEL_FAMILY_ATTN_BILSTM
    return MODEL_FAMILY_ATTN_BILSTM

def _parse_train_window_minutes():
    """Read optional training window (minutes) from env TRAIN_WINDOW_MINUTES."""
    try:
        val = os.environ.get("TRAIN_WINDOW_MINUTES", None)
        if val is None or val == "":
            return None
        return float(val)
    except Exception:
        return None


def _parse_model_family():
    return _normalize_model_family(os.environ.get("PREDICTOR_MODEL_FAMILY", MODEL_FAMILY_ATTN_BILSTM))


def _parse_update_training():
    """Read predictor run mode from env PREDICTOR_UPDATE_TRAINING."""
    raw = str(os.environ.get("PREDICTOR_UPDATE_TRAINING", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _parse_skip_finetune_on_session_env() -> bool:
    """When true, never session-fit a loaded checkpoint (benchmark: 0 min historic, predict-only)."""
    raw = str(os.environ.get("PREDICTOR_SKIP_FINETUNE_ON_SESSION", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _parse_checkpoint_path():
    """Read optional runtime checkpoint path from env PREDICTOR_CHECKPOINT_PATH."""
    raw = str(os.environ.get("PREDICTOR_CHECKPOINT_PATH", "")).strip()
    return raw or None


def _parse_n_future():
    """
    Forecast horizon (number of autoregressive steps).
    Set by app.py (ApplicationTemp) via PREDICTOR_N_FUTURE so one run can cover the full
    gap from last training time to latest actual (sequential catch-up).
    """
    try:
        v = int(os.environ.get("PREDICTOR_N_FUTURE", "100"))
        return max(1, min(v, 200000))
    except Exception:
        return 100


def _parse_cover_until():
    """
    Optional latest timestamp the forecast must reach (exclusive of gaps handled by caller).
    Set by ApplicationTemp during sequential catch-up when predict_input is capped at
    latest_pred but skip/one-step mode would otherwise emit no rows past that cap.
    """
    raw = str(os.environ.get("PREDICTOR_COVER_UNTIL", "")).strip()
    if not raw:
        return None
    try:
        return pd.Timestamp(raw)
    except Exception:
        return None


def _parse_predictor_gru_window_size(default: int = 15) -> int:
    """
    Sequence length W for GRU (past timesteps per step). Set by ApplicationTemp when using
    fresh GRU via PREDICTOR_GRU_WINDOW_SIZE; must match training and rolling inference.
    """
    try:
        v = int(os.environ.get("PREDICTOR_GRU_WINDOW_SIZE", str(int(default))))
        return max(5, min(v, 3600))
    except Exception:
        return max(5, min(int(default), 3600))


def _parse_epochs_per_update() -> int:
    """In-session fit epochs; higher values reduce 'flat forecast' from underfitting (env override)."""
    try:
        v = int(os.environ.get("PREDICTOR_EPOCHS_PER_UPDATE", "40"))
        return max(1, min(v, 500))
    except Exception:
        return 40


def _predictor_meta_path(model_filepath: str) -> str:
    base, _ = os.path.splitext(os.path.abspath(model_filepath))
    return base + "_predictor_meta.json"


def _parse_gru_dropout() -> float:
    try:
        v = float(os.environ.get("PREDICTOR_GRU_DROPOUT", "0.05"))
        return max(0.0, min(v, 0.6))
    except Exception:
        return 0.05


# Eight recurrent layers + four dense layers (32→24→16→1) for deep benchmark runs.
_DEEP_RNN_STACK_UNITS = [48, 40, 36, 32, 28, 24, 20, 16]
_DEEP_RNN_DENSE_UNITS = [32, 24, 16, 1]


def _deep_rnn_benchmark_enabled() -> bool:
    """
    When true, **fresh GRU** and **fresh LSTM** use an 8-layer recurrent stack + 4 dense layers
    (see ``build_model``). Prefer ``MAGNAVIS_DEEP_RNN_BENCHMARK=1``; ``MAGNAVIS_GRU_DEEP_BENCHMARK=1``
    is accepted as a legacy alias for the same behavior.

    Do not set when loading pretrained GRU/LSTM checkpoints whose graphs do not match.
    """
    for key in ("MAGNAVIS_DEEP_RNN_BENCHMARK", "MAGNAVIS_GRU_DEEP_BENCHMARK"):
        raw = str(os.environ.get(key, "")).strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True
    return False


def _append_deep_rnn_stack_sequential(layers_list, RnnClass, dr: float) -> None:
    for i, u in enumerate(_DEEP_RNN_STACK_UNITS):
        ret_seq = i < len(_DEEP_RNN_STACK_UNITS) - 1
        layers_list.append(RnnClass(int(u), return_sequences=ret_seq))
        if dr > 0 and ret_seq and (i in (1, 3, 5)):
            layers_list.append(Dropout(dr))


def _append_deep_dense_head_sequential(layers_list) -> None:
    du = _DEEP_RNN_DENSE_UNITS
    for j in range(len(du) - 1):
        layers_list.append(Dense(int(du[j]), activation="relu"))
    layers_list.append(Dense(int(du[-1])))


def _parse_gru_delta_target() -> bool:
    """
    If true (default), fresh GRU sessions train on one-step Δ(scaled magnitude), which avoids
    collapsing to the series mean when |B| varies slowly. Bundled / legacy checkpoints without
    meta use absolute targets (see _load_predictor_meta).
    """
    raw = str(os.environ.get("PREDICTOR_GRU_DELTA_TARGET", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _parse_learning_rate(model_family: str) -> float:
    raw = os.environ.get("PREDICTOR_LEARNING_RATE", "").strip()
    if raw:
        return float(raw)
    fam = _normalize_model_family(model_family)
    return 0.003 if fam == MODEL_FAMILY_GRU else 0.001


def _parse_online_fit_each_step() -> bool:
    """
    If true, run model.fit() inside each autoregressive step (very slow for large n_future).
    Default off: only the initial fit (before the loop) runs when update_training is on.
    Set PREDICTOR_ONLINE_FIT_EACH_STEP=1 to restore legacy online-learning behavior.
    """
    raw = str(os.environ.get("PREDICTOR_ONLINE_FIT_EACH_STEP", "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _parse_leading_train_minutes() -> float:
    """
    Fresh (train-from-scratch) DL models: fit only on the first N minutes from the
    series start in predict_input, then one-step-ahead predictions on remaining rows
    (aligned to actual timestamps). Set by app via PREDICTOR_LEADING_TRAIN_MINUTES; 0 = off.
    """
    try:
        raw = str(os.environ.get("PREDICTOR_LEADING_TRAIN_MINUTES", "")).strip()
        if not raw:
            return 0.0
        v = float(raw)
        return max(0.0, min(v, 10080.0))
    except Exception:
        return 0.0


def _parse_skip_initial_minutes_predictor() -> float:
    """
    Pretrained / skip-only path: drop the first N minutes from predict_input before scaling
    and inference (no session fit on that segment). PREDICTOR_SKIP_INITIAL_MINUTES; 0 = off.
    """
    try:
        raw = str(os.environ.get("PREDICTOR_SKIP_INITIAL_MINUTES", "")).strip()
        if not raw:
            return 0.0
        v = float(raw)
        return max(0.0, min(v, 10080.0))
    except Exception:
        return 0.0


def _parse_predict_on_input_timestamps() -> bool:
    """
    Force one-step-ahead prediction on input timestamps (aligned with actual data timeline).
    Used by the GUI app so predicted and actual samples overlap for plotting/anomaly checks.
    """
    raw = str(os.environ.get("PREDICTOR_ON_INPUT_TIMESTAMPS", "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _parse_ar_closed_loop() -> bool:
    """
    When true, one-step inference rolls the W-window using predicted magnitudes (closed-loop AR),
    not actual measurements, after an initial bootstrap from actual rows.
    """
    raw = str(os.environ.get("PREDICTOR_AR_CLOSED_LOOP", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _last_index_at_or_before(ts: pd.Series, bound: pd.Timestamp) -> int:
    """Last integer index i with ts.iloc[i] <= bound (ts sorted ascending). Returns -1 if none."""
    s = pd.to_datetime(ts, errors="coerce")
    mask = s <= pd.Timestamp(bound)
    if not bool(mask.any()):
        return -1
    return int(np.flatnonzero(mask.to_numpy())[-1])


class AttnBiLSTMPredictor:
    """
    Recurrent / sequence predictor for magnetic field forecasting.

    Supports:
    - Attention-Bi-LSTM (PREDICTOR_MODEL_FAMILY=attn_bilstm)
    - Vanilla stacked LSTM (PREDICTOR_MODEL_FAMILY=lstm, fresh training; optional 8+4 deep stack)
    - GRU (train-from-scratch or load gru_pretrained_*.keras; optional 8+4 deep stack)
    - Deep 8+4 stack for **both** LSTM and GRU when ``MAGNAVIS_DEEP_RNN_BENCHMARK=1`` (legacy alias: ``MAGNAVIS_GRU_DEEP_BENCHMARK``)
    - Transformer (pre-trained only: load transformer_pretrained_*.keras + SinusoidalPositionEncoding)
    """
    def __init__(self, window_size=5, initial_train_points=3400,
                 epochs_per_update=20, learning_rate=0.001, update_training=True,
                 use_yearly_cycle=False, train_window_minutes=None,
                 model_family=MODEL_FAMILY_ATTN_BILSTM):

        self.window_size = window_size
        self.initial_train_points = initial_train_points
        self.epochs_per_update = epochs_per_update
        self.learning_rate = learning_rate
        self.update_training = update_training
        # Default to StandardScaler for new fits: MinMax on narrow geomagnetic windows squeezes
        # almost all mass into a tiny [0,1] band and encourages a near-constant predictor.
        # load_model() replaces this with whatever was pickled next to a checkpoint.
        self.scaler: object = StandardScaler()
        self.model = None
        self.use_yearly_cycle = use_yearly_cycle
        self.train_window_minutes = train_window_minutes
        self.model_family = _normalize_model_family(model_family)
        # When True, skip in-session model.fit() so pretrained transformer weights are not altered.
        self._skip_finetune_on_session: bool = False
        # GRU only: train/predict one-step delta in scaled magnitude (see _load_predictor_meta).
        self._gru_delta_y: bool = False

    def create_windowed_dataset(self, series, delta_y: bool = False):

        X, y = [], []
        for i in range(len(series) - self.window_size):
            X.append(series[i : i + self.window_size])
            nxt = float(series[i + self.window_size, 0])
            if delta_y:
                prev = float(series[i + self.window_size - 1, 0])
                y.append(nxt - prev)
            else:
                y.append(nxt)
        return np.array(X), np.array(y)

    def build_model(self, feature_dim):
        if self.model_family == MODEL_FAMILY_GRU:
            # Default: four stacked GRU layers (decreasing width); all but the last return full sequences.
            # MAGNAVIS_DEEP_RNN_BENCHMARK=1 (or legacy MAGNAVIS_GRU_DEEP_BENCHMARK): eight GRU + four dense.
            # Dropout between recurrent blocks (rate from PREDICTOR_GRU_DROPOUT, default 0.05).
            # Older gru_pretrained_*.keras with two GRU blocks will not load into this graph — retrain.
            dr = _parse_gru_dropout()
            if _deep_rnn_benchmark_enabled():
                layers_list = [Input(shape=(self.window_size, feature_dim))]
                _append_deep_rnn_stack_sequential(layers_list, GRU, dr)
                _append_deep_dense_head_sequential(layers_list)
                model = Sequential(layers_list)
            else:
                drop_layers = []
                if dr > 0:
                    drop_layers = [Dropout(dr), Dropout(dr)]
                layers_list = [
                    Input(shape=(self.window_size, feature_dim)),
                    GRU(48, return_sequences=True),
                ]
                if dr > 0:
                    layers_list.append(drop_layers[0])
                layers_list.extend(
                    [
                        GRU(32, return_sequences=True),
                    ]
                )
                if dr > 0:
                    layers_list.append(drop_layers[1])
                layers_list.extend(
                    [
                        GRU(24, return_sequences=True),
                        GRU(16),
                        Dense(16, activation="relu"),
                        Dense(1),
                    ]
                )
                model = Sequential(layers_list)
        elif self.model_family == MODEL_FAMILY_TRANSFORMER:
            raise ValueError(
                "Fresh training for MODEL_FAMILY_TRANSFORMER is not implemented. "
                "Use a pre-trained checkpoint (transformer_pretrained_*.keras) via PRETRAINED_MODEL_PATH."
            )
        elif self.model_family == MODEL_FAMILY_LSTM:
            # Default: three stacked LSTM layers + two dense. Deep benchmark: same 8+4 layout as GRU.
            dr = _parse_gru_dropout()
            if _deep_rnn_benchmark_enabled():
                layers_list = [Input(shape=(self.window_size, feature_dim))]
                _append_deep_rnn_stack_sequential(layers_list, LSTM, dr)
                _append_deep_dense_head_sequential(layers_list)
                model = Sequential(layers_list)
            else:
                layers_list = [
                    Input(shape=(self.window_size, feature_dim)),
                    LSTM(48, return_sequences=True),
                ]
                if dr > 0:
                    layers_list.append(Dropout(dr))
                layers_list.append(LSTM(32, return_sequences=True))
                if dr > 0:
                    layers_list.append(Dropout(dr))
                layers_list.extend(
                    [
                        LSTM(16),
                        Dense(16, activation="relu"),
                        Dense(1),
                    ]
                )
                model = Sequential(layers_list)
        elif self.model_family == MODEL_FAMILY_ATTN_BILSTM:
            inputs = Input(shape=(self.window_size, feature_dim))

            # 1. Attention Mechanism (Self-Attention)
            attn_out = Attention()([inputs, inputs])

            # 2. Layer Normalization
            norm_out = LayerNormalization()(attn_out)

            # 3. Bi-LSTM Layer
            bilstm_out = Bidirectional(LSTM(16, return_sequences=True))(norm_out)

            # 4. Residual Connection with Tanh
            projected_norm = Dense(32)(norm_out)
            res_out = Add()([bilstm_out, projected_norm])
            res_out = Activation("tanh")(res_out)

            # 5. Regression Output
            pooled_out = GlobalAveragePooling1D()(res_out)
            fc1 = Dense(16, activation="relu")(pooled_out)
            outputs = Dense(1)(fc1)

            model = Model(inputs=inputs, outputs=outputs)
        else:
            raise ValueError(f"Unsupported model_family for build_model: {self.model_family!r}")

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mean_squared_error")
        self.model = model
        self._skip_finetune_on_session = False

    def save_model(self, filepath):
        """Save the trained model weights and architecture to disk."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        self.model.save(filepath)
        # Also save scaler state for consistent feature scaling
        import pickle
        scaler_path = filepath.replace('.h5', '_scaler.pkl').replace('.keras', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        meta_path = _predictor_meta_path(filepath)
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "gru_delta_y": bool(getattr(self, "_gru_delta_y", False)),
                        "model_family": str(self.model_family),
                    },
                    f,
                )
        except Exception:
            pass

    def load_model(self, filepath):
        """Load a pre-trained model from disk."""
        custom_objects = {"SinusoidalPositionEncoding": SinusoidalPositionEncoding}
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False,
        )
        # Keras 3: weights loaded with compile=False are not trainable until compiled.
        # app.py often runs train+predict (fit on session data); compile enables fit/predict.
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error")
        # Load scaler state if available
        import pickle
        import os
        scaler_path = filepath.replace('.h5', '_scaler.pkl').replace('.keras', '_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        # Align sliding window with checkpoint input (timesteps dimension).
        self._sync_window_size_from_model()
        # A few Adam steps on the full CSV window often destabilizes these checkpoints; inference only.
        self._skip_finetune_on_session = self.model_family == MODEL_FAMILY_TRANSFORMER
        self._load_predictor_meta(filepath)

    def _load_predictor_meta(self, filepath):
        """Sidecar JSON: GRU Δ-target mode for runtime checkpoints (missing file => legacy absolute-y)."""
        self._gru_delta_y = False
        meta_path = _predictor_meta_path(filepath)
        if not os.path.isfile(meta_path):
            return
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            self._gru_delta_y = bool(m.get("gru_delta_y", False))
        except Exception:
            self._gru_delta_y = False

    def _sync_window_size_from_model(self):
        """Set window_size from the model's time dimension so reload/train matches the saved graph."""
        if self.model is None:
            return
        try:
            inp = self.model.input_shape
            if isinstance(inp, list):
                return
            if inp is not None and len(inp) >= 2 and inp[1] is not None:
                self.window_size = int(inp[1])
        except Exception:
            pass

    def _compute_time_features(self, ts_array):
        """Compute cyclic time features (daily, optionally yearly) for an array of pandas Timestamps."""
        seconds_in_day = 24 * 3600
        # Support both DatetimeIndex-like objects (have .hour/.minute/...) and pandas Series (use .dt accessor).
        ts_array = pd.to_datetime(ts_array)
        if isinstance(ts_array, pd.Series):
            hour = ts_array.dt.hour
            minute = ts_array.dt.minute
            second = ts_array.dt.second
            microsecond = ts_array.dt.microsecond
            dayofyear = ts_array.dt.dayofyear
        else:
            hour = ts_array.hour
            minute = ts_array.minute
            second = ts_array.second
            microsecond = ts_array.microsecond
            dayofyear = ts_array.dayofyear

        sec_of_day = hour * 3600 + minute * 60 + second + microsecond / 1e6
        day_angle = 2 * np.pi * (sec_of_day / seconds_in_day)
        sin_day = np.sin(day_angle)
        cos_day = np.cos(day_angle)

        if self.use_yearly_cycle:
            # Day-of-year fraction (including leap-year effect approximately)
            year_angle = 2 * np.pi * (dayofyear / 365.25)
            sin_year = np.sin(year_angle)
            cos_year = np.cos(year_angle)
            return sin_day, cos_day, sin_year, cos_year
        else:
            return sin_day, cos_day

    def forecast(self, timestamps, field_data, n_future, pretrained_model_path=None):
        leading_m = _parse_leading_train_minutes()
        skip_m_raw = _parse_skip_initial_minutes_predictor()
        force_one_step = _parse_predict_on_input_timestamps()
        ar_closed_loop = _parse_ar_closed_loop()
        # Leading-train uses the full timeline from t0; skip-head is for pretrained-only paths.
        skip_m = 0.0 if leading_m > 0 else skip_m_raw
        use_one_step = force_one_step or (leading_m > 0.0) or (skip_m > 0.0)
        if ar_closed_loop and not use_one_step:
            raise ValueError("PREDICTOR_AR_CLOSED_LOOP requires one-step mode (PREDICTOR_ON_INPUT_TIMESTAMPS=1).")

        # Build a clean time series table first so timestamps and values stay aligned.
        ts = pd.to_datetime(timestamps)
        df0 = pd.DataFrame({"ts": ts, "field": field_data})
        df0["ts"] = pd.to_datetime(df0["ts"], errors="coerce")
        df0["field"] = pd.to_numeric(df0["field"], errors="coerce")
        df0 = df0.dropna(subset=["ts", "field"]).sort_values("ts").reset_index(drop=True)

        # Defensive: if duplicate timestamps exist, average them so:
        # - time_delta is non-zero
        # - feature arrays and target arrays have consistent lengths
        df0 = df0.groupby("ts", as_index=False)["field"].mean().sort_values("ts").reset_index(drop=True)

        if skip_m > 0:
            t0 = pd.Timestamp(df0["ts"].iloc[0])
            df0 = df0[df0["ts"] >= t0 + pd.Timedelta(minutes=float(skip_m))].reset_index(drop=True)

        ts = df0["ts"]  # pandas Series of datetime64
        field = df0["field"].to_numpy(dtype=float).reshape(-1, 1)

        # Optional: restrict training data to the most recent N minutes
        if self.train_window_minutes:
            cutoff = ts.max().to_pydatetime() - timedelta(minutes=self.train_window_minutes)
            mask = ts >= cutoff
            if int(mask.sum()) > self.window_size + 1:
                ts = ts[mask].reset_index(drop=True)
                field = field[mask.to_numpy()]

        n_pts = int(len(ts))
        if n_pts < self.window_size + 1:
            raise ValueError(
                f"Not enough points for window_size={self.window_size} (have {n_pts} after trim/skip)."
            )

        # Load pre-trained model and scaler BEFORE scaling, so we use the same scaler as training.
        # Otherwise we would fit_transform on short input and break pre-trained predictions.
        if pretrained_model_path:
            try:
                self.load_model(pretrained_model_path)
                print(f"Using pre-trained model and scaler: {pretrained_model_path}")
            except Exception as e:
                print(f"Warning: Could not load pre-trained model from {pretrained_model_path}: {e}")
                print("Falling back to building new model.")
                self.model = None  # ensure we fit scaler and build model below

        if self.model is not None and (_parse_skip_finetune_on_session_env() or skip_m > 0):
            # Predict-only on loaded weights: no session fit (benchmark 0-min historic or skip-head burn-in).
            self._skip_finetune_on_session = True

        # Loaded checkpoints: meta sidecar sets Δ-target; explicit env can override for ablations.
        if self.model is not None and self.model_family == MODEL_FAMILY_GRU:
            if os.environ.get("PREDICTOR_GRU_DELTA_TARGET", "").strip():
                self._gru_delta_y = _parse_gru_delta_target()

        # Fresh build path: choose GRU Δ-target before scaler fit (loaded checkpoints set this in load_model).
        if self.model is None:
            if self.model_family == MODEL_FAMILY_GRU:
                self._gru_delta_y = _parse_gru_delta_target()
            elif self.model_family == MODEL_FAMILY_LSTM:
                self._gru_delta_y = False
            else:
                self._gru_delta_y = False

        split_ix = -1
        if leading_m > 0:
            bound = pd.Timestamp(ts.iloc[0]) + pd.Timedelta(minutes=float(leading_m))
            split_ix = _last_index_at_or_before(ts, bound)
            if split_ix < self.window_size:
                raise ValueError(
                    f"PREDICTOR_LEADING_TRAIN_MINUTES={leading_m} leaves only split_ix={split_ix} points "
                    f"before the boundary; need at least window_size={self.window_size} for training."
                )

        # Scale magnetic field only; sin/cos are already bounded in [-1, 1]
        # Pre-trained: use loaded scaler (transform). Fresh + leading: fit scaler on training head only.
        if self.model is not None:
            field_scaled = self.scaler.transform(field).flatten()
        else:
            if leading_m > 0 and split_ix >= 0:
                self.scaler.fit(field[: split_ix + 1])
                field_scaled = self.scaler.transform(field).flatten()
            else:
                field_scaled = self.scaler.fit_transform(field).flatten()

        # Build feature matrix: [mag_scaled, sin_time, cos_time, (optional) sin_year, cos_year]
        time_feats = self._compute_time_features(ts)
        if self.use_yearly_cycle:
            sin_day, cos_day, sin_year, cos_year = time_feats
            feature_matrix = np.column_stack([field_scaled, sin_day, cos_day, sin_year, cos_year])
        else:
            sin_day, cos_day = time_feats
            feature_matrix = np.column_stack([field_scaled, sin_day, cos_day])

        # Median timestep for autoregressive extension (shared by one-step tail + legacy path).
        try:
            if len(ts) > 1:
                diffs = pd.Series(ts).diff().dropna()
                diffs = diffs[diffs > pd.Timedelta(0)]
                if len(diffs) > 0:
                    time_delta = diffs.median().to_pytimedelta()
                else:
                    time_delta = timedelta(seconds=1)
            else:
                time_delta = timedelta(seconds=1)
            if time_delta.total_seconds() <= 0:
                time_delta = timedelta(seconds=1)
        except Exception:
            time_delta = timedelta(seconds=1)

        # Legacy autoregressive path uses initial_train_points as the pivot index.
        self.initial_train_points = min(self.initial_train_points, n_pts)
        if self.initial_train_points < self.window_size:
            raise ValueError("initial_train_points must be >= window_size.")
        if self.initial_train_points > n_pts:
            raise ValueError("initial_train_points cannot exceed total data length.")

        # Build model only if we don't have one (pre-trained load failed or not requested)
        if self.model is None:
            if not self.update_training:
                raise ValueError(
                    "Prediction-only mode requested but no trained model is available. "
                    "Provide PRETRAINED_MODEL_PATH or run training first."
                )
            if self.model_family == MODEL_FAMILY_TRANSFORMER:
                raise ValueError(
                    "Transformer predictor could not load PRETRAINED_MODEL_PATH. "
                    "Check that models/transformer_pretrained_<SENSOR>.keras exists and matches this code version."
                )
            self.build_model(feature_matrix.shape[1])
            if self.model_family == MODEL_FAMILY_GRU:
                print(
                    f"GRU training target: {'Δ(scaled magnitude)' if self._gru_delta_y else 'absolute scaled magnitude'}"
                )
            elif self.model_family == MODEL_FAMILY_LSTM:
                print("LSTM training target: absolute scaled magnitude")

        def _inverse_one(next_mag_scaled: float) -> float:
            inv = np.asarray(next_mag_scaled, dtype=np.float64).reshape(1, 1)
            return float(self.scaler.inverse_transform(inv)[0, 0])

        def _next_mag_scaled_from_window(current_window: np.ndarray) -> float:
            raw_head = self.model.predict(np.array([current_window]), verbose=0)[0, 0]
            raw_head = float(np.asarray(raw_head, dtype=np.float64).reshape(-1)[0])
            if self._gru_delta_y:
                last_mag_scaled = float(current_window[-1, 0])
                return last_mag_scaled + raw_head
            return raw_head

        def _predict_from_window(current_window: np.ndarray) -> float:
            return _inverse_one(_next_mag_scaled_from_window(current_window))

        def _feature_row_from_time(next_mag_scaled: float, t_py) -> np.ndarray:
            sec_of_day = (
                t_py.hour * 3600 + t_py.minute * 60 + t_py.second + t_py.microsecond / 1e6
            )
            day_angle = 2 * np.pi * (sec_of_day / (24 * 3600))
            sin_day_new = np.sin(day_angle)
            cos_day_new = np.cos(day_angle)
            if self.use_yearly_cycle:
                day_of_year = t_py.timetuple().tm_yday
                year_angle = 2 * np.pi * (day_of_year / 365.25)
                sin_year_new = np.sin(year_angle)
                cos_year_new = np.cos(year_angle)
                return np.array([next_mag_scaled, sin_day_new, cos_day_new, sin_year_new, cos_year_new])
            return np.array([next_mag_scaled, sin_day_new, cos_day_new])

        # --- One-step-ahead on actual timestamps (leading-train or skip-head pretrained) ---
        if use_one_step:
            if ar_closed_loop:
                print(
                    "Predictor one-step mode: closed-loop AR window (bootstrap from actual, then roll with predictions).",
                    flush=True,
                )
            if leading_m > 0:
                train_slice = feature_matrix[: split_ix + 1]
                X_init, y_init = self.create_windowed_dataset(train_slice, delta_y=self._gru_delta_y)
                if self.update_training and len(X_init) > 0 and not getattr(self, "_skip_finetune_on_session", False):
                    self.model.fit(X_init, y_init, epochs=self.epochs_per_update, verbose=0)
                j_start = split_ix + 1
            else:
                j_start = self.window_size

            if j_start > n_pts - 1:
                return np.array([]), np.array([])

            pred_ts: list = []
            pred_y: list = []
            ar_roll_window = None
            if ar_closed_loop:
                current_window = np.array(
                    feature_matrix[j_start - self.window_size : j_start], copy=True
                )
                for j in range(j_start, n_pts):
                    next_mag_scaled = _next_mag_scaled_from_window(current_window)
                    pred_ts.append(ts.iloc[j])
                    pred_y.append(_inverse_one(next_mag_scaled))
                    t_py = (
                        ts.iloc[j].to_pydatetime()
                        if hasattr(ts, "iloc")
                        else pd.to_datetime(ts[j]).to_pydatetime()
                    )
                    new_feature = _feature_row_from_time(next_mag_scaled, t_py)
                    current_window = np.concatenate(
                        [current_window[1:], new_feature[np.newaxis, :]], axis=0
                    )
                ar_roll_window = current_window
            else:
                for j in range(j_start, n_pts):
                    win = feature_matrix[j - self.window_size : j]
                    pred_ts.append(ts.iloc[j])
                    pred_y.append(_predict_from_window(win))

            # One-step outputs only at CSV row times. Sequential catch-up caps the CSV at
            # latest_pred, so the last row is the last covered time — extend with autoregression
            # until PREDICTOR_COVER_UNTIL (latest actual) when the app sets it.
            cover_ts = _parse_cover_until()
            if cover_ts is not None and pred_ts:
                last_emit = pd.Timestamp(pred_ts[-1])
                if last_emit < cover_ts:
                    td_sec = max(time_delta.total_seconds(), 1e-9)
                    need = int(math.ceil((cover_ts - last_emit).total_seconds() / td_sec)) + 3
                    max_steps = min(200000, max(n_future, need))
                    if ar_roll_window is not None:
                        current_window = np.array(ar_roll_window, copy=True)
                    else:
                        current_window = np.array(feature_matrix[n_pts - self.window_size : n_pts], copy=True)
                    dt_last_py = (
                        ts.iloc[-1].to_pydatetime()
                        if hasattr(ts, "iloc")
                        else pd.to_datetime(ts[-1]).to_pydatetime()
                    )
                    online_fit_each_step = _parse_online_fit_each_step()
                    training_data = (
                        feature_matrix[: self.initial_train_points].copy()
                        if self.update_training and online_fit_each_step
                        else None
                    )
                    for i in range(max_steps):
                        current_window_reshaped = np.array([current_window])
                        raw_head = self.model.predict(current_window_reshaped, verbose=0)[0, 0]
                        raw_head = float(np.asarray(raw_head, dtype=np.float64).reshape(-1)[0])
                        if self._gru_delta_y:
                            last_mag_scaled = float(current_window[-1, 0])
                            next_mag_scaled = last_mag_scaled + raw_head
                        else:
                            next_mag_scaled = raw_head
                        predicted_value = _inverse_one(next_mag_scaled)

                        new_time = dt_last_py + (i + 1) * time_delta
                        sec_of_day = (
                            new_time.hour * 3600
                            + new_time.minute * 60
                            + new_time.second
                            + new_time.microsecond / 1e6
                        )
                        day_angle = 2 * np.pi * (sec_of_day / (24 * 3600))
                        sin_day_new = np.sin(day_angle)
                        cos_day_new = np.cos(day_angle)
                        if self.use_yearly_cycle:
                            day_of_year = new_time.timetuple().tm_yday
                            year_angle = 2 * np.pi * (day_of_year / 365.25)
                            sin_year_new = np.sin(year_angle)
                            cos_year_new = np.cos(year_angle)
                            new_feature = np.array(
                                [next_mag_scaled, sin_day_new, cos_day_new, sin_year_new, cos_year_new]
                            )
                        else:
                            new_feature = np.array([next_mag_scaled, sin_day_new, cos_day_new])

                        pred_ts.append(new_time)
                        pred_y.append(predicted_value)
                        current_window = np.concatenate([current_window[1:], new_feature[np.newaxis, :]], axis=0)

                        if self.update_training and online_fit_each_step and not getattr(
                            self, "_skip_finetune_on_session", False
                        ):
                            training_data = np.concatenate([training_data, new_feature[np.newaxis, :]], axis=0)
                            X_train, y_train = self.create_windowed_dataset(training_data, delta_y=self._gru_delta_y)
                            self.model.fit(X_train, y_train, epochs=self.epochs_per_update, verbose=0)

                        if pd.Timestamp(new_time) >= cover_ts:
                            break

            return np.array(pred_ts), np.array(pred_y)

        # --- Legacy: autoregressive n_future steps beyond the last input timestamp ---
        initial_data = feature_matrix[: self.initial_train_points]
        X_init, y_init = self.create_windowed_dataset(initial_data, delta_y=self._gru_delta_y)
        if self.update_training and len(X_init) > 0 and not getattr(self, "_skip_finetune_on_session", False):
            self.model.fit(X_init, y_init, epochs=self.epochs_per_update, verbose=0)

        try:
            dt_last = ts.iloc[-1].to_pydatetime() if hasattr(ts, "iloc") else pd.to_datetime(ts[-1]).to_pydatetime()
        except Exception as e:
            raise ValueError("Error parsing timestamps for autoregressive forecast.") from e

        predictions = []
        future_timestamps = []
        current_window = feature_matrix[self.initial_train_points - self.window_size : self.initial_train_points]

        online_fit_each_step = _parse_online_fit_each_step()
        if self.update_training and online_fit_each_step:
            training_data = initial_data.copy()

        for i in range(n_future):
            current_window_reshaped = np.array([current_window])
            raw_head = self.model.predict(current_window_reshaped, verbose=0)[0, 0]
            raw_head = float(np.asarray(raw_head, dtype=np.float64).reshape(-1)[0])
            if self._gru_delta_y:
                last_mag_scaled = float(current_window[-1, 0])
                next_mag_scaled = last_mag_scaled + raw_head
            else:
                next_mag_scaled = raw_head
            predicted_value = _inverse_one(next_mag_scaled)
            predictions.append(predicted_value)

            new_time = dt_last + (i + 1) * time_delta
            future_timestamps.append(new_time)

            sec_of_day = new_time.hour * 3600 + new_time.minute * 60 + new_time.second + new_time.microsecond / 1e6
            day_angle = 2 * np.pi * (sec_of_day / (24 * 3600))
            sin_day_new = np.sin(day_angle)
            cos_day_new = np.cos(day_angle)

            if self.use_yearly_cycle:
                day_of_year = new_time.timetuple().tm_yday
                year_angle = 2 * np.pi * (day_of_year / 365.25)
                sin_year_new = np.sin(year_angle)
                cos_year_new = np.cos(year_angle)
                new_feature = np.array([next_mag_scaled, sin_day_new, cos_day_new, sin_year_new, cos_year_new])
            else:
                new_feature = np.array([next_mag_scaled, sin_day_new, cos_day_new])

            current_window = np.concatenate([current_window[1:], new_feature[np.newaxis, :]], axis=0)

            if self.update_training and online_fit_each_step and not getattr(self, "_skip_finetune_on_session", False):
                training_data = np.concatenate([training_data, new_feature[np.newaxis, :]], axis=0)
                X_train, y_train = self.create_windowed_dataset(training_data, delta_y=self._gru_delta_y)
                self.model.fit(X_train, y_train, epochs=self.epochs_per_update, verbose=0)

        # One-step path handles COVER_UNTIL inside its branch; legacy autoregression must extend too
        # when ApplicationTemp caps predict_input and sets PREDICTOR_COVER_UNTIL (sequential catch-up).
        cover_ts = _parse_cover_until()
        if cover_ts is not None and future_timestamps:
            last_emit = pd.Timestamp(future_timestamps[-1])
            if last_emit < cover_ts:
                td_sec = max(time_delta.total_seconds(), 1e-9)
                need = int(math.ceil((cover_ts - last_emit).total_seconds() / td_sec)) + 3
                max_steps = min(200000, max(n_future, need))
                origin = last_emit.to_pydatetime()
                ext_training_data = None
                if self.update_training and online_fit_each_step and not getattr(
                    self, "_skip_finetune_on_session", False
                ):
                    ext_training_data = training_data
                for i in range(max_steps):
                    current_window_reshaped = np.array([current_window])
                    raw_head = self.model.predict(current_window_reshaped, verbose=0)[0, 0]
                    raw_head = float(np.asarray(raw_head, dtype=np.float64).reshape(-1)[0])
                    if self._gru_delta_y:
                        last_mag_scaled = float(current_window[-1, 0])
                        next_mag_scaled = last_mag_scaled + raw_head
                    else:
                        next_mag_scaled = raw_head
                    predicted_value = _inverse_one(next_mag_scaled)

                    new_time = origin + (i + 1) * time_delta
                    sec_of_day = (
                        new_time.hour * 3600
                        + new_time.minute * 60
                        + new_time.second
                        + new_time.microsecond / 1e6
                    )
                    day_angle = 2 * np.pi * (sec_of_day / (24 * 3600))
                    sin_day_new = np.sin(day_angle)
                    cos_day_new = np.cos(day_angle)
                    if self.use_yearly_cycle:
                        day_of_year = new_time.timetuple().tm_yday
                        year_angle = 2 * np.pi * (day_of_year / 365.25)
                        sin_year_new = np.sin(year_angle)
                        cos_year_new = np.cos(year_angle)
                        new_feature = np.array(
                            [next_mag_scaled, sin_day_new, cos_day_new, sin_year_new, cos_year_new]
                        )
                    else:
                        new_feature = np.array([next_mag_scaled, sin_day_new, cos_day_new])

                    predictions.append(predicted_value)
                    future_timestamps.append(new_time)
                    current_window = np.concatenate([current_window[1:], new_feature[np.newaxis, :]], axis=0)

                    if ext_training_data is not None:
                        ext_training_data = np.concatenate([ext_training_data, new_feature[np.newaxis, :]], axis=0)
                        X_train, y_train = self.create_windowed_dataset(ext_training_data, delta_y=self._gru_delta_y)
                        self.model.fit(X_train, y_train, epochs=self.epochs_per_update, verbose=0)

                    if pd.Timestamp(new_time) >= cover_ts:
                        break

        return np.array(future_timestamps), np.array(predictions)

if __name__ == "__main__":

    # start_time = datetime.strptime("01012023000000", "%d%m%Y%H%M%S")
    # timestamps = [(start_time + timedelta(seconds=i)) for i in range(10000)] #.strftime("%d%m%Y%H%M%S") for i in range(10000)]

    # t_numeric = np.arange(10000)
    # field_data = np.sin(0.001 * t_numeric) + 0.05 * np.random.randn(len(t_numeric))

    # read from magdata
    file = sys.argv[1] #r'C:\Users\DELL\Desktop\Projects\quantum\magnavis\src\sessions\c5763b7d-cd79-4bdc-a9b1-c8b8e753e9e7\predict_input.csv'
    print('filein', file)
    # predictor.(train_data)
    df_in = pd.read_csv(file)

    train_window_minutes = _parse_train_window_minutes()
    model_family = _parse_model_family()
    update_training = _parse_update_training()
    checkpoint_path = _parse_checkpoint_path()
    # Optional: path to pre-trained model (set via env var PRETRAINED_MODEL_PATH)
    pretrained_model_path = os.environ.get("PRETRAINED_MODEL_PATH", None)
    gru_window = _parse_predictor_gru_window_size(15)
    epochs_pu = _parse_epochs_per_update()
    lr = _parse_learning_rate(model_family)

    predictor = AttnBiLSTMPredictor(
        window_size=gru_window,
        initial_train_points=len(df_in),
        epochs_per_update=epochs_pu,
        learning_rate=lr,
        update_training=update_training,
        train_window_minutes=train_window_minutes,
        use_yearly_cycle=True,
        model_family=model_family,
    )  # Enable yearly cycle for seasonal patterns
    print(f"Predictor model family: {model_family}")
    if model_family == MODEL_FAMILY_GRU:
        print(f"GRU sequence window W: {predictor.window_size}")
    elif model_family == MODEL_FAMILY_LSTM:
        print(f"LSTM sequence window W: {predictor.window_size}")
    print(f"Predictor run mode: {'train+predict' if update_training else 'predict-only'}")
    print(f"Epochs per update: {epochs_pu}")
    print(f"Learning rate: {lr}")

    df_in['x'] = pd.to_datetime(df_in['x'])
    print('input head for predict', df_in.head())
    timestamps = df_in['x'].to_list()
    field_data = df_in['y'].to_list()
    n_future = _parse_n_future()
    future_times, future_predictions = predictor.forecast(
        timestamps, field_data, n_future=n_future, pretrained_model_path=pretrained_model_path
    )
    if checkpoint_path and predictor.model is not None:
        # Save only when training happened (or if checkpoint does not exist yet).
        if update_training or (not os.path.exists(checkpoint_path)):
            predictor.save_model(checkpoint_path)
            print(f"Saved runtime checkpoint: {checkpoint_path}")
    df_out = pd.DataFrame({'x': future_times, 'y': future_predictions})
    folder = os.path.dirname(file)
    df_out.to_csv(os.path.join(folder, 'predict_out.csv'), index=False)
    # print("Future Timestamps:", future_times)
    # print("Future Magnetic Field Predictions:", future_predictions)
