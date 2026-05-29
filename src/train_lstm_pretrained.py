"""
Pre-training script for vanilla stacked LSTM models on extended magnetic field data.

Mirrors ``train_gru_pretrained.py`` but uses ``MODEL_FAMILY_LSTM`` in
``predictor_ai.AttnBiLSTMPredictor`` and writes:

  lstm_pretrained_<SENSOR>.keras
  lstm_pretrained_<SENSOR>_scaler.pkl

Usage::

  python src/train_lstm_pretrained.py "file1.csv,file2.csv" models/ --epochs 50 \\
    --sensors OBS1_1 OBS1_2 OBS1_3 OBS2_1 OBS2_2 OBS2_3

See ``docs/LSTM_PRETRAINED_MODELS.md`` for runtime loading in the app.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from predictor_ai import AttnBiLSTMPredictor, MODEL_FAMILY_LSTM

OBS_SUFFIX_PATTERN = re.compile(r"(OBS\d+_\d+)$")


def _canonical_sensor_id(sensor_id):
    sid = str(sensor_id).strip()
    m = OBS_SUFFIX_PATTERN.search(sid)
    if m:
        return m.group(1)
    m = re.search(r"(OBS\d+_\d+)", sid)
    if m:
        return m.group(1)
    return sid


def load_magnetic_data_by_sensor(csv_paths):
    """Load magnetic field data from CSV files, grouped by canonical sensor_id (same as GRU trainer)."""
    sensor_data: dict[str, list] = {}

    for csv_path in csv_paths:
        print(f"Loading {csv_path}...")
        try:
            df = pd.read_csv(
                csv_path,
                usecols=lambda x: x
                in ["x", "y", "timestamp", "mag_H_nT", "sensor_id", "b_x", "b_y", "b_z"],
            )
        except Exception as e:
            print(f"Warning: Could not read {csv_path}: {e}")
            continue

        if df.empty:
            print(f"Warning: {csv_path} is empty, skipping.")
            continue

        if "x" in df.columns and "y" in df.columns:
            sensor_id = _canonical_sensor_id(os.path.basename(csv_path).replace(".csv", ""))
            ts = pd.to_datetime(df["x"]).tolist()
            mag = df["y"].astype(float).tolist()
            sensor_data.setdefault(sensor_id, []).extend(list(zip(ts, mag)))

        elif "timestamp" in df.columns and "mag_H_nT" in df.columns:
            sensor_id = _canonical_sensor_id(os.path.basename(csv_path).replace(".csv", ""))
            ts = pd.to_datetime(df["timestamp"]).tolist()
            mag = df["mag_H_nT"].astype(float).tolist()
            sensor_data.setdefault(sensor_id, []).extend(list(zip(ts, mag)))

        elif all(c in df.columns for c in ["sensor_id", "timestamp", "b_x", "b_y", "b_z"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["mag_total_nT"] = np.sqrt(df["b_x"] ** 2 + df["b_y"] ** 2 + df["b_z"] ** 2)

            for raw_sensor_id, group in df.groupby("sensor_id"):
                sensor_id = _canonical_sensor_id(raw_sensor_id)
                group_sorted = group.sort_values("timestamp")
                group_agg = group_sorted.groupby("timestamp", as_index=False)["mag_total_nT"].mean()
                ts = group_agg["timestamp"].tolist()
                mag = group_agg["mag_total_nT"].astype(float).tolist()
                sensor_data.setdefault(sensor_id, []).extend(list(zip(ts, mag)))

        else:
            print(f"Warning: {csv_path} doesn't have expected columns, skipping.")
            continue

    if not sensor_data:
        raise ValueError("No valid data loaded from any CSV files.")

    processed_sensor_data = {}
    for sensor_id, data_list in sensor_data.items():
        df_sensor = pd.DataFrame(data_list, columns=["ts", "field"])
        df_sensor = df_sensor.sort_values("ts").reset_index(drop=True)
        df_sensor = df_sensor.drop_duplicates(subset=["ts"], keep="first")

        timestamps = df_sensor["ts"].tolist()
        field_data = df_sensor["field"].tolist()

        processed_sensor_data[sensor_id] = (timestamps, field_data)
        print(f"  Sensor {sensor_id}: {len(timestamps)} points, range {timestamps[0]} to {timestamps[-1]}")

    return processed_sensor_data


def load_magnetic_data_by_csvs(csv_paths):
    return load_magnetic_data_by_sensor(csv_paths)


def _train_single_sensor_model(
    sensor_id,
    timestamps,
    field_data,
    output_model_dir,
    epochs,
    use_yearly_cycle,
    window_size,
    learning_rate,
    batch_size,
):
    predictor = AttnBiLSTMPredictor(
        window_size=window_size,
        initial_train_points=len(field_data),
        epochs_per_update=epochs,
        learning_rate=learning_rate,
        update_training=False,
        use_yearly_cycle=use_yearly_cycle,
        train_window_minutes=None,
        model_family=MODEL_FAMILY_LSTM,
    )

    print("Building features and training model...")
    ts = pd.to_datetime(timestamps)
    df0 = pd.DataFrame({"ts": ts, "field": field_data})
    df0 = df0.dropna(subset=["ts", "field"]).sort_values("ts").reset_index(drop=True)

    ts = df0["ts"]
    field = df0["field"].to_numpy(dtype=float).reshape(-1, 1)

    field_scaled = predictor.scaler.fit_transform(field).flatten()
    time_feats = predictor._compute_time_features(ts)

    if use_yearly_cycle:
        sin_day, cos_day, sin_year, cos_year = time_feats
        feature_matrix = np.column_stack([field_scaled, sin_day, cos_day, sin_year, cos_year])
    else:
        sin_day, cos_day = time_feats
        feature_matrix = np.column_stack([field_scaled, sin_day, cos_day])

    predictor.build_model(feature_matrix.shape[1])

    X_train, y_train = predictor.create_windowed_dataset(feature_matrix)
    print(f"Training on {len(X_train)} samples (window_size={window_size})")

    print(f"Training for {epochs} epochs...")
    history = predictor.model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )

    safe_sensor_id = sensor_id.replace("/", "_").replace("\\", "_")
    hist_dir = os.path.join(output_model_dir, "training_histories")
    os.makedirs(hist_dir, exist_ok=True)
    hist_path = os.path.join(hist_dir, f"lstm_pretrained_{safe_sensor_id}_fit_history.json")
    serializable = {
        k: [float(x) for x in (v if isinstance(v, list) else list(v))] for k, v in history.history.items()
    }
    with open(hist_path, "w", encoding="utf-8") as hf:
        json.dump(
            {
                "family": "lstm",
                "sensor_id": sensor_id,
                "checkpoint_stem": f"lstm_pretrained_{safe_sensor_id}",
                "epochs_requested": int(epochs),
                "history": serializable,
            },
            hf,
            indent=2,
        )
    print(f"Training history JSON: {hist_path}")

    output_model_path = os.path.join(output_model_dir, f"lstm_pretrained_{safe_sensor_id}.keras")
    predictor.save_model(output_model_path)
    print(f"\nModel saved to: {output_model_path}")

    final_loss = history.history["loss"][-1]
    if "val_loss" in history.history:
        print(f"Final training loss: {final_loss:.6f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
    else:
        print(f"Final training loss: {final_loss:.6f}")


def train_pretrained_model(
    csv_paths,
    output_model_dir,
    epochs=50,
    use_yearly_cycle=True,
    window_size=15,
    learning_rate=0.001,
    batch_size=32,
    sensor_filter=None,
):
    sensor_data = load_magnetic_data_by_csvs(csv_paths)

    if sensor_filter:
        sensor_data = {
            sid: data for sid, data in sensor_data.items() if any(filt in sid for filt in sensor_filter)
        }

    if not sensor_data:
        raise ValueError("No sensor data found after filtering.")

    print(f"\nTraining LSTM models for {len(sensor_data)} sensor(s)...")
    os.makedirs(output_model_dir, exist_ok=True)

    trained_models = []
    for sensor_id, (timestamps, field_data) in sensor_data.items():
        print(f"\n{'=' * 60}")
        print(f"Training LSTM for sensor: {sensor_id}")
        print(f"{'=' * 60}")
        try:
            _train_single_sensor_model(
                sensor_id=sensor_id,
                timestamps=timestamps,
                field_data=field_data,
                output_model_dir=output_model_dir,
                epochs=epochs,
                use_yearly_cycle=use_yearly_cycle,
                window_size=window_size,
                learning_rate=learning_rate,
                batch_size=batch_size,
            )
            trained_models.append(sensor_id)
        except Exception as e:
            print(f"Error training model for {sensor_id}: {e}")
            continue

    print(f"\n{'=' * 60}")
    print(f"LSTM training complete! Trained {len(trained_models)} model(s):")
    for sid in trained_models:
        p = os.path.join(output_model_dir, f"lstm_pretrained_{sid}.keras")
        print(f"  - {sid}: {p}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train LSTM models on extended magnetic field data")
    parser.add_argument(
        "input_csv",
        nargs="?",
        help="CSV path(s): comma-separated list or glob",
    )
    parser.add_argument("output_model_dir", help="Directory for per-sensor checkpoints")
    parser.add_argument("--folder", help="Folder of CSVs (alternative to input_csv)")
    parser.add_argument(
        "--sensors",
        nargs="+",
        help="Only these canonical sensors, e.g. OBS1_1 OBS2_3",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per sensor (default: 50)")
    parser.add_argument("--use-yearly-cycle", action="store_true", default=True)
    parser.add_argument("--no-yearly-cycle", dest="use_yearly_cycle", action="store_false")
    parser.add_argument("--window-size", type=int, default=15, help="Sequence length W (default: 15)")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    if args.folder:
        folder_path = args.folder
        csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        if not csv_files:
            csv_files = sorted(glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True))
        if not csv_files:
            print(f"Error: No CSV files found in folder '{folder_path}'")
            sys.exit(1)
        print(f"Found {len(csv_files)} CSV files in folder '{folder_path}'")
    elif args.input_csv:
        if "," in args.input_csv:
            csv_files = [f.strip() for f in args.input_csv.split(",")]
        else:
            csv_files = sorted(glob.glob(args.input_csv))
            if not csv_files:
                csv_files = [args.input_csv]
        csv_files = sorted(csv_files)
    else:
        parser.error("Either --folder or input_csv must be provided")

    print(f"Will load data from {len(csv_files)} file(s):")
    for f in csv_files:
        print(f"  - {f}")

    train_pretrained_model(
        csv_paths=csv_files,
        output_model_dir=args.output_model_dir,
        epochs=args.epochs,
        use_yearly_cycle=args.use_yearly_cycle,
        window_size=args.window_size,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        sensor_filter=args.sensors,
    )
