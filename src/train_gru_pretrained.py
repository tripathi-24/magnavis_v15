"""
Pre-training script for GRU models on extended magnetic field data (e.g., 2-4 months).

This script trains GRU models on extended historical data to learn daily, weekly,
and seasonal patterns. The trained model can then be used for faster, more accurate
predictions in the main application.

Usage:
    # Single CSV file:
    python train_gru_pretrained.py <input_csv> <output_model_path> [options]
    
    # Multiple CSV files (comma-separated):
    python train_gru_pretrained.py file1.csv,file2.csv,file3.csv <output_model_path> [options]
    
    # Folder containing CSV files:
    python train_gru_pretrained.py --folder "Large Files" <output_model_path> [options]

Example:
    python train_gru_pretrained.py "Large Files/magnetic_data_*.csv" models/attn_bilstm_pretrained.keras --epochs 50
    python train_gru_pretrained.py --folder "Large Files" models/attn_bilstm_pretrained.keras --epochs 50
"""

import os
import sys
import argparse
import glob
import json
import re
import numpy as np
import pandas as pd
from predictor_ai import AttnBiLSTMPredictor, MODEL_FAMILY_GRU

OBS_SUFFIX_PATTERN = re.compile(r"(OBS\d+_\d+)$")


def _canonical_sensor_id(sensor_id):
    """
    Normalize dynamic sensor IDs to a canonical OBS suffix when available.

    Examples:
      - S20260202_124213_OBS1_1 -> OBS1_1
      - S20260217_094000_OBS2_3 -> OBS2_3
    """
    sid = str(sensor_id).strip()
    m = OBS_SUFFIX_PATTERN.search(sid)
    if m:
        return m.group(1)
    # Fallback for malformed variants that still contain OBSx_y somewhere
    m = re.search(r"(OBS\d+_\d+)", sid)
    if m:
        return m.group(1)
    return sid


def load_magnetic_data_by_sensor(csv_paths):
    """
    Load magnetic field data from CSV files, grouped by sensor_id.
    
    Handles CSV files with either:
    - Columns: ('x', 'y') for time series format (single sensor)
    - Columns: ('timestamp', 'mag_H_nT') for processed format (single sensor)
    - Columns: ('sensor_id', 'timestamp', 'b_x', 'b_y', 'b_z') for raw format (multiple sensors)
    
    For raw format, computes magnitude: sqrt(b_x^2 + b_y^2 + b_z^2) per sensor.
    
    Parameters:
    -----------
    csv_paths : list of str
        List of CSV file paths to load
        
    Returns:
    --------
    sensor_data : dict
        Dictionary mapping sensor_id -> (timestamps, field_data)
        where timestamps is list of datetime and field_data is list of float (nT)
    """
    sensor_data = {}  # sensor_id -> list of (timestamp, magnitude) tuples
    
    for csv_path in csv_paths:
        print(f"Loading {csv_path}...")
        try:
            df = pd.read_csv(csv_path, usecols=lambda x: x in ['x', 'y', 'timestamp', 'mag_H_nT', 'sensor_id', 'b_x', 'b_y', 'b_z'])
        except Exception as e:
            print(f"Warning: Could not read {csv_path}: {e}")
            continue
        
        if df.empty:
            print(f"Warning: {csv_path} is empty, skipping.")
            continue
        
        # Case 1: Time series format (x, y) - single sensor, use filename or default
        if 'x' in df.columns and 'y' in df.columns:
            sensor_id = _canonical_sensor_id(os.path.basename(csv_path).replace('.csv', ''))
            ts = pd.to_datetime(df['x']).tolist()
            mag = df['y'].astype(float).tolist()
            if sensor_id not in sensor_data:
                sensor_data[sensor_id] = []
            sensor_data[sensor_id].extend(list(zip(ts, mag)))
        
        # Case 2: Processed format (timestamp, mag_H_nT) - single sensor
        elif 'timestamp' in df.columns and 'mag_H_nT' in df.columns:
            sensor_id = _canonical_sensor_id(os.path.basename(csv_path).replace('.csv', ''))
            ts = pd.to_datetime(df['timestamp']).tolist()
            mag = df['mag_H_nT'].astype(float).tolist()
            if sensor_id not in sensor_data:
                sensor_data[sensor_id] = []
            sensor_data[sensor_id].extend(list(zip(ts, mag)))
        
        # Case 3: Raw format (sensor_id, timestamp, b_x, b_y, b_z) - multiple sensors
        elif all(c in df.columns for c in ['sensor_id', 'timestamp', 'b_x', 'b_y', 'b_z']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['mag_total_nT'] = np.sqrt(df['b_x']**2 + df['b_y']**2 + df['b_z']**2)
            
            # Group by sensor_id and timestamp, take mean if duplicates
            for raw_sensor_id, group in df.groupby('sensor_id'):
                sensor_id = _canonical_sensor_id(raw_sensor_id)
                group_sorted = group.sort_values('timestamp')
                # Average duplicates at same timestamp
                group_agg = group_sorted.groupby('timestamp', as_index=False)['mag_total_nT'].mean()
                ts = group_agg['timestamp'].tolist()
                mag = group_agg['mag_total_nT'].astype(float).tolist()
                
                if sensor_id not in sensor_data:
                    sensor_data[sensor_id] = []
                sensor_data[sensor_id].extend(list(zip(ts, mag)))
        
        else:
            print(f"Warning: {csv_path} doesn't have expected columns, skipping.")
            continue
    
    if not sensor_data:
        raise ValueError("No valid data loaded from any CSV files.")
    
    # Process each sensor: sort by timestamp, remove duplicates
    processed_sensor_data = {}
    for sensor_id, data_list in sensor_data.items():
        df_sensor = pd.DataFrame(data_list, columns=['ts', 'field'])
        df_sensor = df_sensor.sort_values('ts').reset_index(drop=True)
        df_sensor = df_sensor.drop_duplicates(subset=['ts'], keep='first')
        
        timestamps = df_sensor['ts'].tolist()
        field_data = df_sensor['field'].tolist()
        
        processed_sensor_data[sensor_id] = (timestamps, field_data)
        print(f"  Sensor {sensor_id}: {len(timestamps)} points, range {timestamps[0]} to {timestamps[-1]}")
    
    return processed_sensor_data


def load_magnetic_data_by_csvs(csv_paths):
    """Alias for load_magnetic_data_by_sensor for backward compatibility."""
    return load_magnetic_data_by_sensor(csv_paths)


def train_pretrained_model(
    csv_paths,
    output_model_dir,
    epochs=50,
    use_yearly_cycle=True,
    window_size=15,
    learning_rate=0.001,
    batch_size=32,
    sensor_filter=None,
    *,
    delta_target: bool = False,
    subsample_every: int = 1,
):
    """
    Train one GRU model per sensor on extended historical data and save them.
    
    Parameters:
    -----------
    csv_paths : list of str
        List of CSV file paths to load and combine
    output_model_dir : str
        Directory where trained models will be saved (e.g., 'models/')
        Models will be named: gru_pretrained_<sensor_id>.keras
    epochs : int
        Number of training epochs per sensor
    use_yearly_cycle : bool
        Whether to include yearly seasonal features
    window_size : int
        Number of time steps to look back
    learning_rate : float
        Learning rate for Adam optimizer
    batch_size : int
        Batch size for training
    sensor_filter : list of str, optional
        If provided, only train models for sensors matching these IDs (e.g., ['OBS1_1', 'OBS1_2'])
    delta_target : bool
        If True, supervise one-step Δ(scaled magnitude) and write ``gru_delta_y: true`` in
        ``*_predictor_meta.json``. If False, supervise absolute scaled magnitude (legacy default).
    """
    # Load data grouped by sensor
    sensor_data = load_magnetic_data_by_csvs(csv_paths)
    
    # Filter sensors if requested
    if sensor_filter:
        sensor_data = {sid: data for sid, data in sensor_data.items() 
                      if any(filt in sid for filt in sensor_filter)}
    
    if not sensor_data:
        raise ValueError("No sensor data found after filtering.")
    
    print(f"\nTraining models for {len(sensor_data)} sensor(s)...")
    
    # Create output directory
    os.makedirs(output_model_dir, exist_ok=True)
    
    # Train one model per sensor
    trained_models = []
    for sensor_id, (timestamps, field_data) in sensor_data.items():
        print(f"\n{'='*60}")
        print(f"Training model for sensor: {sensor_id}")
        print(f"{'='*60}")
        
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
                delta_target=delta_target,
                subsample_every=subsample_every,
            )
            trained_models.append(sensor_id)
        except Exception as e:
            print(f"Error training model for {sensor_id}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Training complete! Trained {len(trained_models)} model(s):")
    for sid in trained_models:
        model_path = os.path.join(output_model_dir, f"gru_pretrained_{sid}.keras")
        print(f"  - {sid}: {model_path}")
    print(f"{'='*60}")


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
    *,
    delta_target: bool = False,
    subsample_every: int = 1,
):
    """Train a single GRU model for one sensor."""
    target_label = "Δ(scaled magnitude)" if delta_target else "absolute scaled magnitude"
    print(f"GRU training target: {target_label}")

    # Create predictor
    predictor = AttnBiLSTMPredictor(
        window_size=window_size,
        initial_train_points=len(field_data),  # Use all data for pre-training
        epochs_per_update=epochs,
        learning_rate=learning_rate,
        update_training=False,  # Don't update during forecast (we're just training)
        use_yearly_cycle=use_yearly_cycle,
        train_window_minutes=None,  # Use all data
        model_family=MODEL_FAMILY_GRU,
    )
    predictor._gru_delta_y = bool(delta_target)

    # Build feature matrix and train
    print("Building features and training model...")
    ts = pd.to_datetime(timestamps)
    df0 = pd.DataFrame({"ts": ts, "field": field_data})
    df0 = df0.dropna(subset=["ts", "field"]).sort_values("ts").reset_index(drop=True)
    if int(subsample_every) > 1:
        df0 = df0.iloc[:: int(subsample_every)].reset_index(drop=True)
        print(f"Subsampled every {subsample_every} rows → {len(df0)} points")

    ts = df0["ts"]
    field = df0["field"].to_numpy(dtype=float).reshape(-1, 1)
    
    # Scale and build features
    field_scaled = predictor.scaler.fit_transform(field).flatten()
    time_feats = predictor._compute_time_features(ts)
    
    if use_yearly_cycle:
        sin_day, cos_day, sin_year, cos_year = time_feats
        feature_matrix = np.column_stack([field_scaled, sin_day, cos_day, sin_year, cos_year])
    else:
        sin_day, cos_day = time_feats
        feature_matrix = np.column_stack([field_scaled, sin_day, cos_day])
    
    # Build model
    predictor.build_model(feature_matrix.shape[1])
    
    # Create windowed dataset
    X_train, y_train = predictor.create_windowed_dataset(feature_matrix, delta_y=predictor._gru_delta_y)
    print(f"Training on {len(X_train)} samples (window_size={window_size})")
    
    # Train model
    print(f"Training for {epochs} epochs...")
    history = predictor.model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,  # Use 10% for validation
        verbose=2,  # one line per epoch (verbose=1 floods logs and can stall terminals)
    )
    
    # Save one canonical GRU model per sensor (overwrite same name on rerun).
    safe_sensor_id = sensor_id.replace('/', '_').replace('\\', '_')  # Make filename-safe
    hist_dir = os.path.join(output_model_dir, "training_histories")
    os.makedirs(hist_dir, exist_ok=True)
    hist_path = os.path.join(hist_dir, f"gru_pretrained_{safe_sensor_id}_fit_history.json")
    serializable = {
        k: [float(x) for x in (v if isinstance(v, list) else list(v))] for k, v in history.history.items()
    }
    with open(hist_path, "w", encoding="utf-8") as hf:
        json.dump(
            {
                "family": "gru",
                "sensor_id": sensor_id,
                "checkpoint_stem": f"gru_pretrained_{safe_sensor_id}",
                "gru_delta_y": bool(delta_target),
                "epochs_requested": int(epochs),
                "history": serializable,
            },
            hf,
            indent=2,
        )
    print(f"Training history JSON: {hist_path}")

    output_model_path = os.path.join(output_model_dir, f"gru_pretrained_{safe_sensor_id}.keras")
    predictor.save_model(output_model_path)
    print(f"\nModel saved to: {output_model_path}")
    
    # Print final loss
    final_loss = history.history['loss'][-1]
    if 'val_loss' in history.history:
        final_val_loss = history.history['val_loss'][-1]
        print(f"Final training loss: {final_loss:.6f}")
        print(f"Final validation loss: {final_val_loss:.6f}")
    else:
        print(f"Final training loss: {final_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train GRU models on extended magnetic field data")
    parser.add_argument("input_csv", nargs='?', help="Path to CSV file(s) - can be single file, comma-separated list, or glob pattern")
    parser.add_argument("output_model_dir", help="Directory where trained models will be saved (one per sensor)")
    parser.add_argument("--folder", help="Folder containing CSV files to combine (alternative to input_csv)")
    parser.add_argument("--sensors", nargs='+', help="Filter: only train models for sensors containing these strings (e.g., --sensors OBS1_1 OBS1_2)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--use-yearly-cycle", action="store_true", default=True,
                       help="Include yearly seasonal features (default: True)")
    parser.add_argument("--no-yearly-cycle", dest="use_yearly_cycle", action="store_false",
                       help="Disable yearly seasonal features")
    parser.add_argument("--window-size", type=int, default=15, help="Window size for sequences (default: 15)")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument(
        "--delta-target",
        action="store_true",
        help="Train on one-step Δ(scaled magnitude); writes gru_delta_y=true in predictor meta",
    )
    parser.add_argument(
        "--subsample-every",
        type=int,
        default=1,
        help="Keep every Nth row after sorting (default 1 = full series; use 6–10 for faster ablations)",
    )

    args = parser.parse_args()
    
    # Determine CSV files to load
    if args.folder:
        # Load all CSV files from folder
        folder_path = args.folder
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not csv_files:
            csv_files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
        csv_files = sorted(csv_files)  # Sort for consistent ordering
        if not csv_files:
            print(f"Error: No CSV files found in folder '{folder_path}'")
            sys.exit(1)
        print(f"Found {len(csv_files)} CSV files in folder '{folder_path}'")
    elif args.input_csv:
        # Handle comma-separated list or glob pattern
        if ',' in args.input_csv:
            csv_files = [f.strip() for f in args.input_csv.split(',')]
        else:
            # Try glob pattern
            csv_files = glob.glob(args.input_csv)
            if not csv_files:
                # If no glob match, treat as single file
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
        delta_target=bool(args.delta_target),
        subsample_every=max(1, int(args.subsample_every)),
    )
