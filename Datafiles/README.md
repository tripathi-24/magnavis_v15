# Magnetic data exports (`Datafiles/`)

CSV files in this folder are **not committed to Git** (see root `.gitignore`). They are required locally to train models and run benchmarks.

Place exports here using the exact filenames expected by the benchmark scripts, or pass `--csv` / `--manual-csv-basename` when running from another path.

## Files used by the thesis benchmark suite

| File | Approx. size | Used for |
|------|-------------|----------|
| `magnetic_data_20260213_150000_to_20260213_163000.csv` | ~6.4 MB | Feb-13 short-GT k–recall, default `run_suite` |
| `magnetic_data_20260213_150000_to_20260213_180000_synthetic_no_anomaly.csv` | ~7.5 MB | Synthetic no-anomaly scenario |
| `magnetic_data_20260426_060000_to_20260427_090000_1hz.csv` | ~13 MB | Apr-27 long-GT (1 Hz), k–recall & baseline sweep |
| `magnetic_data_20251201_000000_to_20251231_234500.csv` | ~1.5 GB | GRU/LSTM pretraining (Dec-2025 OBS2) |

## Other local copies (optional)

These may exist on your machine for ad-hoc runs but are not required for the canonical k–recall bundle:

- `magnetic_data_20260427_050000_to_20260427_220000_1hz.csv`
- `magnetic_data_20260426_050000_to_20260427_093000.csv`
- `magnetic_data_20251016_000000_to_20251025_234500_downsampled_60x.csv`
- `magnetic_data_ADT_20260222_153000_to_20260222_173000.csv`

## Storage recommendation

Keep full CSV archives on **IITK storage / an external drive** and symlink or copy only what you need into `Datafiles/` for a given experiment.
