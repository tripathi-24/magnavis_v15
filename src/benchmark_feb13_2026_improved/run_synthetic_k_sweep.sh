#!/usr/bin/env bash
# Run improved benchmark on synthetic CSV for k=2,3,4,5 and merge comparison tables.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH="$REPO/src/benchmark_feb13_2026_improved"
PY="${MAGNAVIS_BENCHMARK_PYTHON:-$REPO/.venv/bin/python}"
CSV="$REPO/Datafiles/magnetic_data_20260213_150000_to_20260213_180000_synthetic_no_anomaly.csv"
SWEEP_ROOT="$BENCH/results/synthetic_k_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SWEEP_ROOT"

ONLY="ewma,median,savgol,gru_pretrained,lstm_pretrained,attention_bi_lstm"
COMMON=(
  --csv "$CSV"
  --manual-csv-basename "magnetic_data_20260213_150000_to_20260213_180000_synthetic_no_anomaly.csv"
  --base-date "2026-02-13"
  --no-csv-end
  --only "$ONLY"
  --fast-csv-playback
  --historic-minutes 62
  --skip-initial-minutes 0
)

run_k() {
  local K="$1"
  local DEST="$SWEEP_ROOT/k${K}"
  echo "========== k=${K} =========="
  if [[ "$K" == "4" && -f "$BENCH/results/13Feb26_syntheticData_k4_20260518_081134/comparison_table.csv" ]]; then
    echo "Reusing existing k=4 results."
    mkdir -p "$DEST"
    cp "$BENCH/results/13Feb26_syntheticData_k4_20260518_081134/comparison_table.csv" "$DEST/"
    cp "$BENCH/results/13Feb26_syntheticData_k4_20260518_081134/comparison_table.md" "$DEST/" 2>/dev/null || true
    cp "$BENCH/results/13Feb26_syntheticData_k4_20260518_081134/run_meta.json" "$DEST/" 2>/dev/null || true
    return 0
  fi
  "$PY" "$BENCH/run_suite_improved.py" --k "$K" "${COMMON[@]}"
  local LATEST
  LATEST="$(ls -td "$BENCH/results"/*/comparison_table.csv 2>/dev/null | head -1)"
  LATEST_DIR="$(dirname "$LATEST")"
  mkdir -p "$DEST"
  cp "$LATEST_DIR/comparison_table.csv" "$DEST/"
  cp "$LATEST_DIR/comparison_table.md" "$DEST/" 2>/dev/null || true
  cp "$LATEST_DIR/run_meta.json" "$DEST/" 2>/dev/null || true
  echo "Archived k=${K} -> $DEST"
}

for K in 2 3 4 5; do
  run_k "$K"
done

INPUTS=(
  "k2:$SWEEP_ROOT/k2/comparison_table.csv"
  "k3:$SWEEP_ROOT/k3/comparison_table.csv"
  "k4:$SWEEP_ROOT/k4/comparison_table.csv"
  "k5:$SWEEP_ROOT/k5/comparison_table.csv"
)
"$PY" "$BENCH/merge_k_sweep_tables.py" --out-dir "$SWEEP_ROOT/combined" --inputs "${INPUTS[@]}"

echo "Done. Combined table: $SWEEP_ROOT/combined/comparison_table_all_k.md"
