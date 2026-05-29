#!/usr/bin/env bash
# Wait for LSTM closed-loop k-recall sweep, then regenerate COMPARATIVE_ANALYSIS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CSV="${ROOT}/src/benchmark_feb13_2026_improved/results/k_recall_curves_zero_hist_20260522_170707/runs/apr27_lstm_closed_loop/k_recall_points.csv"
PY="${ROOT}/.venv/bin/python"
GEN="${ROOT}/src/benchmark_feb13_2026_improved/generate_comparative_analysis.py"
LOG=/tmp/k_recall_apr27_lstm_ar.log

echo "Waiting for ${CSV} (5 data rows)..."
while true; do
  if [[ -f "${CSV}" ]]; then
    n=$(tail -n +2 "${CSV}" | wc -l | tr -d ' ')
    if [[ "${n}" -ge 5 ]]; then
      echo "Found ${n} rows — regenerating analysis."
      break
    fi
  fi
  if ! pgrep -f "run_k_recall_curves.py.*apr27_lstm_closed_loop" >/dev/null 2>&1; then
    if [[ -f "${CSV}" ]]; then
      echo "Benchmark process ended; regenerating with available rows."
      break
    fi
    echo "ERROR: benchmark exited without k_recall_points.csv" >&2
    tail -30 "${LOG}" >&2 || true
    exit 1
  fi
  sleep 120
done

"${PY}" "${GEN}"
echo "Done: ${ROOT}/src/benchmark_feb13_2026_improved/results/k_recall_curves_zero_hist_20260522_170707/COMPARATIVE_ANALYSIS.md"
