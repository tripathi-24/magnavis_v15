#!/bin/bash
# Refresh PROGRESS.md from training logs. Run from repo root:
#   bash src/benchmark_feb13_2026_improved/results/gru_delta_ablation_restart/update_progress.sh

ROOT="$(cd "$(dirname "$0")" && pwd)"
ABS_LOG="$ROOT/logs/train_absolute.log"
DEL_LOG="$ROOT/logs/train_delta.log"
OUT="$ROOT/PROGRESS.md"

running() {
  pgrep -f "gru_delta_ablation_restart" >/dev/null 2>&1 && echo "RUNNING" || echo "STOPPED"
}

last_epoch() {
  local f="$1"
  [[ -f "$f" ]] || { echo "(no log yet)"; return; }
  grep -E "Epoch [0-9]+/[0-9]+|Model saved|Training model for sensor" "$f" | tail -6
}

models_done() {
  local d="$ROOT/models/absolute"
  ls "$d"/gru_pretrained_OBS2_*.keras 2>/dev/null | wc -l | tr -d ' '
}

{
  echo "# GRU Delta Ablation — Live Progress"
  echo ""
  echo "**Updated:** $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "**Status:** $(running)"
  echo ""
  echo "## Models saved (absolute): $(models_done)/3"
  echo ""
  echo "### train_absolute.log (last lines)"
  echo '```'
  last_epoch "$ABS_LOG"
  echo '```'
  echo ""
  echo "### train_delta.log (last lines)"
  echo '```'
  last_epoch "$DEL_LOG"
  echo '```'
} > "$OUT"
echo "Wrote $OUT"
