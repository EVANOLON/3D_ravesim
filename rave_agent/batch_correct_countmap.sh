#!/usr/bin/env bash
# Batch-run correct_countmap.py over every timestep (000000NN) of a RAVE-SIM run.
#
# Usage: batch_correct_countmap.sh <sim_dir> [extra correct_countmap.py args...]
#
# Skips timesteps whose detected_corrected.npy is already newer than detected.npy.
# All output is appended to <sim_dir>/batch_correct.log.
set -uo pipefail

SIM_DIR="${1:?usage: batch_correct_countmap.sh <sim_dir> [extra args...]}"
shift

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/rave_agent/correct_countmap.py"
LOG="$SIM_DIR/batch_correct.log"

done=0; skipped=0; failed=0; missing=0

for d in "$SIM_DIR"/000000*/; do
  [ -d "$d" ] || continue
  ts="$(basename "$d")"
  det="$d/detected.npy"
  corr="$d/detected_corrected.npy"

  if [ ! -f "$det" ]; then
    echo "[$ts] no detected.npy - skip" | tee -a "$LOG"
    missing=$((missing+1)); continue
  fi
  if [ -f "$corr" ] && [ "$corr" -nt "$det" ]; then
    echo "[$ts] already corrected - skip" | tee -a "$LOG"
    skipped=$((skipped+1)); continue
  fi

  echo "[$ts] correcting..." | tee -a "$LOG"
  if python "$SCRIPT" --sim_dir "$SIM_DIR" --path "$det" "$@" >> "$LOG" 2>&1; then
    echo "[$ts] OK -> $corr" | tee -a "$LOG"
    done=$((done+1))
  else
    echo "[$ts] FAILED (see $LOG)" | tee -a "$LOG"
    failed=$((failed+1))
  fi
done

echo "SUMMARY corrected=$done skipped=$skipped failed=$failed missing=$missing" | tee -a "$LOG"
