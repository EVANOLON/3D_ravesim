#!/bin/bash
# Run all 50 CB-BPM sources per shell on the z-ALIGNED grids (current binary,
# post-fba9c5d area kernel + source.y explicit). Run dirs get config/computed/
# subconfigs/grid (symlink) copied from the prep dirs once; fastwave -s i is then
# invoked directly for every source like the historical run_cbbpm_bulk.sh.
set -u
FW=/mnt/d/rave-sim-main/rave-sim-main/fast-wave/build-Release/fastwave
ROOT=/mnt/d/rave-sim-main/rave-sim-main
PREP_BASE=$ROOT/output/_agent_runs/align_full
RUN_BASE=$ROOT/output/_agent_runs/align_full_run
export CUDA_VISIBLE_DEVICES=0

START=${1:-0}
END=${2:-49}
mkdir -p "$RUN_BASE"

for shell in perfect l1 l2 varT; do
  RD="$RUN_BASE/${shell}_run"
  if [ ! -f "$RD/config.yaml" ]; then
    echo "===== setting up $shell run dir ====="
    rm -rf "$RD"; mkdir -p "$RD"
    cp -r "$PREP_BASE/${shell}_prep/." "$RD/"
  fi
  echo "===== shell $shell: sources $START..$END ====="
  for i in $(seq "$START" "$END"); do
    echo "--- $shell source $i ---"
    if ! "$FW" -s "$i" "$RD" >/dev/null 2>>"$RUN_BASE/${shell}_bulk_err.log"; then
      echo "[FAIL] $shell source $i exit $?"
    fi
  done
  echo "===== done $shell ====="
done
echo "ALL ALIGNED SOURCES FINISHED"
