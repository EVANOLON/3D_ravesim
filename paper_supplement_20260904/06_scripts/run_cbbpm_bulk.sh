#!/bin/bash
# Run remaining CB-BPM sources for all four shells.
# Usage: run_cbbpm_bulk.sh <start_source> <end_source>
# Each shell's run dir must already exist with source 0..start-1 done.
set -u
FW=/mnt/d/rave-sim-main/rave-sim-main/fast-wave/build-Release/fastwave
RUN_BASE=/mnt/d/rave-sim-main/rave-sim-main/output/_agent_runs
START=${1:-1}
END=${2:-49}
export CUDA_VISIBLE_DEVICES=0

shells=(perfect_sphere l1_m0_eps002 l2_m0_eps002 l1_varT_er002_et005)

# Map each shell to its run dir (created by rave_sim_run copies)
declare -A RUNDIR=(
  [perfect_sphere]="$RUN_BASE/perfect_sphere__agentrun_20260901061636"
)

for label in "${shells[@]}"; do
  rd="${RUNDIR[$label]:-}"
  if [ -z "$rd" ]; then
    echo "[skip] $label: no run dir registered"
    continue
  fi
  if [ ! -d "$rd" ]; then
    echo "[skip] $label: run dir missing $rd"
    continue
  fi
  echo "===== shell $label: sources $START..$END in $rd ====="
  for i in $(seq "$START" "$END"); do
    echo "--- $label source $i ---"
    if ! "$FW" -s "$i" "$rd"; then
      echo "[FAIL] $label source $i exit $?"
    fi
  done
  echo "===== done $label ====="
done
echo "ALL BULK SOURCES FINISHED"
