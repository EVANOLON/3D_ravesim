#!/bin/bash
# M2 (axial dz convergence, varT_z coarsened 2um/4um, sources 0-14) + M4 (aligned
# vacuum, all 50 sources). Current binary. Run dirs created from prep dirs.
set -u
FW=/mnt/d/rave-sim-main/rave-sim-main/fast-wave/build-Release/fastwave
ROOT=/mnt/d/rave-sim-main/rave-sim-main
BASE=$ROOT/output/_agent_runs
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$BASE/m2m4_run"

run_sources () {  # $1=prep  $2=rundir  $3=first  $4=last
  local prep=$1 rd=$2 f=$3 l=$4
  if [ ! -f "$rd/config.yaml" ]; then rm -rf "$rd"; mkdir -p "$rd"; cp -r "$prep/." "$rd/"; fi
  for i in $(seq "$f" "$l"); do
    echo "--- $(basename $prep) source $i ---"
    if ! "$FW" -s "$i" "$rd" >/dev/null 2>>"$BASE/m2m4_run/bulk_err.log"; then
      echo "[FAIL] source $i exit $?"
    fi
  done
}

# M2: varT_z coarsened grids, dz=2um & 4um, sources 0..14 (1um 50-src reference exists)
run_sources "$BASE/align_converge/varT_dz2_prep" "$BASE/m2m4_run/varT_dz2_run" 0 14
run_sources "$BASE/align_converge/varT_dz4_prep" "$BASE/m2m4_run/varT_dz4_run" 0 14
# M4: aligned-geometry vacuum, sources 0..49
run_sources "$BASE/align_vacuum/vacuum_prep" "$BASE/m2m4_run/vacuum_run" 0 49
echo "M2+M4 ALL FINISHED"
