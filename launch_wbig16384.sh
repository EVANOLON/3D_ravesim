#!/bin/bash
# Windows-side launcher for the big-wave 16384^2 W case (survives harness restarts).
# NOTE: scratch and checkpoints are NOT deleted here — the runner resumes from the
# last ready checkpoint if one exists. To force a clean run, remove the
# checkpoint directory manually first.
export LD_LIBRARY_PATH=/home/taylor/anaconda3/lib:$LD_LIBRARY_PATH
export PYTHONPATH=nist_lookup:rave_agent/_bfpy:big-wave
cd /mnt/d/rave-sim-main/rave-sim-main || exit 1
mkdir -p /tmp/rave-sim/wbig16384
/home/taylor/anaconda3/bin/python3 -u run_wbig16384.py > output/_agent_runs/bigwave_w_compare/wbig7.log 2>&1
