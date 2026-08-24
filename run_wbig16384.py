#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run big-wave 16384^2 W large case (bfpy_ooc + checkpoint/resume).

Scratch lives on the WSL-native /tmp filesystem (fast OOC I/O); checkpoints and
final results stay on D: (survive VM reboots). Resumes automatically when a
ready checkpoint manifest exists.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, 'nist_lookup')
sys.path.insert(0, 'rave_agent/_bfpy')
sys.path.insert(0, 'big-wave')

import multisim  # noqa: E402

sim = Path('output/_agent_runs/bigwave_w_compare/w_origscale_bigwave')
scratch = Path('/tmp/rave-sim/wbig16384')
scratch.mkdir(parents=True, exist_ok=True)
ckpt_dir = Path('output/_agent_runs/bigwave_w_compare/checkpoint')
resume = (ckpt_dir / '00000000' / 'manifest.json').exists()
print(f'big-wave 16384 starting | scratch={scratch} | checkpoint={ckpt_dir} | resume={resume}',
      flush=True)
t0 = time.time()
multisim.run_single_simulation(
    sim, 0, scratch, save_keypoints_path=None,
    checkpoint_dir=ckpt_dir, resume_checkpoint=resume,
)
print('WBIG16384 DONE %.1f s' % (time.time() - t0), flush=True)
