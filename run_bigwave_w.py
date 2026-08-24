#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run big-wave 2D simulation for the scaled W capsule test (engine comparison)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, 'nist_lookup')
sys.path.insert(0, 'rave_agent/_bfpy')
sys.path.insert(0, 'big-wave')

import multisim  # noqa: E402  (import order breaks the circular import)

sim = Path('output/_agent_runs/bigwave_w_compare/w_scaled_bigwave')
scratch = Path('output/_agent_runs/bigwave_w_compare/scratch')
scratch.mkdir(exist_ok=True)

t0 = time.time()
print('big-wave starting', flush=True)
multisim.run_single_simulation(sim, 0, scratch, save_keypoints_path=None)
print('BIGWAVE DONE in %.1f s' % (time.time() - t0), flush=True)
