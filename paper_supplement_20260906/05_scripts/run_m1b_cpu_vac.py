import sys, time
sys.path.insert(0, 'nist_lookup'); sys.path.insert(0, 'rave_agent/_bfpy'); sys.path.insert(0, 'big-wave')
import multisim
from pathlib import Path
sim = Path('output/_agent_runs/m1b_case/offvac_prep')
scratch = Path('output/_agent_runs/m1b_run2/cpu_scratch'); scratch.mkdir(exist_ok=True)
t0=time.time(); print('big-wave CPU off-vac starting', flush=True)
multisim.run_single_simulation(sim, 0, scratch, save_keypoints_path=None)
print('DONE %.1f s'%(time.time()-t0), flush=True)
