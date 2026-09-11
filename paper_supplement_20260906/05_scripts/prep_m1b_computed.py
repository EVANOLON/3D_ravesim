import sys, traceback
sys.path.insert(0, 'nist_lookup'); sys.path.insert(0, 'rave_agent/_bfpy'); sys.path.insert(0, 'big-wave')
from pathlib import Path
try:
    import multisim
    import config as cfg
    for tag in ('cbbpm','off'):
        d = Path('output/_agent_runs/m1b_case')/f'{tag}_prep'
        dct = cfg.load(d/'config.yaml')
        print(tag, 'config loaded, is2d', dct.get('sim_params',{}).get('is2d'))
        multisim.setup_simulation(dct, d, d)
        print(tag, 'setup ok -> computed.yaml written')
except Exception:
    traceback.print_exc()
