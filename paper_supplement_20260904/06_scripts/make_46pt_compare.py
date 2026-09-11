"""June FS_On + correct_countmap vs CB-BPM, 46pt sum.
46pt = 50 sources - {|x|>2um: 5,11,47} - {source 0 (overwritten)} = 46.
June side: per-source correct_countmap (I/count, renorm to source mean) then sum.
CB-BPM side: sum existing per-source detected_corrected.npy.
"""
import numpy as np, yaml, sys
from pathlib import Path
sys.path.insert(0,'/mnt/d/rave-sim-main/rave-sim-main')
sys.path.insert(0,'/mnt/d/rave-sim-main/rave-sim-main/rave_agent')
from correct_countmap import compute_count_map

OUT=Path('/mnt/d/rave-sim-main/rave-sim-main/output/_agent_runs/46pt_compare')
OUT.mkdir(exist_ok=True)

JUN={'perfect':Path('/mnt/d/rave-sim-main/rave-sim-main/output/2026/06/20260627_194236291561'),
     'l1_m0':Path('/mnt/d/rave-sim-main/rave-sim-main/output/2026/06/20260627_194515977270'),
     'l2_m0':Path('/mnt/d/rave-sim-main/rave-sim-main/output/2026/06/20260627_194829543751'),
     'varT':Path('/mnt/d/rave-sim-main/rave-sim-main/output/2026/06/20260627_195114086223')}
CB={'perfect':Path('/mnt/d/rave-sim-main/rave-sim-main/output/_agent_runs/perfect_sphere__agentrun_20260901063739__agentrun_20260901071544'),
    'l1_m0':Path('/mnt/d/rave-sim-main/rave-sim-main/output/_agent_runs/l1_m0_eps002__agentrun_20260901101237__agentrun_20260901121843'),
    'l2_m0':Path('/mnt/d/rave-sim-main/rave-sim-main/output/_agent_runs/l2_m0_eps002__agentrun_20260901133514'),
    'varT':Path('/mnt/d/rave-sim-main/rave-sim-main/output/_agent_runs/l1_varT_er002_et005__agentrun_20260901154505')}

exc={5,11,47,0}
srcs=[i for i in range(50) if i not in exc]
print('46 sources:',srcs)

def load_det(path):
    a=np.load(path,mmap_mode='r'); return (a[0] if a.ndim==3 else a).astype(np.float64)

# ---- June: per-source correct then sum ----
for nm,d in JUN.items():
    cfg=yaml.safe_load((d/'config.yaml').read_text())
    cm=compute_count_map(cfg)
    cnt=cm['count']
    tot=None
    for i in srcs:
        a=load_det(d/f'{i:08d}/detected.npy')
        orig_mean=a.mean()
        corr=np.where(cnt>0, a/np.where(cnt>0,cnt,1), 0.0)
        m=corr.mean()
        if m>0: corr*=orig_mean/m
        tot=corr if tot is None else tot+corr
    np.save(OUT/f'june_{nm}_46pt.npy',tot)
    print(f'june {nm}: mean={tot.mean():.3e} max={tot.max():.3e}')

# ---- CB-BPM: sum existing corrected ----
for nm,d in CB.items():
    tot=None
    for i in srcs:
        p=d/f'{i:08d}/detected_corrected.npy'
        if not p.exists(): p=d/f'{i:08d}/detected.npy'
        a=load_det(p)
        tot=a if tot is None else tot+a
    np.save(OUT/f'cbbpm_{nm}_46pt.npy',tot)
    print(f'cbbpm {nm}: mean={tot.mean():.3e} max={tot.max():.3e}')
print('DONE')
