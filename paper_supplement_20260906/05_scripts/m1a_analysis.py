import numpy as np
from pathlib import Path
AR=Path('output/_agent_runs')
def ld(p):
    a=np.load(p); return (a[0] if a.ndim==3 else a).astype(np.float32)
plate={m:ld(AR/f'thin_family_prep/C10__{m}/00000000/detected.npy') for m in ['fson','fsoff','cbbpm']}
vac  ={m:ld(AR/f'thin_vacuum_run/C10vac__{m}/00000000/detected.npy') for m in ['fson','fsoff','cbbpm']}
mask=(vac['fson']>np.percentile(vac['fson'],1))
def L2(A,B):
    A=A.astype(np.float64); B=B.astype(np.float64)
    A,B=A/A.mean(),B/B.mean()
    d=A[mask]-B[mask]
    return np.sqrt(np.mean(d**2))
print('raw mean-norm L2:', flush=True)
for a,b in [('fson','cbbpm'),('fsoff','fson'),('fsoff','cbbpm')]:
    print(f'  {a}-{b}: {L2(plate[a],plate[b]):.4e}', flush=True)
print('vacuum field (mean-norm):', flush=True)
for m in ['fson','fsoff','cbbpm']:
    v=vac[m].astype(np.float64); v=v/v.mean()
    print(f'  {m}: min {v.min():.3f} max {v.max():.3f} std {v.std():.3e}', flush=True)
print('contrast plate/vac L2:', flush=True)
for a,b in [('fson','cbbpm'),('fsoff','fson'),('fsoff','cbbpm')]:
    ca=(plate[a].astype(np.float64)/vac[a].astype(np.float64)); cb=(plate[b].astype(np.float64)/vac[b].astype(np.float64))
    ca=ca/ca.mean(); cb=cb/cb.mean()
    d=ca[mask]-cb[mask]
    print(f'  {a}-{b}: L2 {np.sqrt(np.mean(d**2)):.4e}  max|d| {np.abs(d).max():.3e}', flush=True)
