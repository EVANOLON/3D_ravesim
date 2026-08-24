#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare fast-wave vs big-wave on the thin-W plate (1 um, material validation)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FDIR = 'output/_agent_runs/20260818_120635983579__agentrun_20260818040704/00000000'
BDIR = 'output/_agent_runs/thin_w_cmp/keypoints'
BDET = 'output/_agent_runs/thin_w_cmp/2026/08/20260818_120635983579/00000000/detected.npy'

# theory (from subconfig values)
DELTA, BETA = 2.792861120466358e-05, 1.7839768621442877e-06
ENERGY = 9999.834044009405
LAM = 1.239841984e-6 / ENERGY
T = 1e-6  # 1 um
PHI_THEORY = -2 * np.pi * DELTA * T / LAM
TRANS_THEORY = np.exp(-4 * np.pi * BETA * T / LAM)
print(f'theory: dphi = {PHI_THEORY:.4f} rad, |T|^2 = {TRANS_THEORY:.4f}')

def load(f):
    return np.load(f).reshape(8192, 8192)

ub_f = load(f'{FDIR}/wave_before_sample.npy')
ua_f = load(f'{FDIR}/wave_after_sample.npy')
ub_b = load(f'{BDIR}/keypoint_00_0.npy')
ua_b = load(f'{BDIR}/keypoint_00_1.npy')

# 1) before/after wavefield correlation
for name, x, y in [('before', ub_f, ub_b), ('after', ua_f, ua_b)]:
    cf = np.abs(x)**2; cb = np.abs(y)**2
    cc = np.corrcoef(cf.ravel(), cb.ravel())[0, 1]
    mf = cf.mean(); mb = cb.mean()
    dph = np.angle(x * np.conj(y))
    w = cf
    ph_mean = np.average(dph, weights=w)
    ph_std = np.sqrt(np.average((dph - ph_mean)**2, weights=w))
    print(f'wave_{name}: |u|2 corr={cc:.6f}  mean ratio={mf/mb:.4f}  phase offset={ph_mean:.4f} rad (std {ph_std:.4f})')

# 2) material factor T in plate region (|x|,|y| < 90 um -> 750 wavefield px)
n = 8192; half = 750
s = slice(n//2 - half, n//2 + half)
Tf = ua_f[s, s] / ub_f[s, s]
Tb = ua_b[s, s] / ub_b[s, s]
for name, T in [('fastwave', Tf), ('bigwave', Tb)]:
    ph = np.angle(T)
    mag2 = np.abs(T)**2
    print(f'{name}: T phase mean={ph.mean():.4f} rad (theory {PHI_THEORY:.4f}, err {ph.mean()-PHI_THEORY:+.4f}) | std={ph.std():.4f}')
    print(f'{name}: |T|^2 mean={mag2.mean():.4f} (theory {TRANS_THEORY:.4f}, err {mag2.mean()-TRANS_THEORY:+.4f}) | std={mag2.std():.4f}')
print('T(fastwave) vs T(bigwave): phase corr =', np.corrcoef(np.angle(Tf).ravel(), np.angle(Tb).ravel())[0, 1])

# 3) detector images
det_f = np.load(f'{FDIR}/detected.npy')[0].astype(float)
det_b = np.load(BDET)[0].astype(float)
print(f'detector: fastwave mean={det_f.mean():.3e} bigwave mean={det_b.mean():.3e}')
print(f'detector corr = {np.corrcoef(det_f.ravel(), det_b.ravel())[0,1]:.4f}')
# center intensity transmission: plate is at center; use center ROI
roi = 20
cf = det_f[det_f.shape[0]//2-roi:det_f.shape[0]//2+roi, det_f.shape[1]//2-roi:det_f.shape[1]//2+roi].mean()
cb = det_b[det_b.shape[0]//2-roi:det_b.shape[0]//2+roi, det_b.shape[1]//2-roi:det_b.shape[1]//2+roi].mean()
bgf = det_f[det_f.shape[0]//2-200:det_f.shape[0]//2-150, det_f.shape[1]//2-200:det_f.shape[1]//2-150].mean()
bgb = det_b[det_b.shape[0]//2-200:det_b.shape[0]//2-150, det_b.shape[1]//2-200:det_b.shape[1]//2-150].mean()
print(f'I_center/I_bg: fastwave={cf/bgf:.4f} (theory {TRANS_THEORY:.4f})  bigwave={cb/bgb:.4f}')

# 4) figure: phase maps + detector
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, T, ttl in [(axes[0,0], Tf, 'fastwave T phase'), (axes[0,1], Tb, 'bigwave T phase')]:
    im = ax.imshow(np.angle(T), cmap='RdBu', vmin=-1.6, vmax=-1.2)
    ax.set_title(ttl); plt.colorbar(im, ax=ax, fraction=0.046)
im = axes[0,2].imshow(np.angle(Tf) - np.angle(Tb), cmap='RdBu', vmin=-0.2, vmax=0.2)
axes[0,2].set_title('phase diff (fast-big)'); plt.colorbar(im, ax=axes[0,2], fraction=0.046)
for ax, d, ttl in [(axes[1,0], det_f, 'fastwave detector'), (axes[1,1], det_b, 'bigwave detector')]:
    im = ax.imshow(d, cmap='inferno', vmax=np.percentile(d, 99))
    ax.set_title(ttl); plt.colorbar(im, ax=ax, fraction=0.046)
ax = axes[1,2]
c = det_f.shape[0]//2
ax.plot(det_f[c], 'r-', lw=0.8, label='fastwave')
ax.plot(det_b[c], 'b--', lw=0.8, label='bigwave')
ax.set_title('detector center row'); ax.legend()
plt.tight_layout()
plt.savefig('output/_agent_runs/plots/thin_w_engine_compare.png', dpi=110)
print('saved output/_agent_runs/plots/thin_w_engine_compare.png')
