#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare fast-wave vs big-wave detector images for the scaled W test."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FAST = Path('output/_agent_runs/w_scaled_bigwave__agentrun_20260816142327/00000000/detected.npy')
BIG = Path('output/_agent_runs/bigwave_w_compare/w_scaled_bigwave/00000000/detected.npy')

a = np.load(FAST)[0].astype(np.float64)
b = np.load(BIG)[0].astype(np.float64)
assert a.shape == b.shape, f'shape mismatch: {a.shape} vs {b.shape}'
n = a.shape[0]
print(f'shape: {a.shape}')
print(f'fastwave: min={a.min():.4e} max={a.max():.4e} mean={a.mean():.4e}')
print(f'bigwave : min={b.min():.4e} max={b.max():.4e} mean={b.mean():.4e}')

# global correlation
cc = np.corrcoef(a.ravel(), b.ravel())[0, 1]
print(f'correlation: {cc:.6f}')

# relative difference (normalized)
denom = np.maximum(a, 1e-30)
rel = np.abs(a - b) / denom
print(f'rel diff: median={np.median(rel):.4%} p95={np.percentile(rel,95):.4%} max={rel.max():.4%}')

# center lineout + a background lineout
cy = n // 2
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
vmax = np.percentile(a, 99.5)
im0 = axes[0, 0].imshow(a, cmap='inferno', vmin=0, vmax=vmax)
axes[0, 0].set_title('fast-wave (GPU)')
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
im1 = axes[0, 1].imshow(b, cmap='inferno', vmin=0, vmax=vmax)
axes[0, 1].set_title('big-wave (CPU)')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
axes[1, 0].plot(a[cy], 'r-', lw=0.8, label='fastwave')
axes[1, 0].plot(b[cy], 'b--', lw=0.8, label='bigwave')
axes[1, 0].set_title(f'center row {cy} lineout'); axes[1, 0].legend()
axes[1, 1].plot(a[cy + 800], 'r-', lw=0.8, label='fastwave')
axes[1, 1].plot(b[cy + 800], 'b--', lw=0.8, label='bigwave')
axes[1, 1].set_title(f'off-center row {cy + 800} lineout'); axes[1, 1].legend()
plt.tight_layout()
out = 'output/_agent_runs/plots/w_bigwave_vs_fastwave.png'
plt.savefig(out, dpi=110)
print('saved', out)

# difference map stats by region
diff = np.abs(a - b) / np.maximum(a, 1e-30)
yy, xx = np.mgrid[0:n, 0:n]
r = np.sqrt((xx - n // 2) ** 2 + (yy - n // 2) ** 2)
for name, mask in [('center(r<500)', r < 500), ('mid(500-1200)', (r >= 500) & (r < 1200)), ('outer(>1200)', r >= 1200)]:
    print(f'rel diff {name}: median={np.median(rel[mask]):.4%} mean={rel[mask].mean():.4%}')
