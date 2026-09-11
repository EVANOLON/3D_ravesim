#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1 Case B analysis: CB-BPM (fastwave) vs direct Cartesian spherical wave
(big-wave CPU, independent implementation) on the reduced-FOV thin-plate case.
Legs: fastwave cbbpm, fastwave off, big-wave off. Metrics: mean-normalized
relative L2, max abs error, center lineouts + figure."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/d/rave-sim-main/rave-sim-main")
RUN = ROOT / "output/_agent_runs/m1b_run"
PLOTS = ROOT / "output/_agent_runs/plots"

legs = {}
for tag, path in [("fastwave_cbbpm", RUN / "cbbpm_run/00000000/detected.npy"),
                  ("fastwave_off", RUN / "off_run/00000000/detected.npy"),
                  ("bigwave_off", RUN / "off_prep/00000000/detected.npy")]:
    if path.exists():
        a = np.load(path)
        legs[tag] = (a[0] if a.ndim == 3 else a).astype(np.float64)
        print(f"{tag}: shape {legs[tag].shape} mean {legs[tag].mean():.4e} "
              f"max {legs[tag].max():.4e}")
    else:
        print(f"{tag}: MISSING {path}")


def rel_l2(A, B):
    A, B = A / A.mean(), B / B.mean()
    ny, nx = min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1])
    d = np.abs(A[:ny, :nx] - B[:ny, :nx])
    return float(np.sqrt(np.mean(d ** 2))), float(d.max())


if len(legs) >= 2:
    names = list(legs)
    print("\npairwise relative L2 / max-abs:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            l2, mx = rel_l2(legs[names[i]], legs[names[j]])
            print(f"  {names[i]:15s} vs {names[j]:15s}: L2 {l2:.4e}  max|d| {mx:.4e}")
    # figure: images + center lineouts
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    vmax = max(v.max() for v in legs.values())
    for ax, (k, v) in zip(axes[0], legs.items()):
        im = ax.imshow(v, origin="lower", cmap="rainbow", vmin=0, vmax=vmax)
        ax.set_title(k); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)
    for ax, (k, v) in zip(axes[1], legs.items()):
        row = v[v.shape[0] // 2, :] / v.mean()
        x = (np.arange(len(row)) - len(row) / 2) * 0.2  # um, physical det px
        ax.plot(x, row, lw=1.1)
        ax.set_title(f"{k} center row"); ax.set_xlabel("detector um")
    fig.tight_layout()
    out = PLOTS / "m1b_reduced_fov_cross_engine.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved:", out)
