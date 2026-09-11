#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FINAL comparison: z-ALIGNED 50pt (current binary) vs stored UNALIGNED 46pt
(old-era corrected) CB-BPM residuals, per defect pair.

Band definitions (aligned images, shells at 4.112 mm, R_in=500 um, R_out=600 um):
  inner band r in [255, 272] px (inner surface ~267 px)
  outer band r in [295, 335] px (outer surface ~320-323 px)
Old side (unaligned): 46pt intensity sums from 46pt_compare/cbbpm_*_46pt.npy
(corrected countmap, mixed-era kernel runs but ripple-free). New side (aligned):
50pt sums from align_full_46pt/align_*_sum50.npy (current binary, raw).
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/d/rave-sim-main/rave-sim-main")
OLD = ROOT / "output/_agent_runs/46pt_compare"
NEW = ROOT / "output/_agent_runs/align_full_46pt"
PLOTS = ROOT / "output/_agent_runs/plots"
NAMES = {"perfect": "perfect", "l1": "l1_m0", "l2": "l2_m0", "varT": "varT"}

old = {k: np.load(OLD / f"cbbpm_{NAMES[k]}_46pt.npy").astype(float) for k in NAMES}
new = {k: np.load(NEW / f"align_{k}_sum50.npy").astype(float) for k in NAMES}


def rmap_of(img):
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    return np.sqrt((xx - (nx - 1) / 2) ** 2 + (yy - (ny - 1) / 2) ** 2).astype(int)


def radial_prof(img, rmap):
    return np.bincount(rmap.ravel(), weights=img.ravel()) / np.maximum(np.bincount(rmap.ravel()), 1)


def band_rms(d, rmap, r1, r2):
    m = (rmap >= r1) & (rmap <= r2)
    return np.sqrt(np.mean(d[m] ** 2)) if m.any() else 0.0


rmap = rmap_of(old["perfect"])
pairs = [("l1", "perfect"), ("l2", "perfect"), ("varT", "perfect"),
         ("l1", "l2"), ("l1", "varT"), ("l2", "varT")]

print(f"{'pair':16s} | {'unaligned 46pt':>28s} | {'aligned 50pt':>28s}")
print(f"{'':16s} | {'full RMS':>10s} {'inner':>10s} {'outer':>10s} | "
      f"{'full RMS':>10s} {'inner':>10s} {'outer':>10s}")
rows = {}
for a, b in pairs:
    ro = dict()
    for tag, src in (("old", old), ("new", new)):
        dA, dB = src[a], src[b]
        d = (dA / dA.mean()) - (dB / dB.mean())
        ro[tag] = (np.sqrt(np.mean(d ** 2)), band_rms(d, rmap, 255, 272),
                   band_rms(d, rmap, 295, 335))
    rows[(a, b)] = ro
    print(f"{a+'-'+b:16s} | {ro['old'][0]:10.4e} {ro['old'][1]:10.4e} {ro['old'][2]:10.4e} | "
          f"{ro['new'][0]:10.4e} {ro['new'][1]:10.4e} {ro['new'][2]:10.4e}")
    print(f"{'':16s} | {'':>10} ratio inner old/new: {ro['old'][1]/max(ro['new'][1],1e-12):.1f}x "
          f"| {'':>10} ")

# ---- figure 1: radial |residual| profiles, l1-perfect & l1-varT, old vs new ----
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
for ax, (a, b) in zip(axes, [("l1", "perfect"), ("l1", "varT")]):
    for tag, src, ls in (("unaligned 46pt", old, "--"), ("aligned 50pt", new, "-")):
        dA, dB = src[a], src[b]
        d = (dA / dA.mean()) - (dB / dB.mean())
        p = radial_prof(np.abs(d), rmap)
        ax.plot(p, ls=ls, lw=1.6 if ls == "-" else 1.2,
                label=f"{tag}")
    ax.axvspan(255, 272, color="r", alpha=0.12)
    ax.axvspan(295, 335, color="b", alpha=0.12)
    ax.set_xlim(200, 360)
    ax.set_xlabel("radius px"); ax.set_ylabel("mean |resid| (rel)")
    ax.set_title(f"{a} - {b} : radial |residual|")
    ax.legend(fontsize=8)
fig.suptitle("red band = inner surface (r 255-272), blue band = outer (295-335)", fontsize=10)
fig.tight_layout()
f1 = PLOTS / "align50_radial_vs_unaligned.png"
fig.savefig(f1, dpi=140, bbox_inches="tight")
print("saved:", f1)

# ---- figure 2: aligned 6-panel residual maps ----
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
for ax, (a, b) in zip(axes2.ravel(), pairs):
    dA, dB = new[a], new[b]
    d = (dA / dA.mean()) - (dB / dB.mean())
    vmax = np.percentile(np.abs(d), 99.5)
    im = ax.imshow(d, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"{a} - {b}"); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
fig2.suptitle("z-aligned CB-BPM residuals (50pt, current binary)", fontsize=13)
fig2.tight_layout()
f2 = PLOTS / "align50_residuals_grid.png"
fig2.savefig(f2, dpi=140, bbox_inches="tight")
print("saved:", f2)

# persist table
(PLOTS / "align50_table.txt").write_text("\n".join(
    [f"{a}-{b} unaligned {rows[(a,b)]['old']} aligned {rows[(a,b)]['new']}"
     for a, b in pairs]))
