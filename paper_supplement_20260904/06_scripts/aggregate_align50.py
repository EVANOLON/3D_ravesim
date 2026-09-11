#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Post-processing for the z-ALIGNED 50-source CB-BPM rerun (current binary).

Run when the bulk (run_align50_bulk.sh) has finished all four shells.

Produces under output/_agent_runs/plots/:
  align50_residuals_grid.png       6-panel relative residual maps (confound-free)
  align50_radial_vs_unaligned.png  radial |residual| profiles: aligned 50pt (new,
    current binary) vs stored unaligned 46pt sums (old-era corrected) for the
    l1-perfect and l1-varT pairs -> demonstrates the inner-band collapse
  align50_inner_outer_bars.png     inner/outer band RMS table figure
Also saves per-shell 50-source sums + residual npy under align_full_46pt/.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/d/rave-sim-main/rave-sim-main")
RUN_BASE = ROOT / "output/_agent_runs/align_full_run"
OLD = ROOT / "output/_agent_runs/46pt_compare"
OUT = ROOT / "output/_agent_runs/align_full_46pt"
PLOTS = ROOT / "output/_agent_runs/plots"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

shells = ["perfect", "l1", "l2", "varT"]
labels = {"perfect": "perfect", "l1": "l1_m0", "l2": "l2_m0", "varT": "varT"}


def load_source(rd, i):
    p = Path(rd) / f"{i:08d}" / "detected.npy"
    if not p.exists():
        return None
    a = np.load(p)
    return (a[0] if a.ndim == 3 else a).astype(np.float64)


def radial_map(img, cx=416.0, cy=416.0):
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(int)


def radial_prof(img, rmap=None):
    if rmap is None:
        rmap = radial_map(img)
    return np.bincount(rmap.ravel(), weights=img.ravel()) / np.maximum(np.bincount(rmap.ravel()), 1)


def band_rms(d, rmap, r1, r2):
    m = (rmap >= r1) & (rmap <= r2)
    return np.sqrt(np.mean(d[m] ** 2)) if m.any() else 0.0


# ---------- gather aligned 50-source sums ----------
sums, counts = {}, {}
for shell in shells:
    rd = RUN_BASE / f"{shell}_run"
    parts, miss = [], []
    for i in range(50):
        s = load_source(rd, i)
        if s is None:
            miss.append(i)
            continue
        parts.append(s)
    n = len(parts)
    counts[shell] = n
    if n == 0:
        print(f"[SKIP] {shell}: no sources yet"); continue
    t = np.sum(parts, axis=0)
    sums[shell] = t
    np.save(OUT / f"align_{shell}_sum50.npy", t)
    print(f"{shell}: {n}/50 sources, mean {t.mean():.4e}, max {t.max():.4e}"
          + (f"  MISSING {miss}" if miss else ""))

if len(sums) < 2:
    print("Not enough shells done yet; rerun when bulk finishes.")
    raise SystemExit(0)

# ---------- relative residual maps (per-image mean normalized) ----------
P = sums["perfect"]
rel = {k: (sums[k] / sums[k].mean()) - (P / P.mean()) for k in sums if k != "perfect"}
pairs = [("l1", "l2"), ("l1", "varT"), ("l2", "varT")]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
panels = [(f"{k} - perfect", rel[k]) for k in ("l1", "l2", "varT")] + \
         [(f"{a} - {b}", (sums[a] / sums[a].mean()) - (sums[b] / sums[b].mean())) for a, b in pairs]
for ax, (ttl, d) in zip(axes.ravel(), panels):
    vmax = np.percentile(np.abs(d), 99.5)
    im = ax.imshow(d, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(ttl); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)
    print(f"{ttl:14s} rel-RMS {np.sqrt(np.mean(d**2)):.4e}")
fig.suptitle("z-aligned CB-BPM residuals, 50-source sums, current binary", fontsize=13)
fig.tight_layout()
f1 = PLOTS / "align50_residuals_grid.png"; fig.savefig(f1, dpi=140, bbox_inches="tight")
print("saved:", f1)

# ---------- radial comparison: aligned vs stored unaligned 46pt ----------
rmap = radial_map(P)
oe_unal = None
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
for ax, (a, b, oldf) in zip(axes2,
                            [("l1", "perfect", "cbbpm_l1_m0_46pt.npy"),
                             ("l1", "varT", "cbbpm_varT_46pt.npy")]):
    oldA = np.load(OLD / f"cbbpm_{'l1_m0' if a=='l1' else a}_46pt.npy").astype(float)
    oldB = np.load(OLD / oldf).astype(float)
    d_old = (oldA / oldA.mean()) - (oldB / oldB.mean())
    d_new = (sums[a] / sums[a].mean()) - (sums[b] / sums[b].mean())
    p_old = radial_prof(np.abs(d_old), rmap)
    p_new = radial_prof(np.abs(d_new), rmap)
    ax.plot(p_old, label=f"unaligned 46pt (old-era, corrected)")
    ax.plot(p_new, label=f"aligned 50pt (current)")
    ax.set_xlim(200, 360); ax.set_xlabel("radius px"); ax.set_ylabel("mean |resid|")
    ax.set_title(f"{a} - {b}: radial |residual|"); ax.legend(fontsize=8)
    # inner/outer band numbers for the OLD pair & NEW pair
    o = {"l1": None}
    for tag, d in (("unaligned", d_old), ("aligned", d_new)):
        ib = band_rms(d, rmap, 255, 272)
        ob = band_rms(d, rmap, 295, 335)
        print(f"{a}-{b} {tag}: inner-band RMS {ib:.4e}  outer-band RMS {ob:.4e}")
fig2.tight_layout()
f2 = PLOTS / "align50_radial_vs_unaligned.png"; fig2.savefig(f2, dpi=140, bbox_inches="tight")
print("saved:", f2)
