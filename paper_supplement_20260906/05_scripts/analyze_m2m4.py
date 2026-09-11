#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M2 + M4 analysis.
M2: axial dz convergence on the aligned varT_z geometry (coarsened grids).
    legs: dz=1 um (existing align_full_run/varT_run, 50 src) vs new dz=2/4 um
    runs (m2m4_run/varT_dz2/dz4_run, 15 src each). Image error vs the finest
    (dz=1 um) reference on the SAME 15-source subset.
M4: aligned-geometry vacuum open field (50 src) -> I_open; open-field-normalized
    four-shell images (aligned 50pt sums / I_open)."""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/d/rave-sim-main/rave-sim-main")
AR = ROOT / "output/_agent_runs"
PLOTS = AR / "plots"
SUBSET = list(range(15))          # sources 0..14


def load(rd, i):
    a = np.load(Path(rd) / f"{i:08d}" / "detected.npy")
    return (a[0] if a.ndim == 3 else a).astype(np.float64)


def sum_sources(rd, idx):
    return np.sum([load(rd, i) for i in idx], axis=0)


# ---------------- M2 ----------------
ref = sum_sources(AR / "align_full_run/varT_run", SUBSET)      # dz=1 um
legs = {}
for dz, rd in [("dz2um", AR / "m2m4_run/varT_dz2_run"),
               ("dz4um", AR / "m2m4_run/varT_dz4_run")]:
    legs[dz] = sum_sources(rd, SUBSET)

def rel_l2(A, B):
    A, B = A / A.mean(), B / B.mean()
    ny = min(A.shape[0], B.shape[0]); nx = min(A.shape[1], B.shape[1])
    return float(np.sqrt(np.mean((A[:ny, :nx] - B[:ny, :nx]) ** 2)))

def rel_l2_band(A, B, r1, r2):
    A, B = A / A.mean(), B / B.mean()
    ny, nx = A.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    R = np.sqrt((xx - (nx - 1) / 2) ** 2 + (yy - (ny - 1) / 2) ** 2)
    m = (R >= r1) & (R <= r2)
    return float(np.sqrt(np.mean((A[m] - B[m]) ** 2)))

print("M2 axial convergence (varT_z aligned, 15-source sums, ref = dz=1um):")
print(f"{'dz':>6s} {'full L2':>10s} {'outer-band L2':>14s}")
rows = []
for dz, I in legs.items():
    rows.append((dz, rel_l2(I, ref), rel_l2_band(I, ref, 295, 335)))
    print(f"{dz:>6s} {rows[-1][1]:10.4e} {rows[-1][2]:14.4e}")

fig, ax = plt.subplots(figsize=(6.5, 5))
dzu = np.array([1.0, 2.0, 4.0])
errs = np.array([0.0] + [r[1] for r in rows])   # dz=1 is the reference -> 0 by def.
errs_out = np.array([0.0] + [r[2] for r in rows])
ax.loglog(dzu, errs + 1e-12, "o-", label="full-frame rel L2 vs dz=1um")
ax.loglog(dzu, errs_out + 1e-12, "s--", label="outer-band (295-335 px) rel L2")
ax.set_xlabel("axial step dz [um]"); ax.set_ylabel("relative L2 (ref dz=1 um)")
ax.set_title("M2 axial-slice convergence, varT_z aligned, 15 sources")
ax.legend(); ax.grid(alpha=0.3, which="both")
fig.tight_layout()
f2 = PLOTS / "m2_axial_convergence.png"
fig.savefig(f2, dpi=140, bbox_inches="tight")
print("saved:", f2)

# ---------------- M4 ----------------
print("\nM4 vacuum open field:")
Ivac = sum_sources(AR / "m2m4_run/vacuum_run", list(range(50)))
np.save(AR / "align_full_46pt/align_vacuum_sum50.npy", Ivac)
print(f"vacuum 50-src sum: mean {Ivac.mean():.4e} min {Ivac.min():.4e} "
      f"max {Ivac.max():.4e} rel-std {Ivac.std()/Ivac.mean():.3e}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
im = axes[0].imshow(Ivac, origin="lower", cmap="rainbow")
axes[0].set_title("aligned CB-BPM open field (vacuum, 50 src)")
plt.colorbar(im, ax=axes[0], fraction=0.046)
v = Ivac / Ivac.mean()
im = axes[1].imshow(v, origin="lower", cmap="RdBu_r", vmin=0.99, vmax=1.01)
axes[1].set_title("open field / mean (color range ±1%)")
plt.colorbar(im, ax=axes[1], fraction=0.046)
fig.tight_layout()
f4 = PLOTS / "m4_open_field.png"
fig.savefig(f4, dpi=140, bbox_inches="tight")
print("saved:", f4)

# open-field normalized four-shell sums
print("\ndefect / open-field normalization (aligned 50pt):")
for shell in ("perfect", "l1", "l2", "varT"):
    I = np.load(AR / f"align_full_46pt/align_{shell}_sum50.npy").astype(np.float64)
    In = I / Ivac
    In = In / In.mean()
    print(f"  {shell}: I/I_open rel-std {In.std():.3e}")
