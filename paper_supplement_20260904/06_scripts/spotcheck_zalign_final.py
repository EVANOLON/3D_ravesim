#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FINAL z-alignment spot check, all images from the CURRENT fastwave binary
(post-fba9c5d area kernel), raw detected.npy, source 0, per-image mean-normalized.

Shell center depths (grid layer 0 = z_start = 3.5 mm, 1 um/layer):
  perfect_ref  center layer 599.5 -> 4.0995 mm
  l1_ref       center layer 612.0 -> 4.1120 mm   (reference depth)
  varT_ref     center layer 603.0 -> 4.1030 mm
  perfect_z    center layer 611.5 -> 4.1115 mm   (0.5 um shallower than l1)
  varT_z       center layer 612.0 -> 4.1120 mm   (exactly l1's depth)

Questions:
  Q1 (l1 - perfect): RMS & inner/outer band residual drops when the 12 um axial
     mismatch is removed (perfect_ref -> perfect_z)?
  Q2 (l1 - varT):    does the inner-wall residual (which has NO geometric cause:
     inner surfaces identical in both grids) survive removal of the 9 um mismatch
     (varT_ref -> varT_z)?
  Q3 edge radii: does each shell's outer edge shrink by ~0.9 px (12 um) / ~0.7 px
     (9 um) when pushed deeper?
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/mnt/d/rave-sim-main/rave-sim-main/output/_agent_runs")
DIRS = {
    "perfect_ref": BASE / "perfect_ref_prep__agentrun_20260903200001",
    "l1_ref":      BASE / "l1_ref_prep__agentrun_20260903200627",
    "varT_ref":    BASE / "varT_ref_prep__agentrun_20260903201252",
    "perfect_z":   BASE / "perfect_zalign_prep__agentrun_20260903193914",
    "varT_z":      BASE / "varT_zalign_prep__agentrun_20260903194539",
}
DEPTH = {  # shell center depth in mm
    "perfect_ref": 4.0995, "l1_ref": 4.1120, "varT_ref": 4.1030,
    "perfect_z": 4.1115, "varT_z": 4.1120,
}
PLOTS = BASE / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def load(rd):
    a = np.load(Path(rd) / "00000000/detected.npy")
    a = (a[0] if a.ndim == 3 else a).astype(np.float64)
    return a / a.mean()


def radial(img, cy=416.0, cx=416.0):
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(int)
    prof = np.bincount(r.ravel(), weights=img.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    return prof, r


def outer_edge(prof, lo=280, hi=345):
    """Outer edge = radius of steepest intensity fall in the shell outer band."""
    d = np.diff(np.nan_to_num(prof))
    j = int(np.nanargmin(d[lo:hi])) + lo
    return j + 0.5


data = {k: load(v) for k, v in DIRS.items()}
profs = {k: radial(v)[0] for k, v in data.items()}
edges = {k: outer_edge(profs[k]) for k in data}
for k in edges:
    print(f"{k:12s} outer edge ~ {edges[k]:6.2f} px   (center depth {DEPTH[k]:.4f} mm)")
print("  edge shrink perfect: ref->z {:.2f} px (pred ~0.93); varT: ref->z {:.2f} px (pred ~0.70)"
      .format(edges["perfect_ref"] - edges["perfect_z"], edges["varT_ref"] - edges["varT_z"]))

pairs = {
    "l1_minus_perfect_ref": ("l1_ref", "perfect_ref"),
    "l1_minus_perfect_z":   ("l1_ref", "perfect_z"),
    "l1_minus_varT_ref":    ("l1_ref", "varT_ref"),
    "l1_minus_varT_z":      ("l1_ref", "varT_z"),
}
res = {}
ny = nx = 833
yy, xx = np.mgrid[0:ny, 0:nx]
rmap = np.sqrt((xx - 416) ** 2 + (yy - 416) ** 2).astype(int)
oe_l1 = edges["l1_ref"]
for name, (A, B) in pairs.items():
    dI = data[A] - data[B]
    rms_full = np.sqrt(np.mean(dI ** 2))
    absd = np.abs(dI)
    rp = np.bincount(rmap.ravel(), weights=absd.ravel()) / np.maximum(np.bincount(rmap.ravel()), 1)
    rp = np.nan_to_num(rp)
    # bands around l1's inner (263-268) & outer (315-325) surfaces
    def band(r1, r2):
        m = (rmap >= r1) & (rmap <= r2)
        return np.sqrt(np.mean(dI[m] ** 2))
    inner = band(int(oe_l1 * 0.83), int(oe_l1 - 48))      # ~ (255, 272) for oe~320
    outer = band(int(oe_l1 - 48), int(oe_l1 + 8))
    # radial profile peaks in same bands
    pk_in = rp[int(oe_l1 * 0.83):int(oe_l1 - 48)].max() if oe_l1 * 0.83 < oe_l1 - 48 else 0
    pk_out = rp[int(oe_l1 - 48):int(oe_l1 + 8)].max()
    res[name] = dict(rms=rms_full, inner=inner, outer=outer, pk_in=pk_in, pk_out=pk_out, rp=rp)
    print(f"{name:26s} RMS={rms_full:.4e}  inner-band RMS={inner:.4e} (pk {pk_in:.2e})  "
          f"outer-band RMS={outer:.4e} (pk {pk_out:.2e})")

print("\n=== axial-shift attribution ===")
for tag, (nz, nref) in {"perfect": ("l1_minus_perfect_z", "l1_minus_perfect_ref"),
                        "varT": ("l1_minus_varT_z", "l1_minus_varT_ref")}.items():
    r = res[nz]
    print(f"{tag}: aligned RMS {r['rms']:.4e} = {100*r['rms']/res[nref]['rms']:.1f}% of unaligned; "
          f"inner-band {r['inner']:.4e} vs unaligned {res[nref]['inner']:.4e} "
          f"({100*r['inner']/res[nref]['inner']:.1f}%); "
          f"inner pk {r['pk_in']:.2e} vs {res[nref]['pk_in']:.2e}")

# ---------------- figure ----------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
ax = axes[0, 0]
for nm, lbl in [("l1_minus_perfect_ref", "l1 - perfect_ref (12 um z off)"),
                ("l1_minus_perfect_z", "l1 - perfect_z (aligned)")]:
    ax.plot(res[nm]["rp"], label=lbl)
ax.axvline(oe_l1, color="k", ls=":", lw=1)
ax.axvline(edges["perfect_ref"], color="b", ls=":", lw=0.8)
ax.set_xlim(200, 360); ax.set_ylim(0, None)
ax.set_xlabel("radius px"); ax.set_ylabel("mean |dI|")
ax.set_title("l1 - perfect: radial |residual|"); ax.legend(fontsize=8)
ax = axes[0, 1]
for nm, lbl in [("l1_minus_varT_ref", "l1 - varT_ref (9 um z off)"),
                ("l1_minus_varT_z", "l1 - varT_z (aligned)")]:
    ax.plot(res[nm]["rp"], label=lbl)
ax.axvline(oe_l1, color="k", ls=":", lw=1)
ax.set_xlim(200, 360); ax.set_ylim(0, None)
ax.set_xlabel("radius px"); ax.set_ylabel("mean |dI|")
ax.set_title("l1 - varT: radial |residual|"); ax.legend(fontsize=8)
for ax, (nm, title) in zip([axes[1, 0], axes[1, 1]],
                           [("l1_minus_varT_ref", "l1 - varT_ref (9 um z off)"),
                            ("l1_minus_varT_z", "l1 - varT_z (aligned)")]):
    dI = res[nm]["rp"] * 0  # placeholder
    A, B = pairs[nm]
    dI = data[A] - data[B]
    vmax = np.percentile(np.abs(dI), 99.8)
    im = ax.imshow(dI, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
out = PLOTS / "spotcheck_zalign_final.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
