#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-generate the thin-plate three-mode comparison figures from raw detected.npy.

Input : data/<CASE>__<mode>/00000000/detected.npy
        CASE in {C10,C20,W1,W2,W5,C1layer,W1layer}, mode in {fson,fsoff,cbbpm}
Output: figures/regenerated/
  - <CASE>_3modes.png       2D images of each present mode (same scale) + residuals
  - <CASE>_centerline.png   center row & column lineouts overlaid
  - engine_agreement_table.csv   rel-RMS(fson-cbbpm), rel-RMS(fsoff-fson) etc.
  - engine_agreement.png    rel-RMS vs plate thickness

Metric (identical to the report table): mean-normalized full-image relative RMS on
the common min-crop:
    rel-RMS(A,B) = sqrt(mean(((A/<A>)-(B/<B>))^2))
All mode arrays share the same detector sampling (4250x4250 @ 0.2 um/px physical),
so element-wise comparison is physically aligned.

Usage: python3 regenerate_figures.py   (run from this folder or anywhere)
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures" / "regenerated"
FIG.mkdir(parents=True, exist_ok=True)

CASES = ["C10", "C20", "W1", "W2", "W5", "C1layer", "W1layer"]
MODES = ["fson", "fsoff", "cbbpm"]
THICKNESS_UM = {"C10": 10, "C20": 20, "W1": 1, "W2": 2, "W5": 5,
                "C1layer": 10, "W1layer": 1}   # 1-layer variants labelled by parent thickness
MATERIAL = {"C10": "C", "C20": "C", "W1": "W", "W2": "W", "W5": "W",
            "C1layer": "C", "W1layer": "W"}
DET_PX_UM = 0.2   # 0.85 mm / 4250 px, physical detector pixel
# Object (plate) geometric shadow as reference, measured on the DETECTOR plane.
# Grid truth: C plates are 25x25 um squares, W plates 200x200 um (50 nm voxels);
# point source at z=0, plate front at z_start=1.0 m, detector at z=4.0 m => M=4.
# Shadow half-width on detector = (plate half-width) * 4
#   C: 12.5 um * 4 = 50 um  = 250 px ;  W: 100 um * 4 = 400 um = 2000 px
PLATE_UM = {   # half-width of the plate itself [um]
    "C10": 12.5, "C20": 12.5, "C1layer": 12.5,
    "W1": 100.0, "W2": 100.0, "W5": 100.0, "W1layer": 100.0,
}
MAG = 4.0
SHADOW_HALF_DET_UM = {c: w * MAG for c, w in PLATE_UM.items()}   # detector-plane um
SHADOW_HALF_PX = {c: s / DET_PX_UM for c, s in SHADOW_HALF_DET_UM.items()}  # array px


def load(case, mode):
    p = DATA / f"{case}__{mode}" / "00000000" / "detected.npy"
    if not p.exists():
        return None
    a = np.load(p)
    return (a[0] if a.ndim == 3 else a).astype(np.float64)


def rel_rms(A, B):
    A, B = A / A.mean(), B / B.mean()
    ny = min(A.shape[0], B.shape[0]); nx = min(A.shape[1], B.shape[1])
    d = np.abs(A[:ny, :nx] - B[:ny, :nx])
    return float(np.sqrt(np.mean(d ** 2)))


def lineout(img, axis):
    return img[img.shape[0] // 2, :] if axis == "x" else img[:, img.shape[1] // 2]


rows = []
for case in CASES:
    im = {m: load(case, m) for m in MODES}
    im = {m: a for m, a in im.items() if a is not None}
    if len(im) < 2:
        print(f"[skip] {case}: only {list(im)}")
        continue
    # ---- 2D panels + residuals ----
    names = list(im)
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 8))
    axes = np.atleast_2d(axes)
    vmax = max(a.max() for a in im.values())
    sh = SHADOW_HALF_PX[case]                       # reference box half-width [px]
    for j, m in enumerate(names):
        axes[0, j].imshow(im[m], origin="lower", cmap="rainbow", vmin=0, vmax=vmax)
        axes[0, j].set_title(f"{case} {m}")
        axes[0, j].axis("off")
        # geometric-shadow reference box (plate actual extent, x and y)
        cy, cx = im[m].shape[0] / 2, im[m].shape[1] / 2
        rect = plt.Rectangle((cx - sh, cy - sh), 2 * sh, 2 * sh, fill=False,
                             edgecolor="w", ls="--", lw=1.4, alpha=0.9)
        axes[0, j].add_patch(rect)
        axes[0, j].text(cx, cy + sh + 0.02 * im[m].shape[0],
                        f"plate {2*PLATE_UM[case]:.0f}x{2*PLATE_UM[case]:.0f} um shadow",
                        color="w", ha="center", va="bottom", fontsize=8,
                        bbox=dict(fc="k", alpha=0.35, pad=1))
    # residual panels: mode - first mode
    ref = names[0]
    for j, m in enumerate(names[1:], start=1):
        d = im[m] / im[m].mean() - im[ref] / im[ref].mean()
        vm = np.percentile(np.abs(d), 99.5)
        axes[1, j].imshow(d, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
        axes[1, j].set_title(f"{m} - {ref} (rel)")
        axes[1, j].axis("off")
    axes[1, 0].text(0.02, 0.5, "images (top) / relative residuals (bottom)",
                    transform=axes[1, 0].transAxes, rotation=90, va="center", fontsize=9)
    fig.suptitle(f"{case}: {MATERIAL[case]} plate {THICKNESS_UM[case]} um", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / f"{case}_3modes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- center lineouts (with plate shadow reference) ----
    sh_um = SHADOW_HALF_DET_UM[case]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for m, a in im.items():
        x = (np.arange(a.shape[1]) - a.shape[1] / 2) * DET_PX_UM
        y = (np.arange(a.shape[0]) - a.shape[0] / 2) * DET_PX_UM
        axes[0].plot(x, a[a.shape[0] // 2, :] / a.mean(), lw=1.2, label=m)
        axes[1].plot(y, a[:, a.shape[1] // 2] / a.mean(), lw=1.2, label=m)
    for ax, t in zip(axes, ["center row (x)", "center column (y)"]):
        ax.axvspan(-sh_um, sh_um, color="0.85", alpha=0.45, lw=0,
                   label=f"plate shadow $\\pm${sh_um:.0f} um")
        ax.axvline(-sh_um, color="k", ls="--", lw=1.0, alpha=0.8)
        ax.axvline(sh_um, color="k", ls="--", lw=1.0, alpha=0.8)
        ax.set_xlabel("position on detector [um]"); ax.set_ylabel("I / <I>")
        ax.set_title(t); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"{case}: center lineouts, {MATERIAL[case]} {THICKNESS_UM[case]} um "
                 f"(grey = plate geometric shadow)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / f"{case}_centerline.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- agreement metrics ----
    row = {"case": case, "material": MATERIAL[case], "thickness_um": THICKNESS_UM[case]}
    for a, b in [("fson", "cbbpm"), ("fsoff", "fson"), ("fsoff", "cbbpm")]:
        if a in im and b in im:
            row[f"relrms_{a}-{b}"] = rel_rms(im[a], im[b])
    rows.append(row)
    print(row)

# ---- CSV + agreement figure ----
with open(FIG / "engine_agreement_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["case", "material", "thickness_um",
                                      "relrms_fson-cbbpm", "relrms_fsoff-fson",
                                      "relrms_fsoff-cbbpm"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

fig, ax = plt.subplots(figsize=(8, 5))
xs, ys_fc, ys_fo = [], [], []
for r in rows:
    if "relrms_fson-cbbpm" not in r:
        continue
    xs.append(r["thickness_um"]); ys_fc.append(r["relrms_fson-cbbpm"])
    if "relrms_fsoff-fson" in r:
        ys_fo.append(r["relrms_fsoff-fson"])
ax.semilogy(xs, ys_fc, "o-", label="fson vs cbbpm")
ax.plot([r["thickness_um"] for r in rows if "relrms_fsoff-fson" in r],
        [r["relrms_fsoff-fson"] for r in rows if "relrms_fsoff-fson" in r],
        "s--", label="fsoff vs fson")
ax.set_xlabel("plate thickness [um]"); ax.set_ylabel("relative RMS")
ax.set_title("thin-plate engine agreement (1-layer variants marked open)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "engine_agreement.png", dpi=130, bbox_inches="tight")
print("figures ->", FIG)
