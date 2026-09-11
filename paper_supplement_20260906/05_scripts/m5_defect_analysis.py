#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M5: defect-quantification analysis on the z-aligned 50-source CB-BPM sums.
(2b deliverables)
  1. Azimuthal Fourier decomposition of the outer-interface edge displacement
     dR(phi) = R_defect(phi) - R_perfect(phi): l1 (dipole Y10, body shift) should
     peak at azimuthal harmonic m=1; l2 (P2/Y20) at m=2; varT outer (Y10 eps_t) at
     m=1 with smaller amplitude.
  2. Inner/outer interface residual-ratio table (bands 255-272 / 295-335 px).
  3. Local zoom of the residual near the inner & outer interfaces.
  4. Center lineouts + radial mean profiles (defect vs perfect, normalized).
Figures -> plots/m5_*.png ; table printed & saved."""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/d/rave-sim-main/rave-sim-main")
DATA = ROOT / "output/_agent_runs/align_full_46pt"
PLOTS = ROOT / "output/_agent_runs/plots"
SHELLS = {"perfect": "perfect", "l1": "l1_m0", "l2": "l2_m0", "varT": "varT"}

im = {k: np.load(DATA / f"align_{k}_sum50.npy").astype(np.float64) for k in SHELLS}
for k in im:
    im[k] = im[k] / im[k].mean()
NY = NX = 833
CY = CX = (NY - 1) / 2.0
yy, xx = np.mgrid[0:NY, 0:NX]
R = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
PHI = np.arctan2(yy - CY, xx - CX)   # radians, -pi..pi


def ray_edge(img, phi_deg, r0=280, r1=350):
    """outer edge radius along one azimuth by max |dI/dr| (subpixel via parabola)."""
    ph = np.deg2rad(phi_deg)
    rs = np.arange(r0, r1, 1.0)
    xs = CX + rs * np.cos(ph)
    ys = CY + rs * np.sin(ph)
    vals = []
    for x, y in zip(xs, ys):
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        if not (0 <= x0 < NX - 1 and 0 <= y0 < NY - 1):
            vals.append(np.nan); continue
        fx, fy = x - x0, y - y0
        v = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) +
             img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
        vals.append(v)
    vals = np.array(vals)
    d = np.diff(vals)
    j = int(np.nanargmax(-d))
    if 1 <= j < len(d) - 1:
        den = d[j - 1] - 2 * d[j] + d[j + 1]
        off = 0.5 * (d[j - 1] - d[j + 1]) / den if den != 0 else 0
    else:
        off = 0
    return r0 + j + 0.5 + off


PHIS = np.arange(0, 360, 2.0)   # every 2 deg
# ---- per-azimuth outer edge radius (single image cost ~180*70 interps) ----
print("computing per-azimuth outer edges ...", flush=True)
edges = {}
for k, img in im.items():
    edges[k] = np.array([ray_edge(img, p) for p in PHIS])
    print(f"  {k}: mean edge {edges[k].mean():.2f} px, std {edges[k].std():.2f}", flush=True)

dR = {k: edges[k] - edges["perfect"] for k in ("l1", "l2", "varT")}
# FFT azimuthal harmonics (periodic in phi over 360 deg -> integer m)
amps = {}
for k in ("l1", "l2", "varT"):
    f = np.fft.rfft(dR[k]) / len(PHIS)
    amps[k] = 2 * np.abs(f)          # m = 0..180 ; 2x for single-sided
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, k in zip(axes, ("l1", "l2", "varT")):
    ax.plot(PHIS, dR[k], lw=1)
    ax.set_title(f"{k}: outer-edge dR(phi) [px]"); ax.set_xlabel("azimuth deg")
    ax.set_ylabel("px"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "m5_azimuth_edge_dR.png", dpi=140, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, k in zip(axes, ("l1", "l2", "varT")):
    m = np.arange(len(amps[k]))
    ax.bar(m[:9], amps[k][:9])
    ax.set_title(f"{k}: |dR| azimuthal harmonics"); ax.set_xlabel("m")
    ax.set_ylabel("px")
    print(f"{k}: m=1 amp {amps[k][1]:.3f} px, m=2 amp {amps[k][2]:.3f} px, "
          f"ratio m1/m2 {amps[k][1]/max(amps[k][2],1e-9):.2f}", flush=True)
fig.tight_layout()
fig.savefig(PLOTS / "m5_azimuthal_harmonics.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ---- interface residual ratio table ----
def band_rms(A, B, r1, r2):
    d = (A - B)
    m = (R >= r1) & (R <= r2)
    return float(np.sqrt(np.mean(d[m] ** 2)))

rows = []
for k in ("l1", "l2", "varT"):
    inner = band_rms(im[k], im["perfect"], 255, 272)
    outer = band_rms(im[k], im["perfect"], 295, 335)
    rows.append((k, inner, outer, inner / outer))
    print(f"{k}-perfect: inner-band {inner:.4e}  outer-band {outer:.4e}  "
          f"inner/outer {inner/outer:.3f}")
with open(PLOTS / "m5_interface_ratio_table.txt", "w") as f:
    f.write("defect  inner_RMS_255_272  outer_RMS_295_335  inner/outer\n")
    for k, i, o, r in rows:
        f.write(f"{k}  {i:.4e}  {o:.4e}  {r:.3f}\n")

# ---- local zoom at interfaces (residual maps, arc sectors) ----
fig, axes = plt.subplots(3, 2, figsize=(12, 13))
for row, k in zip(axes, ("l1", "l2", "varT")):
    d = im[k] - im["perfect"]
    for ax, (ttl, rc, rad) in zip(row, [("inner interface r=255-272", 264, 20),
                                        ("outer interface r=295-335", 318, 25)]):
        m = (R >= rc - rad) & (R <= rc + rad)
        vm = np.percentile(np.abs(d[m]), 99)
        ax.imshow(np.where(m, d, np.nan), origin="lower", cmap="RdBu_r",
                  vmin=-vm, vmax=vm)
        ax.set_title(f"{k}-perfect: {ttl}"); ax.axis("off")
fig.suptitle("local interface residual zoom (aligned 50pt)", fontsize=13)
fig.tight_layout()
fig.savefig(PLOTS / "m5_local_zoom_interfaces.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ---- center lineouts & radial mean profiles ----
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for col, k in zip(range(3), ("l1", "l2", "varT")):
    ax = axes[0, col]
    for ref, ls in [("perfect", "--"), (k, "-")]:
        row = im[ref][416, :]
        ax.plot((np.arange(NX) - CX) * 0.048, row, ls=ls, lw=1, label=ref)
    ax.set_title(f"{k}: center-row lineout (mm)"); ax.legend(fontsize=8)
    ax = axes[1, col]
    for ref in ("perfect", k):
        r = np.round(R).astype(int)
        p = np.bincount(r.ravel(), weights=im[ref].ravel()) / \
            np.maximum(np.bincount(r.ravel()), 1)
        ax.plot(p[:370], label=ref)
    ax.axvspan(255, 272, color="r", alpha=0.12); ax.axvspan(295, 335, color="b", alpha=0.12)
    ax.set_title(f"{k}: radial mean (red=inner band, blue=outer)"); ax.legend(fontsize=8)
    ax.set_xlabel("radius px")
fig.tight_layout()
fig.savefig(PLOTS / "m5_profiles_lineouts.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("M5 figures saved under plots/m5_*.png")
