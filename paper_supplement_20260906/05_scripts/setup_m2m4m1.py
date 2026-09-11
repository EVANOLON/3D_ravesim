#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup for M2 (axial dz convergence on varT_z) and M4 (aligned-geometry vacuum)
and M1-case-A (thin C10 vacuum-normalization diagnosis).

M2: coarsen the aligned varT_z grid (1224,1206,1206)@1um along z by 2x/4x
    (occupancy-OR, material 0/1) -> (612,1206,1206)@2um, (306,1206,1206)@4um.
    Preps carry pixel_size_z = 2e-6 / 4e-6, same config/computed/subconfigs.
M4: all-zero grid (1224,1200,1200) like the aligned perfect_z layout; 50 sources.
M1A: zero grid (200,500,500) thin C10 vacuum preps for fson/fsoff/cbbpm.
"""
import shutil
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/d/rave-sim-main/rave-sim-main")
AR = ROOT / "output/_agent_runs"
VAR_T_Z = AR / "align_full/grids/500um_100um_shell_l1_varT_zalign1224.npy"
TPL = AR / "align_full/varT_prep"          # config/computed/subconfigs template
TPL_CFG = (TPL / "config.yaml").read_text()


def subconfigs(dst, src, n=50):
    for i in range(n):
        sub = dst / f"{i:08d}"
        sub.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / f"{i:08d}/subconfig.yaml", sub / "subconfig.yaml")


def write_prep(d, cfg_text, grid_name, grid_src, n_sub=50, sub_src=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(cfg_text)
    shutil.copy(TPL / "computed.yaml", d / "computed.yaml")
    subconfigs(d, sub_src or TPL, n_sub)
    shutil.copy(grid_src, d / grid_name)
    print("prep:", d)


def coarsen_z(src_path, factor, out_path):
    """Occupancy-OR coarsening along axis 0; keeps uint32 material semantics."""
    a = np.load(src_path, mmap_mode="r")
    nz, ny, nx = a.shape
    nzo = nz // factor
    out = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.uint32,
                                    shape=(nzo, ny, nx))
    for z in range(nzo):
        block = np.asarray(a[z * factor:(z + 1) * factor]).any(axis=0).astype(np.uint32)
        out[z] = block
    out.flush()
    del out, a
    print(f"coarsened {nz}->{nzo} layers -> {out_path} ({out_path.stat().st_size/1e9:.2f} GB)")


# ---------- M2 preps ----------
M2 = AR / "align_converge"
M2.mkdir(parents=True, exist_ok=True)
g2 = M2 / "varT_z_dz2_grid.npy"
g4 = M2 / "varT_z_dz4_grid.npy"
coarsen_z(VAR_T_Z, 2, g2)
coarsen_z(VAR_T_Z, 4, g4)
for tag, gname, gsrc, pz in [("varT_dz2", "varT_z_dz2_grid.npy", g2, "2e-06"),
                             ("varT_dz4", "varT_z_dz4_grid.npy", g4, "4e-06")]:
    cfg = TPL_CFG
    for key in ("500um_100um_shell_l1_varT_zalign1224.npy",
                "500um_100um_shell_l1_varT_er002_et005_rot90.npy",
                "500um_100um_perfect_shell_zalign1224.npy",
                "500um_100um_perfect_shell.npy",
                "500um_100um_shell_l1_m0_eps002_rot90.npy",
                "500um_100um_shell_l2_m0_eps002_rot90.npy"):
        cfg = cfg.replace(f"grid_path: {key}", f"grid_path: {gname}")
    cfg = cfg.replace("pixel_size_z: 1e-06", f"pixel_size_z: {pz}")
    write_prep(M2 / f"{tag}_prep", cfg, gname, gsrc)

# ---------- M4 vacuum prep (aligned geometry, zero grid) ----------
VAC = AR / "align_vacuum"
VAC.mkdir(parents=True, exist_ok=True)
vacg = VAC / "vacuum_zalign1224.npy"
v = np.lib.format.open_memmap(str(vacg), mode="w+", dtype=np.uint32,
                              shape=(1224, 1200, 1200))
v.flush()  # zero-filled
del v
cfg = TPL_CFG.replace("grid_path: 500um_100um_shell_l1_varT_zalign1224.npy",
                      "grid_path: vacuum_zalign1224.npy")
write_prep(VAC / "vacuum_prep", cfg, "vacuum_zalign1224.npy", vacg)

# ---------- M1A thin C10 vacuum preps ----------
T = AR / "thin_family_prep"
V = AR / "thin_vacuum"
V.mkdir(parents=True, exist_ok=True)
c10 = np.load(T / "C10__fson/thin_c_plate_10um_25um.npy", mmap_mode="r")
c10v = V / "thin_c10_vacuum.npy"
cv = np.lib.format.open_memmap(str(c10v), mode="w+", dtype=np.uint32, shape=c10.shape)
cv.flush(); del cv, c10
for mode in ("fson", "fsoff", "cbbpm"):
    src = T / f"C10__{mode}"
    cfg = (src / "config.yaml").read_text().replace(
        "grid_path: thin_c_plate_10um_25um.npy", "grid_path: thin_c10_vacuum.npy")
    d = V / f"C10vac__{mode}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(cfg)
    shutil.copy(src / "computed.yaml", d / "computed.yaml")
    sub = d / "00000000"; sub.mkdir(exist_ok=True)
    shutil.copy(src / "00000000/subconfig.yaml", sub / "subconfig.yaml")
    shutil.copy(c10v, d / "thin_c10_vacuum.npy")
    print("prep:", d)
print("ALL SETUP DONE")
