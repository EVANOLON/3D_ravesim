#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Continuation: build the M2/M4/M1A prep dirs (coarse grids already exist)."""
import shutil
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/d/rave-sim-main/rave-sim-main")
AR = ROOT / "output/_agent_runs"
TPL = AR / "align_full/varT_prep"
TPL_CFG = (TPL / "config.yaml").read_text()
M2 = AR / "align_converge"


def subconfigs(dst, n=50):
    for i in range(n):
        sub = dst / f"{i:08d}"
        sub.mkdir(parents=True, exist_ok=True)
        shutil.copy(TPL / f"{i:08d}/subconfig.yaml", sub / "subconfig.yaml")


def write_prep(d, cfg_text, grid_name, grid_src, n_sub=50):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(cfg_text)
    shutil.copy(TPL / "computed.yaml", d / "computed.yaml")
    subconfigs(d, n_sub)
    shutil.copy(grid_src, d / grid_name)
    print("prep:", d)


for tag, gname, pz in [("varT_dz2", "varT_z_dz2_grid.npy", "2e-06"),
                       ("varT_dz4", "varT_z_dz4_grid.npy", "4e-06")]:
    cfg = TPL_CFG
    for key in ("500um_100um_shell_l1_varT_zalign1224.npy",
                "500um_100um_shell_l1_varT_er002_et005_rot90.npy",
                "500um_100um_perfect_shell_zalign1224.npy",
                "500um_100um_perfect_shell.npy",
                "500um_100um_shell_l1_m0_eps002_rot90.npy",
                "500um_100um_shell_l2_m0_eps002_rot90.npy"):
        cfg = cfg.replace(f"grid_path: {key}", f"grid_path: {gname}")
    cfg = cfg.replace("pixel_size_z: 1e-06", f"pixel_size_z: {pz}")
    write_prep(M2 / f"{tag}_prep", cfg, gname, M2 / gname)

VAC = AR / "align_vacuum"
VAC.mkdir(parents=True, exist_ok=True)
vacg = VAC / "vacuum_zalign1224.npy"
if not vacg.exists():
    v = np.lib.format.open_memmap(str(vacg), mode="w+", dtype=np.uint32,
                                  shape=(1224, 1200, 1200))
    v.flush(); del v
    print("vacuum grid written")
cfg = TPL_CFG.replace("grid_path: 500um_100um_shell_l1_varT_zalign1224.npy",
                      "grid_path: vacuum_zalign1224.npy")
write_prep(VAC / "vacuum_prep", cfg, "vacuum_zalign1224.npy", vacg)

T = AR / "thin_family_prep"
V = AR / "thin_vacuum"
V.mkdir(parents=True, exist_ok=True)
c10v = V / "thin_c10_vacuum.npy"
if not c10v.exists():
    c10 = np.load(T / "C10__fson/thin_c_plate_10um_25um.npy", mmap_mode="r")
    cv = np.lib.format.open_memmap(str(c10v), mode="w+", dtype=np.uint32, shape=c10.shape)
    cv.flush(); del cv, c10
    print("thin c10 vacuum grid written")
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
