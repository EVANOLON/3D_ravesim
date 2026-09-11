#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the four full (50-source) aligned prep dirs for the controlled rerun.

Grids (all content z-centers aligned to layer 612.0 == l1/l2's measured center):
  perfect_z : original 1200^3 front-padded 12 -> (1224,1200,1200), center 611.5
  l1        : original 1224^3, center 612.0  (reference, unchanged)
  l2        : original 1224^3, center 612.0  (unchanged, verify via bash-13)
  varT_z    : original 1206^3 front-padded 9 -> (1224,1206,1206), center 612.0

Each prep dir: config.yaml + computed.yaml + 00000000..49/subconfig.yaml +
grid file (symlink to shared staged file, so rave_sim_run's cp -r copy stays cheap
and target dirs carry a working symlink).
"""
import shutil
from pathlib import Path

ROOT = Path("/mnt/d/rave-sim-main/rave-sim-main")
SUBCFG_SRC = ROOT / "output/_agent_runs/l1_m0_eps002__agentrun_20260901101237__agentrun_20260901121843"
GRIDDIR = ROOT / "output/_agent_runs/align_full/grids"
PREP = ROOT / "output/_agent_runs/align_full"
CFG_TEMPLATE = (SUBCFG_SRC / "config.yaml").read_text()
COMPUTED = (SUBCFG_SRC / "computed.yaml").read_bytes()

SHELLS = {
    "perfect": dict(grid_name="500um_100um_perfect_shell_zalign1224.npy",
                    grid_src=GRIDDIR / "500um_100um_perfect_shell_zalign1224.npy",
                    old_key="grid_path: 500um_100um_perfect_shell.npy"),
    "l1":      dict(grid_name="500um_100um_shell_l1_m0_eps002_rot90.npy",
                    grid_src=ROOT / "grid/500um_100um_shell_l1_m0_eps002_rot90.npy",
                    old_key="grid_path: 500um_100um_shell_l1_m0_eps002_rot90.npy"),
    "l2":      dict(grid_name="500um_100um_shell_l2_m0_eps002_rot90.npy",
                    grid_src=ROOT / "grid/500um_100um_shell_l2_m0_eps002_rot90.npy",
                    old_key="grid_path: 500um_100um_shell_l2_m0_eps002_rot90.npy"),
    "varT":    dict(grid_name="500um_100um_shell_l1_varT_zalign1224.npy",
                    grid_src=GRIDDIR / "500um_100um_shell_l1_varT_zalign1224.npy",
                    old_key="grid_path: 500um_100um_shell_l1_varT_er002_et005_rot90.npy"),
}
for shell, info in SHELLS.items():
    d = PREP / f"{shell}_prep"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / "computed.yaml").write_bytes(COMPUTED)
    # 50 per-source subconfigs (identical across shells; only subconfig.yaml files)
    for i in range(50):
        sub = d / f"{i:08d}"
        sub.mkdir()
        shutil.copy(SUBCFG_SRC / f"{i:08d}/subconfig.yaml", sub / "subconfig.yaml")
    # config: ensure grid_path points at this shell's grid name
    cfg = CFG_TEMPLATE
    for key in ("500um_100um_perfect_shell.npy",
                "500um_100um_shell_l1_m0_eps002_rot90.npy",
                "500um_100um_shell_l2_m0_eps002_rot90.npy",
                "500um_100um_shell_l1_varT_er002_et005_rot90.npy",
                "500um_100um_perfect_shell_zalign1224.npy",
                "500um_100um_shell_l1_varT_zalign1224.npy"):
        cfg = cfg.replace(f"grid_path: {key}", f"grid_path: {info['grid_name']}")
    (d / "config.yaml").write_text(cfg)
    # grid symlink
    (d / info["grid_name"]).symlink_to(info["grid_src"])
    print(shell, "prep ready:", d, "->", info["grid_name"])
print("ALL FULL PREPS READY")
