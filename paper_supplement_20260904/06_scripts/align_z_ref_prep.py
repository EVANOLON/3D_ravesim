#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build reference (unpadded, original-depth) prep dirs for perfect / l1 / varT
under the CURRENT fastwave binary, so all spot-check images share one code era
(post-fba9c5d area kernel). Config/computed/subconfig copied from the zalign prep
template (generic); only grid_path and the grid file change."""
import shutil
from pathlib import Path

ROOT = Path("/mnt/d/rave-sim-main/rave-sim-main")
TEMPLATE = ROOT / "output/_agent_runs/align_z/perfect_zalign_prep"
PREP = ROOT / "output/_agent_runs/align_z"
GRIDS = {
    "perfect_ref": ("500um_100um_perfect_shell.npy", ROOT / "grid/500um_100um_perfect_shell.npy"),
    "l1_ref":      ("500um_100um_shell_l1_m0_eps002_rot90.npy",
                    ROOT / "grid/500um_100um_shell_l1_m0_eps002_rot90.npy"),
    "varT_ref":    ("500um_100um_shell_l1_varT_er002_et005_rot90.npy",
                    ROOT / "grid/500um_100um_shell_l1_varT_er002_et005_rot90.npy"),
}
for label, (gname, gsrc) in GRIDS.items():
    d = PREP / f"{label}_prep"
    d.mkdir(parents=True, exist_ok=True)
    (d / "computed.yaml").write_bytes((TEMPLATE / "computed.yaml").read_bytes())
    sub = d / "00000000"
    sub.mkdir(exist_ok=True)
    (sub / "subconfig.yaml").write_bytes((TEMPLATE / "00000000/subconfig.yaml").read_bytes())
    txt = (TEMPLATE / "config.yaml").read_text()
    txt = txt.replace("grid_path: 500um_100um_perfect_shell_zalign1224.npy", f"grid_path: {gname}")
    (d / "config.yaml").write_text(txt)
    # hardlink the grid (same fs) to avoid duplicating 7 GB; rave_sim_run copy will materialize it
    dst = d / gname
    if not dst.exists():
        try:
            dst.hardlink_to(gsrc)
        except OSError:
            shutil.copy(gsrc, dst)
    print(label, "prep ready:", d)
