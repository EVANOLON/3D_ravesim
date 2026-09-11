#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Measure bounding boxes / shell centers of the four capsule grids (memmap-safe)."""
import numpy as np
from pathlib import Path

GRID = Path("/mnt/d/rave-sim-main/rave-sim-main/grid")
files = {
    "perfect": GRID / "500um_100um_perfect_shell.npy",
    "l1":      GRID / "500um_100um_shell_l1_m0_eps002_rot90.npy",
    "l2":      GRID / "500um_100um_shell_l2_m0_eps002_rot90.npy",
    "varT":    GRID / "500um_100um_shell_l1_varT_er002_et005_rot90.npy",
}


def bbox_axis(arr, ax):
    """Return (min,max) index of nonzero content along axis ax."""
    if ax == 0:
        mask = np.any(arr, axis=(1, 2))
    elif ax == 1:
        mask = np.any(arr, axis=(0, 2))
    else:
        mask = np.any(arr, axis=(0, 1))
    idx = np.nonzero(mask)[0]
    return int(idx[0]), int(idx[-1]) if idx.size else (None, None)


for name, p in files.items():
    arr = np.load(p, mmap_mode="r")
    print(f"== {name}: {p.name} shape={arr.shape} dtype={arr.dtype}")
    b = [bbox_axis(arr, ax) for ax in (0, 1, 2)]
    print("   bbox z,y,x:", b)
    centers = [(lo + hi) / 2.0 for (lo, hi) in b]
    print("   centers z,y,x:", centers, "  (voxel units, 1um/voxel)")
    # sample middle-layer stats to understand value convention
    z0 = arr.shape[0] // 2
    sl = np.asarray(arr[z0])
    uniq, counts = np.unique(sl, return_counts=True)
    print(f"   layer z={z0}: nonzero frac={np.mean(sl>0):.4f}, uniq values:",
          list(zip(uniq[:8].tolist(), counts[:8].tolist())))
    del arr, sl
