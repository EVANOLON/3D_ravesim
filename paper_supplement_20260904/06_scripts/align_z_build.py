#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build z-aligned capsule grids: pad vacuum layers at the grid FRONT (axis 0,
the near-source end, since grid layer i sits at z = z_start + i*1um) so that all
shell centers land at z-layer 611.5 (depth 3.5mm + 611.5um = 4.1115mm), matching
the l1 (1224-layer) grid. Front pad P = (1224 - nz)/2:
  perfect: nz=1200 -> P=12 ; varT: nz=1206 -> P=9 ; l1: unchanged (nz=1224).
Also build fresh prep dirs with config/computed copied from the l1 run dir and the
grid_path swapped to the aligned file.  Material index 0 = vacuum (padding value).
"""
import numpy as np
from pathlib import Path
import shutil
import sys

ROOT = Path("/mnt/d/rave-sim-main/rave-sim-main")
SRC_GRIDS = {
    "perfect": ROOT / "grid" / "500um_100um_perfect_shell.npy",
    "varT":    ROOT / "grid" / "500um_100um_shell_l1_varT_er002_et005_rot90.npy",
}
L1_CFG = ROOT / "output/_agent_runs/l1_m0_eps002__agentrun_20260901101237__agentrun_20260901121843"
PREP = ROOT / "output/_agent_runs/align_z"
PREP.mkdir(parents=True, exist_ok=True)

NZ_TARGET = 1224  # all aligned grids span the same 1224 z-layers as l1


def bounds_from_stream(src):
    """Single sequential pass over z layers; return z/y/x bounds of nonzero content."""
    nz, ny, nx = src.shape
    row_any = np.zeros(ny, dtype=bool)
    col_any = np.zeros(nx, dtype=bool)
    zany = np.zeros(nz, dtype=bool)
    for i in range(nz):
        sl = np.asarray(src[i])
        a = sl.any()
        zany[i] = a
        if a:
            r = sl.any(axis=1)
            row_any |= r
            col_any |= sl.any(axis=0)
    zz = np.nonzero(zany)[0]
    yy = np.nonzero(row_any)[0]
    xx = np.nonzero(col_any)[0]
    return (int(zz[0]), int(zz[-1])), (int(yy[0]), int(yy[-1])), (int(xx[0]), int(xx[-1]))


def build(label, src_path, out_name, P, center_target=611.5):
    src = np.load(src_path, mmap_mode="r")
    nz, ny, nx = src.shape
    print(f"[{label}] src {src_path.name}: shape={src.shape} dtype={src.dtype} P_front={P}", flush=True)
    assert src.dtype == np.uint32, src.dtype
    assert nz + 2 * P == NZ_TARGET, (nz, P)
    out_path = PREP / out_name
    dst = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.uint32,
                                    shape=(NZ_TARGET, ny, nx))
    row_any = np.zeros(ny, dtype=bool)
    col_any = np.zeros(nx, dtype=bool)
    zany = np.zeros(NZ_TARGET, dtype=bool)
    for i in range(nz):
        sl = np.asarray(src[i])
        dst[P + i] = sl
        zany[P + i] = bool(sl.any())
        if zany[P + i]:
            row_any |= sl.any(axis=1)
            col_any |= sl.any(axis=0)
    dst.flush()
    zz = np.nonzero(zany)[0]
    yy = np.nonzero(row_any)[0]
    xx = np.nonzero(col_any)[0]
    zb, yb, xb = (int(zz[0]), int(zz[-1])), (int(yy[0]), int(yy[-1])), (int(xx[0]), int(xx[-1]))
    zc, yc, xc = (zb[0]+zb[1])/2, (yb[0]+yb[1])/2, (xb[0]+xb[1])/2
    print(f"[{label}] aligned bbox z={zb} y={yb} x={xb}", flush=True)
    print(f"[{label}] aligned centers z={zc:.1f} y={yc:.1f} x={xc:.1f}  (target {center_target})",
          flush=True)
    del dst, src
    return zc, yc, xc, out_path


def make_prep(label, grid_basename):
    d = PREP / f"{label}_prep"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(L1_CFG / "computed.yaml", d / "computed.yaml")
    txt = (L1_CFG / "config.yaml").read_text()
    txt = txt.replace("grid_path: 500um_100um_shell_l1_m0_eps002_rot90.npy",
                      f"grid_path: {grid_basename}")
    (d / "config.yaml").write_text(txt)
    print(f"[{label}] prep dir ready: {d}", flush=True)
    return d


if __name__ == "__main__":
    jobs = [
        ("perfect_zalign", SRC_GRIDS["perfect"], "500um_100um_perfect_shell_zalign1224.npy", 12),
        ("varT_zalign", SRC_GRIDS["varT"], "500um_100um_shell_l1_varT_zalign1224.npy", 9),
    ]
    for label, sp, out_name, P in jobs:
        zc, yc, xc, out_path = build(label, sp, out_name, P)
        ok = abs(zc - 611.5) < 2.0
        print(f"[{label}] z-center check {'OK' if ok else 'FAIL'} (zc={zc:.1f})", flush=True)
        if not ok:
            sys.exit(1)
        make_prep(label, out_name)
    print("ALL ALIGNED GRIDS BUILT", flush=True)
