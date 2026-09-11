#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1 Case B: reduced-FOV CB-BPM vs direct-Cartesian-spherical-wave correctness
case. Geometry: point source z=0, 10um C plate (25um x 25um, 200 layers @50nm) at
z=1.0 m, detector z=4.0 m (M=4). Reduced sim: nx=2048, dx=200 nm (FOV 410 um >=
detector 200 um, carrier-sampled at z=1/4 m). Three engine legs share the exact
same config except mode flags:
  fastwave: cone_beam_bpm (use_fresnel_scaling+use_cone_beam_bpm)
  fastwave: off          (neither)
  big-wave CPU: off      (direct spherical exp(ikr)/r; independent implementation)
"""
import shutil
from pathlib import Path

ROOT = Path("/mnt/d/rave-sim-main/rave-sim-main")
AR = ROOT / "output/_agent_runs"
BASE = AR / "m1b_case"
BASE.mkdir(parents=True, exist_ok=True)

GRID_SRC = AR / "thin_family_prep/C10__fson/thin_c_plate_10um_25um.npy"
SUBCFG = AR / "thin_family_prep/C10__fson/00000000/subconfig.yaml"

COMMON = """
sim_params:
  is2d: 'true'
  use_fresnel_scaling: {fres}
  use_cone_beam_bpm: {cone}
  N: 4194304
  nx: 2048
  ny: 2048
  dx: 2e-07
  dy: 2e-07
  z_detector: 4.0
  detector_size: 0.0002
  detector_size_y: 0.0002
  detector_pixel_size_x: 2e-07
  detector_pixel_size_y: 2e-07
  chunk_size: 4194304
use_disk_vector: false
save_debug_wavefields: false
save_final_u_vectors: false
dtype: c8
multisource:
  type: points
  energy_range:
    - 9999
    - 10001
  x_range:
    - -1e-06
    - 1e-06
  y_range:
    - -1e-06
    - 1e-06
  z: 0.0
  nr_source_points: 50
  seed: 1
elements:
  - type: sample
    z_start: 1.0
    pixel_size_x: 5e-08
    pixel_size_y: 5e-08
    pixel_size_z: 5e-08
    grid_path: thin_c_plate_10um_25um.npy
    materials:
      -   - H
          - 0.255
    x_positions:
      - 0
    y_positions:
      - 0
"""

for tag, fres, cone in [("cbbpm", "true", "true"), ("off", "false", "false")]:
    d = BASE / f"{tag}_prep"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(COMMON.format(fres=fres, cone=cone))
    shutil.copy(GRID_SRC, d / "thin_c_plate_10um_25um.npy")
    sub = d / "00000000"; sub.mkdir(exist_ok=True)
    shutil.copy(SUBCFG, sub / "subconfig.yaml")
    # 50 subconfigs identical (source ensemble like the thin runs)
    for i in range(1, 50):
        s2 = d / f"{i:08d}"; s2.mkdir(exist_ok=True)
        shutil.copy(SUBCFG, s2 / "subconfig.yaml")
    print("prep:", d)
print("M1B PREPS READY")
