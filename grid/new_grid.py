from new_grid_generation import non_spherical_shell, non_spherical_shell_variable_thickness
import numpy as np
import os

SAVE_DIR = "D:/rave-sim-main/rave-sim-main/grid"
R = 500e-6    # 平均半径 500 um
T = 100e-6    # 壳厚 100 um
SCALE = 1e-6  # 1 um/voxel

os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1. 理想样品 ──
print("[1/1] Generating l=0, m=0, eps=0.00 ")
grid1 = non_spherical_shell(
    base_radius=R, thickness=T,
    l=0, m=0, epsilon=0.00,
    scale_x=SCALE, scale_y=SCALE, scale_z=SCALE
)
path1 = os.path.join(SAVE_DIR, "500um_100um_shell_l0_m0_eps000.npy")
np.save(path1, grid1)
print(f"    Saved -> {path1}, shape={grid1.shape}")