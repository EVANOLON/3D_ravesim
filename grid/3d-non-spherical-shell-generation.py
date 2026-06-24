"""生成三种典型 ICF 靶丸静态制造缺陷的 3D 壳层网格 (uint32)

1. 整体椭球度 l=1 (偶极偏心率) — 靶丸沿 z 轴偏移
2. 整体椭球度 l=2 (四极椭球度) — 椭球 / 扁椭球
3. 壁厚不均匀 (同心度偏差) — 内腔偏移导致一侧壁厚一侧壁薄
"""
from new_grid_generation import non_spherical_shell, non_spherical_shell_variable_thickness
import numpy as np
import os

SAVE_DIR = "D:/rave-sim-main/rave-sim-main/grid"
R = 500e-6    # 平均半径 500 um
T = 100e-6    # 壳厚 100 um
SCALE = 1e-6  # 1 um/voxel

os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1. 整体椭球度 l=1 (偶极偏心率) ──
# 靶丸沿 z 轴偏移，偏心量 ~ eps*R*sqrt(3/(4pi)) ~ 0.02*500*0.489 ~ 4.9 um
print("[1/3] Generating l=1, m=0, eps=0.02 (dipole eccentricity)...")
grid1 = non_spherical_shell(
    base_radius=R, thickness=T,
    l=1, m=0, epsilon=0.02,
    scale_x=SCALE, scale_y=SCALE, scale_z=SCALE
)
path1 = os.path.join(SAVE_DIR, "500um_100um_shell_l1_m0_eps002.npy")
np.save(path1, grid1)
print(f"    Saved -> {path1}, shape={grid1.shape}")

# ── 2. 整体椭球度 l=2 (四极椭球度) ──
# P2 模椭球，赤道比两极厚/薄 ~ 2%
print("[2/3] Generating l=2, m=0, eps=0.02 (P2 ellipsoid)...")
grid2 = non_spherical_shell(
    base_radius=R, thickness=T,
    l=2, m=0, epsilon=0.02,
    scale_x=SCALE, scale_y=SCALE, scale_z=SCALE
)
path2 = os.path.join(SAVE_DIR, "500um_100um_shell_l2_m0_eps002.npy")
np.save(path2, grid2)
print(f"    Saved -> {path2}, shape={grid2.shape}")

# ── 3. 壁厚不均匀 (同心度偏差, l=1) ──
# 内表面 eps_r=0.02 (内腔偏移 ~4.9 um)，外表面 eps_t=0.005 (外壳偏移 ~1.2 um)
# 内腔比外壳偏移更大 -> 一侧壁厚一侧壁薄
print("[3/3] Generating variable thickness: l=1, eps_r=0.02, eps_t=0.005...")
grid3 = non_spherical_shell_variable_thickness(
    base_radius=R, thickness0=T,
    l=1, m=0,
    epsilon_r=0.02, epsilon_t=0.005,
    scale_x=SCALE, scale_y=SCALE, scale_z=SCALE
)
path3 = os.path.join(SAVE_DIR, "500um_100um_shell_l1_varT_er002_et005.npy")
np.save(path3, grid3)
print(f"    Saved -> {path3}, shape={grid3.shape}")

print("\nAll done!")
