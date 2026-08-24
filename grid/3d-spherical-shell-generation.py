from new_grid_generation import hollow_sphere
import numpy as np
# 600µm 平均半径，10µm 壳厚，l=2,m=2 ε=0.15 → 四极变形壳层
grid = hollow_sphere(600e-6, 500e-6, 1e-6, 1e-6, 1e-6)
np.save("D:/rave-sim-main/rave-sim-main/grid/500um_100um_perfect_shell.npy", grid)
# grid shape ≈ (2501, 2501, 2501), dtype=uint8