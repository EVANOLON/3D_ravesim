"""3D isosurface rendering of 500um_100um_perfect_shell.npy."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from skimage import measure
from pathlib import Path
import sys

GRID_DIR = Path("D:/rave-sim-main/rave-sim-main/grid")

# --- Load ---
path = GRID_DIR / "500um_100um_perfect_shell.npy"
print(f"Loading {path} ...")
arr = np.load(path, mmap_mode='r')
print(f"  Shape: {arr.shape}, dtype: {arr.dtype}, filled: {arr.sum()} / {arr.size}")

# --- Downsample to ~100^3 for marching cubes ---
if max(arr.shape) > 150:
    step = 2  # 600^3, much finer surface
    small = arr[::step, ::step, ::step].astype(np.float64)
    print(f"  Downsampled to {small.shape} for 3D render (step={step})")
else:
    small = arr.astype(np.float64)
    step = 1

# --- Marching cubes ---
print("  Running marching cubes ...")
verts, faces, _, _ = measure.marching_cubes(small, level=0.5)

# Scale to physical units (um)
# Voxel pitch = 1 um, and after step downsampling each voxel = step * 1 um
pitch = 1e-6
verts_phys = verts * step * pitch * 1e6
print(f"  Vertices: {len(verts)}, Faces: {len(faces)}")

# --- Plot ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts_phys[:, 0], verts_phys[:, 1], faces, verts_phys[:, 2],
                cmap='viridis', lw=0.1, edgecolor='none', alpha=0.9)
ax.set_xlabel('x (um)')
ax.set_ylabel('y (um)')
ax.set_zlabel('z (um)')
ax.set_title('Perfect shell (500um outer radius, 100um wall)')

# Equal aspect
max_range = max(np.ptp(verts_phys[:, 0]), np.ptp(verts_phys[:, 1]), np.ptp(verts_phys[:, 2]))
mid = [verts_phys[:, i].mean() for i in range(3)]
ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)
ax.view_init(elev=25, azim=45)

plt.tight_layout()
out_path = GRID_DIR / "500um_100um_perfect_shell_3d_preview.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Saved: {out_path}")
print("Done!")
