"""Preview a 3D binary array: orthogonal slices + 3D isosurface."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from skimage import measure

# ─── Load ───────────────────────────────────────────────────────────────
arr = np.load("D:/rave-sim-main/rave-sim-main/grid/100um_half_hollow_millisphere.npy")
print(f"Shape: {arr.shape}, dtype: {arr.dtype}, filled: {arr.sum()}/{arr.size}")

nz, ny, nx = arr.shape
cx, cy, cz = nx // 2, ny // 2, nz // 2

# ─── 1. Orthogonal mid-plane slices ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# xy-plane (z mid)
axes[0].imshow(arr[cz, :, :], cmap='gray', origin='lower')
axes[0].set_title(f'XY slice at z={cz}')
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')

# yz-plane (x mid)
axes[1].imshow(arr[:, :, cx], cmap='gray', origin='lower')
axes[1].set_title(f'YZ slice at x={cx}')
axes[1].set_xlabel('y'); axes[1].set_ylabel('z')

# xz-plane (y mid)
axes[2].imshow(arr[:, cy, :], cmap='gray', origin='lower')
axes[2].set_title(f'XZ slice at y={cy}')
axes[2].set_xlabel('x'); axes[2].set_ylabel('z')

plt.tight_layout()
plt.savefig("preview_slices.png", dpi=150)
print("Saved preview_slices.png")

# ─── 2. Downsample + isosurface 3D ─────────────────────────────────────
# If full resolution, downsample to ~100³ for speed
if max(arr.shape) > 150:
    step = max(1, max(arr.shape) // 100)
    small = arr[::step, ::step, ::step].astype(np.float64)
    print(f"Downsampled to {small.shape} for 3D render (step={step})")
else:
    small = arr.astype(np.float64)
    step = 1

verts, faces, _, _ = measure.marching_cubes(small, level=0.5)

# Scale vertices back to physical units (µm)
scale_xyz = 5e-7  # from generation params
verts_phys = verts * step * scale_xyz * 1e6  # m → µm

fig2 = plt.figure(figsize=(10, 8))
ax = fig2.add_subplot(111, projection='3d')
ax.plot_trisurf(verts_phys[:, 0], verts_phys[:, 1], faces, verts_phys[:, 2],
                cmap='viridis', lw=0.1, edgecolor='none', alpha=0.9)
ax.set_xlabel('x (µm)')
ax.set_ylabel('y (µm)')
ax.set_zlabel('z (µm)')
ax.set_title(f'3D shell (l=2, m=2, ε=0.15) — downsampled {small.shape[0]}³')
# Equal aspect
max_range = max(np.ptp(verts_phys[:, 0]), np.ptp(verts_phys[:, 1]), np.ptp(verts_phys[:, 2]))
mid_x = verts_phys[:, 0].mean()
mid_y = verts_phys[:, 1].mean()
mid_z = verts_phys[:, 2].mean()
ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.savefig("preview_3d.png", dpi=150)
print("Saved preview_3d.png")
print("Done!")
