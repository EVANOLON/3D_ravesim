"""
Generate 3D grid for 900μm diameter, 50μm thick HDC spherical shell.
Outer radius = 450μm, inner radius = 400μm, voxel size = 1μm.
Grid: ~900³ voxels, uint32, ~2.9 GB.

Usage: python generate_hdc_shell_grid.py
"""

import numpy as np
import math
import time

OUTER_RADIUS = 450e-6   # 450 μm
INNER_RADIUS = 400e-6   # 400 μm (50 μm shell thickness)
SCALE = 1e-6            # 1 μm voxel size

# Grid dimensions (must contain the full sphere)
grid_len = int(math.ceil(OUTER_RADIUS / SCALE * 2))  # 900
print(f"Grid dimensions: {grid_len} x {grid_len} x {grid_len}")
print(f"Estimated size: {grid_len**3 * 4 / 1e9:.2f} GB")

# Create coordinate arrays (centered)
t0 = time.time()
coords = (np.arange(grid_len, dtype=np.float64) - grid_len / 2.0) * SCALE
print(f"Coordinate array: {coords[0]:.1e} to {coords[-1]:.1e}, step {SCALE:.1e}")

# Use broadcasting to compute r² for all voxels.
# Process in chunks to reduce memory usage.
CHUNK_SIZE = 200  # Process CHUNK_SIZE z-slices at a time
grid = np.zeros((grid_len, grid_len, grid_len), dtype=np.uint32)

for z_start in range(0, grid_len, CHUNK_SIZE):
    z_end = min(z_start + CHUNK_SIZE, grid_len)
    z_slice = coords[z_start:z_end]

    # Create 3D r² for this chunk using broadcasting
    Z = z_slice[:, np.newaxis, np.newaxis]  # shape: (chunk, 1, 1)
    Y = coords[np.newaxis, :, np.newaxis]    # shape: (1, grid_len, 1)
    X = coords[np.newaxis, np.newaxis, :]    # shape: (1, 1, grid_len)

    r2 = Z**2 + Y**2 + X**2  # shape: (chunk, grid_len, grid_len)

    # Shell condition: inner_radius² ≤ r² ≤ outer_radius²
    mask = (r2 >= INNER_RADIUS**2) & (r2 <= OUTER_RADIUS**2)
    grid[z_start:z_end, :, :] = mask.astype(np.uint32)

    elapsed = time.time() - t0
    print(f"  z slices {z_start}-{z_end-1} done ({z_end}/{grid_len}), elapsed: {elapsed:.1f}s")

total_elapsed = time.time() - t0
voxels_filled = np.sum(grid)
print(f"\nGrid generation complete in {total_elapsed:.1f}s")
print(f"Grid shape: {grid.shape}, dtype: {grid.dtype}")
print(f"Voxels filled (shell=1): {voxels_filled:,}")
print(f"Shell volume estimate: {4/3*np.pi*(OUTER_RADIUS**3 - INNER_RADIUS**3) / SCALE**3:.0f} voxels")

# Save grid
output_path = "D:/rave-sim-main/rave-sim-main/grid/900um_50um_hdc_shell.npy"
np.save(output_path, grid)
print(f"Saved to: {output_path}")
