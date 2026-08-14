"""
Generate thin square plate 3D grid files for absorption/phase validation.

Each grid is a solid square plate (material index 1) with 0 = vacuum background.
Grid shape: (nz, ny, nx) matching the Sample class convention grid[z, y, x].
Voxel size: 50 nm isotropic (matching existing capsule notebooks).
Lateral extent: 25 um x 25 um (500 x 500 voxels).
"""

import numpy as np
from pathlib import Path

GRID_DIR = Path(__file__).parent
VOXEL = 5e-8  # 50 nm isotropic


def thin_square_plate_3d(thickness: float, lateral: float, voxel: float = VOXEL) -> np.ndarray:
    """Generate a 3D uint32 grid for a thin square plate.

    Args:
        thickness: Plate thickness in meters (along z)
        lateral:   Plate lateral size in meters (along x and y)
        voxel:     Voxel size in meters (default 50 nm)

    Returns:
        np.uint32 array of shape (nz, ny, nx), all 1s (material index 1).
    """
    nz = int(round(thickness / voxel))
    ny = int(round(lateral / voxel))
    nx = int(round(lateral / voxel))
    return np.ones((nz, ny, nx), dtype=np.uint32)


if __name__ == "__main__":
    specs = [
        # (filename_suffix, thickness_m, lateral_m, material_label)
        ("thin_w_plate_1um_200um",   1e-6,  200e-6, "W"),
        ("thin_w_plate_2um_200um",   2e-6,  200e-6, "W"),
        ("thin_w_plate_5um_200um",   5e-6,  200e-6, "W"),
        ("thin_c_plate_10um_25um",   10e-6, 25e-6, "C"),
        ("thin_c_plate_20um_25um",   20e-6, 25e-6, "C"),
    ]

    print("=" * 60)
    print("Generating thin square plate 3D grids")
    print("=" * 60)
    print(f"Voxel size: {VOXEL*1e9:.0f} nm")
    total_size_mb = 0
    for name, thick, lateral, mat in specs:
        grid = thin_square_plate_3d(thick, lateral)
        path = GRID_DIR / f"{name}.npy"
        np.save(str(path), grid)
        size_mb = grid.nbytes / 1e6
        total_size_mb += size_mb
        print(f"  {name}.npy")
        print(f"    shape={list(grid.shape)}  ({thick*1e6:.0f} um x {lateral*1e6:.0f} um x {lateral*1e6:.0f} um)")
        print(f"    dtype={grid.dtype}  {size_mb:.0f} MB  material={mat}")
    print("-" * 60)
    print(f"Total: {len(specs)} files, {total_size_mb:.0f} MB")
    print("Done.")
