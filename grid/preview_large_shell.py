"""Preview large 3D shell arrays (>1GB) using memory-mapped sampling."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FILES = [
    ("500um_100um_hollow_spherical.npy", "Spherical shell"),
    ("500um_100um_shell_l1_m0_eps002.npy", "Non-spherical l=1,m=0 ε=0.02"),
    ("500um_100um_hollow_non_spherical_l1_m0_epsilon005.npy", "Non-spherical l=1,m=0 ε=0.05"),
]

for fname, label in FILES:
    print(f"\n{'='*60}")
    print(f"File: {fname}")
    print(f"Label: {label}")

    arr = np.load(fname, mmap_mode='r')
    nz, ny, nx = arr.shape
    dt = arr.dtype
    filled = np.sum(arr)
    total = arr.size
    fill_pct = 100 * filled / total
    print(f"  Shape: ({nz}, {ny}, {nx}), dtype: {dt}")
    print(f"  Filled: {filled}/{total} = {fill_pct:.2f}%")

    # Expected shell volume fraction for a thin shell:
    # V_shell / V_cube ≈ (4π/3 * (R³ - r³)) / (2R)³
    # R=500, r=400 → V_shell ≈ 4π/3*(125-64)e6 = 255.5e6
    # cube ≈ 1000³ = 1e9 → ~25.5%
    if "spherical" in fname.lower():
        expected_pct = 100 * (4*np.pi/3 * (500**3 - 400**3)) / 1000**3
        print(f"  Expected fill (spherical approx): {expected_pct:.1f}%")

    # Mid-plane slice stats
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    print(f"  Center slice z={cz}: {arr[cz].sum()}/{arr[cz].size} filled ({100*arr[cz].sum()/arr[cz].size:.1f}%)")

    # Check values
    vals = set()
    for i in range(0, nz, nz//10):
        vals.update(np.unique(arr[i]))
    print(f"  Sampled unique values: {sorted(vals)}")

    del arr
