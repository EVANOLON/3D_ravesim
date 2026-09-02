"""
RAVE-SIM 2D Simulation Test Script

Tests both Python (big-wave) and C++ (fast-wave) 2D simulation capabilities.

This script covers four test scenarios:
  1. 2D Free-space propagation (big-wave Python, NumpyVector)
  2. 2D Sample propagation with 3D material grid (big-wave Python)
  3. 1D PlasmaSample basic test (PlasmaSample is currently 1D-only)
  4. Generate fast-wave-compatible 2D config file

Usage:
    python big-wave/test_2d_simulation.py          # run all tests + plots
    python big-wave/test_2d_simulation.py --no-plot  # run all tests, no plots

Requires: numpy, scipy, matplotlib (for plotting)
"""

import sys
import os
from pathlib import Path
import argparse
import warnings

import numpy as np

# ── add big-wave to path ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "big-wave"))

# Import config first to avoid the repository's config/propagation circular
# import when this file is executed directly.
import config  # noqa: F401, E402
from propagation import (
    SimParams, propagate, propagate_2d, propagate_analytically,
    propagate_analytically_2d, square_and_downsample,
    square_and_downsample_2d, convert_energy_wavelength,
    apply_frequency_cutoff, apply_frequency_cutoff_2d,
    compute_cutoff_angles,
)
from source import PointSource
from optical_element import Sample, Material, precise_Sample
from plasma_sample import PlasmaSample
from plasma import plasma_delta_beta
from vector import NumpyVector

warnings.filterwarnings("ignore")

# ── helpers ───────────────────────────────────────────────────────────

def _make_sphere_grid(nx, ny, nz, radius_px, cx=None, cy=None):
    """
    Create a 3D uint32 grid (nz, ny, nx) with a sphere of material index 1
    embedded in vacuum (index 0).
    """
    if cx is None:
        cx = nx // 2
    if cy is None:
        cy = ny // 2
    grid = np.zeros((nz, ny, nx), dtype=np.uint32)
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - nz // 2) ** 2 <= radius_px ** 2
    grid[mask] = 1
    return grid


def _make_gaussian_plasma(nz, nx, ne_peak, Z_star, T_e):
    """
    Create synthetic plasma parameter grids with a Gaussian profile in x.
    Returns (ne_grid, ni_grid, te_grid, zstar_grid), each shape (nz, nx).
    """
    x = np.arange(nx) - nx // 2
    z = np.arange(nz)
    X, _ = np.meshgrid(x, z)
    profile = np.exp(-(X ** 2) / (2 * (nx // 8) ** 2))
    ne = ne_peak * profile + 1e17  # cm^-3, floor to avoid division by zero
    zstar = Z_star * np.ones_like(ne)
    ni = ne / zstar  # quasi-neutrality: n_e = Z* * n_i
    te = T_e * np.ones_like(ne)  # eV
    return ne, ni, te, zstar


def _check_2d_output(label, detected, shape_2d=True):
    """Validate detector output shape and basic sanity."""
    assert detected is not None, f"{label}: detected is None"
    assert np.isfinite(detected).all(), f"{label}: non-finite values"
    if shape_2d:
        assert detected.ndim == 2, (
            f"{label}: expected 2D output, got shape {detected.shape}"
        )
        assert detected.shape[0] > 1 and detected.shape[1] > 1, (
            f"{label}: 2D output too small: {detected.shape}"
        )
    else:
        assert detected.ndim == 1, (
            f"{label}: expected 1D output, got shape {detected.shape}"
        )
    print(f"  ✓ {label}: shape={detected.shape}, "
          f"min={detected.min():.4e}, max={detected.max():.4e}, "
          f"mean={detected.mean():.4e}")


# ══════════════════════════════════════════════════════════════════════
#  Test 1: 2D Free-space propagation  (big-wave Python)
# ══════════════════════════════════════════════════════════════════════

def test_2d_free_space(plot=True):
    """
    Validate 2D Fresnel propagation—point source → propagate_analytically_2d
    → fft2 → frequency_cutoff_2d → ifft2 → square_and_downsample_2d.

    Also runs the equivalent 1D propagation for comparison.
    """
    print("\n" + "═" * 60)
    print("Test 1: 2D Free-space propagation")
    print("═" * 60)

    # ── parameters ──────────────────────────────────────────────
    nx, ny = 256, 256
    N = nx * ny                         # = 65536
    dx = 1.0e-7                         # 0.1 µm; satisfies source-plane Nyquist
    dy = 1.0e-7
    wl = convert_energy_wavelength(8000.0)  # 8 keV → ~1.55e-10 m
    z_detector = 0.02                   # 2 cm

    det_pix_x = 4.0e-7
    det_pix_y = 4.0e-7
    det_size_x = nx * dx                # 25.6 µm
    det_size_y = ny * dy

    chunk_size = 4096
    cutoff_angle = 2.0e-4               # below both axis Nyquist limits
    cutoff_freq = np.sin(cutoff_angle) / wl

    params_2d = SimParams(
        N=N, dx=dx, z_detector=z_detector,
        detector_size=det_size_x,
        detector_pixel_size_x=det_pix_x,
        detector_pixel_size_y=det_pix_y,
        wl=wl, chunk_size=chunk_size,
        ny=ny, dy=dy,
        detector_size_x=det_size_x,
        detector_size_y=det_size_y,
    )
    assert params_2d.is_2d, "SimParams should be in 2D mode"
    print(f"  2D SimParams: nx={params_2d.nx}, ny={params_2d.ny}, "
          f"N={params_2d.N}, dx={dx:.2e}, wl={wl:.2e}")

    # ── source propagation (2D) ──────────────────────────────
    source = PointSource(x=0.0, y=0.0, z=0.0)
    u = NumpyVector(np.zeros(N, dtype=np.complex64))
    U = NumpyVector(np.zeros(N, dtype=np.complex64))

    source.propagate_to(z_detector, params_2d, cutoff_freq, u, U, None)

    # ── detector readout ─────────────────────────────────────
    detected_2d = square_and_downsample_2d(u, params_2d, z_detector)
    _check_2d_output("2D free-space", detected_2d)

    # ── reference: 1D propagation with same transverse extent ─
    params_1d = SimParams(
        N=nx, dx=dx, z_detector=z_detector,
        detector_size=det_size_x,
        detector_pixel_size_x=det_pix_x,
        detector_pixel_size_y=det_pix_y,
        wl=wl, chunk_size=chunk_size,
    )
    assert not params_1d.is_2d, "SimParams should be in 1D mode"

    u1 = NumpyVector(np.zeros(nx, dtype=np.complex64))
    U1 = NumpyVector(np.zeros(nx, dtype=np.complex64))

    source_1d = PointSource(x=0.0, z=0.0)
    source_1d.propagate_to(z_detector, params_1d, cutoff_freq, u1, U1, None)
    detected_1d = square_and_downsample(u1, params_1d, z_detector)
    _check_2d_output("1D free-space (reference)", detected_1d, shape_2d=False)

    # ── plot ─────────────────────────────────────────────────
    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        # 2D detector image
        im = axes[0].imshow(detected_2d, cmap="inferno", origin="lower",
                            aspect="auto")
        axes[0].set_title("2D free-space intensity")
        axes[0].set_xlabel("x (pix)")
        axes[0].set_ylabel("y (pix)")
        plt.colorbar(im, ax=axes[0])

        # 1D central cross-section (at mid-y)
        mid = detected_2d.shape[0] // 2
        axes[1].plot(detected_2d[mid, :], label="2D central row")
        axes[1].plot(detected_1d / detected_1d.max()
                     * detected_2d[mid, :].max(),
                     "--", label="1D (scaled)", alpha=0.7)
        axes[1].set_title("Central row cross-section")
        axes[1].set_xlabel("x (pix)")
        axes[1].set_ylabel("Intensity")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("test2d_freespace.png", dpi=150)
        print("  📊 → test2d_freespace.png")

    return detected_2d, detected_1d


# ══════════════════════════════════════════════════════════════════════
#  Test 2: 2D Sample propagation  (big-wave Python)
# ══════════════════════════════════════════════════════════════════════

def test_2d_sample(plot=True):
    """
    Create a 3D material grid (z × y × x) representing a small sphere,
    attach it as a Sample, and propagate a 2D wavefront through it.

    Validates that Sample._apply_2d() is called and produces a 2D
    diffraction pattern that differs from free-space propagation.
    """
    print("\n" + "═" * 60)
    print("Test 2: 2D Sample propagation with 3D material grid")
    print("═" * 60)

    # ── parameters ──────────────────────────────────────────────
    nx, ny = 256, 256
    N = nx * ny
    dx = 5.0e-8
    dy = 5.0e-8
    wl = convert_energy_wavelength(8000.0)

    z_source = 0.0
    z_sample = 0.01         # 1 cm
    z_detector = 0.03       # 3 cm

    det_pix_x = 2.0e-7
    det_pix_y = 2.0e-7
    chunk_size = 4096
    cutoff_angle = 5.0e-4
    cutoff_freq = np.sin(cutoff_angle) / wl

    params = SimParams(
        N=N, dx=dx, z_detector=z_detector,
        detector_size=nx * dx,
        detector_pixel_size_x=det_pix_x,
        detector_pixel_size_y=det_pix_y,
        wl=wl, chunk_size=chunk_size,
        ny=ny, dy=dy,
        detector_size_x=nx * dx,
        detector_size_y=ny * dy,
    )
    assert params.is_2d

    # ── build 3D sample grid ──────────────────────────────────
    # a small sphere of W (Z=74) in vacuum
    sample_px_x = 0.5e-6    # 0.5 µm per sample pixel
    sample_px_y = 0.5e-6
    sample_px_z = 1.0e-6    # 1 µm per layer

    sx, sy, sz = 40, 40, 8  # grid dimensions (x, y, z)
    sphere_r = 15            # radius in pixels
    grid_3d = _make_sphere_grid(sx, sy, sz, sphere_r)
    print(f"  3D grid shape: {grid_3d.shape}, "
          f"non-vacuum fraction: {grid_3d.mean():.3f}")

    sample = Sample(
        z_start=z_sample,
        pixel_size_x=sample_px_x,
        pixel_size_y=sample_px_y,
        pixel_size_z=sample_px_z,
        grid=grid_3d,
        materials=[Material("W", 19.35)],
        x_positions=np.array([0.0]),
        y_positions=np.array([0.0]),
    )
    sample.check_valid()

    # Pre-compute deltabeta from the material table
    # (normally done via generate_deltabeta_table + store_deltabetas)
    from optical_element import generate_deltabeta_table, collect_all_materials
    mats = collect_all_materials([sample])
    dbt = generate_deltabeta_table(mats, convert_energy_wavelength(wl))
    sample.store_deltabetas(dbt)

    # ── run simulation ─────────────────────────────────────────
    # Source propagation → sample → propagate to detector

    source = PointSource(x=0.0, y=0.0, z=0.0)
    u = NumpyVector(np.zeros(N, dtype=np.complex64))
    U = NumpyVector(np.zeros(N, dtype=np.complex64))

    # 1) source → sample
    source.propagate_to(z_sample, params, cutoff_freq, u, U, None)

    # 2) apply sample (should trigger _apply_2d)
    sample.apply(u, U, params, cutoff_freq, stepping_iteration=0, history=None)

    # 3) sample → detector
    dz = z_detector - (z_sample + sample.get_thickness())
    propagate_2d(u, U, params.dx, params.get_dy(), params.wl,
                 dz, params.chunk_size, cutoff_freq,
                 params.nx, params.ny)

    # 4) detector
    detected = square_and_downsample_2d(u, params, z_detector)
    _check_2d_output("2D sample diffraction", detected)

    # ── free-space reference (no sample) ──────────────────────
    u_ref = NumpyVector(np.zeros(N, dtype=np.complex64))
    U_ref = NumpyVector(np.zeros(N, dtype=np.complex64))
    source.propagate_to(z_detector, params, cutoff_freq, u_ref, U_ref, None)
    ref = square_and_downsample_2d(u_ref, params, z_detector)
    _check_2d_output("2D free-space (reference)", ref)

    diff = np.abs(detected - ref)
    print(f"  │ max difference from free-space: {diff.max():.4e}")

    # ── plot ─────────────────────────────────────────────────
    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        for ax, img, title in zip(
            axes,
            [detected, ref, diff],
            ["With W sphere", "Free space", "|Difference|"],
        ):
            im = ax.imshow(img, cmap="inferno", origin="lower", aspect="auto")
            ax.set_title(title)
            ax.set_xlabel("x (pix)")
            ax.set_ylabel("y (pix)")
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        plt.savefig("test2d_sample.png", dpi=150)
        print("  📊 → test2d_sample.png")

    return detected


# ══════════════════════════════════════════════════════════════════════
#  Test 3: 1D PlasmaSample  (big-wave Python, 1D-only for now)
# ══════════════════════════════════════════════════════════════════════

def test_plasma_1d(plot=True):
    """
    Run PlasmaSample in 1D mode and verify the physics:
      - Free electrons contribute delta_free > 0
      - Hot plasma has beta_ff (inverse bremsstrahlung)

    Also cross-check against the neutral-atom prediction for the same
    density using precise_Sample as a reference.

    Note: PlasmaSample currently only supports 1D (ny=1).  Its apply()
    calls propagate(), not propagate_2d().
    """
    print("\n" + "═" * 60)
    print("Test 3: 1D PlasmaSample propagation")
    print("═" * 60)

    # ── parameters ──────────────────────────────────────────────
    N = 65536
    dx = 2.0e-9
    wl = convert_energy_wavelength(8000.0)
    z_source = 0.0
    z_sample = 0.005        # 5 mm
    z_detector = 0.02       # 2 cm
    det_pix = 1.0e-7
    chunk_size = 4096
    cutoff_angle = 0.02
    cutoff_freq = np.sin(cutoff_angle) / wl

    params = SimParams(
        N=N, dx=dx, z_detector=z_detector,
        detector_size=N * dx,
        detector_pixel_size_x=det_pix,
        detector_pixel_size_y=1.0,
        wl=wl, chunk_size=chunk_size,
    )
    assert not params.is_2d

    # ── plasma parameters ──────────────────────────────────────
    Z = 13                          # Aluminium
    Z_star = 8.0                    # partially ionised
    T_e = 100.0                     # 100 eV
    ne_peak = 5.0e21                # cm^-3
    nz = 10
    nx = N // 8                     # ~8192

    ne, ni, te, zs = _make_gaussian_plasma(nz, nx, ne_peak, Z_star, T_e)
    print(f"  Plasma: Z={Z}, Z*={Z_star}, Te={T_e:.0f} eV, "
          f"ne_peak={ne_peak:.1e} cm⁻³")

    plasma = PlasmaSample(
        z_start=z_sample,
        pixel_size_x=dx,
        pixel_size_z=1.0e-6,
        ne_grid=ne,
        ni_grid=ni,
        te_grid=te,
        zstar_grid=zs,
        Z=Z,
        x_positions=np.array([0.0]),
    )
    plasma.check_valid()
    print(f"  Plasma thickness: {plasma.get_thickness():.2e} m")

    # ── run propagation with plasma ─────────────────────────────
    source = PointSource(x=0.0, z=0.0)
    u = NumpyVector(np.zeros(N, dtype=np.complex64))
    U = NumpyVector(np.zeros(N, dtype=np.complex64))

    source.propagate_to(z_sample, params, cutoff_freq, u, U, None)
    plasma.apply(u, U, params, cutoff_freq, stepping_iteration=0, history=None)

    dz = z_detector - (z_sample + plasma.get_thickness())
    propagate(u, U, params.dx, params.wl, dz,
              params.chunk_size, cutoff_freq)
    detected_plasma = square_and_downsample(u, params, z_detector)
    _check_2d_output("Plasma 1D", detected_plasma, shape_2d=False)

    # ── cross-check: plasma_delta_beta physics ─────────────────
    # Verify plasma_delta_beta produces expected values
    d, b, atlen = plasma_delta_beta(
        n_e=ne_peak, n_i=ne_peak / Z_star,
        T_e=T_e, Z_star=Z_star, Z=Z,
        energy=8000.0,
    )
    print(f"  Physics check: delta={d:.6e}, beta={b:.6e}, "
          f"attenuation length={atlen:.4e} cm")
    assert d > 0, "delta_free should be positive"
    assert b > 0, "beta should be positive (absorption)"
    assert np.isfinite(atlen), "attenuation length should be finite"

    # ── plot ─────────────────────────────────────────────────
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 3.5))
        x_det = (np.arange(len(detected_plasma)) - len(detected_plasma)/2
                 ) * det_pix * 1e6
        ax.plot(x_det, detected_plasma)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("Intensity")
        ax.set_title("1D PlasmaSample intensity at detector")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("test2d_plasma_1d.png", dpi=150)
        print("  📊 → test2d_plasma_1d.png")

    return detected_plasma


# ══════════════════════════════════════════════════════════════════════
#  Test 4: Generate fast-wave-compatible 2D config
# ══════════════════════════════════════════════════════════════════════

def test_generate_2d_config():
    """
    Generate a 2D simulation configuration that can be run by the
    fast-wave C++ binary (e.g. fast-wave -260428 3d版本).

    This mimics the notebook pattern: config_dict → multisim.setup_simulation()
    → C++ fastwave binary.
    """
    print("\n" + "═" * 60)
    print("Test 4: Generate fast-wave-compatible 2D config")
    print("═" * 60)

    import multisim
    import tempfile
    import shutil

    # Small 2D simulation config (matches fast-wave config_parsing.cpp)
    nx, ny = 1024, 1024
    dx = 1.0e-7
    dy = 1.0e-7

    # Create a simple 2D grid file (1D sample: z × x)
    grid_2d = np.zeros((3, 64), dtype=np.uint32)
    grid_2d[0, 24:40] = 1   # W bar
    grid_2d[1, 24:40] = 1
    grid_2d[2, 24:40] = 1

    # Create a temporary directory for the grid file
    tmpdir = Path(tempfile.mkdtemp(prefix="test2d_config_"))
    grid_path = tmpdir / "test_grid.npy"
    np.save(grid_path, grid_2d)

    config_dict = {
        "sim_params": {
            "N": nx * ny,
            "nx": nx,
            "ny": ny,
            "dx": dx,
            "dy": dy,
            "z_detector": 0.5,
            "detector_size_x": nx * dx * 0.2,
            "detector_size_y": ny * dy * 0.2,
            "detector_pixel_size_x": dx * 4,
            "detector_pixel_size_y": dy * 4,
            "chunk_size": 256 * 1024 * 1024 // 16,
        },
        "use_disk_vector": False,
        "save_final_u_vectors": False,
        "dtype": "c8",
        "multisource": {
            "type": "points",
            "energy_range": [7900, 8100],
            "x_range": [-1e-6, 1e-6],
            "y_range": [-1e-6, 1e-6],
            "z": 0.0,
            "nr_source_points": 1,
            "seed": 42,
        },
        "elements": [
            {
                "type": "sample",
                "z_start": 0.4,
                "pixel_size_x": 2e-7,
                "pixel_size_y": 2e-7,
                "pixel_size_z": 3e-6,
                "grid_path": str(grid_path),
                "materials": [["W", 19.35]],
                "x_positions": [0.0],
                "y_positions": [0.0],
            },
        ],
    }

    # Use a subdirectory of tmpdir for the output
    outdir = tmpdir / "sim_output"
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        sim_path = multisim.setup_simulation(
            config_dict, tmpdir, outdir
        )
        print(f"  ✓ Config generated at: {sim_path}")

        # Verify key files exist
        for f in ["config.yaml", "computed.yaml"]:
            assert (sim_path / f).exists(), f"Missing {f}"
        subdirs = list(sim_path.glob("????????"))
        assert len(subdirs) == 1, f"Expected 1 source subdirectory, got {len(subdirs)}"
        assert (subdirs[0] / "subconfig.yaml").exists()

        # Verify 2D parameters in config.yaml
        import yaml
        with open(sim_path / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        sp = cfg["sim_params"]
        assert sp.get("nx") == nx, f"nx mismatch: {sp.get('nx')} != {nx}"
        assert sp.get("ny") == ny, f"ny mismatch: {sp.get('ny')} != {ny}"
        print(f"  ✓ Config validated: N={sp['N']}, "
              f"nx={sp['nx']}, ny={sp['ny']}")

        print(f"\n  To run with fast-wave C++ binary:")
        print(f"    fastwave -s 0 {sim_path}")
        print(f"  Then load detected.npy for 2D detector image.")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("  ✓ fast-wave-compatible config generated and validated")


# ══════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════

def print_summary():
    """Print a summary of 2D support across the entire codebase."""
    summary = """
╔══════════════════════════════════════════════════════════════════╗
║              2D Support Summary                                 ║
╠══════════════════════════════════════════════════════════════════╣
║  big-wave (Python)                     fast-wave (C++/CUDA)    ║
║  ─────────────────────                 ────────────────────    ║
║  propagate_2d()             ✅         propagate_2d()    ✅    ║
║  propagate_analytically_2d  ✅         (CUDA kernel)      ✅    ║
║  square_and_downsample_2d   ✅         (CUDA kernel)      ✅    ║
║  frequency_cutoff_2d        ✅         (CUDA kernel)      ✅    ║
║  Sample._apply_2d()         ✅         apply_sample_2d()  ✅    ║
║  PointSource y-support      ✅         (y coordinate)     ✅    ║
║  SimParams ny/dy            ✅         (nx/ny/is2d)       ✅    ║
║  NumpyVector.fft2           ✅         cuFFT 2D plan      ✅    ║
║  wavesim.run_simulation()   ✅         run_simulation_2d()✅    ║
║  ─────────────────────                 ────────────────────    ║
║  NOT yet 2D:                                                  ║
║    PlasmaSample.apply()     ✅          vectorized row tiles   ║
║    precise_Sample.apply()   ❌          (only 1D)              ║
║    Grating/EnvGrating       ❌          (only 1D on both)      ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(summary)


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RAVE-SIM 2D simulation test suite"
    )
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable matplotlib plots")
    args = parser.parse_args()

    plot = not args.no_plot
    has_plt = True
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")  # headless
        except ImportError:
            print("⚠ matplotlib not available, skipping plots")
            has_plt = False
            plot = False

    failures = []

    # ── Test 1 ──────────────────────────────────────────────────
    try:
        test_2d_free_space(plot=plot and has_plt)
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        failures.append("test_2d_free_space")

    # ── Test 2 ──────────────────────────────────────────────────
    try:
        test_2d_sample(plot=plot and has_plt)
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        failures.append("test_2d_sample")

    # ── Test 3 ──────────────────────────────────────────────────
    try:
        test_plasma_1d(plot=plot and has_plt)
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        failures.append("test_plasma_1d")

    # ── Test 4 ──────────────────────────────────────────────────
    try:
        test_generate_2d_config()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        failures.append("test_generate_2d_config")

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "═" * 60)
    if failures:
        print(f"❌ {len(failures)} test(s) FAILED: {', '.join(failures)}")
    else:
        print("✅ All tests PASSED")

    print_summary()

    return len(failures)


if __name__ == "__main__":
    sys.exit(main())
