"""
PlasmaSample Phase & Absorption Validation Tests
=================================================

Follows the 3D test notebook design pattern (C_thin_sample_phase,
W_thin_sample_absorption), adapted for plasma physics verification.

Test design principles (from 3D_test_notebooks):
  1. Thin slab geometry — uniform, simple, clean extraction
  2. Thickness scan — enables linear-fit validation against theory
  3. Phase:  T = u_after / u_before  with vacuum reference → pure material phase
  4. Absorption: I0 from empty-field sim → Beer-Lambert law
  5. Analytical comparison against known physics

Tests:
  A. Phase validation — fully ionized plasma (delta = delta_free, exact analytic)
  B. Absorption validation — partially ionized plasma vs plasma_delta_beta()
  C. Kramers IB isolation — compare fully vs partially ionized at same ne

Usage:
    python big-wave/test_plasma_sample.py               # run all
    python big-wave/test_plasma_sample.py --1d-only      # 1D mode (fast)
    python big-wave/test_plasma_sample.py --2d           # 2D mode (slow)
"""

import sys
import os
from pathlib import Path
import argparse
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "big-wave"))

import config  # noqa: F401, E402
from propagation import (
    SimParams, propagate, propagate_2d, convert_energy_wavelength,
    square_and_downsample, square_and_downsample_2d,
)
from source import PointSource
from plasma_sample import PlasmaSample
from plasma import plasma_delta_beta, _kramers_beta_ff
from vector import NumpyVector

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
#  Physics constants
# ─────────────────────────────────────────────────────────────────────────
R_E_CM = 2.8179403262e-13  # classical electron radius [cm]


def _make_plasma_slab(nz, nx, ny, ne_val, Z, Z_star, T_e):
    """Create uniform plasma slab grids of shape (nz, ny, nx)."""
    ne = np.full((nz, ny, nx), ne_val, dtype=np.float64)
    ni = np.full((nz, ny, nx), ne_val / Z_star, dtype=np.float64)
    te = np.full((nz, ny, nx), T_e, dtype=np.float64)
    zs = np.full((nz, ny, nx), Z_star, dtype=np.float64)
    return ne, ni, te, zs


def _make_plasma_slab_1d(nz, nx, ne_val, Z, Z_star, T_e):
    """Create uniform plasma slab grids of shape (nz, nx) for 1D mode."""
    ne = np.full((nz, nx), ne_val, dtype=np.float64)
    ni = np.full((nz, nx), ne_val / Z_star, dtype=np.float64)
    te = np.full((nz, nx), T_e, dtype=np.float64)
    zs = np.full((nz, nx), Z_star, dtype=np.float64)
    return ne, ni, te, zs


def _compute_expected(ne, ni, T_e, Z_star, Z, energy, thickness):
    """Compute expected delta, beta, phase shift, and transmission."""
    d, b, atlen = plasma_delta_beta(ne, ni, T_e, Z_star, Z, energy)
    wl = convert_energy_wavelength(energy)
    phase = 2 * np.pi * d * thickness / wl
    transmission = np.exp(-4 * np.pi * b * thickness / wl)
    return d, b, phase, transmission, atlen


# ═══════════════════════════════════════════════════════════════════════════
#  Test A: Phase validation — fully ionized plasma
# ═══════════════════════════════════════════════════════════════════════════

def test_plasma_phase_fully_ionized(plot=True, mode="1d"):
    """
    Verify plasma phase shift for a fully ionized plasma.

    When Z* = Z, fraction_bound = 0, so:
      delta = delta_free = n_e * r_e * lambda^2 / (2*pi)      (exact)

    Test: thickness scan → linear fit of phase vs thickness → compare slope
    with the analytic delta_free formula.

    Pattern: C_thin_sample_phase.ipynb (C thin plates, 3D_test_notebooks)
    """
    print("\n" + "=" * 70)
    print("Test A: Plasma Phase Validation (fully ionized, Z* = Z)")
    print("=" * 70)

    energy = 8000.0
    wl = convert_energy_wavelength(energy)

    Z, Z_star = 13, 13       # fully ionized Al → fraction_bound = 0
    T_e = 100.0
    ne_val = 1.0e23           # cm^-3
    ni_val = ne_val / Z_star

    thicknesses_um = [20, 40, 60]
    thicknesses_m = [t * 1e-6 for t in thicknesses_um]

    # Analytic delta_free (exact, no Chantler dependency)
    lamb_cm = wl * 100.0
    delta_free = ne_val * R_E_CM * lamb_cm * lamb_cm / (2.0 * np.pi)
    slope_theory = -2 * np.pi * delta_free / wl     # rad/m

    if mode == "1d":
        N, dx = 65536, 1.0e-7
        z_sample, z_detector = 0.05, 0.15
        cutoff_angle, det_pix = 0.015, 8.0e-6
        chunk_size = 4096
        sim_nx = N // 8
        params = SimParams(N=N, dx=dx, z_detector=z_detector,
                           detector_size=20e-3,
                           detector_pixel_size_x=det_pix,
                           detector_pixel_size_y=1.0,
                           wl=wl, chunk_size=chunk_size)
        cutoff_freq = np.sin(cutoff_angle) / wl
    else:
        N, nx, ny = 8192 * 8192, 8192, 8192
        dx, dy = 1.7e-7, 1.7e-7
        z_sample, z_detector = 0.5, 2.0
        cutoff_angle, det_pix = 0.01, 4.0e-7
        chunk_size = 512 * 1024 * 1024 // 16
        params = SimParams(N=N, nx=nx, ny=ny, dx=dx, dy=dy,
                           z_detector=z_detector,
                           detector_size=800e-6,
                           detector_size_x=800e-6,
                           detector_size_y=800e-6,
                           detector_pixel_size_x=det_pix,
                           detector_pixel_size_y=det_pix,
                           wl=wl, chunk_size=chunk_size)
        cutoff_freq = np.sin(cutoff_angle) / wl
        sim_nx = nx // 8
        sim_ny = ny // 8

    measured_phases = []

    for t_um, t_m in zip(thicknesses_um, thicknesses_m):
        nz = int(t_m / 1.0e-6)

        if mode == "1d":
            ne, ni, te, zs = _make_plasma_slab_1d(nz, sim_nx, ne_val, Z, Z_star, T_e)
            plasma = PlasmaSample(
                z_start=z_sample, pixel_size_x=dx, pixel_size_z=1.0e-6,
                ne_grid=ne, ni_grid=ni, te_grid=te, zstar_grid=zs, Z=Z,
                x_positions=np.array([0.0]),
            )
            source = PointSource(x=0.0, z=0.0)
            u = NumpyVector(np.zeros(params.N, dtype=np.complex64))
            U = NumpyVector(np.zeros(params.N, dtype=np.complex64))
            source.propagate_to(z_sample, params, cutoff_freq, u, U, None)
            u_before = u.to_numpy().copy()
            plasma.apply(u, U, params, cutoff_freq, stepping_iteration=0, history=None)
            u_after = u.to_numpy().copy()

            # Extract material phase at centre
            cx = len(u_before) // 2
            T_centre = u_after[cx] / u_before[cx]
            vacuum_hw = 500
            T_vac_left = np.mean(u_after[cx - vacuum_hw - 200:cx - vacuum_hw] /
                                 u_before[cx - vacuum_hw - 200:cx - vacuum_hw])
            T_vac_right = np.mean(u_after[cx + vacuum_hw:cx + vacuum_hw + 200] /
                                  u_before[cx + vacuum_hw:cx + vacuum_hw + 200])
            phi_sample = np.angle(T_centre)
            phi_vac = (np.angle(T_vac_left) + np.angle(T_vac_right)) / 2
            phase_shift = phi_sample - phi_vac
        else:
            ne, ni, te, zs = _make_plasma_slab(nz, sim_nx, sim_ny, ne_val, Z, Z_star, T_e)
            plasma = PlasmaSample(
                z_start=z_sample, pixel_size_x=dx, pixel_size_y=dy,
                pixel_size_z=1.0e-6,
                ne_grid=ne, ni_grid=ni, te_grid=te, zstar_grid=zs, Z=Z,
                x_positions=np.array([0.0]), y_positions=np.array([0.0]),
            )
            source = PointSource(x=0.0, z=0.0, y=0.0)
            u = NumpyVector(np.zeros(params.N, dtype=np.complex64))
            U = NumpyVector(np.zeros(params.N, dtype=np.complex64))
            source.propagate_to(z_sample, params, cutoff_freq, u, U, None)
            u_before_2d = u.to_numpy().reshape(params.ny, params.nx).copy()
            plasma.apply(u, U, params, cutoff_freq, stepping_iteration=0, history=None)
            u_after_2d = u.to_numpy().reshape(params.ny, params.nx).copy()

            cx2, cy2 = params.nx // 2, params.ny // 2
            roi_hw = sim_nx // 4
            vacuum_hw_2d = sim_nx // 3
            T_sample = u_after_2d[cy2 - roi_hw//2:cy2 + roi_hw//2,
                                  cx2 - roi_hw//2:cx2 + roi_hw//2] / \
                       u_before_2d[cy2 - roi_hw//2:cy2 + roi_hw//2,
                                   cx2 - roi_hw//2:cx2 + roi_hw//2]
            T_left = u_after_2d[cy2 - roi_hw//2:cy2 + roi_hw//2,
                                cx2 - vacuum_hw_2d - 200:cx2 - vacuum_hw_2d] / \
                     u_before_2d[cy2 - roi_hw//2:cy2 + roi_hw//2,
                                 cx2 - vacuum_hw_2d - 200:cx2 - vacuum_hw_2d]
            T_right = u_after_2d[cy2 - roi_hw//2:cy2 + roi_hw//2,
                                 cx2 + vacuum_hw_2d:cx2 + vacuum_hw_2d + 200] / \
                      u_before_2d[cy2 - roi_hw//2:cy2 + roi_hw//2,
                                  cx2 + vacuum_hw_2d:cx2 + vacuum_hw_2d + 200]
            phi_sample = np.angle(np.mean(T_sample))
            phi_vac = (np.angle(np.mean(T_left)) + np.angle(np.mean(T_right))) / 2
            phase_shift = phi_sample - phi_vac

        measured_phases.append(phase_shift)
        d, b, phase_exp, trans, _ = _compute_expected(
            ne_val, ni_val, T_e, Z_star, Z, energy, t_m)
        print(f"  t={t_um:3d} um:  Delta_phi = {phase_shift:+8.4f} rad  "
              f"(expected {phase_exp:+8.4f} rad, diff {phase_shift-phase_exp:+8.4f})")

    # Linear fit
    coeffs = np.polyfit(thicknesses_m, measured_phases, 1)
    slope_meas = coeffs[0]
    slope_err = abs(slope_meas - slope_theory) / abs(slope_theory) * 100

    print(f"\n  Analytic slope  = {slope_theory:.4e} rad/m  (delta_free = {delta_free:.4e})")
    print(f"  Measured slope  = {slope_meas:.4e} rad/m  (intercept = {coeffs[1]:.4f} rad)")
    print(f"  Slope deviation = {slope_err:.2f}%")

    assert slope_err < 5.0, f"Phase slope deviation {slope_err:.2f}% exceeds 5% threshold"
    print(f"  PASS: slope deviation {slope_err:.2f}% < 5%")

    return measured_phases, slope_meas, slope_theory


# ═══════════════════════════════════════════════════════════════════════════
#  Test B: Absorption validation — partially ionized plasma
# ═══════════════════════════════════════════════════════════════════════════

def test_plasma_absorption(plot=True, mode="1d"):
    """
    Verify plasma absorption via Beer-Lambert law.

    Partially ionized Al plasma: Z=13, Z*=8 → both beta_bound and beta_ff.
    Thickness scan → fit mu from I/I0 → compare with plasma_delta_beta().

    Pattern: W_thin_sample_absorption.ipynb (W thin plates, 3D_test_notebooks)
    """
    print("\n" + "=" * 70)
    print("Test B: Plasma Absorption Validation (partially ionized Al)")
    print("=" * 70)

    energy = 8000.0
    wl = convert_energy_wavelength(energy)

    Z, Z_star = 13, 8          # partially ionized Al
    T_e = 100.0
    ne_val = 1.0e23             # cm^-3
    ni_val = ne_val / Z_star

    thicknesses_um = [50, 100, 150]
    thicknesses_m = [t * 1e-6 for t in thicknesses_um]

    if mode == "1d":
        N, dx = 65536, 1.0e-7
        z_sample, z_detector = 0.05, 0.15
        cutoff_angle, det_pix = 0.015, 8.0e-6
        chunk_size = 4096
        sim_nx = N // 8
        params = SimParams(N=N, dx=dx, z_detector=z_detector,
                           detector_size=20e-3,
                           detector_pixel_size_x=det_pix,
                           detector_pixel_size_y=1.0,
                           wl=wl, chunk_size=chunk_size)
        cutoff_freq = np.sin(cutoff_angle) / wl
    else:
        N, nx, ny = 8192 * 8192, 8192, 8192
        dx, dy = 1.7e-7, 1.7e-7
        z_sample, z_detector = 0.5, 2.0
        cutoff_angle, det_pix = 0.01, 4.0e-7
        chunk_size = 512 * 1024 * 1024 // 16
        params = SimParams(N=N, nx=nx, ny=ny, dx=dx, dy=dy,
                           z_detector=z_detector,
                           detector_size=800e-6,
                           detector_size_x=800e-6,
                           detector_size_y=800e-6,
                           detector_pixel_size_x=det_pix,
                           detector_pixel_size_y=det_pix,
                           wl=wl, chunk_size=chunk_size)
        cutoff_freq = np.sin(cutoff_angle) / wl
        sim_nx = nx // 8
        sim_ny = ny // 8

    # ── I0 reference: free-space propagation (no sample) ──
    source = PointSource(x=0.0, z=0.0, y=0.0 if mode == "2d" else None)
    if mode == "1d":
        u0 = NumpyVector(np.zeros(params.N, dtype=np.complex64))
        U0 = NumpyVector(np.zeros(params.N, dtype=np.complex64))
        source.propagate_to(z_detector, params, cutoff_freq, u0, U0, None)
        det_empty = square_and_downsample(u0, params, z_detector)
        I0 = np.mean(det_empty[len(det_empty)//2 - 20:len(det_empty)//2 + 20])
    else:
        u0 = NumpyVector(np.zeros(params.N, dtype=np.complex64))
        U0 = NumpyVector(np.zeros(params.N, dtype=np.complex64))
        source.propagate_to(z_detector, params, cutoff_freq, u0, U0, None)
        det_empty_2d = square_and_downsample_2d(u0, params, z_detector)
        cy_d, cx_d = det_empty_2d.shape[0] // 2, det_empty_2d.shape[1] // 2
        I0 = np.mean(det_empty_2d[cy_d - 20:cy_d + 20, cx_d - 20:cx_d + 20])

    print(f"  I0 (empty-field centre) = {I0:.6e}")

    transmissions = []
    d_expected, b_expected = None, None

    for t_um, t_m in zip(thicknesses_um, thicknesses_m):
        nz = int(t_m / 1.0e-6)

        if mode == "1d":
            ne, ni, te, zs = _make_plasma_slab_1d(nz, sim_nx, ne_val, Z, Z_star, T_e)
            plasma = PlasmaSample(
                z_start=z_sample, pixel_size_x=dx, pixel_size_z=1.0e-6,
                ne_grid=ne, ni_grid=ni, te_grid=te, zstar_grid=zs, Z=Z,
                x_positions=np.array([0.0]),
            )
            u = NumpyVector(np.zeros(params.N, dtype=np.complex64))
            U = NumpyVector(np.zeros(params.N, dtype=np.complex64))
            source.propagate_to(z_sample, params, cutoff_freq, u, U, None)
            plasma.apply(u, U, params, cutoff_freq, stepping_iteration=0, history=None)
            dz = z_detector - (z_sample + t_m)
            propagate(u, U, params.dx, params.wl, dz, params.chunk_size, cutoff_freq)
            det = square_and_downsample(u, params, z_detector)
            I_centre = np.mean(det[len(det)//2 - 20:len(det)//2 + 20])
        else:
            ne, ni, te, zs = _make_plasma_slab(nz, sim_nx, sim_ny, ne_val, Z, Z_star, T_e)
            plasma = PlasmaSample(
                z_start=z_sample, pixel_size_x=dx, pixel_size_y=dy,
                pixel_size_z=1.0e-6,
                ne_grid=ne, ni_grid=ni, te_grid=te, zstar_grid=zs, Z=Z,
                x_positions=np.array([0.0]), y_positions=np.array([0.0]),
            )
            u = NumpyVector(np.zeros(params.N, dtype=np.complex64))
            U = NumpyVector(np.zeros(params.N, dtype=np.complex64))
            source.propagate_to(z_sample, params, cutoff_freq, u, U, None)
            plasma.apply(u, U, params, cutoff_freq, stepping_iteration=0, history=None)
            dz = z_detector - (z_sample + t_m)
            propagate_2d(u, U, params.dx, params.get_dy(), params.wl, dz,
                         params.chunk_size, cutoff_freq, params.nx, params.ny)
            det_2d = square_and_downsample_2d(u, params, z_detector)
            cy_d2, cx_d2 = det_2d.shape[0] // 2, det_2d.shape[1] // 2
            I_centre = np.mean(det_2d[cy_d2 - 20:cy_d2 + 20, cx_d2 - 20:cx_d2 + 20])

        T_meas = I_centre / I0
        transmissions.append(T_meas)

        d, b, phase_exp, T_exp, _ = _compute_expected(
            ne_val, ni_val, T_e, Z_star, Z, energy, t_m)
        d_expected, b_expected = d, b
        print(f"  t={t_um:3d} um:  I/I0 = {T_meas:.4f}  "
              f"(expected {T_exp:.4f}, diff {T_meas - T_exp:+.4f})")

    # Fit mu from Beer-Lambert
    mu_fitted = -np.log(np.array(transmissions)) / np.array(thicknesses_m)
    mu_theory = 4 * np.pi * b_expected / wl
    mu_mean = mu_fitted.mean()
    mu_err = abs(mu_mean - mu_theory) / mu_theory * 100

    print(f"\n  mu_theory = {mu_theory:.4e} m^-1  (beta = {b_expected:.4e})")
    print(f"  mu_fitted = {mu_mean:.4e} m^-1  (per-point: {mu_fitted})")
    print(f"  Deviation  = {mu_err:.2f}%")

    assert mu_err < 10.0, f"Absorption mu deviation {mu_err:.2f}% exceeds 10% threshold"
    print(f"  PASS: mu deviation {mu_err:.2f}% < 10%")

    return transmissions, mu_mean, mu_theory


# ═══════════════════════════════════════════════════════════════════════════
#  Test C: Kramers IB isolation
# ═══════════════════════════════════════════════════════════════════════════

def test_plasma_kramers_isolation(plot=True, mode="1d"):
    """
    Isolate Kramers inverse bremsstrahlung by comparing two plasmas:

      (a) Fully ionized (Z* = Z):  beta = beta_ff only
      (b) Partially ionized (Z* < Z): beta = beta_bound + beta_ff

    At the same n_e and T_e, (b) - (a) = beta_bound (photoabsorption).
    The residual in (a) gives beta_ff directly.

    Cross-check: compare measured beta_ff with analytic _kramers_beta_ff().
    """
    print("\n" + "=" * 70)
    print("Test C: Kramers IB Isolation")
    print("=" * 70)

    energy = 8000.0
    wl = convert_energy_wavelength(energy)

    Z = 13
    T_e = 100.0
    ne_val = 1.0e23
    thickness_um = 100
    thickness_m = thickness_um * 1e-6
    nz = int(thickness_m / 1.0e-6)

    if mode == "1d":
        N, dx = 65536, 1.0e-7
        z_sample, z_detector = 0.05, 0.15
        cutoff_angle, det_pix = 0.015, 8.0e-6
        chunk_size = 4096
        sim_nx = N // 8
        params = SimParams(N=N, dx=dx, z_detector=z_detector,
                           detector_size=20e-3,
                           detector_pixel_size_x=det_pix,
                           detector_pixel_size_y=1.0,
                           wl=wl, chunk_size=chunk_size)
        cutoff_freq = np.sin(cutoff_angle) / wl
    else:
        N, nx, ny = 8192 * 8192, 8192, 8192
        dx, dy = 1.7e-7, 1.7e-7
        z_sample, z_detector = 0.5, 2.0
        cutoff_angle, det_pix = 0.01, 4.0e-7
        chunk_size = 512 * 1024 * 1024 // 16
        params = SimParams(N=N, nx=nx, ny=ny, dx=dx, dy=dy,
                           z_detector=z_detector,
                           detector_size=800e-6,
                           detector_size_x=800e-6,
                           detector_size_y=800e-6,
                           detector_pixel_size_x=det_pix,
                           detector_pixel_size_y=det_pix,
                           wl=wl, chunk_size=chunk_size)
        cutoff_freq = np.sin(cutoff_angle) / wl
        sim_nx = nx // 8
        sim_ny = ny // 8

    source = PointSource(x=0.0, z=0.0, y=0.0 if mode == "2d" else None)

    # I0 reference
    if mode == "1d":
        u0 = NumpyVector(np.zeros(params.N, dtype=np.complex64))
        U0 = NumpyVector(np.zeros(params.N, dtype=np.complex64))
        source.propagate_to(z_detector, params, cutoff_freq, u0, U0, None)
        det_empty = square_and_downsample(u0, params, z_detector)
        I0 = np.mean(det_empty[len(det_empty)//2 - 20:len(det_empty)//2 + 20])
    else:
        u0 = NumpyVector(np.zeros(params.N, dtype=np.complex64))
        U0 = NumpyVector(np.zeros(params.N, dtype=np.complex64))
        source.propagate_to(z_detector, params, cutoff_freq, u0, U0, None)
        det_empty_2d = square_and_downsample_2d(u0, params, z_detector)
        cy_d, cx_d = det_empty_2d.shape[0] // 2, det_empty_2d.shape[1] // 2
        I0 = np.mean(det_empty_2d[cy_d - 20:cy_d + 20, cx_d - 20:cx_d + 20])

    results = {}
    for label, Z_star in [("fully ionized (Z*=13)", 13),
                           ("partially ionized (Z*=8)", 8)]:
        ni_val = ne_val / Z_star

        if mode == "1d":
            ne, ni, te, zs = _make_plasma_slab_1d(nz, sim_nx, ne_val, Z, Z_star, T_e)
            plasma = PlasmaSample(
                z_start=z_sample, pixel_size_x=dx, pixel_size_z=1.0e-6,
                ne_grid=ne, ni_grid=ni, te_grid=te, zstar_grid=zs, Z=Z,
                x_positions=np.array([0.0]),
            )
            u = NumpyVector(np.zeros(params.N, dtype=np.complex64))
            U = NumpyVector(np.zeros(params.N, dtype=np.complex64))
            source.propagate_to(z_sample, params, cutoff_freq, u, U, None)
            plasma.apply(u, U, params, cutoff_freq, stepping_iteration=0, history=None)
            dz = z_detector - (z_sample + thickness_m)
            propagate(u, U, params.dx, params.wl, dz, params.chunk_size, cutoff_freq)
            det = square_and_downsample(u, params, z_detector)
            I_centre = np.mean(det[len(det)//2 - 20:len(det)//2 + 20])
        else:
            ne, ni, te, zs = _make_plasma_slab(nz, sim_nx, sim_ny, ne_val, Z, Z_star, T_e)
            plasma = PlasmaSample(
                z_start=z_sample, pixel_size_x=dx, pixel_size_y=dy,
                pixel_size_z=1.0e-6,
                ne_grid=ne, ni_grid=ni, te_grid=te, zstar_grid=zs, Z=Z,
                x_positions=np.array([0.0]), y_positions=np.array([0.0]),
            )
            u = NumpyVector(np.zeros(params.N, dtype=np.complex64))
            U = NumpyVector(np.zeros(params.N, dtype=np.complex64))
            source.propagate_to(z_sample, params, cutoff_freq, u, U, None)
            plasma.apply(u, U, params, cutoff_freq, stepping_iteration=0, history=None)
            dz = z_detector - (z_sample + thickness_m)
            propagate_2d(u, U, params.dx, params.get_dy(), params.wl, dz,
                         params.chunk_size, cutoff_freq, params.nx, params.ny)
            det_2d = square_and_downsample_2d(u, params, z_detector)
            cy_d2, cx_d2 = det_2d.shape[0] // 2, det_2d.shape[1] // 2
            I_centre = np.mean(det_2d[cy_d2 - 20:cy_d2 + 20, cx_d2 - 20:cx_d2 + 20])

        T_meas = I_centre / I0
        mu_meas = -np.log(T_meas) / thickness_m
        beta_meas = mu_meas * wl / (4 * np.pi)

        d_exp, b_exp, _, _, _ = _compute_expected(
            ne_val, ni_val, T_e, Z_star, Z, energy, thickness_m)

        # Analytic Kramers beta_ff
        b_ff = _kramers_beta_ff(ne_val, ni_val, Z_star, T_e, energy)
        b_bound = b_exp - b_ff

        results[label] = {
            "T": T_meas, "mu": mu_meas, "beta_meas": beta_meas,
            "beta_exp": b_exp, "beta_ff": b_ff, "beta_bound": b_bound,
        }

        print(f"\n  {label}:")
        print(f"    I/I0 = {T_meas:.4f}")
        print(f"    beta_measured = {beta_meas:.4e}")
        print(f"    beta_expected = {b_exp:.4e}  (ff={b_ff:.4e}, bound={b_bound:.4e})")
        print(f"    beta_ff/beta_total = {b_ff/b_exp*100:.2f}%")

    # Cross-check: difference in beta between the two cases
    beta_diff_meas = results["partially ionized (Z*=8)"]["beta_meas"] - \
                     results["fully ionized (Z*=13)"]["beta_meas"]
    beta_diff_exp = results["partially ionized (Z*=8)"]["beta_bound"] - \
                    results["fully ionized (Z*=13)"]["beta_bound"]
    # For fully ionized, beta_bound = 0, so difference = beta_bound of partial

    print(f"\n  Cross-check (partial - fully ionized):")
    print(f"    beta_diff_measured = {beta_diff_meas:.4e}")
    print(f"    beta_bound_expected = {beta_diff_exp:.4e}")
    print(f"    (should equal beta_bound for partially ionized case)")

    # Verify Kramers directly
    beta_ff_meas = results["fully ionized (Z*=13)"]["beta_meas"]
    beta_ff_exp = results["fully ionized (Z*=13)"]["beta_ff"]
    ff_err = abs(beta_ff_meas - beta_ff_exp) / max(beta_ff_exp, 1e-20) * 100 \
        if beta_ff_exp > 1e-20 else 0

    print(f"\n  Kramers beta_ff verification (fully ionized):")
    print(f"    measured = {beta_ff_meas:.4e}")
    print(f"    expected = {beta_ff_exp:.4e}  (analytic _kramers_beta_ff)")
    if beta_ff_exp > 1e-15:
        print(f"    deviation = {ff_err:.2f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlasmaSample validation tests")
    parser.add_argument("--1d-only", action="store_true", help="Run 1D tests only")
    parser.add_argument("--2d", action="store_true", help="Run 2D tests (slow)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plots")
    args = parser.parse_args()

    mode = "1d"
    if args.__dict__.get("2d"):
        mode = "2d"
    plot = not args.__dict__.get("no_plot", False)

    print("PlasmaSample Phase & Absorption Validation")
    print(f"Mode: {mode}")
    print(f"Tests follow 3D_test_notebooks design patterns:")
    print(f"  - Thin slab geometry")
    print(f"  - Thickness scan + linear fit")
    print(f"  - Phase: T = u_after/u_before, vacuum reference")
    print(f"  - Absorption: I0 from empty-field, Beer-Lambert")

    # Test A: Phase validation
    phases, slope_m, slope_t = test_plasma_phase_fully_ionized(plot=plot, mode=mode)

    # Test B: Absorption validation
    transmissions, mu_m, mu_t = test_plasma_absorption(plot=plot, mode=mode)

    # Test C: Kramers IB isolation
    kramers_results = test_plasma_kramers_isolation(plot=plot, mode=mode)

    print("\n" + "=" * 70)
    print("All PlasmaSample validation tests completed.")
    print("=" * 70)
