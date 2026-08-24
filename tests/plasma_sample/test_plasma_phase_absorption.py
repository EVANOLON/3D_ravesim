"""
PlasmaSample Phase & Absorption Validation
===========================================

Method: T = (u_after_plasma / u_after_empty) at the same spatial position.

For each test case we run TWO propagations from the same u_before:
  1. plasma.apply()  →  u_after_plasma  (material + Fresnel propagation)
  2. propagate(dz)   →  u_after_empty   (Fresnel propagation only)

Then T_material = u_after_plasma / u_after_empty at centre.  Because both
fields start from the identical u_before and propagate through the same
Fresnel kernel, ALL propagation effects (spherical curvature, diffraction,
frequency cutoff) cancel exactly.  Only the material transfer function remains.

This is the gold-standard method validated in quantify_phase_error.py.

Modulo-2pi design (Test A):
  Fully ionised H (Z=Z*=1): delta = delta_free — exact analytic.
  Single layer at exact thickness; phases: pi/2, pi, 3pi/2, 2pi.
  phi=2pi wraps to 0 — only ~0.01% residual Kramers absorption.

Python convention: T = exp(+2*pi*i * delta * t / lambda)  (positive phase).

Usage:
    python tests/plasma_sample/test_plasma_phase_absorption.py           # 1D
    python tests/plasma_sample/test_plasma_phase_absorption.py --2d      # 2D
"""
import sys, types
from pathlib import Path
import numpy as np

BIGWAVE = Path(__file__).resolve().parents[2] / "big-wave"
sys.path.insert(0, str(BIGWAVE))

# bfpy shim (required by vector.py)
_b = types.ModuleType('bfpy')
class _CE:
    def __init__(self, *a, **kw): pass
    def position(self): return 0
    def read_chunk_c16(self, b): return len(b)
    def read_chunk_c8(self, b): return len(b)
    def write_chunk_c16(self, b): pass
    def write_chunk_c8(self, b): pass
    def advance(self, n): pass
    def __len__(self): return 0
    def dtype(self): return 'c16'
_b.ChunkedEditor = _CE
for m in ['generate_header_c16','generate_header_c8','fft_c16','fft_c8','ifft_c16','ifft_c8']:
    setattr(_b, m, lambda *a, **kw: None)
sys.modules['bfpy'] = _b

# Notebook-verified import order
import multisim          # noqa: F401
import config            # noqa: F401
import util              # noqa: F401
import propagation       # noqa: F401

from propagation import (
    SimParams, convert_energy_wavelength, propagate, propagate_2d,
)
from source import PointSource
from plasma_sample import PlasmaSample
from plasma import plasma_delta_beta, _kramers_beta_ff
from vector import NumpyVector

R_E_CM = 2.8179403262e-13
R_E_M  = 2.8179403262e-15


# ═══════════════════════════════════════════════════════════════════════════
#  Gold-standard measurement
# ═══════════════════════════════════════════════════════════════════════════

def _measure_T_material(u_before, plasma, params, cutoff_freq, centre, t_m, mode):
    """
    Gold-standard material transfer function.

    Runs plasma.apply() AND an empty propagation from the same u_before,
    then returns T = u_after_plasma(centre) / u_after_empty(centre).

    Both propagations share the same Fresnel kernel, so all propagation
    effects cancel — only the material transfer function remains.
    """
    # Plasma propagation
    u_p = NumpyVector(u_before.copy())
    U_p = NumpyVector(np.zeros_like(u_before))
    plasma.apply(u_p, U_p, params, cutoff_freq, stepping_iteration=0, history=None)
    u_after_plasma = u_p.vec

    # Empty propagation (same dz = t_m, same starting field)
    u_e = NumpyVector(u_before.copy())
    U_e = NumpyVector(np.zeros_like(u_before))
    if mode == "1d":
        propagate(u_e, U_e, params.dx, params.wl, t_m,
                  params.chunk_size, cutoff_freq)
    else:
        propagate_2d(u_e, U_e, params.dx, params.get_dy(), params.wl, t_m,
                     params.chunk_size, cutoff_freq, params.nx, params.ny)
    u_after_empty = u_e.vec

    # Ratio at centre — all propagation effects cancel
    if mode == "1d":
        return u_after_plasma[centre] / u_after_empty[centre]
    else:
        nx, ny = params.nx, params.ny
        cx, cy = nx // 2, ny // 2
        p = u_after_plasma.reshape(ny, nx)
        e = u_after_empty.reshape(ny, nx)
        # Use a small patch at centre to average out any residual noise
        roi = 3
        T_patch = p[cy - roi:cy + roi, cx - roi:cx + roi] / \
                  e[cy - roi:cy + roi, cx - roi:cx + roi]
        return np.mean(T_patch)


# ═══════════════════════════════════════════════════════════════════════════
#  Simulation parameters
# ═══════════════════════════════════════════════════════════════════════════

def _build_params(mode, wl):
    """Return (params, z_sample, z_detector, dx, dy, sim_nx, sim_ny)."""
    if mode == "1d":
        z_sample, z_detector = 0.05, 0.15
        N = 65536
        dx = propagation.max_dx(z_sample, 0.0, N, wl)
        params = SimParams(N=N, dx=dx, z_detector=z_detector,
                           detector_size=10e-3,
                           detector_pixel_size_x=4.0e-6,
                           detector_pixel_size_y=1.0,
                           wl=wl, chunk_size=4096)
        return params, z_sample, dx, N // 8, 1
    else:
        z_sample, z_detector = 1.0, 2.0
        # Use moderate grid: large enough for accurate 2D bilinear interpolation,
        # small enough to run in reasonable time (~2 min per test on CPU).
        nx, ny = 2048, 2048
        N = nx * ny
        dx = propagation.max_dx(z_sample, 0.0, nx, wl)
        dy = dx
        params = SimParams(N=N, ny=ny, dx=dx, dy=dy,
                           z_detector=z_detector,
                           detector_size=400e-6,
                           detector_size_x=400e-6,
                           detector_size_y=400e-6,
                           detector_pixel_size_x=2.0e-7,
                           detector_pixel_size_y=2.0e-7,
                           wl=wl, chunk_size=64*1024*1024//16)
        return params, z_sample, dx, nx // 4, ny // 4


def _make_plasma(mode, nz, sim_nx, sim_ny, ne_val, Z, Z_star, T_e,
                 z_sample, dx, dy, t_m):
    """Build a PlasmaSample for a uniform slab."""
    if mode == "1d":
        ne = np.full((nz, sim_nx), ne_val, dtype=np.float64)
        ni = np.full((nz, sim_nx), ne_val / Z_star, dtype=np.float64)
        te = np.full((nz, sim_nx), T_e, dtype=np.float64)
        zs = np.full((nz, sim_nx), Z_star, dtype=np.float64)
        return PlasmaSample(
            z_start=z_sample, pixel_size_x=dx, pixel_size_z=t_m / nz,
            ne_grid=ne, ni_grid=ni, te_grid=te, zstar_grid=zs, Z=Z,
            x_positions=np.array([0.0]),
        )
    else:
        ne = np.full((nz, sim_ny, sim_nx), ne_val, dtype=np.float64)
        ni = np.full((nz, sim_ny, sim_nx), ne_val / Z_star, dtype=np.float64)
        te = np.full((nz, sim_ny, sim_nx), T_e, dtype=np.float64)
        zs = np.full((nz, sim_ny, sim_nx), Z_star, dtype=np.float64)
        return PlasmaSample(
            z_start=z_sample, pixel_size_x=dx, pixel_size_y=dy,
            pixel_size_z=t_m / nz,
            ne_grid=ne, ni_grid=ni, te_grid=te, zstar_grid=zs, Z=Z,
            x_positions=np.array([0.0]), y_positions=np.array([0.0]),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Test A: Clean-phase validation
# ═══════════════════════════════════════════════════════════════════════════

def test_phase_clean(mode="1d"):
    """
    Phase validation with modulo-2pi-friendly phases.

    Fully ionised H (Z=Z*=1): delta = delta_free — exact analytic,
    no Chantler table lookup needed.

    Gold-standard method: T = u_after_plasma / u_after_empty at centre.
    Fresnel propagation effects cancel exactly; material phase remains.

    Phases: pi/2, pi, 3pi/2, 2pi.  phi=2pi wraps to 0.
    """
    print("\n" + "=" * 70)
    print("Test A: Clean-phase validation (fully ionised H, Z=Z*=1)")
    print("=" * 70)

    energy = 8000.0
    wl = convert_energy_wavelength(energy)
    params, z_sample, dx, sim_nx, sim_ny = _build_params(mode, wl)
    dy = params.get_dy() if mode == "2d" else dx
    cutoff_freq = np.sin(0.015) / wl

    Z, Z_star = 1, 1           # fully ionised hydrogen
    T_e = 100.0
    ne_val = 1.0e24             # cm⁻³

    # Phase coefficient (Python convention: +ve)
    ne_m3 = 1.0e30
    dphi_dt = ne_m3 * R_E_M * wl     # = ne * re * lambda  [rad/m]

    # Physics verification
    d_free = ne_val * R_E_CM * (wl * 100)**2 / (2 * np.pi)
    d_py, b_py, _ = plasma_delta_beta(ne_val, ne_val, T_e, Z_star, Z, energy)

    print(f"  ne = {ne_val:.1e} cm-3   E = {energy:.0f} eV")
    print(f"  delta_free (exact)        = {d_free:.6e}")
    print(f"  delta (plasma_delta_beta) = {d_py:.6e}")
    print(f"  beta  = {b_py:.4e}  (H at 8 keV: negligible)")
    print(f"  dphi/dt = {dphi_dt:.4e} rad/m")
    print()

    targets = [
        ("+pi/2",  np.pi / 2,   +1j),
        ("+pi",    np.pi,       -1.0 + 0j),
        ("+3pi/2", 3*np.pi/2,   -1j),
        ("+2pi",   2*np.pi,      1.0 + 0j),   # ← wraps to 0
    ]

    # Source and u_before (same for all test cases)
    source = PointSource(x=0.0, z=0.0, y=0.0 if mode == "2d" else None)
    u_init = NumpyVector(np.zeros(params.N, dtype=np.complex64))
    U_init = NumpyVector(np.zeros(params.N, dtype=np.complex64))
    source.propagate_to(z_sample, params, cutoff_freq, u_init, U_init, None)
    u_before = u_init.vec.copy()

    centre = params.N // 2 if mode == "1d" else 0

    all_pass = True
    for label, phi_t, T_expected in targets:
        t_m = phi_t / dphi_dt
        plasma = _make_plasma(mode, 1, sim_nx, sim_ny, ne_val, Z, Z_star,
                              T_e, z_sample, dx, dy, t_m)

        T_mat = _measure_T_material(u_before, plasma, params, cutoff_freq,
                                    centre, t_m, mode)

        phi_m = np.angle(T_mat)
        abs_m = np.abs(T_mat)

        # Compare modulo 2pi
        pw = np.arctan2(np.sin(phi_m), np.cos(phi_m))
        tw = np.arctan2(np.sin(phi_t), np.cos(phi_t))
        err = min(abs(pw - tw), 2*np.pi - abs(pw - tw))

        abs_exp = np.exp(-2 * np.pi * b_py * t_m / wl)

        wrap = "  <- wraps to 0!" if abs(phi_t - 2*np.pi) < 0.01 else ""
        ok_phi = err * 180 / np.pi < 2.0
        ok_abs = abs(abs_m - abs_exp) < 0.005
        status = "PASS" if (ok_phi and ok_abs) else "FAIL"
        if not (ok_phi and ok_abs):
            all_pass = False

        print(f"  {label:>6s}: t={t_m*1e6:.1f} um  "
              f"phi={pw/np.pi:+.4f}pi  err={err*180/np.pi:.2f}deg  "
              f"|T|={abs_m:.4f} (exp {abs_exp:.4f})  "
              f"{status}{wrap}")

    if all_pass:
        print(f"\n  All phase checks PASSED (< 2 deg, < 0.005 |T|).")
    else:
        print(f"\n  Some checks FAILED.")

    return all_pass


# ═══════════════════════════════════════════════════════════════════════════
#  Test B: Absorption via |T| (Beer-Lambert)
# ═══════════════════════════════════════════════════════════════════════════

def test_absorption(mode="1d"):
    """
    Absorption validation: |T| = exp(-2*pi*beta*t/lambda).

    Partially ionised Al: Z=13, Z*=8.  Thickness scan → fit mu.
    Both phase AND absorption are extracted from T_material.

    Kramers beta_ff contributes ~0.04% of total beta at these parameters.
    """
    print("\n" + "=" * 70)
    print("Test B: Absorption via |T| (partially ionised Al, Z=13, Z*=8)")
    print("=" * 70)

    energy = 8000.0
    wl = convert_energy_wavelength(energy)
    params, z_sample, dx, sim_nx, sim_ny = _build_params(mode, wl)
    dy = params.get_dy() if mode == "2d" else dx
    cutoff_freq = np.sin(0.015) / wl

    Z, Z_star = 13, 8
    T_e = 100.0
    ne_val = 1.0e23              # cm⁻³
    ni_val = ne_val / Z_star

    thicknesses_um = [50, 100, 150]

    _, b_exp, _ = plasma_delta_beta(ne_val, ni_val, T_e, Z_star, Z, energy)
    b_ff = _kramers_beta_ff(ne_val, ni_val, Z_star, T_e, energy)
    mu_theory = 4 * np.pi * b_exp / wl     # intensity attenuation

    print(f"  ne={ne_val:.1e} cm-3  Z={Z}  Z*={Z_star}")
    print(f"  beta_total = {b_exp:.4e}")
    print(f"    beta_ff  = {b_ff:.4e}  ({b_ff/b_exp*100:.2f}% of total)")
    print(f"    beta_bound = {b_exp - b_ff:.4e}")
    print(f"  mu_theory = {mu_theory:.1f} m-1  (intensity)")
    print()

    # Source (same u_before for all)
    source = PointSource(x=0.0, z=0.0, y=0.0 if mode == "2d" else None)
    u_init = NumpyVector(np.zeros(params.N, dtype=np.complex64))
    U_init = NumpyVector(np.zeros(params.N, dtype=np.complex64))
    source.propagate_to(z_sample, params, cutoff_freq, u_init, U_init, None)
    u_before = u_init.vec.copy()
    centre = params.N // 2 if mode == "1d" else 0

    Ts_field = []
    for t_um in thicknesses_um:
        t_m = t_um * 1e-6
        nz = max(1, int(t_m / 1.0e-6))
        plasma = _make_plasma(mode, nz, sim_nx, sim_ny, ne_val, Z, Z_star,
                              T_e, z_sample, dx, dy, t_m)

        T_mat = _measure_T_material(u_before, plasma, params, cutoff_freq,
                                    centre, t_m, mode)
        abs_field = np.abs(T_mat)
        abs_field_exp = np.exp(-2 * np.pi * b_exp * t_m / wl)
        Ts_field.append(abs_field)

        T_int = abs_field ** 2
        T_int_exp = abs_field_exp ** 2
        mu_meas = -np.log(T_int) / t_m

        print(f"  t={t_um:3d} um:  |T_field|={abs_field:.4f} (exp {abs_field_exp:.4f})  "
              f"I/I0={T_int:.4f} (exp {T_int_exp:.4f})  "
              f"mu={mu_meas:.1f} m-1")

    # Fit mu from intensity
    Ts_int = np.array(Ts_field) ** 2
    mu_fitted = -np.log(Ts_int) / np.array([t * 1e-6 for t in thicknesses_um])
    mu_mean = mu_fitted.mean()
    mu_err = abs(mu_mean - mu_theory) / mu_theory * 100

    print(f"\n  mu_theory = {mu_theory:.1f} m-1")
    print(f"  mu_fitted = {mu_mean:.1f} m-1  (per-pt: {[f'{m:.1f}' for m in mu_fitted]})")
    print(f"  deviation = {mu_err:.2f}%")
    print(f"  {'PASS' if mu_err < 5 else 'FAIL'} (threshold: 5%)")

    return mu_err < 5


# ═══════════════════════════════════════════════════════════════════════════
#  Test C: Kramers IB isolation (Z* contrast)
# ═══════════════════════════════════════════════════════════════════════════

def test_kramers_isolation(mode="1d"):
    """
    Isolate Kramers IB by Z* contrast at fixed ne.

    (a) Z*=Z=13 (fully ionised):  beta = beta_ff        (pure IB)
    (b) Z*=8   (partially):       beta = beta_ff + beta_bound

    Difference isolates beta_bound; residual in (a) verifies beta_ff.
    """
    print("\n" + "=" * 70)
    print("Test C: Kramers IB isolation (Z* contrast)")
    print("=" * 70)

    energy = 8000.0
    wl = convert_energy_wavelength(energy)
    params, z_sample, dx, sim_nx, sim_ny = _build_params(mode, wl)
    dy = params.get_dy() if mode == "2d" else dx
    cutoff_freq = np.sin(0.015) / wl

    Z = 13
    T_e = 100.0
    ne_val = 1.0e23
    t_um = 100
    t_m = t_um * 1e-6
    nz = int(t_m / 1.0e-6)

    # Source
    source = PointSource(x=0.0, z=0.0)
    u_init = NumpyVector(np.zeros(params.N, dtype=np.complex64))
    U_init = NumpyVector(np.zeros(params.N, dtype=np.complex64))
    source.propagate_to(z_sample, params, cutoff_freq, u_init, U_init, None)
    u_before = u_init.vec.copy()
    centre = params.N // 2 if mode == "1d" else 0

    results = {}
    for label, Z_star in [("Z*=13 (fully ionised)", 13),
                           ("Z*=8  (partially)",      8)]:
        ni_val = ne_val / Z_star
        plasma = _make_plasma(mode, nz, sim_nx, sim_ny, ne_val, Z, Z_star,
                              T_e, z_sample, dx, dy, t_m)

        T_mat = _measure_T_material(u_before, plasma, params, cutoff_freq,
                                    centre, t_m, mode)
        abs_field = np.abs(T_mat)
        beta_meas = -np.log(abs_field) * wl / (2 * np.pi * t_m)

        _, b_exp, _ = plasma_delta_beta(ne_val, ni_val, T_e, Z_star, Z, energy)
        b_ff = _kramers_beta_ff(ne_val, ni_val, Z_star, T_e, energy)

        results[label] = {
            "|T|": abs_field,
            "beta_meas": beta_meas,
            "beta_exp": b_exp,
            "beta_ff": b_ff,
            "beta_bound": b_exp - b_ff,
        }

        print(f"\n  {label}:")
        print(f"    |T_field| = {abs_field:.4f}")
        print(f"    beta_meas  = {beta_meas:.4e}")
        print(f"    beta_exp   = {b_exp:.4e}")
        print(f"      beta_ff    = {b_ff:.4e}  (Kramers IB)")
        print(f"      beta_bound = {b_exp - b_ff:.4e}  (photoabsorption)")

    # Cross-check
    fully = results["Z*=13 (fully ionised)"]
    partial = results["Z*=8  (partially)"]

    beta_bound_meas = partial["beta_meas"] - fully["beta_meas"]
    beta_bound_exp  = partial["beta_bound"]

    print(f"\n  Z* contrast (partial - fully):")
    print(f"    beta_bound (measured) = {beta_bound_meas:.4e}")
    print(f"    beta_bound (expected) = {beta_bound_exp:.4e}")
    print(f"    beta_ff    (measured) = {fully['beta_meas']:.4e}")
    print(f"    beta_ff    (analytic) = {fully['beta_ff']:.4e}")
    print(f"\n  Note: Kramers beta_ff ~ {fully['beta_ff']:.2e} is "
          f"{fully['beta_ff']/partial['beta_exp']*100:.2f}% of total beta "
          f"at ne={ne_val:.1e}, 8 keV.")

    return results


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--2d", action="store_true")
    args = parser.parse_args()

    mode = "2d" if getattr(args, "2d", False) else "1d"

    print("=" * 70)
    print("PlasmaSample Phase & Absorption Validation Suite")
    print(f"Mode: {mode}")
    print("Method: T = u_after_plasma / u_after_empty (same-position)")
    print("  Fresnel propagation cancels exactly → pure material T")
    print("  Test A — Clean-phase (pi/2, pi, 3pi/2, 2pi), modulo-2pi")
    print("  Test B — Absorption via |T|, Beer-Lambert thickness scan")
    print("  Test C — Kramers IB isolation via Z* contrast")
    print("=" * 70)

    test_phase_clean(mode=mode)
    test_absorption(mode=mode)
    test_kramers_isolation(mode=mode)

    print("\n" + "=" * 70)
    print("All tests completed.")
    print("=" * 70)
