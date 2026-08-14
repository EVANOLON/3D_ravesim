"""
Quantify the phase error contribution in Test A.

Strategy: run an EMPTY simulation (no plasma, just free-space propagation
by the same dz) alongside the plasma simulation.  Then:

  T_plasma = u_after_plasma / u_before    (material + propagation)
  T_empty  = u_after_empty  / u_before    (propagation only)
  T_material_true = T_plasma / T_empty    (pure material, propagation cancelled)

This is the TRUE material transfer function, independent of vacuum-reference
errors. The vacuum-reference method approximates T_empty ~ T_vac.

Compare:
  (a) T_material_true = T_plasma(centre) / T_empty(centre)          [gold standard]
  (b) T_vac_ref       = T_plasma(centre) / T_plasma(vacuum)        [current method]
  (c) T_ideal          = exp(i * 2*pi * delta * t / lambda)        [analytic]
"""
import sys, types, os
from pathlib import Path
import numpy as np

BIGWAVE = Path(__file__).resolve().parents[2] / "big-wave"
sys.path.insert(0, str(BIGWAVE))

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

import multisim, config, util, propagation
from propagation import SimParams, convert_energy_wavelength, propagate
from source import PointSource
from plasma_sample import PlasmaSample
from plasma import plasma_delta_beta
from vector import NumpyVector

R_E_M = 2.8179403262e-15

energy, wl = 8000.0, convert_energy_wavelength(8000.0)
z_sample, N = 0.05, 65536
dx = propagation.max_dx(z_sample, 0.0, N, wl)
params = SimParams(N=N, dx=dx, z_detector=0.15, detector_size=10e-3,
                   detector_pixel_size_x=4.0e-6, detector_pixel_size_y=1.0,
                   wl=wl, chunk_size=4096)
cutoff_freq = np.sin(0.015) / wl
sim_nx = N // 8

Z, Z_star, T_e, ne_val = 1, 1, 100.0, 1.0e24
ne_m3 = 1.0e30
dphi_dt = ne_m3 * R_E_M * wl

d_val, b_val, _ = plasma_delta_beta(ne_val, ne_val, T_e, Z_star, Z, energy)

print("=" * 70)
print("Quantifying phase error sources in Test A")
print("=" * 70)
print(f"  ne={ne_val:.1e} cm-3  E={energy:.0f} eV  lambda={wl:.4e} m")
print(f"  dphi/dt = {dphi_dt:.4e} rad/m")
print(f"  dx = {dx:.4e} m  N = {N}")
print()

centre = N // 2
slab_hw = sim_nx // 2

targets = [
    ("+pi/2",  np.pi / 2),
    ("+pi",    np.pi),
    ("+3pi/2", 3 * np.pi / 2),
    ("+2pi",   2 * np.pi),
]

print(f"  {'phase':<8} {'t(um)':<8} "
      f"{'true_err':<10} {'vacref_err':<12} {'extra_err':<12} "
      f"{'tfrac_vacref':<14}")
print(f"  {'-'*64}")

for label, phi_t in targets:
    t_m = phi_t / dphi_dt

    # --- Plasma simulation ---
    ne_g = np.full((1, sim_nx), ne_val, dtype=np.float64)
    ni_g = np.full((1, sim_nx), ne_val / Z_star, dtype=np.float64)
    te_g = np.full((1, sim_nx), T_e, dtype=np.float64)
    zs_g = np.full((1, sim_nx), Z_star, dtype=np.float64)
    plasma = PlasmaSample(
        z_start=z_sample, pixel_size_x=dx, pixel_size_z=t_m,
        ne_grid=ne_g, ni_grid=ni_g, te_grid=te_g, zstar_grid=zs_g, Z=Z,
        x_positions=np.array([0.0]),
    )
    source = PointSource(x=0.0, z=0.0)
    u_p = NumpyVector(np.zeros(N, dtype=np.complex64))
    U_p = NumpyVector(np.zeros(N, dtype=np.complex64))
    source.propagate_to(z_sample, params, cutoff_freq, u_p, U_p, None)
    u_before = u_p.vec.copy()
    plasma.apply(u_p, U_p, params, cutoff_freq, stepping_iteration=0, history=None)
    u_after_plasma = u_p.vec.copy()

    # --- Empty simulation (free-space propagation by same dz) ---
    u_e = NumpyVector(np.zeros(N, dtype=np.complex64))
    U_e = NumpyVector(np.zeros(N, dtype=np.complex64))
    # Copy u_before into u_e
    u_e.vec = u_before.copy()
    propagate(u_e, U_e, params.dx, params.wl, t_m, params.chunk_size, cutoff_freq)
    u_after_empty = u_e.vec.copy()

    # --- Compare methods ---
    # (a) Gold standard: T_plasma / T_empty at centre
    T_plasma_c = u_after_plasma[centre] / u_before[centre]
    T_empty_c  = u_after_empty[centre] / u_before[centre]
    T_true = T_plasma_c / T_empty_c
    phi_true = np.angle(T_true)

    # (b) Vacuum reference method
    ref_hw = slab_hw + 500
    ref_w = 200
    T_plasma_l = np.mean(u_after_plasma[centre - ref_hw - ref_w:centre - ref_hw] /
                         u_before[centre - ref_hw - ref_w:centre - ref_hw])
    T_plasma_r = np.mean(u_after_plasma[centre + ref_hw:centre + ref_hw + ref_w] /
                         u_before[centre + ref_hw:centre + ref_hw + ref_w])
    T_vac = (T_plasma_l + T_plasma_r) / 2
    T_vacref = T_plasma_c / T_vac
    phi_vacref = np.angle(T_vacref)

    # (c) Analytic
    phi_analytic = 2 * np.pi * d_val * t_m / wl

    # Compute errors
    def phase_err(phi, target):
        pw = np.arctan2(np.sin(phi), np.cos(phi))
        tw = np.arctan2(np.sin(target), np.cos(target))
        e = abs(pw - tw)
        return min(e, 2*np.pi - e) * 180 / np.pi

    err_true = phase_err(phi_true, phi_t)
    err_vacref = phase_err(phi_vacref, phi_t)
    extra = err_vacref - err_true

    # What fraction of vacuum-ref error is from the method itself?
    tfrac = extra / err_vacref * 100 if err_vacref > 0.1 else 0

    print(f"  {label:<8} {t_m*1e6:<8.1f} "
          f"{err_true:<10.1f} {err_vacref:<12.1f} {extra:<+12.1f} "
          f"{tfrac:<14.0f}")

print()
print(f"  true_err:    error of T_plasma/T_empty (gold standard)")
print(f"  vacref_err:  error of T_plasma(c)/T_plasma(vac) (current method)")
print(f"  extra_err:   vacref_err - true_err (error from vacuum-reference method)")
print(f"  tfrac:       extra_err / vacref_err * 100%")

# Also verify: T_empty at centre vs T_empty at vacuum
print(f"\n{'='*70}")
print("Verification: does T_empty vary with position?")
print("=" * 70)
print("  If T_empty(x) were constant, vacref ≡ gold standard.")
print()

for label, phi_t in targets[:2]:  # just check first two for brevity
    t_m = phi_t / dphi_dt
    u_e = NumpyVector(np.zeros(N, dtype=np.complex64))
    U_e = NumpyVector(np.zeros(N, dtype=np.complex64))
    # need u_before again
    source2 = PointSource(x=0.0, z=0.0)
    u_tmp = NumpyVector(np.zeros(N, dtype=np.complex64))
    U_tmp = NumpyVector(np.zeros(N, dtype=np.complex64))
    source2.propagate_to(z_sample, params, cutoff_freq, u_tmp, U_tmp, None)
    u_e.vec = u_tmp.vec.copy()
    propagate(u_e, U_e, params.dx, params.wl, t_m, params.chunk_size, cutoff_freq)
    u_after_e = u_e.vec.copy()
    u_b = u_tmp.vec.copy()

    T_e_centre = u_after_e[centre] / u_b[centre]
    T_e_vac = u_after_e[centre - ref_hw] / u_b[centre - ref_hw]
    ratio = T_e_vac / T_e_centre

    print(f"  {label}: t={t_m*1e6:.1f} um")
    print(f"    T_empty(centre) = {T_e_centre:.6f}")
    print(f"    T_empty(vacuum) = {T_e_vac:.6f}")
    print(f"    ratio = {ratio:.6f}")
    print(f"    |ratio| = {abs(ratio):.6f}  (should be 1.0)")
    print(f"    angle(ratio)/pi = {np.angle(ratio)/np.pi:+.6f}  (should be 0.0)")
