"""Demonstrate Fresnel leakage into vacuum region."""
import sys, types, numpy as np
from pathlib import Path
BIGWAVE = Path('big-wave')
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
from propagation import SimParams, convert_energy_wavelength
from source import PointSource
from plasma_sample import PlasmaSample
from vector import NumpyVector

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
dphi_dt = ne_m3 * 2.8179403262e-15 * wl
t_m = np.pi / dphi_dt

ne = np.full((1, sim_nx), ne_val, dtype=np.float64)
ni = np.full((1, sim_nx), ne_val / Z_star, dtype=np.float64)
te = np.full((1, sim_nx), T_e, dtype=np.float64)
zs = np.full((1, sim_nx), Z_star, dtype=np.float64)
plasma = PlasmaSample(z_start=z_sample, pixel_size_x=dx, pixel_size_z=t_m,
                      ne_grid=ne, ni_grid=ni, te_grid=te, zstar_grid=zs, Z=Z,
                      x_positions=np.array([0.0]))
source = PointSource(x=0.0, z=0.0)
u = NumpyVector(np.zeros(N, dtype=np.complex64))
U = NumpyVector(np.zeros(N, dtype=np.complex64))
source.propagate_to(z_sample, params, cutoff_freq, u, U, None)
u_before = u.vec.copy()
plasma.apply(u, U, params, cutoff_freq, stepping_iteration=0, history=None)
u_after = u.vec.copy()

T_full = u_after / u_before
amp_T = np.abs(T_full)
phase_T = np.angle(T_full)

centre = N // 2
slab_hw = sim_nx // 2
edge = centre - slab_hw

print("=" * 70)
print("T = u_after/u_before  across the slab-vacuum boundary")
print("=" * 70)
print(f"  Slab half-width: {slab_hw} px = {slab_hw*dx*1e6:.1f} um")
print(f"  Propagation per layer: {t_m*1e6:.1f} um")
print(f"  Slab T_expected = exp(i*pi) = -1")
print(f"  Vacuum T_expected = 1")
print()
print(f"  {'offset(px)':<12} {'x(um)':<10} {'inside?':<10} {'|T|':<10} {'angle(T)/pi':<14}")
print(f"  {'-'*58}")

for offset in [-50, -20, -10, -5, -3, -1, 0, 1, 3, 5, 10, 20, 50]:
    idx = edge + offset
    x_um = (idx - centre) * dx * 1e6
    inside = "YES" if 0 <= offset < sim_nx else "no"
    print(f"  {offset:<12} {x_um:<10.1f} {inside:<10} {amp_T[idx]:<10.6f} {phase_T[idx]/np.pi:<+14.6f}")

# Quantify leakage into reference region
print(f"\n  --- Vacuum reference region (500-700 px from edge) ---")
ref_s = edge - 700
ref_e = edge - 500
T_vac_region = T_full[ref_s:ref_e]
print(f"  T_vac mean = {np.mean(T_vac_region):.6f}")
print(f"  |T_vac|    = {np.abs(np.mean(T_vac_region)):.6f}  (expected 1.0)")
print(f"  angle(T_vac)/pi = {np.angle(np.mean(T_vac_region))/np.pi:+.6f}  (expected 0.0)")
leak = abs(np.mean(T_vac_region) - 1.0)
print(f"  leakage magnitude = {leak:.4f}")
print(f"  This {leak*100:.2f}% deviation in T_vac is what causes the phase error")
print(f"  in T_centre/T_vacuum.")

# Show why: Fresnel kernel width
print(f"\n  --- Why does this happen? ---")
fresnel_width = np.sqrt(wl * t_m)
print(f"  Fresnel kernel width ~ sqrt(lambda*dz) = {fresnel_width*1e6:.2f} um")
print(f"  This is the characteristic distance over which the Fresnel")
print(f"  propagation 'smears' the field.  Points within ~{fresnel_width*1e6:.1f} um")
print(f"  of the slab edge are affected by diffraction from the edge.")
