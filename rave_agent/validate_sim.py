#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAVE-SIM simulation validation & feasibility checker.

Two modes:
  python validate_sim.py --validate <sim_dir>
      Physical/logical checks on an existing simulation directory:
      config parse, FOV, Nyquist sampling, power-of-two N, z-layout,
      cutoff-angle monotonicity, phase-step consistency.

  python validate_sim.py --feasibility <sim_dir> [--engine fast-wave|big-wave]
      Everything from --validate plus hardware feasibility:
      GPU memory estimate vs actual free VRAM, disk estimate, rough runtime.

Output: single JSON object on stdout. Exit code 0 iff all checks pass.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

RAVE_ROOT = '/mnt/d/rave-sim-main/rave-sim-main'
BIG_WAVE = os.path.join(RAVE_ROOT, 'big-wave')
NIST = os.path.join(RAVE_ROOT, 'nist_lookup')
for p in (BIG_WAVE, NIST):
    if p not in sys.path:
        sys.path.insert(0, p)

# Config is the safe import entry (importing history/propagation/source first
# triggers the repo's latent circular import); multisim brings the rest.
import config as rave_config  # noqa: E402
import multisim  # noqa: E402
from propagation import SimParams, convert_energy_wavelength, grid_density_check, grid_density_check_2d  # noqa: E402

NVIDIA_SMI_CANDIDATES = [
    '/usr/lib/wsl/lib/nvidia-smi',
    '/usr/bin/nvidia-smi',
    '/usr/local/bin/nvidia-smi',
]


def is_pow2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def load_sim(sim_dir):
    sim_dir = Path(sim_dir)
    dct = rave_config.load(sim_dir / 'config.yaml')
    params = rave_config.parse_sim_params(dct['sim_params'])
    computed = rave_config.load(sim_dir / 'computed.yaml')
    elements = [rave_config.parse_optical_element(el, sim_dir) for el in dct.get('elements', [])]
    return dct, params, computed, elements


def nvidia_smi(query):
    for smi in NVIDIA_SMI_CANDIDATES:
        if not os.path.exists(smi):
            continue
        try:
            r = subprocess.run(
                [smi, '--query-gpu=' + query, '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                return r.stdout.strip()
        except Exception:
            pass
    return None


def check(name, ok, detail):
    return {'name': name, 'ok': bool(ok), 'detail': str(detail)}


def validate(sim_dir):
    checks = []
    meta = {'sim_dir': sim_dir}
    try:
        dct, params, computed, elements = load_sim(sim_dir)
    except Exception as e:  # config unreadable
        checks.append(check('config_parse', False, f'{type(e).__name__}: {e}'))
        return {'ok': False, 'checks': checks, **meta}

    # 1. config parses
    checks.append(check('config_parse', True, 'YAML + sim_params parsed'))

    is_2d = bool(params.is_2d)
    N = params.N
    nx = params.nx if is_2d else N
    ny = params.ny if is_2d else 1
    dx = params.dx
    dy = params.get_dy()

    # 2. power-of-two N (rustfft / big-fourier requirement; fast-wave cuFFT
    #    tolerates it too, but the repo convention keeps N a power of two)
    if is_2d:
        checks.append(check('power_of_two', is_pow2(nx) and is_pow2(ny),
                            f'nx={nx}, ny={ny}'))
    else:
        checks.append(check('power_of_two', is_pow2(N), f'N={N}'))

    # 3. FOV: simulation window must cover the detector
    if is_2d:
        fov = (nx * dx >= params.get_detector_size_x()) and \
              (ny * dy >= params.get_detector_size_y())
        detail = (f'nx*dx={nx * dx:.3e} vs det_x={params.get_detector_size_x():.3e}, '
                  f'ny*dy={ny * dy:.3e} vs det_y={params.get_detector_size_y():.3e}')
    else:
        fov = N * dx >= params.detector_size
        detail = f'N*dx={N * dx:.3e} vs detector_size={params.detector_size:.3e}'
    checks.append(check('fov', fov, detail))

    # 4. z-layout + cutoff monotonicity via the repo's own validator
    multisource = dct.get('multisource', {})
    z_source = float(multisource.get('z', 0.0))
    energy_range = tuple(map(float, computed.get('energy_range', [0.0, 0.0])))
    cutoff_angles = list(map(float, computed.get('cutoff_angles', [])))
    try:
        multisim.check_simulation_inputs(params, z_source, elements, cutoff_angles)
        checks.append(check('z_layout', True, 'z monotonic, no overlaps, detector after last element'))
    except (AssertionError, ValueError) as e:
        checks.append(check('z_layout', False, str(e)))

    if cutoff_angles:
        mono = all(a <= b for a, b in zip(cutoff_angles, cutoff_angles[1:])) and \
               all(0 <= a <= math.pi / 2 for a in cutoff_angles)
        checks.append(check('cutoff_angles', mono, f'{cutoff_angles}'))

    # 5. Nyquist sampling (shortest wavelength => strictest)
    try:
        if energy_range[1] > 0:
            wl = convert_energy_wavelength(energy_range[1])
            first_z = elements[0].z_start if elements else params.z_detector
            dz = max(first_z - z_source, 1e-12)
            x_range = multisource.get('x_range', [0.0, 0.0])
            x_source = max(abs(float(x_range[0])), abs(float(x_range[1])))
            if is_2d:
                y_range = multisource.get('y_range', [0.0, 0.0])
                y_source = max(abs(float(y_range[0])), abs(float(y_range[1])))
                grid_density_check_2d(dz, x_source, nx, dx, y_source, ny, dy, wl)
            else:
                grid_density_check(dz, x_source, N, dx, wl)
            checks.append(check('nyquist', True,
                                f'dx={dx:.3e} satisfies Nyquist at E={energy_range[1]:.0f} eV'))
        else:
            checks.append(check('nyquist', True, 'no energy info; skipped'))
    except (ValueError, AssertionError) as e:
        checks.append(check('nyquist', False, str(e)))

    # 6. phase steps consistent across elements
    steps = [len(getattr(el, 'x_positions', [1])) for el in elements if hasattr(el, 'x_positions')]
    if steps:
        ok = len(set(steps)) <= 1
        checks.append(check('phase_steps', ok, f'per-element steps: {steps}'))

    meta['is_2d'] = is_2d
    meta['N'] = N
    meta['nx'] = nx
    meta['ny'] = ny
    meta['dx'] = dx
    meta['detector_size'] = params.detector_size
    meta['energy_range'] = list(energy_range)
    meta['z_source'] = z_source
    meta['nr_elements'] = len(elements)
    meta['elements'] = [getattr(el, 'type', type(el).__name__) for el in elements]
    meta['nr_source_points'] = int(multisource.get('nr_source_points', 1))
    meta['engine_hint'] = 'fast-wave' if is_2d else 'either'

    ok = all(c['ok'] for c in checks)
    return {'ok': ok, 'checks': checks, **meta}


def gpu_memory():
    """Return (free_mib, total_mib) or (None, None)."""
    raw = nvidia_smi('memory.free,memory.total')
    if not raw:
        return None, None
    try:
        parts = [p.strip() for p in raw.split(',')]
        return float(parts[0]), float(parts[1])
    except Exception:
        return None, None


def feasibility(sim_dir, engine='fast-wave'):
    v = validate(sim_dir)
    result = dict(v)
    if not v['ok']:
        result['feasible'] = False
        result['blocked_by'] = [c['name'] for c in v['checks'] if not c['ok']]
        return result

    is_2d = v['is_2d']
    N = v['N']
    nx, ny = v['nx'], v['ny']

    # VRAM estimate (c8 = complex64 = 8 bytes; u + U + transient workspaces)
    wave_bytes = (nx * ny if is_2d else N) * 8
    vram_est_bytes = wave_bytes * 3
    vram_est_gb = vram_est_bytes / (1024 ** 3)

    free_mib, total_mib = gpu_memory()
    if free_mib is None:
        gpu_block = {'ok': False, 'detail': 'nvidia-smi unavailable / GPU not visible from this process'}
        free_gb = None
    else:
        free_gb = free_mib / 1024.0
        headroom = free_gb * 0.85  # keep 15% for the display/other tenants
        gpu_block = {
            'ok': vram_est_gb <= headroom,
            'detail': (f'est_vram={vram_est_gb:.2f} GB, free={free_gb:.2f} GB '
                       f'(total {total_mib / 1024:.2f} GB, 85% usable={headroom:.2f} GB)'),
        }
    result['gpu'] = {'est_vram_gb': round(vram_est_gb, 3), 'free_gb': None if free_gb is None else round(free_gb, 2), **gpu_block}

    # Disk estimate (big-wave out-of-core mode keeps u+U+scratch on disk)
    use_disk = bool(dct_use_disk(sim_dir))
    if use_disk:
        disk_est_gb = vram_est_bytes / (1024 ** 3)
        try:
            usage = shutil.disk_usage(sim_dir)
            free_disk_gb = usage.free / (1024 ** 3)
            disk_ok = disk_est_gb < free_disk_gb * 0.9
        except Exception:
            free_disk_gb = None
            disk_ok = None
        result['disk'] = {'est_disk_gb': round(disk_est_gb, 3),
                          'free_disk_gb': None if free_disk_gb is None else round(free_disk_gb, 2),
                          'ok': disk_ok}
    else:
        result['disk'] = {'est_disk_gb': 0.0, 'ok': True, 'note': 'in-memory mode (use_disk_vector=false)'}

    # Rough runtime estimate, calibrated on the 2.68e8-point 1D shockwave run
    # (~192 s on an RTX 5070 Ti): scale linearly with point count, double for 2D FFT cost.
    points = (nx * ny if is_2d else N)
    est_time_min = points / 2.68e8 * 3.2 * (2.0 if is_2d else 1.0)
    result['est_time_min'] = round(est_time_min, 2)

    result['feasible'] = (gpu_block.get('ok', False) if engine == 'fast-wave' else True) and \
                         (result['disk'].get('ok') is not False)
    if not result['feasible']:
        result['blocked_by'] = []
        if not gpu_block.get('ok', False) and engine == 'fast-wave':
            result['blocked_by'].append('gpu_vram')
        if result['disk'].get('ok') is False:
            result['blocked_by'].append('disk')
    return result


def dct_use_disk(sim_dir):
    try:
        with open(os.path.join(str(sim_dir), 'config.yaml'), 'r') as f:
            import re
            m = re.search(r'use_disk_vector\s*:\s*(true|false)', f.read(), re.IGNORECASE)
            return m.group(1).lower() == 'true' if m else False
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', metavar='SIM_DIR')
    ap.add_argument('--feasibility', metavar='SIM_DIR')
    ap.add_argument('--engine', default='fast-wave', choices=['fast-wave', 'big-wave'])
    args = ap.parse_args()

    if args.validate:
        out = validate(args.validate)
    elif args.feasibility:
        out = feasibility(args.feasibility, args.engine)
    else:
        ap.error('one of --validate / --feasibility is required')

    print(json.dumps(out, indent=2, ensure_ascii=False))
    sys.exit(0 if out.get('ok', False) else 2)


if __name__ == '__main__':
    main()
