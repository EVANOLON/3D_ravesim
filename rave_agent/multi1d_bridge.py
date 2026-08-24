#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi1d_bridge.py — subprocess CLI used by the DSH `multi1d-bridge-tools` plugin.

Thin wrapper around the multi1d_loader / plasma_grid_builder / rave_config_gen
modules. Reads a single JSON command word + JSON payload and prints a JSON
result to stdout (safe: no logs pollute stdout; logging goes to stderr).

Usage:
  python multi1d_bridge.py load {"case_path": "...", "parse_materials": true}
  python multi1d_bridge.py build_grids {"case_path": "...", "timestep": 200, ...}
  python multi1d_bridge.py gen_config {"case_path": "...", "timestep": 200, ...}
"""
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# The canonical bridge modules live next to the Multi1D++ distribution.
BRIDGE_DIR = Path('/mnt/d/rave-sim-main/Multi1D++Portable20241128')
if str(BRIDGE_DIR) not in sys.path:
    sys.path.insert(0, str(BRIDGE_DIR))

logging.basicConfig(level=logging.WARNING)


def _load(case_path, parse_materials=True):
    from multi1d_loader import load_multi1d_output
    return load_multi1d_output(case_path, parse_materials=parse_materials)


def cmd_load(p):
    data = _load(p['case_path'], bool(p.get('parse_materials', True)))
    def rng(name):
        a = data.get(name)
        if a is None or a.size == 0:
            return None
        return float(a.min()), float(a.max())
    return {
        'ok': True,
        'nt': data['nt'],
        'ncell': data['ncell'],
        'ngroups': data.get('ngroups', 1),
        'time': rng('TIME1D'),
        'xc': rng('XC'),
        'rho': rng('R'),
        'te': rng('T'),
        'n_materials': len(data.get('materials_map', {})),
        'vars_scalars': [k for k in data if isinstance(data[k], np.ndarray) and data[k].ndim == 1],
        'vars_center': [k for k in data if isinstance(data[k], np.ndarray) and data[k].ndim == 2],
        'available_timesteps': list(range(data['nt'])),
    }


def _grid_config(p):
    from plasma_grid_builder import GridConfig
    return GridConfig(
        geometry=p.get('geometry', 'side-on'),
        nx=int(p.get('nx', 256)),
        transverse_size_um=float(p.get('transverse_size_um', 200.0)),
        los_thickness_um=float(p.get('los_thickness_um', 100.0)),
        pixel_size_z_um=float(p.get('pixel_size_z_um', 1.0)),
        max_nz=int(p.get('max_nz', 500)),
        Te_threshold_eV=float(p.get('te_threshold_eV', GridConfig.Te_threshold_eV)),
        Zstar_threshold_frac=float(p.get('zstar_threshold_frac', GridConfig.Zstar_threshold_frac)),
        default_Z=int(p.get('default_Z', 1)),
    )


def _summarize_grids(grids):
    out = {}
    meta = grids.get('meta', {})
    out['meta'] = {k: meta[k] for k in ('timestep', 'time', 'geometry', 'effective_Z', 'xc_range') if k in meta}
    for section in ('plasma', 'solid'):
        if section not in grids:
            continue
        s = grids[section]
        rec = {}
        if 'ne' in s:
            m = s['ne'] > 0
            rec['shape'] = list(s['ne'].shape)
            rec['ne'] = [float(s['ne'][m].min()), float(s['ne'][m].max())] if m.any() else None
            rec['te'] = [float(s['te'][m].min()), float(s['te'][m].max())] if m.any() else None
            rec['zstar'] = [float(s['zstar'][m].min()), float(s['zstar'][m].max())] if m.any() else None
            rec['Z'] = int(s.get('Z', 0))
            rec['pixel_size_x_m'] = s.get('pixel_size_x_m')
            rec['pixel_size_z_m'] = s.get('pixel_size_z_m')
            rec['z_start_m'] = s.get('z_start_m')
        if 'material_grid' in s:
            rec['shape'] = list(s['material_grid'].shape)
            rec['materials'] = s.get('materials')
        out[section] = rec
    return out


def cmd_build_grids(p):
    data = _load(p['case_path'])
    from plasma_grid_builder import build_hybrid_grids, save_grids, grid_summary
    output = Path(p['output_dir']).resolve()
    ts = int(p.get('timestep', data['nt'] - 1))
    grids = build_hybrid_grids(data, timestep=ts, config=_grid_config(p))
    paths = save_grids(grids, output)
    return {
        'ok': True,
        'output_dir': str(output),
        'paths': paths,
        'grids': _summarize_grids(grids),
        'grid_summary': grid_summary(grids),
    }


def cmd_gen_config(p):
    data = _load(p['case_path'])
    from plasma_grid_builder import build_hybrid_grids, save_grids
    from rave_config_gen import generate_config, save_config, SimConfig
    output = Path(p['output_dir']).resolve()
    ts = int(p.get('timestep', data['nt'] - 1))
    grids = build_hybrid_grids(data, timestep=ts, config=_grid_config(p))
    save_grids(grids, output)

    sc = SimConfig(
        dimension=p.get('dimension', '1d'),
        energy_min_eV=float(p.get('energy_min_eV', 8000.0)),
        energy_max_eV=float(p.get('energy_max_eV', 10000.0)),
        source_z_m=float(p.get('source_z_m', 0.0)),
        z_target_m=float(p.get('z_target_m', 0.5)),
        z_detector_m=float(p.get('z_detector_m', 4.8)),
        detector_size_x_m=float(p.get('detector_size_x_m', 0.003)),
        detector_pixel_size_x_m=float(p.get('detector_pixel_size_x_m', 2.0e-5)),
        N=int(p.get('N', 33554432)),
        dx_m=float(p.get('dx_m', 3.0e-10)),
        chunk_size=int(p.get('chunk_size', 16777216)),
        nr_source_points=int(p.get('nr_source_points', 100)),
        seed=int(p.get('seed', 42)),
        ny=int(p.get('ny', 512)),
        dy_m=float(p.get('dy_m', 2.0e-7)),
    )
    cfg = generate_config(output, grids, sc)
    cfg_path = save_config(cfg, output)
    # Verify it loads back through RAVE-SIM's own loader.
    load_ok = False
    load_err = None
    try:
        sys.path.insert(0, '/mnt/d/rave-sim-main/rave-sim-main/big-wave')
        import config as rave_config
        rave_config.load(cfg_path)
        load_ok = True
    except Exception as e:
        load_err = f'{type(e).__name__}: {e}'
    return {
        'ok': True,
        'output_dir': str(output),
        'config_path': str(cfg_path),
        'dimension': sc.dimension,
        'n_elements': len(cfg['elements']),
        'element_types': [el['type'] for el in cfg['elements']],
        'energy_eV': [sc.energy_min_eV, sc.energy_max_eV],
        'z_target_m': sc.z_target_m,
        'z_detector_m': sc.z_detector_m,
        'N': sc.N,
        'config_reload_ok': load_ok,
        'config_reload_error': load_err,
        'grids': _summarize_grids(grids),
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({'ok': False, 'error': 'subcommand required (load|build_grids|gen_config)'}))
        sys.exit(2)
    cmd = sys.argv[1]
    try:
        p = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    except Exception as e:
        print(json.dumps({'ok': False, 'error': f'bad JSON payload: {e}'}))
        sys.exit(2)
    try:
        res = {'load': cmd_load, 'build_grids': cmd_build_grids, 'gen_config': cmd_gen_config}[cmd](p)
        print(json.dumps(res, ensure_ascii=False))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({'ok': False, 'error': f'{type(e).__name__}: {e}'}))
        sys.exit(1)


if __name__ == '__main__':
    main()
