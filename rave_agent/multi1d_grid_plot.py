#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi1D++ -> RAVE-SIM grid renderer.

Renders the plasma/cold grids produced by plasma_grid_builder into PNGs under
output/_agent_runs/plots so the agent can embed them in the conversation via
markdown images (the plot server serves that directory at
http://127.0.0.1:8811).

Recognised grid files (by suffix, in grid_dir, unless --grid overrides):
  ne_grid.npy / ni_grid.npy / te_grid.npy / zstar_grid.npy   (plasma_sample)
  material_grid.npy / density_grid.npy                       (precise_sample)

Usage:
  python multi1d_grid_plot.py --grid_dir <dir> [--fields ne,te,zstar]
                              [--grid <file.npy>] [--out <out.png>]
                              [--pixel_size_x_um X] [--pixel_size_z_um Z]

Output: single JSON object on stdout:
  {ok, sim_dir, grids: [{name, png_path, url, shape, dtype, field,
                         min, max, mean}]}
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PLOTS_DIR = Path('/mnt/d/rave-sim-main/rave-sim-main/output/_agent_runs/plots')
BASE_URL = 'http://127.0.0.1:8811'

# field -> (filename, label, unit)
FIELDS = {
    'ne': ('ne_grid.npy', 'Electron density n_e', 'cm^-3'),
    'ni': ('ni_grid.npy', 'Ion density n_i', 'cm^-3'),
    'te': ('te_grid.npy', 'Electron temperature T_e', 'eV'),
    'zstar': ('zstar_grid.npy', 'Avg ionisation Z*', ''),
    'material': ('material_grid.npy', 'Material index', ''),
    'density': ('density_grid.npy', 'Density', 'g/cm^3'),
}


def resolve_pixel_size(grid_dir, field):
    """Try to read pixel_size_x/z from config.yaml element; else None."""
    cfg = Path(grid_dir) / 'config.yaml'
    if cfg.is_file():
        try:
            sys.path.insert(0, '/mnt/d/rave-sim-main/rave-sim-main/big-wave')
            import config as rave_config
            dct = rave_config.load(cfg)
            for el in dct.get('elements', []):
                if el.get('type') in ('plasma_sample', 'precise_sample'):
                    return (
                        float(el.get('pixel_size_x', 1e-6)),
                        float(el.get('pixel_size_z', 1e-6)),
                    )
        except Exception:
            pass
    return None, None


def plot_1d(arr, out, px, title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(arr.shape[-1]) * (px * 1e6)
    use_log = arr.max() > 1e3 and arr.min() >= 0
    if use_log:
        # Clamp the log floor so vacuum/edge values don't dominate the axis.
        floor = max(arr.max() * 1e-4, 1e-300)
        arr = np.maximum(arr, floor)
        ax.semilogy(x, arr)
    else:
        ax.plot(x, arr)
    ax.set_xlabel('x (µm)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_2d(arr, out, px, pz, title, ylabel):
    nz, nx = arr.shape
    extent = [0, nx * px * 1e6, 0, nz * pz * 1e6]
    fig, ax = plt.subplots(figsize=(max(6, nx / max(nz, 1) * 5), 5))
    im = ax.imshow(arr, extent=extent, aspect='auto', cmap='inferno', origin='lower')
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('z (µm, along beam)')
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(ylabel)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def render_one(grid_path, out, px, pz, field):
    grid = np.load(grid_path)
    _fn, label, unit = FIELDS[field]
    name = Path(grid_path).name
    kind = '2d' if grid.ndim >= 2 and min(grid.shape) > 1 else '1d'
    if grid.ndim == 3:
        grid = grid[:, grid.shape[1] // 2, :]
        kind = '2d'
    if kind == '1d':
        # squeeze leading singlet dims (e.g. (1, nx) side-on)
        grid = grid.reshape(-1)
    gf = grid.astype(np.float64)
    if kind == '2d':
        plot_2d(gf, out, px, pz, f'{name} — {label}', unit or label)
    else:
        plot_1d(gf, out, px, f'{name} — {label}', unit or label)
    return {
        'name': name,
        'field': field,
        'png_path': str(out),
        'url': BASE_URL + '/' + out.name,
        'shape': list(np.atleast_2d(grid).shape) if kind == '1d' else list(grid.shape),
        'dtype': str(grid.dtype),
        'kind': kind,
        'min': float(gf.min()),
        'max': float(gf.max()),
        'mean': float(gf.mean()),
    }


def main():
    ap = argparse.ArgumentParser(description='Multi1D++ grid renderer')
    ap.add_argument('--grid_dir', required=True, help='directory containing the grid .npy files')
    ap.add_argument('--fields', default='ne,te,zstar', help='comma-separated fields to render (default ne,te,zstar)')
    ap.add_argument('--grid', default=None, help='explicit grid .npy path (renders a single field)')
    ap.add_argument('--out', default=None, help='optional output PNG path (single field only)')
    ap.add_argument('--pixel_size_x_um', type=float, default=None)
    ap.add_argument('--pixel_size_z_um', type=float, default=None)
    args = ap.parse_args()

    grid_dir = str(Path(args.grid_dir).resolve())
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    base_name = Path(grid_dir).name or 'multi1d'

    if args.grid:
        gp = Path(args.grid)
        if not gp.is_file():
            print(json.dumps({'ok': False, 'error': 'grid file not found: ' + str(gp)}))
            sys.exit(2)
        field = sorted(FIELDS.items(), key=lambda kv: gp.name.startswith(kv[1][0]))[0][0]
        for k, (fn, *_ ) in FIELDS.items():
            if gp.name == fn:
                field = k
                break
        px = args.pixel_size_x_um * 1e-6 if args.pixel_size_x_um else 1e-6
        pz = args.pixel_size_z_um * 1e-6 if args.pixel_size_z_um else px
        out = Path(args.out) if args.out else PLOTS_DIR / f'{base_name}__{field}_{stamp}.png'
        try:
            info = render_one(gp, out, px, pz, field)
            print(json.dumps({'ok': True, 'sim_dir': grid_dir, 'grids': [info]}, ensure_ascii=False))
            sys.exit(0)
        except Exception as e:
            print(json.dumps({'ok': False, 'error': f'{type(e).__name__}: {e}'}))
            sys.exit(2)

    px_default, pz_default = resolve_pixel_size(grid_dir, None)
    px = args.pixel_size_x_um * 1e-6 if args.pixel_size_x_um else (px_default or 1e-6)
    pz = args.pixel_size_z_um * 1e-6 if args.pixel_size_z_um else (pz_default or px)

    results = []
    for field in [f.strip() for f in args.fields.split(',') if f.strip()]:
        if field not in FIELDS:
            results.append({'field': field, 'error': f'unknown field: {field}'})
            continue
        fn = FIELDS[field][0]
        gp = Path(grid_dir) / fn
        if not gp.is_file():
            results.append({'field': field, 'error': f'grid file not found: {gp}'})
            continue
        out = PLOTS_DIR / f'{base_name}__{field}_{stamp}.png'
        try:
            results.append(render_one(gp, out, px, pz, field))
        except Exception as e:
            results.append({'field': field, 'error': f'{type(e).__name__}: {e}'})

    ok = any('error' not in r for r in results)
    print(json.dumps({'ok': ok, 'sim_dir': grid_dir, 'grids': results}, ensure_ascii=False))
    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()
