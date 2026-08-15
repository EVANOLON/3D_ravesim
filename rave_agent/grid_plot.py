#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVE-SIM sample-grid renderer.

Plots the material/density grids referenced by a simulation directory's
config.yaml (elements[].grid_path) into PNG files under
output/_agent_runs/plots, so the agent can embed them in the conversation via
markdown images (the plot server serves that directory at
http://127.0.0.1:8811).

Usage:
  python grid_plot.py --sim_dir <sim_dir> [--grid <grid.npy>] [--out <out.png>]

Output: single JSON object on stdout:
  {ok, grids: [{name, png_path, url, shape, dtype, kind,
                grid_min, grid_max, grid_mean,
                density_min, density_max, density_mean}]}
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
RAVE_ROOT = '/mnt/d/rave-sim-main/rave-sim-main'


def load_config(sim_dir):
    sys.path.insert(0, os.path.join(RAVE_ROOT, 'big-wave'))
    import config as rave_config
    return rave_config.load(Path(sim_dir) / 'config.yaml')


def grid_to_density(grid, materials):
    """Map a 1-based material-index grid to density (g/cm^3) via the config materials table."""
    dens = np.zeros(grid.shape, dtype=np.float64)
    valid = (grid >= 1) & (grid <= len(materials))
    idx = grid[valid].astype(np.int64) - 1
    dens[valid] = np.array([m[1] for m in materials], dtype=np.float64)[idx]
    return dens, valid


def plot_2d(arr, out, px, pz, title, cbar_label):
    nz, nx = arr.shape
    extent = [0, nx * px * 1e6, 0, nz * pz * 1e6]  # µm (grids are (z, x))
    fig, ax = plt.subplots(figsize=(max(6, nx / max(nz, 1) * 5), 5))
    im = ax.imshow(arr, extent=extent, aspect='auto', cmap='inferno', origin='lower')
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('z (µm, along beam)')
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_1d(arr, out, px, title, cbar_label):
    x = np.arange(arr.shape[-1]) * px * 1e6  # µm
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, arr)
    ax.set_xlabel('x (µm)')
    ax.set_ylabel(cbar_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def render(grid_path, materials, px, pz, out):
    grid = np.load(grid_path)
    name = Path(grid_path).name
    kind = '2d' if grid.ndim >= 2 and min(grid.shape) > 1 else '1d'
    if grid.ndim == 3:
        # 3D (z, y, x): render the central y slice
        grid = grid[:, grid.shape[1] // 2, :]
        kind = '2d'
    grid_f = grid.astype(np.float64)
    info = {
        'name': name,
        'png_path': str(out),
        'url': BASE_URL + '/' + out.name,
        'shape': list(grid.shape),
        'dtype': str(grid.dtype),
        'kind': kind,
        'grid_min': float(grid_f.min()),
        'grid_max': float(grid_f.max()),
        'grid_mean': float(grid_f.mean()),
        'density_min': None,
        'density_max': None,
        'density_mean': None,
    }
    title = name
    if materials:
        dens, valid = grid_to_density(grid, materials)
        info['density_min'] = float(dens[valid].min()) if valid.any() else None
        info['density_max'] = float(dens[valid].max()) if valid.any() else None
        info['density_mean'] = float(dens[valid].mean()) if valid.any() else None
        arr = dens
        cbar_label = 'density (g/cm³)'
        title = name + ' — density'
    else:
        arr = grid_f
        cbar_label = 'material index'
        title = name + ' — material index'
    if kind == '2d':
        plot_2d(arr, out, px, pz, title, cbar_label)
    else:
        plot_1d(arr, out, px, title, cbar_label)
    return info


def collect_grids(sim_dir, explicit):
    dct = load_config(sim_dir)
    out = []
    if explicit:
        out.append({'grid_path': explicit, 'px': None, 'pz': None, 'materials': None})
        return out
    for el in dct.get('elements', []):
        gp = el.get('grid_path')
        if not gp:
            continue
        p = Path(gp)
        if not p.is_absolute():
            p = Path(sim_dir) / p
        out.append({
            'grid_path': str(p),
            'px': float(el.get('pixel_size_x', el.get('pixel_size', 1e-6))),
            'pz': float(el.get('pixel_size_z', el.get('pixel_size', 1e-6))),
            'materials': el.get('materials') or None,
        })
    return out


def main():
    ap = argparse.ArgumentParser(description='RAVE-SIM sample-grid renderer')
    ap.add_argument('--sim_dir', required=True, help='simulation directory (contains config.yaml)')
    ap.add_argument('--grid', default=None, help='optional explicit grid .npy path')
    ap.add_argument('--out', default=None, help='optional output PNG path (first grid only)')
    args = ap.parse_args()

    sim_dir = str(Path(args.sim_dir).resolve())
    grids = collect_grids(sim_dir, args.grid)
    if not grids:
        print(json.dumps({'ok': False, 'error': 'no grid_path found in config.yaml elements'}))
        sys.exit(2)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sim_name = Path(sim_dir).name
    stamp = time.strftime('%Y%m%d_%H%M%S')
    results = []
    for i, g in enumerate(grids):
        gpath = Path(g['grid_path'])
        if not gpath.is_file():
            results.append({'name': gpath.name, 'error': 'grid file not found: ' + str(gpath)})
            continue
        if args.out and i == 0:
            out = Path(args.out)
        else:
            out = PLOTS_DIR / '{0}__grid_{1}_{2}.png'.format(sim_name, i, stamp)
        px = g['px'] if g['px'] else 1e-6
        pz = g['pz'] if g['pz'] else px
        try:
            info = render(gpath, g['materials'], px, pz, out)
            results.append(info)
        except Exception as e:
            results.append({'name': gpath.name, 'error': '{0}: {1}'.format(type(e).__name__, e)})

    ok = any('error' not in r for r in results)
    print(json.dumps({'ok': ok, 'sim_dir': sim_dir, 'grids': results}, ensure_ascii=False))
    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()
