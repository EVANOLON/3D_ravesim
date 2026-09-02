#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAVE-SIM result plotter: render detected.npy (or any .npy) to a PNG.

Usage:
  python plot_result.py --path <detected.npy> --out <out.png> [--sim_dir <sim_dir>] [--kind auto|1d|2d]

Output: JSON on stdout: {png_path, shape, dtype, min, max, mean, kind}
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_config_geometry(sim_dir):
    """Try to read detector geometry from the sim dir config for x-axis labels."""
    if not sim_dir:
        return None
    try:
        sys.path.insert(0, '/mnt/d/rave-sim-main/rave-sim-main/big-wave')
        import config as rave_config
        dct = rave_config.load(Path(sim_dir) / 'config.yaml')
        sp = dct['sim_params']
        return sp
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True, help='path to the .npy file')
    ap.add_argument('--out', required=True, help='output PNG path')
    ap.add_argument('--sim_dir', default=None, help='optional sim dir for detector geometry')
    ap.add_argument('--kind', default='auto', choices=['auto', '1d', '2d'])
    args = ap.parse_args()

    data = np.load(args.path)
    sp = load_config_geometry(args.sim_dir)

    arr = data.squeeze()
    kind = args.kind
    if kind == 'auto':
        kind = '2d' if arr.ndim >= 2 and min(arr.shape) > 1 else '1d'

    fig = plt.figure(figsize=(10, 4) if kind == '1d' else (6, 6))
    if kind == '1d':
        if sp is not None:
            try:
                det_size = float(sp.get('detector_size', sp.get('detector_size_x')))
                px = float(sp.get('detector_pixel_size_x'))
                n = arr.shape[-1]
                x = (np.arange(n) - n / 2) * px * 1e3  # mm
                xlabel = 'detector x (mm)'
            except Exception:
                x = np.arange(arr.shape[-1])
                xlabel = 'pixel'
        else:
            x = np.arange(arr.shape[-1])
            xlabel = 'pixel'
        plt.plot(x, arr)
        plt.xlabel(xlabel)
        plt.ylabel('intensity')
        plt.title(f'{Path(args.path).name} — 1D detector profile')
    else:
        plt.imshow(arr, aspect='equal', cmap='inferno')
        plt.colorbar(label='intensity')
        plt.xlabel('x pixel')
        plt.ylabel('y pixel')
        plt.title(f'{Path(args.path).name} — 2D detector image')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close(fig)

    print(json.dumps({
        'png_path': args.out,
        'shape': list(data.shape),
        'dtype': str(data.dtype),
        'kind': kind,
        'min': float(data.min()),
        'max': float(data.max()),
        'mean': float(data.mean()),
    }))


if __name__ == '__main__':
    main()
