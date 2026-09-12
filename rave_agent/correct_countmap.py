#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Correct the detector downsampling "count-map" ripple in historical fast-wave 2D
outputs created before the CUDA area-weighting fix.

The historical square_and_downsample_2d_kernel integrated |u|^2 over an
integer-truncated range of grid points per detector
pixel. When the effective detector pixel (detector_pixel_size / M) is not an
integer multiple of the grid spacing dx, the number of covered grid points
alternates (e.g. 6/7/8 for the 4096->416 capsule runs), imprinting a periodic
multiplicative ripple on the image (dominant FFT peak k=151 -> 2.75 px).

This script replicates the kernel's index arithmetic EXACTLY (C++ static_cast<int>
= truncation toward zero, clamps to [0, nx]), computes the per-pixel coverage
count map, and divides it out:

    I_corrected = I_detected / count                     (ripple removal)
    I_corrected = I_detected / (count * cos_angle)       (--obliquity: also
                  removes the kernel's radial obliquity factor)

then renormalizes to the original image mean so the intensity scale is kept.

Usage:
  python correct_countmap.py --legacy-output --sim_dir <sim_dir> [--path <detected.npy>]
                             [--out <out.npy>] [--obliquity] [--no-renorm]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def load_config(sim_dir):
    with open(Path(sim_dir) / 'config.yaml') as f:
        return yaml.safe_load(f)


def compute_count_map(cfg):
    """Replicate square_and_downsample_2d_kernel index arithmetic."""
    sp = cfg['sim_params']
    nx = int(sp['nx'])
    ny = int(sp.get('ny', nx))
    dx = float(sp['dx'])
    dy = float(sp.get('dy', dx))
    z_det = float(sp['z_detector'])

    det_px_x = float(sp['detector_pixel_size_x'])
    det_px_y = float(sp.get('detector_pixel_size_y', det_px_x))
    det_size_x = float(sp.get('detector_size_x', sp.get('detector_size')))
    det_size_y = float(sp.get('detector_size_y', det_size_x))
    out_x = int(det_size_x / det_px_x)
    out_y = int(det_size_y / det_px_y)

    use_fresnel = str(sp.get('use_fresnel_scaling', 'false')).lower() in ('true', '1', 'yes')

    if use_fresnel:
        z_src = float(cfg['multisource'].get('z', 0.0))
        z_sample = float(cfg['elements'][0]['z_start'])
        z_s = z_sample - z_src
        z_d = z_det - z_sample
        if z_s > 0.0 and z_d > 0.0:
            z_eff = (z_s * z_d) / (z_s + z_d)
            M = (z_s + z_d) / z_s
        else:
            z_eff, M = z_d, 1.0
        # kernel divides detector pixel sizes by magnification
        ds_px = det_px_x / M
        ds_py = det_px_y / M
        current_z = z_eff
    else:
        ds_px, ds_py, current_z, M = det_px_x, det_px_y, z_det, 1.0

    # --- vectorized kernel replica ---
    ix = np.arange(out_x, dtype=np.float64)
    iy = np.arange(out_y, dtype=np.float64)
    x_det = (ix - out_x // 2) * ds_px          # detector-plane coordinates [m]
    y_det = (iy - out_y // 2) * ds_py

    def trunc(v):
        return np.trunc(v).astype(np.int64)    # C++ static_cast<int>: toward zero

    i_min = np.clip(trunc((x_det - ds_px * 0.5) / dx) + nx // 2, 0, nx)
    i_max = np.clip(trunc((x_det + ds_px * 0.5) / dx) + nx // 2, 0, nx)
    j_min = np.clip(trunc((y_det - ds_py * 0.5) / dy) + ny // 2, 0, ny)
    j_max = np.clip(trunc((y_det + ds_py * 0.5) / dy) + ny // 2, 0, ny)

    cx = np.maximum(i_max - i_min, 0)          # covered grid points along x
    cy = np.maximum(j_max - j_min, 0)
    count = np.outer(cy, cx)                   # 2D coverage count (per detector pixel)

    # obliquity factor from the kernel: cos(atan(r / current_z))
    xx, yy = np.meshgrid(x_det, y_det)
    r = np.sqrt(xx * xx + yy * yy)
    cos_angle = 1.0 / np.sqrt(1.0 + (r / current_z) ** 2)

    return dict(count=count, cos_angle=cos_angle, out=(out_y, out_x),
                M=M, z_eff=current_z, ds_px=ds_px, use_fresnel=use_fresnel)


def ripple_stats(img):
    """Background (outer border) rel std and dominant periodic FFT peak amplitude."""
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = ny // 2, nx // 2
    bg_mask = (np.abs(xx - cx) > 0.3 * nx) | (np.abs(yy - cy) > 0.3 * ny)
    bg = img[bg_mask]
    rel_std = float(bg.std() / bg.mean()) if bg.size > 0 and bg.mean() != 0 else float('nan')
    F = np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean())))
    F[cy - 3:cy + 4, cx - 3:cx + 4] = 0
    pk = np.unravel_index(np.argmax(F), F.shape)
    k = (pk[0] - cy, pk[1] - cx)
    amp = float(F[pk] / np.median(F))
    return rel_std, amp, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--legacy-output',
        action='store_true',
        help='confirm that the input was produced by the historical truncation kernel',
    )
    ap.add_argument('--sim_dir', required=True)
    ap.add_argument('--path', default=None, help='detected.npy; default <sim_dir>/00000000/detected.npy')
    ap.add_argument('--out', default=None, help='output npy; default <dir>/detected_corrected.npy')
    ap.add_argument('--obliquity', action='store_true', help='also divide by cos_angle')
    ap.add_argument('--no-renorm', action='store_true', help='do not renormalize to original mean')
    ap.add_argument('--fill-zeros', action='store_true',
                    help='fill count==0 pixels (empty integration range, e.g. the '
                         'center zero-cross) from the 8-neighbourhood mean')
    args = ap.parse_args()

    if not args.legacy_output:
        ap.error(
            'this correction is only for historical pre-area-weighted fast-wave '
            'outputs; pass --legacy-output after confirming the input provenance'
        )

    cfg = load_config(args.sim_dir)
    cm = compute_count_map(cfg)

    path = Path(args.path) if args.path else Path(args.sim_dir) / '00000000' / 'detected.npy'
    out = Path(args.out) if args.out else path.with_name('detected_corrected.npy')

    img = np.load(path)
    keep_shape = img.shape
    a = img.squeeze()
    if a.shape != cm['out']:
        print(f'ERROR: detected {a.shape} != kernel output {cm["out"]}', file=sys.stderr)
        sys.exit(2)

    count = cm['count']
    denom = count * cm['cos_angle'] if args.obliquity else count
    corr = np.where(denom > 0, a / np.where(denom > 0, denom, 1.0), a)
    if args.fill_zeros:
        zc = count <= 0
        if zc.any():
            med = np.median(corr[~zc])
            filled = corr.copy()
            for i, j in np.argwhere(zc):
                sl_i, sl_j = slice(max(0, i - 1), i + 2), slice(max(0, j - 1), j + 2)
                nb = corr[sl_i, sl_j][count[sl_i, sl_j] > 0]
                filled[i, j] = nb.mean() if nb.size else med
            corr = filled
    if not args.no_renorm:
        corr = corr * (a.mean() / corr.mean())

    np.save(out, corr.reshape(keep_shape).astype(a.dtype))

    before = ripple_stats(a)
    after = ripple_stats(corr)
    print(json.dumps({
        'sim_dir': args.sim_dir,
        'M': cm['M'], 'z_eff': cm['z_eff'], 'effective_px': cm['ds_px'],
        'fresnel': cm['use_fresnel'],
        'count_unique': [int(v) for v in np.unique(count)],
        'count_fft_peak_k': int(np.argmax(np.abs(np.fft.fft(count[0] - count[0].mean()))[1:]) + 1),
        'before': {'bg_rel_std': before[0], 'fft_peak_k': (int(before[2][0]), int(before[2][1])), 'fft_peak_rel_amp': before[1]},
        'after': {'bg_rel_std': after[0], 'fft_peak_k': (int(after[2][0]), int(after[2][1])), 'fft_peak_rel_amp': after[1]},
        'mean_before': float(a.mean()), 'mean_after': float(corr.mean()),
        'out': str(out),
    }, indent=2))


if __name__ == '__main__':
    main()
