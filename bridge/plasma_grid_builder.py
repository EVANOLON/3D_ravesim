"""
等离子体混合网格构建器 (v2 — 几何修正版)
===========================================
将 Multi1D++ 1D 剖面转换为 RAVE-SIM 所需的 2D/3D 网格。

几何模式:
  side-on:   激光 ⊥ X射线. Multi1D 剖面 → x 轴(横向).
             z=1 层(视线厚度). 冷/热在 x 方向并排.
             单 PlasmaSample 覆盖全部 x.
  face-on:   激光 ∥ X射线. Multi1D 剖面 → z 轴(深度).
             x 方向均匀 slab. 冷/热在 z 方向分层.
             可选 precise_sample(冷) + plasma_sample(热) 双元素.

用法:
    from plasma_grid_builder import build_hybrid_grids

    grids = build_hybrid_grids(multi1d_data, timestep=100,
                                geometry="side-on", nx=256)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger("plasma_grid_builder")


# ─── 配置 ────────────────────────────────────────────────────────────────────


@dataclass
class GridConfig:
    """网格构建配置。"""
    # 几何模式: "side-on" | "face-on"
    geometry: str = "side-on"

    # ── 横向网格 (x 方向 — RAVE-SIM 的 x 轴) ──
    nx: int = 256               # 横向像素数
    transverse_size_um: float = 200.0  # 横向物理范围 [μm]
    edge_ramp_pixels: int = 8   # 边缘过渡像素数 (soft edge)

    # ── side-on: 视线方向 (z) 厚度 ──
    los_thickness_um: float = 100.0  # 等离子体沿视线方向的厚度 [μm]
    # 对于圆柱靶: ≈直径; 对于平面靶: ≈片厚

    # ── face-on: z 方向分辨率 ──
    pixel_size_z_um: float = 1.0  # z 方向像素 [μm]
    max_nz: int = 500             # 最大 z 层数

    # ── 冷/热分割 (仅 face-on 需要) ──
    Te_threshold_eV: float = 5.0
    Zstar_threshold_frac: float = 0.1

    # ── 等离子体 ──
    default_Z: int = 1

    # ── 输出 ──
    save_dir: Optional[Path] = None


# ─── 主入口 ──────────────────────────────────────────────────────────────────


def build_hybrid_grids(
    multi1d_data: dict,
    timestep: int,
    config: Optional[GridConfig] = None,
) -> dict:
    """
    从 Multi1D 数据构建网格。

    返回:
        dict:
          "plasma":  {ne, ni, te, zstar, pixel_size_x_m, pixel_size_z_m, Z, z_start_m}
          "solid":   {material_grid, density_grid, materials, ...}  ← 仅 face-on + 有冷区
          "meta":    {timestep, time, geometry, xc_range, ...}
    """
    if config is None:
        config = GridConfig()

    from multi1d_loader import get_timestep
    profile = get_timestep(multi1d_data, timestep)

    materials_map = multi1d_data.get("materials_map", {})
    plasma_Z = _get_effective_Z(profile, materials_map, config)

    logger.info(
        f"构建网格: t={timestep}/{multi1d_data['nt']}, "
        f"geometry={config.geometry}, Z={plasma_Z}"
    )

    if config.geometry == "side-on":
        result = _build_side_on(profile, config, plasma_Z, materials_map)
    elif config.geometry == "face-on":
        result = _build_face_on(profile, config, plasma_Z, materials_map)
    else:
        raise ValueError(f"未知几何模式: {config.geometry}")

    result["meta"] = {
        "timestep": timestep,
        "time": float(profile["time"]),
        "xc_range": (float(profile["XC"].min()), float(profile["XC"].max())),
        "geometry": config.geometry,
        "effective_Z": plasma_Z,
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SIDE-ON: 激光 ⊥ X 射线
#   Multi1D 剖面 → x 轴 (横向)
#   z = 1 层 (视线穿过等离子体柱)
#   冷/热在 x 方向并排 → 单 PlasmaSample 覆盖全部
# ═══════════════════════════════════════════════════════════════════════════════


def _build_side_on(
    profile: dict,
    config: GridConfig,
    Z: int,
    materials_map: dict,
) -> dict:
    """
    Side-on 几何:

      X射线 →→→ (z, 垂直于纸面)
        │
        │  视线穿过等离子体柱
        │  厚度 = los_thickness
        │
        ▼ 探测器看到 x 方向的强度变化

      RAVE-SIM grid: (nz=1, nx)
        - x 轴: Multi1D 剖面 (密度/温度梯度方向)
        - z 轴: 1 层, pixel_size_z = 视线厚度
    """
    xc = profile["XC"]
    r = profile["R"]
    te = profile["T"]
    dene = profile["DENE"]
    zi = profile["ZI"]

    valid = (dene > 1e10) & (r > 1e-15)
    if not valid.any():
        logger.warning("side-on: 无有效等离子体")
        return _empty_result(config, "plasma")

    xc_v = xc[valid]
    dene_v = dene[valid]
    zi_v = zi[valid]
    te_v = te[valid]

    # ni = DENE / ZI
    ni_v = np.where(zi_v > 0.01, dene_v / zi_v, dene_v)

    # ── x 轴: 插值 Multi1D 剖面到均匀网格 ──
    nx = config.nx
    x_min = xc_v.min()
    x_max = xc_v.max()

    if x_max - x_min < 1e-8:
        logger.warning("side-on: 剖面太薄")
        return _empty_result(config, "plasma")

    x_uniform = np.linspace(x_min, x_max, nx)
    pixel_size_x_cm = (x_max - x_min) / (nx - 1) if nx > 1 else 1e-4

    ne_x = _safe_interp(x_uniform, xc_v, dene_v)
    ni_x = _safe_interp(x_uniform, xc_v, ni_v)
    te_x = _safe_interp(x_uniform, xc_v, te_v)
    zi_x = _safe_interp(x_uniform, xc_v, zi_v)

    # ── z 方向: 1 层, 厚度 = 视线厚度 ──
    nz = 1
    pixel_size_z_cm = config.los_thickness_um * 1e-4  # μm → cm

    # 构建 2D 网格 (nz=1, nx)
    ne_grid = ne_x.reshape(1, nx).astype(np.float64)
    ni_grid = ni_x.reshape(1, nx).astype(np.float64)
    te_grid = te_x.reshape(1, nx).astype(np.float64)
    zstar_grid = zi_x.reshape(1, nx).astype(np.float64)

    # 真空处理
    for g in [ne_grid, ni_grid, te_grid, zstar_grid]:
        g[g < 0] = 0.0

    # 边缘软过渡 (在 x 方向两端)
    edge = config.edge_ramp_pixels
    if edge > 0 and nx > 2 * edge:
        ramp = np.sin(np.linspace(0, np.pi / 2, edge)) ** 2
        envelope = np.ones(nx)
        envelope[:edge] = ramp
        envelope[-edge:] = ramp[::-1]
        ne_grid[0, :] *= envelope
        ni_grid[0, :] *= envelope

    pixel_size_x_m = pixel_size_x_cm * 1e-2  # cm → m
    pixel_size_z_m = pixel_size_z_cm * 1e-2

    logger.info(
        f"  side-on: ({nz}×{nx}), "
        f"dx={pixel_size_x_m*1e6:.1f}μm, dz(los)={pixel_size_z_m*1e6:.1f}μm, "
        f"ne=[{ne_grid.max():.2e}] cm⁻³"
    )

    return {
        "plasma": {
            "ne": ne_grid,
            "ni": ni_grid,
            "te": te_grid,
            "zstar": zstar_grid,
            "pixel_size_x_m": pixel_size_x_m,
            "pixel_size_z_m": pixel_size_z_m,
            "pixel_size_y_m": 0.0,
            "Z": Z,
            "z_start_m": 0.5,
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FACE-ON: 激光 ∥ X 射线
#   Multi1D 剖面 → z 轴 (传播深度)
#   x 方向均匀 slab
#   冷固体/热等离子体在 z 方向分层 → 可选双元素
# ═══════════════════════════════════════════════════════════════════════════════


def _build_face_on(
    profile: dict,
    config: GridConfig,
    Z: int,
    materials_map: dict,
) -> dict:
    """
    Face-on 几何:

      X射线 →→→ (z 方向, 穿过不同深度)
        │
        ▼ 先穿过热等离子体, 再穿过冷固体

      RAVE-SIM grid: (nz, nx)
        - z 轴: Multi1D 剖面 (激光方向 = X射线深度)
        - x 轴: 均匀 slab
    """
    cold_mask, hot_mask = _classify_cells(profile, config)
    n_cold = cold_mask.sum()
    n_hot = hot_mask.sum()

    result: dict = {}

    if n_hot == 0 and n_cold == 0:
        logger.warning("face-on: 无有效区域")
        return _empty_result(config, "plasma")

    # ── 等离子体网格 ──
    if n_hot > 0:
        hot_profile = _mask_profile(profile, hot_mask)
        result["plasma"] = _build_face_on_zstack(hot_profile, config, Z, "plasma")

    # ── 固体网格 ──
    if n_cold > 0:
        cold_profile = _mask_profile(profile, cold_mask)
        result["solid"] = _build_face_on_zstack(cold_profile, config, Z, "solid",
                                                  materials_map)

    # ── 设置 z_start ──
    base_z = 0.5  # m
    if "plasma" in result and "solid" in result:
        # 确定哪个在前 (XC 较小的 → 更靠近源)
        p = result["plasma"]
        s = result["solid"]
        p_thick = p["ne"].shape[0] * p["pixel_size_z_m"]
        # 默认 plasma 在前 (通常热等离子体在激光入射侧)
        p["z_start_m"] = base_z
        s["z_start_m"] = base_z + p_thick
        logger.info(f"  face-on: plasma(z={p['z_start_m']:.3f}→{p['z_start_m']+p_thick:.3f}m) "
                     f"+ solid(z={s['z_start_m']:.3f}m)")

    return result


def _build_face_on_zstack(
    profile: dict,
    config: GridConfig,
    Z: int,
    kind: str,  # "plasma" | "solid"
    materials_map: Optional[dict] = None,
) -> dict:
    """Face-on: 构建 z-stacked 网格。"""
    xc = profile["XC"]
    r = profile["R"]
    te = profile["T"]
    dene = profile["DENE"]
    zi = profile["ZI"]
    mid = profile.get("MID", np.zeros_like(xc, dtype=int))

    valid = (dene > 1e10) & (r > 1e-15)
    if not valid.any():
        return _empty_result(config, kind)

    xc_v = xc[valid]
    dene_v = dene[valid]
    zi_v = zi[valid]
    te_v = te[valid]
    r_v = r[valid]
    mid_v = mid[valid].astype(int)

    z_min = xc_v.min()
    z_max = xc_v.max()

    pixel_size_z_cm = config.pixel_size_z_um * 1e-4
    nz = max(10, int((z_max - z_min) / pixel_size_z_cm))
    if nz > config.max_nz:
        nz = config.max_nz
    z_uniform = np.linspace(z_min, z_max, nz)

    nx = config.nx
    pixel_size_x_m = config.transverse_size_um * 1e-6 / nx
    pixel_size_z_m = (z_max - z_min) / (nz - 1) * 1e-2 if nz > 1 else pixel_size_z_cm * 1e-2

    if kind == "plasma":
        ni_v = np.where(zi_v > 0.01, dene_v / zi_v, dene_v)

        ne_z = _safe_interp(z_uniform, xc_v, dene_v)
        ni_z = _safe_interp(z_uniform, xc_v, ni_v)
        te_z = _safe_interp(z_uniform, xc_v, te_v)
        zi_z = _safe_interp(z_uniform, xc_v, zi_v)

        return {
            "ne": np.outer(ne_z, np.ones(nx)).astype(np.float64),
            "ni": np.outer(ni_z, np.ones(nx)).astype(np.float64),
            "te": np.outer(te_z, np.ones(nx)).astype(np.float64),
            "zstar": np.outer(zi_z, np.ones(nx)).astype(np.float64),
            "pixel_size_x_m": pixel_size_x_m,
            "pixel_size_z_m": pixel_size_z_m,
            "pixel_size_y_m": 0.0,
            "Z": Z,
            "z_start_m": 0.5,
        }
    else:
        # Solid: precise_sample
        density_z = _safe_interp(z_uniform, xc_v, r_v)
        unique_mids = np.unique(mid_v)
        mid_to_idx = {0: 0}
        materials_list = []
        for m in unique_mids:
            if m == 0:
                continue
            info = (materials_map or {}).get(m, {})
            materials_list.append([info.get("formula", "C"), float(info.get("rho", 1.0))])
            mid_to_idx[int(m)] = len(materials_list)

        material_z = np.zeros(nz, dtype=np.uint32)
        for i, z in enumerate(z_uniform):
            idx = np.argmin(np.abs(xc_v - z))
            material_z[i] = mid_to_idx.get(int(mid_v[idx]), 0)

        return {
            "material_grid": np.outer(material_z, np.ones(nx)).astype(np.uint32),
            "density_grid": np.outer(density_z, np.ones(nx)).astype(np.float32),
            "materials": materials_list if materials_list else [["C", 1.0]],
            "pixel_size_x_m": pixel_size_x_m,
            "pixel_size_z_m": pixel_size_z_m,
            "z_start_m": 0.5,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════


def _classify_cells(profile: dict, config: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    """将 cells 分为冷固体和热等离子体 (仅 face-on 使用)。"""
    te = profile["T"]
    zi = profile["ZI"]
    r = profile["R"]
    zi_max = zi.max() if zi.max() > 1e-6 else 1.0
    cold = (te < config.Te_threshold_eV) & (zi < config.Zstar_threshold_frac * zi_max) & (r > 1e-15)
    hot = (~cold) & (r > 1e-15)
    return cold, hot


def _mask_profile(profile: dict, mask: np.ndarray) -> dict:
    """对 1D 剖面应用布尔掩码。"""
    return {k: v[mask] for k, v in profile.items()
            if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == len(mask)}


def _get_effective_Z(profile: dict, materials_map: dict, config: GridConfig) -> int:
    """从 MID 推断有效原子序数。"""
    mid = profile.get("MID", None)
    if mid is None:
        return config.default_Z
    mids = mid[mid > 0].astype(int)
    if len(mids) == 0:
        return config.default_Z
    unique, counts = np.unique(mids, return_counts=True)
    dominant = unique[np.argmax(counts)]
    info = materials_map.get(dominant, {})
    z = info.get("z", config.default_Z)
    return int(z) if z > 0 else config.default_Z


def _safe_interp(
    x_new: np.ndarray, x_old: np.ndarray, y_old: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """安全 1D 插值: 去重 + 单调排序 + 外推。"""
    if len(x_old) < 2:
        return np.full_like(x_new, fill_value, dtype=np.float64)

    order = np.argsort(x_old)
    xs, ys = x_old[order], y_old[order]
    uniq = np.diff(xs, prepend=xs[0] - 1) > 1e-15
    xs, ys = xs[uniq], ys[uniq]

    if len(xs) < 2:
        return np.full_like(x_new, fill_value, dtype=np.float64)

    try:
        f = interp1d(xs, ys, kind="linear", bounds_error=False,
                      fill_value=(float(ys[0]), float(ys[-1])))
        result = f(x_new)
    except Exception:
        idx = np.searchsorted(xs, x_new).clip(1, len(xs) - 1)
        result = ys[idx]

    return np.nan_to_num(np.asarray(result, dtype=np.float64),
                         nan=fill_value, posinf=fill_value, neginf=fill_value)


def _empty_result(config: GridConfig, kind: str) -> dict:
    """返回空结果。"""
    nx = config.nx
    if kind == "plasma":
        return {"plasma": {
            "ne": np.zeros((1, nx)), "ni": np.zeros((1, nx)),
            "te": np.zeros((1, nx)), "zstar": np.zeros((1, nx)),
            "pixel_size_x_m": config.transverse_size_um * 1e-6 / nx,
            "pixel_size_z_m": config.los_thickness_um * 1e-6,
            "pixel_size_y_m": 0.0, "Z": config.default_Z, "z_start_m": 0.5,
        }}
    else:
        return {"solid": {
            "material_grid": np.zeros((1, nx), dtype=np.uint32),
            "density_grid": np.zeros((1, nx), dtype=np.float32),
            "materials": [["C", 1.0]],
            "pixel_size_x_m": config.transverse_size_um * 1e-6 / nx,
            "pixel_size_z_m": config.pixel_size_z_um * 1e-6,
            "z_start_m": 0.5,
        }}


# ─── 便利函数 ────────────────────────────────────────────────────────────────


def save_grids(grids: dict, output_dir: Union[str, Path]) -> dict:
    """将网格保存为 .npy 文件。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict = {}

    for section in ["plasma", "solid"]:
        if section not in grids:
            continue
        p = grids[section]
        for key in p:
            val = p[key]
            if isinstance(val, np.ndarray):
                fname = f"{key}_grid.npy" if section == "plasma" else f"{key}.npy"
                np.save(output_dir / fname, val)
                paths[f"{key}_grid_path" if section == "plasma" else f"{key}_path"] = fname

    logger.info(f"网格已保存: {output_dir} ({len(paths)} files)")
    return paths


def grid_summary(grids: dict) -> str:
    """生成网格摘要。"""
    lines = ["网格摘要", "─" * 40]
    meta = grids.get("meta", {})
    for k in ["timestep", "time", "geometry", "effective_Z"]:
        if k in meta:
            lines.append(f"  {k}: {meta[k]}")

    for section in ["plasma", "solid"]:
        if section not in grids:
            continue
        p = grids[section]
        shape = p.get("ne", p.get("material_grid")).shape
        lines.append(f"  {section}: {shape} (nz×nx)")
        if section == "plasma":
            mask = p["ne"] > 0
            if mask.any():
                lines.append(f"    ne: [{p['ne'][mask].min():.2e}, {p['ne'][mask].max():.2e}]")
                lines.append(f"    Te: [{p['te'][mask].min():.1f}, {p['te'][mask].max():.1f}] eV")
            lines.append(f"    dx: {p['pixel_size_x_m']*1e6:.2f}μm  dz: {p['pixel_size_z_m']*1e6:.2f}μm")
        else:
            lines.append(f"    materials: {p.get('materials', [])}")

    return "\n".join(lines)
