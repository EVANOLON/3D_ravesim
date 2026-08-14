"""
RAVE-SIM 配置生成器
===================
从 Multi1D 数据和等离子体网格自动生成 RAVE-SIM 的 config.yaml。

支持:
  - 1D (柱对称) 和 2D (球面波) 双模式
  - 混合元素: precise_sample (冷固体) + plasma_sample (热等离子体)
  - 自动计算 z_start 位置 (基于 Multi1D 空间坐标)
  - 8-10 keV 默认 X 射线源参数

关键设计 — 混合网格的 RAVE-SIM 实现:

    混合网格通过 elements 列表中的多个元素实现:

    elements:
      - type: plasma_sample     ← 热等离子体 (先被 X 射线穿过)
        z_start: 0.50
        ...
      - type: precise_sample    ← 冷固体 (后被 X 射线穿过)
        z_start: 0.500004        ← = 等离子体后端位置
        ...

    RAVE-SIM 按顺序处理 elements: source → free space → element[0]
    → free space → element[1] → ... → detector。 两个相邻元素之间
    若 z 坐标连续, 则无需自由空间传播 (直接紧贴)。

用法:
    from rave_config_gen import generate_config, SimConfig

    cfg = generate_config(grid_dir, grids_dict, sim_config)
    # 生成 config.yaml + subconfig.yaml 模板
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger("rave_config_gen")


# ─── 配置数据类 ──────────────────────────────────────────────────────────────


@dataclass
class SimConfig:
    """RAVE-SIM 仿真参数配置。"""

    # ── 仿真模式 ──
    dimension: str = "1d"  # "1d" | "2d"

    # ── X 射线源 ──
    energy_min_eV: float = 8000.0
    energy_max_eV: float = 10000.0
    source_z_m: float = 0.0
    x_range_um: tuple = (-3.0, 3.0)
    y_range_um: tuple = (-3.0, 3.0)  # 仅 2D 模式
    nr_source_points: int = 100
    seed: int = 42

    # ── 几何 ──
    z_target_m: float = 0.5       # 源到靶第一面距离 [m]
    z_detector_m: float = 4.8     # 源到探测器距离 [m]
    detector_size_x_m: float = 0.003
    detector_pixel_size_x_m: float = 2.0e-5

    # ── 数值参数 ──
    N: int = 33554432
    dx_m: float = 3.0e-10
    chunk_size: int = 16777216
    dtype: str = "c8"
    use_disk_vector: bool = False

    # ── 1D 模式 ──
    detector_pixel_size_y_m: float = 2.0e-5

    # ── 2D 模式 ──
    ny: int = 512
    dy_m: float = 2.0e-7
    detector_size_y_m: float = 0.001

    # ── 输出选项 ──
    save_final_u_vectors: bool = False
    phase_stepping: list = field(default_factory=lambda: [0.0])


# ─── 主入口 ──────────────────────────────────────────────────────────────────


def generate_config(
    grid_dir: Union[str, Path],
    grids: dict,
    sim_config: Optional[SimConfig] = None,
    *,
    spectrum_path: Optional[str] = None,
) -> dict:
    """
    生成 RAVE-SIM 配置。

    参数:
        grid_dir:      网格文件 (.npy) 所在目录 (相对路径基准)
        grids:         build_hybrid_grids() 的返回值
        sim_config:    仿真参数 (None=使用默认)
        spectrum_path: X 射线谱文件路径 (可选 HDF5)

    返回:
        {"config": dict, "elements": list, "yaml_str": str}
        可直接用 ruamel.yaml 写入文件
    """
    if sim_config is None:
        sim_config = SimConfig()

    grid_dir = Path(grid_dir)
    meta = grids.get("meta", {})
    is_2d = sim_config.dimension == "2d"

    # ── 构建 sim_params ──
    sim_params = _build_sim_params(sim_config, is_2d)

    # ── 构建 multisource ──
    multisource = _build_multisource(sim_config, is_2d, spectrum_path)

    # ── 构建 elements 列表 ──
    elements = _build_elements(grids, grid_dir, sim_config)

    # ── 组装完整配置 ──
    config = {
        "sim_params": sim_params,
        "multisource": multisource,
        "elements": elements,
    }

    if sim_config.use_disk_vector:
        config["use_disk_vector"] = True
    if sim_config.save_final_u_vectors:
        config["save_final_u_vectors"] = True
    if sim_config.dtype != "c16":
        config["dtype"] = sim_config.dtype

    logger.info(
        f"生成配置: {len(elements)} 个光学元件, "
        f"{'2D' if is_2d else '1D'} 模式, "
        f"能量 {sim_config.energy_min_eV/1000:.0f}-{sim_config.energy_max_eV/1000:.0f} keV"
    )

    return {
        "config": config,
        "elements": elements,
        "sim_params": sim_params,
        "multisource": multisource,
    }


def save_config(
    config_data: dict,
    output_dir: Union[str, Path],
    *,
    filename: str = "config.yaml",
) -> Path:
    """将配置写入 YAML 文件。"""
    try:
        from ruamel.yaml import YAML
        yaml = YAML(typ="rt")
        yaml.indent(mapping=2, sequence=4, offset=2)
    except ImportError:
        # 降级: 使用标准 yaml (不会保留注释, 但可用)
        import yaml as _yaml
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            _yaml.dump(config_data["config"], f, default_flow_style=False, sort_keys=False)
        return output_path

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yaml.dump(config_data["config"], output_path)
    return output_path


# ─── 内部: sim_params ───────────────────────────────────────────────────────


def _build_sim_params(cfg: SimConfig, is_2d: bool) -> dict:
    """构建 sim_params 配置块。"""
    params = {
        "N": cfg.N,
        "dx": cfg.dx_m,
        "z_detector": cfg.z_detector_m,
        "detector_size": cfg.detector_size_x_m,
        "detector_pixel_size_x": cfg.detector_pixel_size_x_m,
        "detector_pixel_size_y": cfg.detector_pixel_size_y_m,
        "chunk_size": cfg.chunk_size,
    }

    if is_2d:
        params["ny"] = cfg.ny
        params["dy"] = cfg.dy_m
        params["detector_size_x"] = cfg.detector_size_x_m
        params["detector_size_y"] = cfg.detector_size_y_m
        params["is_2d"] = 1

    return params


# ─── 内部: multisource ──────────────────────────────────────────────────────


def _build_multisource(
    cfg: SimConfig,
    is_2d: bool,
    spectrum_path: Optional[str],
) -> dict:
    """构建 multisource 配置块。"""
    ms = {
        "type": "points",
        "energy_range": [float(cfg.energy_min_eV), float(cfg.energy_max_eV)],
        "x_range": [float(cfg.x_range_um[0]) * 1e-6, float(cfg.x_range_um[1]) * 1e-6],
        "z": cfg.source_z_m,
        "nr_source_points": cfg.nr_source_points,
        "seed": cfg.seed,
    }

    if is_2d:
        ms["y_range"] = [float(cfg.y_range_um[0]) * 1e-6, float(cfg.y_range_um[1]) * 1e-6]

    if spectrum_path:
        ms["spectrum"] = spectrum_path

    return ms


# ─── 内部: elements 构建 (核心: 混合网格实现) ─────────────────────────────


def _build_elements(
    grids: dict,
    grid_dir: Path,
    cfg: SimConfig,
) -> list[dict]:
    """
    构建 elements 列表。

    Side-on (激光 ⊥ X射线):
      冷固体和热等离子体在 x 方向并排 (同一 z 位置)
      → 单 plasma_sample 覆盖全部, plasma 模型通过 Z* 自然处理冷→热过渡

    Face-on (激光 ∥ X射线):
      冷固体和热等离子体在 z 方向分层
      → plasma_sample (热) + precise_sample (冷) 双元素
    """
    elements = []
    meta = grids.get("meta", {})
    geometry = meta.get("geometry", "side-on")
    base_z = cfg.z_target_m

    has_plasma = "plasma" in grids
    has_solid = "solid" in grids

    # ═══ side-on: 单 plasma_sample ═══
    if geometry == "side-on":
        if not has_plasma:
            logger.warning("side-on: 无等离子体网格")
            return elements

        p = grids["plasma"]
        elem = {
            "type": "plasma_sample",
            "z_start": float(base_z),
            "pixel_size_x": float(p["pixel_size_x_m"]),
            "pixel_size_z": float(p["pixel_size_z_m"]),
            "ne_grid_path": _rel_path(grid_dir, "ne_grid.npy"),
            "ni_grid_path": _rel_path(grid_dir, "ni_grid.npy"),
            "te_grid_path": _rel_path(grid_dir, "te_grid.npy"),
            "zstar_grid_path": _rel_path(grid_dir, "zstar_grid.npy"),
            "Z": int(p["Z"]),
            "x_positions": cfg.phase_stepping if cfg.phase_stepping else [0.0],
        }
        if p.get("pixel_size_y_m", 0.0) > 0:
            elem["pixel_size_y"] = float(p["pixel_size_y_m"])
            elem["y_positions"] = cfg.phase_stepping if cfg.phase_stepping else [0.0]
        elements.append(elem)

    # ═══ face-on: plasma_sample + precise_sample ═══
    else:
        if has_plasma:
            p = grids["plasma"]
            plasma_thick = p["ne"].shape[0] * p["pixel_size_z_m"]
            elements.append({
                "type": "plasma_sample",
                "z_start": float(base_z),
                "pixel_size_x": float(p["pixel_size_x_m"]),
                "pixel_size_z": float(p["pixel_size_z_m"]),
                "ne_grid_path": _rel_path(grid_dir, "ne_grid.npy"),
                "ni_grid_path": _rel_path(grid_dir, "ni_grid.npy"),
                "te_grid_path": _rel_path(grid_dir, "te_grid.npy"),
                "zstar_grid_path": _rel_path(grid_dir, "zstar_grid.npy"),
                "Z": int(p["Z"]),
                "x_positions": cfg.phase_stepping if cfg.phase_stepping else [0.0],
            })
        else:
            plasma_thick = 0.0

        if has_solid:
            s = grids["solid"]
            elements.append({
                "type": "precise_sample",
                "z_start": float(base_z + plasma_thick),
                "pixel_size_x": float(s["pixel_size_x_m"]),
                "pixel_size_z": float(s["pixel_size_z_m"]),
                "material_grid_path": _rel_path(grid_dir, "material_grid.npy"),
                "density_grid_path": _rel_path(grid_dir, "density_grid.npy"),
                "materials": s.get("materials", [["C", 1.0]]),
                "x_positions": cfg.phase_stepping if cfg.phase_stepping else [0.0],
            })

    return elements


# ─── 辅助 ────────────────────────────────────────────────────────────────────


def _rel_path(base_dir: Path, filename: str) -> str:
    """生成相对路径字符串 (用于 YAML)。"""
    return filename  # 网格文件与 config.yaml 在同一目录


def generate_subconfig(
    energy_eV: float,
    source_x_m: float = 0.0,
    source_y_m: float = 0.0,
    deltabeta_table: Optional[list] = None,
) -> dict:
    """
    生成单个源点的 subconfig.yaml。

    RAVE-SIM multisim 会为每个 Monte Carlo 源点生成一个子目录,
    内含 subconfig.yaml。
    """
    sub = {
        "source": {
            "type": "point",
            "x": float(source_x_m),
            "z": 0.0,
        },
        "energy": float(energy_eV),
    }

    if source_y_m != 0.0:
        sub["source"]["y"] = float(source_y_m)

    if deltabeta_table:
        sub["deltabeta_table"] = deltabeta_table

    return sub


# ─── 便利函数 ────────────────────────────────────────────────────────────────


def quick_config_for_timestep(
    grid_dir: Union[str, Path],
    grids: dict,
    *,
    energy_eV: float = 9000.0,
    z_target_m: float = 0.5,
    z_detector_m: float = 4.8,
    N: int = 33554432,
    dimension: str = "1d",
) -> dict:
    """
    快速生成单时间步、单能配置 (用于测试/调试)。

    参数:
        grid_dir: 网格文件目录
        grids:    build_hybrid_grids() 的返回值
        energy_eV: 单能光子能量
        z_target_m: 源到靶距离
        z_detector_m: 源到探测器距离
        N: 波前采样点数 (调试用小值如 8192)
        dimension: "1d" | "2d"

    返回:
        config_data dict (可直接传给 save_config)
    """
    sim_cfg = SimConfig(
        dimension=dimension,
        energy_min_eV=energy_eV,
        energy_max_eV=energy_eV,
        z_target_m=z_target_m,
        z_detector_m=z_detector_m,
        N=N,
        nr_source_points=1,
        seed=42,
    )

    return generate_config(grid_dir, grids, sim_cfg)
