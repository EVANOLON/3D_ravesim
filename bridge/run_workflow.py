"""
Multi1D++ → RAVE-SIM 批量工作流编排器
======================================
命令行入口，串联整个工作流:

  [1] 加载 Multi1D 数据
  [2] 循环时间步: 构建网格 → 生成配置 → (可选)运行 RAVE-SIM
  [3] 汇总结果

用法:
  # 生成模式 (仅网格+配置, 不运行仿真)
  python -m bridge.run_workflow case_dir/ --timesteps 0:10:100 --mode generate

  # 运行模式 (生成 + 执行 RAVE-SIM)
  python -m bridge.run_workflow case_dir/ --timesteps 0,50,100 --mode run

  # 完整批量
  python -m bridge.run_workflow case_dir/ --timesteps 0:5:344 --energy 8000 10000
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Union

# ─── 日志 ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_workflow")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_timesteps(spec: str, nt: int) -> list[int]:
    """
    解析时间步规格。

    "all"           → 全部
    "0:10:100"      → range(0, 100, 10)
    "0,50,100,200"  → [0, 50, 100, 200]
    "100"           → [100]
    """
    spec = spec.strip()
    if spec.lower() == "all":
        return list(range(nt))

    if "," in spec:
        return [int(x) for x in spec.split(",")]

    parts = spec.split(":")
    if len(parts) == 1:
        return [int(parts[0])]
    elif len(parts) == 2:
        return list(range(int(parts[0]), int(parts[1])))
    elif len(parts) == 3:
        return list(range(int(parts[0]), int(parts[2]), int(parts[1])))
    else:
        raise ValueError(f"无法解析时间步规格: {spec}")


def parse_energy(spec: str) -> tuple[float, float]:
    """解析能量规格: "9000" → (9000,9000), "8000 10000" → (8000,10000)"""
    parts = spec.split()
    if len(parts) == 1:
        e = float(parts[0])
        return (e, e)
    elif len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    else:
        raise ValueError(f"无法解析能量规格: {spec}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi1D++ → RAVE-SIM 批量工作流",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s 251222test/ --timesteps 0,114,229 --mode generate
  %(prog)s 251222test/ --timesteps 0:10:100 --energy 9000 --nx 128
  %(prog)s 251222test/ --timesteps all --mode run --output ./results/
        """,
    )

    p.add_argument("case_path", help="Multi1D++ 输出目录或 case 文件前缀")
    p.add_argument(
        "--timesteps", default="all",
        help="时间步: 'all', '0:10:100', '0,50,100', '100'"
    )
    p.add_argument(
        "--mode", choices=["generate", "run"], default="generate",
        help="generate=仅网格+配置, run=生成+运行RAVE-SIM"
    )
    p.add_argument(
        "--geometry", choices=["side-on", "face-on"], default="side-on",
        help="实验几何 (默认: side-on)"
    )
    p.add_argument(
        "--dimension", choices=["1d", "2d"], default="1d",
        help="RAVE-SIM 仿真维度"
    )

    # 网格参数
    g = p.add_argument_group("网格参数")
    g.add_argument("--nx", type=int, default=256, help="横向像素数 (默认: 256)")
    g.add_argument("--transverse-size", type=float, default=200.0,
                   help="横向物理范围 [μm] (默认: 200)")
    g.add_argument("--los-thickness", type=float, default=100.0,
                   help="视线厚度 [μm], 仅 side-on (默认: 100)")
    g.add_argument("--pixel-size-z", type=float, default=1.0,
                   help="z 像素 [μm], 仅 face-on (默认: 1.0)")
    g.add_argument("--max-nz", type=int, default=500,
                   help="最大 z 层数, 仅 face-on (默认: 500)")

    # X 射线源
    s = p.add_argument_group("X 射线源")
    s.add_argument("--energy", default="9000",
                   help="光子能量 [eV]: '9000' 或 '8000 10000'")
    s.add_argument("--nr-sources", type=int, default=100,
                   help="Monte Carlo 源点数 (默认: 100)")
    s.add_argument("--x-range", type=float, nargs=2, default=[-3.0, 3.0],
                   help="源 x 范围 [μm] (默认: -3 3)")

    # 几何
    d = p.add_argument_group("探测器几何")
    d.add_argument("--z-target", type=float, default=0.5,
                   help="源到靶距离 [m] (默认: 0.5)")
    d.add_argument("--z-detector", type=float, default=4.8,
                   help="源到探测器距离 [m] (默认: 4.8)")

    # 数值
    n = p.add_argument_group("数值参数")
    n.add_argument("--N", type=int, default=33554432,
                   help="波前采样点 (默认: 33554432, 调试用 8192)")
    n.add_argument("--dx", type=float, default=3e-10,
                   help="横向像素间距 [m] (默认: 3e-10)")
    n.add_argument("--chunk-size", type=int, default=16777216)

    # 输出
    o = p.add_argument_group("输出")
    o.add_argument("--output", default="./workflow_output",
                   help="输出根目录 (默认: ./workflow_output)")
    o.add_argument("--keep-grids", action="store_true", default=True,
                   help="保留中间网格文件")
    o.add_argument("--resume", action="store_true",
                   help="跳过已存在输出的时间步")

    # 调试
    dbg = p.add_argument_group("调试")
    dbg.add_argument("--dry-run", action="store_true",
                     help="验证但不写入文件")
    dbg.add_argument("--verbose", "-v", action="store_true",
                     help="详细日志")

    return p


# ─── 主逻辑 ──────────────────────────────────────────────────────────────────


def run_workflow(args: argparse.Namespace) -> int:
    """执行工作流。返回 0=成功, 1=失败。"""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── 导入 bridge 模块 ──
    _ensure_bridge_on_path()

    from multi1d_loader import load_multi1d_output, summary as loader_summary
    from plasma_grid_builder import build_hybrid_grids, GridConfig, save_grids, grid_summary
    from rave_config_gen import generate_config, save_config, SimConfig

    # ── 1. 加载数据 ──
    logger.info(f"加载: {args.case_path}")
    try:
        data = load_multi1d_output(args.case_path, parse_materials=True)
    except Exception as e:
        logger.error(f"加载失败: {e}")
        return 1

    nt = data["nt"]
    logger.info(loader_summary(data))

    # ── 2. 解析时间步 ──
    timesteps = parse_timesteps(args.timesteps, nt)
    timesteps = [t for t in timesteps if 0 <= t < nt]
    if not timesteps:
        logger.error(f"无有效时间步 (nt={nt}): {args.timesteps}")
        return 1

    logger.info(f"时间步: {len(timesteps)} 个 ({timesteps[0]}→{timesteps[-1]})")

    # ── 3. 解析能量 ──
    energy_range = parse_energy(args.energy)

    # ── 4. 准备输出目录 ──
    output_root = Path(args.output)
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    # ── 5. 构建配置 ──
    grid_config = GridConfig(
        geometry=args.geometry,
        nx=args.nx,
        transverse_size_um=args.transverse_size,
        los_thickness_um=args.los_thickness,
        pixel_size_z_um=args.pixel_size_z,
        max_nz=args.max_nz,
    )

    sim_config = SimConfig(
        dimension=args.dimension,
        energy_min_eV=energy_range[0],
        energy_max_eV=energy_range[1],
        z_target_m=args.z_target,
        z_detector_m=args.z_detector,
        N=args.N,
        dx_m=args.dx,
        chunk_size=args.chunk_size,
        nr_source_points=args.nr_sources,
        x_range_um=tuple(args.x_range),
    )

    # ── 6. 循环时间步 ──
    succeeded = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    for i_ts, ts in enumerate(timesteps):
        ts_dir = output_root / f"t{ts:05d}"
        detected_path = ts_dir / "detected.npy"

        # 断点续传
        if args.resume and detected_path.exists():
            logger.info(f"[{i_ts+1}/{len(timesteps)}] t={ts} — 跳过 (已存在)")
            skipped += 1
            continue

        logger.info(f"[{i_ts+1}/{len(timesteps)}] t={ts}")

        try:
            # Step A: 构建网格
            grids = build_hybrid_grids(data, timestep=ts, config=grid_config)

            if args.verbose:
                logger.debug(grid_summary(grids))

            if args.dry_run:
                succeeded += 1
                continue

            # Step B: 保存网格
            ts_dir.mkdir(parents=True, exist_ok=True)
            save_grids(grids, ts_dir)

            # Step C: 生成配置
            config_data = generate_config(ts_dir, grids, sim_config)

            # 手动覆盖 z_start (用户指定)
            for el in config_data["elements"]:
                el["z_start"] = float(args.z_target)

            save_config(config_data, ts_dir)

            # Step D: 运行 RAVE-SIM (可选)
            if args.mode == "run":
                _run_rave_sim(ts_dir, config_data, args)

            succeeded += 1

        except Exception as e:
            logger.error(f"t={ts} 失败: {e}", exc_info=args.verbose)
            failed += 1
            if not args.resume:
                # 非 resume 模式下, 第一个失败就停止
                logger.error("终止 (使用 --resume 跳过失败项)")
                break

    elapsed = time.time() - t_start

    # ── 7. 汇总 ──
    print(f"\n{'='*50}")
    print(f"工作流完成: {elapsed:.1f}s")
    print(f"  成功: {succeeded}")
    print(f"  跳过: {skipped}")
    print(f"  失败: {failed}")
    print(f"  输出: {output_root.absolute()}")
    print(f"{'='*50}")

    return 0 if failed == 0 else 1


# ─── RAVE-SIM 执行 ───────────────────────────────────────────────────────────


def _run_rave_sim(ts_dir: Path, config_data: dict, args: argparse.Namespace) -> None:
    """
    通过 subprocess 运行 RAVE-SIM big-wave。

    预期环境: RAVE-SIM big-wave 目录在项目根下,
    Python 路径包含 big-wave/ 和 nist_lookup/。
    """
    rave_root = _find_rave_root()
    config_path = ts_dir / "config.yaml"

    cmd = [
        sys.executable, "-c", _RAVE_RUNNER_SCRIPT,
        str(config_path),
        str(ts_dir),
        str(args.N),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(rave_root / "big-wave"),
        str(rave_root / "nist_lookup" / "nist_lookup"),
        env.get("PYTHONPATH", ""),
    ])

    logger.info(f"  运行 RAVE-SIM (N={args.N})...")
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(rave_root), env=env,
        timeout=3600,  # 1 hour max
    )

    if result.returncode != 0:
        # 打印部分 stderr 用于诊断
        stderr_tail = "\n".join(result.stderr.splitlines()[-10:])
        logger.error(f"  RAVE-SIM 失败 (code={result.returncode}):\n{stderr_tail}")
        raise RuntimeError(f"RAVE-SIM exited with {result.returncode}")
    else:
        logger.info(f"  → {ts_dir / 'detected.npy'}")


# RAVE-SIM 内联运行脚本 (避免模块导入问题)
_RAVE_RUNNER_SCRIPT = r"""
import sys, os, json
from pathlib import Path
import numpy as np

config_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
N = int(sys.argv[3])

sys.path.insert(0, 'big-wave')
sys.path.insert(0, 'nist_lookup/nist_lookup')

from config import load, parse_optical_element, parse_sim_params, parse_source
from wavesim import run_simulation

# Load config
cfg = load(config_path)
sim_params = parse_sim_params(cfg['sim_params'])
sim_params.wl = 1.239842e-6 / 9000.0  # 9 keV default (overridden by subconfig if present)
sim_params.N = N

elements = [parse_optical_element(el, config_path.parent) for el in cfg['elements']]

# Create point source at centre
from source import PointSource
source = PointSource(x=0.0, z=0.0)

# Run
detected = run_simulation(sim_params, elements, source)

# Save
np.save(str(output_dir / 'detected.npy'), detected)
print(f'Detected shape: {detected.shape}, max={detected.max():.4e}')
"""


# ─── 辅助 ────────────────────────────────────────────────────────────────────


def _ensure_bridge_on_path() -> None:
    """确保 bridge 模块可导入。"""
    bridge_dir = Path(__file__).resolve().parent
    if str(bridge_dir) not in sys.path:
        sys.path.insert(0, str(bridge_dir))


def _find_rave_root() -> Path:
    """查找 RAVE-SIM 项目根目录。"""
    # 从 bridge/ 的父目录推断
    bridge_dir = Path(__file__).resolve().parent
    rave_root = bridge_dir.parent
    if (rave_root / "big-wave").exists():
        return rave_root
    # 降级: 搜索
    for d in [Path.cwd(), Path("/mnt/d/rave-sim-main/rave-sim-main")]:
        if (d / "big-wave").exists():
            return d
    raise FileNotFoundError("找不到 RAVE-SIM 项目根目录 (需含 big-wave/)")


# ─── 入口 ────────────────────────────────────────────────────────────────────


def main():
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run_workflow(args))


if __name__ == "__main__":
    main()
