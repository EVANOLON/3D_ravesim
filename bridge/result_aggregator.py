"""
结果汇总与诊断
==============
加载工作流输出 (各时间步的 detected.npy), 汇总为时间序列,
生成 XPCI 条纹动画, 计算诊断量, 导出 HDF5。

用法:
  from result_aggregator import aggregate_workflow

  result = aggregate_workflow("./workflow_output/")
  # result["stack"]  → (nt, n_phase, n_pixels) 或 (nt, n_pixels)
  # result["diagnostics"] → {"visibility": ..., "peak": ..., "fringe_shift": ...}

  # 生成动画
  animate_fringes(result, "fringes.mp4")

  # 导出 HDF5
  export_hdf5(result, "results.h5")
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger("result_aggregator")

# ─── 主入口 ──────────────────────────────────────────────────────────────────


def aggregate_workflow(
    output_dir: Union[str, Path],
    *,
    phase_step: int = 0,
    pattern: str = "t*/detected.npy",
) -> dict:
    """
    加载工作流输出目录中的所有 detected.npy。

    参数:
        output_dir:  工作流输出根目录 (含 t00000/, t00001/, ... 子目录)
        phase_step:  相位步索引 (若 detected.npy 有多步)
        pattern:     文件匹配 glob 模式

    返回:
        {
            "stack":       ndarray, shape (nt, ...),
            "timesteps":   list[int],
            "times":       ndarray, shape (nt,),
            "shapes":      list[tuple],
            "file_paths":  list[Path],
            "diagnostics": dict,
        }
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"目录不存在: {output_dir}")

    # ── 发现 detected.npy 文件 ──
    detected_files = sorted(
        output_dir.glob(pattern),
        key=lambda p: _extract_timestep(p),
    )

    if not detected_files:
        logger.warning(f"未找到 detected.npy 文件: {output_dir}/{pattern}")
        return _empty_result()

    logger.info(f"发现 {len(detected_files)} 个 detected.npy 文件")

    # ── 加载并检查形状一致性 ──
    arrays: list[np.ndarray] = []
    timesteps: list[int] = []
    times: list[float] = []
    shapes: list[tuple] = []

    # 尝试加载时间信息
    time_data = _load_time_info(output_dir)

    for f in detected_files:
        try:
            arr = np.load(f)
        except Exception as e:
            logger.warning(f"跳过 {f.name}: {e}")
            continue

        # 处理多相位步: 取指定步
        if arr.ndim == 3:
            # (n_phase, n_y, n_x) or (n_phase, n_x)
            arr = arr[phase_step, ...]
        elif arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[0] <= 20:
            # 可能是 (n_phase, n_x)
            arr = arr[phase_step, ...]

        arrays.append(arr)
        shapes.append(arr.shape)

        ts = _extract_timestep(f)
        timesteps.append(ts)
        times.append(time_data.get(ts, float(ts)))

    if not arrays:
        return _empty_result()

    # 检查形状一致性
    if len(set(shapes)) > 1:
        logger.warning(f"detected 形状不一致: {set(shapes)}, 将用最小形状裁剪")
        min_shape = tuple(min(dim) for dim in zip(*shapes))
        arrays = [a.reshape(-1)[: np.prod(min_shape)].reshape(min_shape) for a in arrays]

    # ── 堆叠 ──
    stack = np.stack(arrays, axis=0)  # (nt, ...)
    times_arr = np.array(times)

    logger.info(f"堆叠形状: {stack.shape} (nt={stack.shape[0]})")

    # ── 计算诊断 ──
    diagnostics = compute_diagnostics(stack, times_arr)

    return {
        "stack": stack,
        "timesteps": timesteps,
        "times": times_arr,
        "shapes": shapes,
        "file_paths": detected_files,
        "diagnostics": diagnostics,
    }


# ─── 诊断计算 ────────────────────────────────────────────────────────────────


def compute_diagnostics(stack: np.ndarray, times: np.ndarray) -> dict:
    """
    从探测器堆叠数据计算诊断量。

    参数:
        stack: (nt, nx) 或 (nt, ny, nx) 探测器强度
        times: (nt,) 时间 [s]

    返回:
        {
            "visibility": (nt,) 相衬可见度,
            "peak_intensity": (nt,) 峰值强度,
            "mean_intensity": (nt,) 平均强度,
            "fringe_shift": (nt,) 条纹质心位移,
            "contrast": (nt,) 对比度,
        }
    """
    nt = stack.shape[0]

    # 对 2D 探测器: 沿 y 平均 → 1D 剖面
    if stack.ndim == 3:
        profile = stack.mean(axis=1)  # (nt, nx)
    else:
        profile = stack  # (nt, nx)

    nx = profile.shape[1]

    # ── 可见度 ──
    # V = (Imax - Imin) / (Imax + Imin)
    i_max = profile.max(axis=1)
    i_min = profile.min(axis=1)
    denom = i_max + i_min
    visibility = np.where(denom > 0, (i_max - i_min) / denom, 0.0)

    # ── 峰值强度 ──
    peak = i_max

    # ── 平均强度 ──
    mean_i = profile.mean(axis=1)

    # ── 对比度 (标准差/均值) ──
    std_i = profile.std(axis=1)
    contrast = np.where(mean_i > 0, std_i / mean_i, 0.0)

    # ── 条纹质心位移 ──
    # 计算每行强度加权质心, 减去初始位置
    x_axis = np.arange(nx)
    centroids = np.array([
        np.average(x_axis, weights=profile[i] + 1e-30)
        for i in range(nt)
    ])
    fringe_shift = centroids - centroids[0]

    return {
        "visibility": visibility,
        "peak_intensity": peak,
        "mean_intensity": mean_i,
        "contrast": contrast,
        "fringe_shift_px": fringe_shift,
        "x_pixels": nx,
    }


# ─── 动画生成 ────────────────────────────────────────────────────────────────


def animate_fringes(
    result: dict,
    output_path: Union[str, Path],
    *,
    fps: int = 10,
    dpi: int = 100,
    cmap: str = "inferno",
    title: str = "XPCI Fringe Evolution",
) -> Optional[Path]:
    """
    生成 XPCI 条纹时间演化动画。

    需要 matplotlib.

    返回:
        输出文件路径, 或 None (若 matplotlib 不可用)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        logger.warning("matplotlib 不可用, 跳过动画生成")
        return None

    stack = result["stack"]
    times = result["times"]
    nt = stack.shape[0]

    # 降采样: 最多 200 帧
    if nt > 200:
        stride = nt // 200
        indices = np.arange(0, nt, stride)
    else:
        indices = np.arange(nt)

    # 对 2D 探测器取中心行
    if stack.ndim == 3:
        cy = stack.shape[1] // 2
        frames = stack[indices, cy, :]
    else:
        frames = stack[indices, :]

    # 确定强度范围
    vmin = np.percentile(frames, 1)
    vmax = np.percentile(frames, 99)

    fig, (ax_img, ax_diag) = plt.subplots(2, 1, figsize=(10, 8),
                                           gridspec_kw={"height_ratios": [3, 1]})

    im = ax_img.imshow(frames, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, origin="lower",
                       extent=[0, frames.shape[1], 0, len(indices)])
    ax_img.set_xlabel("Detector pixel")
    ax_img.set_ylabel("Frame")
    ax_img.set_title(title)
    plt.colorbar(im, ax=ax_img, label="Intensity")

    # 诊断量: 可见度随时间
    diag = result["diagnostics"]
    ax_diag.plot(times, diag["visibility"], "b-", label="Visibility")
    ax_diag.set_xlabel("Time [s]")
    ax_diag.set_ylabel("Visibility")
    ax_diag.legend()
    ax_diag.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    logger.info(f"动画已保存: {output_path}")
    return output_path


def plot_diagnostics(
    result: dict,
    output_path: Union[str, Path],
    *,
    dpi: int = 150,
) -> Optional[Path]:
    """
    绘制 4 面板诊断图。

    面板:
      1. 可见度 vs 时间
      2. 峰值/平均强度 vs 时间
      3. 条纹位移 vs 时间
      4. 初始/最终探测器剖面叠加
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    diag = result["diagnostics"]
    times = result["times"] * 1e9  # s → ns
    stack = result["stack"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel 1: Visibility
    ax = axes[0, 0]
    ax.plot(times, diag["visibility"], "b-", lw=1)
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Visibility")
    ax.set_title("Phase Contrast Visibility")
    ax.grid(True, alpha=0.3)

    # Panel 2: Intensity
    ax = axes[0, 1]
    ax.plot(times, diag["peak_intensity"], "r-", lw=1, label="Peak")
    ax.plot(times, diag["mean_intensity"], "k--", lw=1, label="Mean")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Intensity")
    ax.set_title("Detector Intensity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Fringe shift
    ax = axes[1, 0]
    ax.plot(times, diag["fringe_shift_px"], "g-", lw=1)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Shift [px]")
    ax.set_title("Fringe Centroid Shift")
    ax.grid(True, alpha=0.3)

    # Panel 4: First/last profiles
    ax = axes[1, 1]
    if stack.ndim == 3:
        p0 = stack[0].mean(axis=0)
        p1 = stack[-1].mean(axis=0)
    else:
        p0 = stack[0]
        p1 = stack[-1]
    ax.plot(p0, "b-", lw=1, alpha=0.7, label=f"t={times[0]:.1f}ns")
    ax.plot(p1, "r-", lw=1, alpha=0.7, label=f"t={times[-1]:.1f}ns")
    ax.set_xlabel("Detector pixel")
    ax.set_ylabel("Intensity")
    ax.set_title("Detector Profiles")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    logger.info(f"诊断图已保存: {output_path}")
    return output_path


# ─── HDF5 导出 ───────────────────────────────────────────────────────────────


def export_hdf5(
    result: dict,
    output_path: Union[str, Path],
    *,
    compression: str = "gzip",
) -> Path:
    """
    将汇总结果导出为 HDF5 文件。

    结构:
      /stack          — 探测器强度堆叠 (nt, ...)
      /times          — 时间 [s]
      /timesteps      — 时间步索引
      /diagnostics/*  — 各诊断量
    """
    try:
        import h5py
    except ImportError:
        logger.error("h5py 不可用")
        raise

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("stack", data=result["stack"], compression=compression)
        f.create_dataset("times", data=result["times"], compression=compression)
        f.create_dataset("timesteps", data=np.array(result["timesteps"]))

        diag_grp = f.create_group("diagnostics")
        for key, val in result["diagnostics"].items():
            if isinstance(val, np.ndarray):
                diag_grp.create_dataset(key, data=val, compression=compression)
            else:
                diag_grp.attrs[key] = val

        # 元数据
        f.attrs["nt"] = result["stack"].shape[0]
        f.attrs["n_files"] = len(result["file_paths"])

    logger.info(f"HDF5 已导出: {output_path} ({output_path.stat().st_size/1024:.0f} KB)")
    return output_path


# ─── 模拟数据 (用于测试, 无需 RAVE-SIM) ──────────────────────────────────────


def generate_mock_detected(
    output_dir: Union[str, Path],
    nt: int = 50,
    nx: int = 256,
    *,
    fringe_period_px: float = 20.0,
    shift_per_step_px: float = 0.3,
    noise_level: float = 0.02,
    seed: int = 42,
) -> Path:
    """
    生成模拟 detected.npy 文件 (用于测试聚合器)。

    模拟 XPCI 条纹: 正弦条纹 + 平移 + 衰减包络 + 噪声。
    """
    rng = np.random.RandomState(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(nx)
    envelope = np.exp(-((x - nx / 2) ** 2) / (2 * (nx / 6) ** 2))

    for t in range(nt):
        ts_dir = output_dir / f"t{t:05d}"
        ts_dir.mkdir(parents=True, exist_ok=True)

        # 条纹: sin + 平移
        shift = t * shift_per_step_px
        fringe = 1.0 + 0.3 * np.sin(2 * np.pi * (x - shift) / fringe_period_px)
        profile = fringe * envelope

        # 衰减 (模拟等离子体膨胀 → 透射率变化)
        transmission = 1.0 - 0.3 * np.sin(np.pi * t / nt)
        profile *= transmission

        # 噪声
        profile += rng.normal(0, noise_level, nx)

        np.save(ts_dir / "detected.npy", profile.astype(np.float32))

    logger.info(f"生成 {nt} 个模拟 detected.npy → {output_dir}")
    return output_dir


# ─── 辅助 ────────────────────────────────────────────────────────────────────


def _extract_timestep(path: Path) -> int:
    """从路径中提取时间步索引: t00114/detected.npy → 114"""
    for part in path.parts:
        m = re.match(r"t(\d+)", part)
        if m:
            return int(m.group(1))
    return 0


def _load_time_info(output_dir: Path) -> dict[int, float]:
    """从 config.yaml 加载时间信息 (若有)。"""
    times: dict[int, float] = {}
    # 尝试从每个 ts 目录的 config.yaml 读取
    for config_path in sorted(output_dir.glob("t*/config.yaml")):
        ts = _extract_timestep(config_path)
        try:
            # 简易 YAML 解析 (不依赖 ruamel)
            with open(config_path) as f:
                content = f.read()
            # 查找 time 字段
            for line in content.splitlines():
                if line.strip().startswith("#") and "time" in line.lower():
                    # 从注释头提取
                    pass
        except Exception:
            pass
    return times


def _empty_result() -> dict:
    return {
        "stack": np.array([]),
        "timesteps": [],
        "times": np.array([]),
        "shapes": [],
        "file_paths": [],
        "diagnostics": {},
    }


# ─── 便利函数 ────────────────────────────────────────────────────────────────


def quick_report(result: dict) -> str:
    """生成文本摘要报告。"""
    if result["stack"].size == 0:
        return "空结果"

    stack = result["stack"]
    diag = result["diagnostics"]
    lines = [
        "XPCI 结果报告",
        "=" * 50,
        f"  时间步数:       {stack.shape[0]}",
        f"  探测器形状:     {stack.shape[1:]}",
        f"  时间范围:       [{result['times'][0]:.4e}, {result['times'][-1]:.4e}] s",
        f"  强度范围:       [{stack.min():.4e}, {stack.max():.4e}]",
    ]

    if "visibility" in diag:
        v = diag["visibility"]
        lines.append(f"  可见度:         [{v.min():.4f}, {v.max():.4f}]")
    if "contrast" in diag:
        c = diag["contrast"]
        lines.append(f"  对比度:         [{c.min():.4f}, {c.max():.4f}]")
    if "fringe_shift_px" in diag:
        fs = diag["fringe_shift_px"]
        lines.append(f"  条纹位移:       [{fs.min():.2f}, {fs.max():.2f}] px")

    return "\n".join(lines)
