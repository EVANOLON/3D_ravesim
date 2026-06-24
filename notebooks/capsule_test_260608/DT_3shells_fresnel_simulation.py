"""
DT 靶丸菲涅耳衍射成像仿真 — 三组静态制造缺陷对比

依次对三种典型 ICF 靶丸制造缺陷进行菲涅耳衍射成像仿真：
  1. 偶极偏心率 (l=1, m=0, ε=0.02) — 靶丸沿 z 轴平移偏移 ~4.9 µm
  2. P₂ 椭球度   (l=2, m=0, ε=0.02) — 极向拉伸 1%，赤道收缩 0.6%
  3. 壁厚不均匀  (l=1, ε_r=0.02, ε_t=0.005) — 内腔偏移 > 外壳偏移

用法：
  python DT_3shells_fresnel_simulation.py          # 完整运行（设置+仿真+出图）
  python DT_3shells_fresnel_simulation.py --nosim   # 仅设置+出图（跳过仿真，用于已有结果时重新绘图）
"""

import sys
import copy
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ─── 路径 ──────────────────────────────────────────────────────────────
RAVE_SIM_DIR = Path('/mnt/d/rave-sim-main/rave-sim-main')
SIMULATIONS_DIR = RAVE_SIM_DIR / 'output'
GRID_DIR = RAVE_SIM_DIR / 'grid'

sys.path.insert(0, str(RAVE_SIM_DIR / "big-wave"))
import multisim
import config
import util

# ─── 三组靶丸配置 ─────────────────────────────────────────────────────
SHELL_CONFIGS = [
    {
        "label": "l1_m0_eps002",
        "grid_path": str(GRID_DIR / "500um_100um_shell_l1_m0_eps002.npy"),
        "desc": "偶极偏心 (l=1, m=0, eps=0.02) — 靶丸沿 z 轴平移偏移 ~4.9um",
    },
    {
        "label": "l2_m0_eps002",
        "grid_path": str(GRID_DIR / "500um_100um_shell_l2_m0_eps002.npy"),
        "desc": "P2 椭球 (l=2, m=0, eps=0.02) — 极向拉伸 1%，赤道收缩 0.6%",
    },
    {
        "label": "l1_varT_er002_et005",
        "grid_path": str(GRID_DIR / "500um_100um_shell_l1_varT_er002_et005.npy"),
        "desc": "壁厚不均匀 (l=1, eps_r=0.02, eps_t=0.005) — 内腔偏移 > 外壳偏移",
    },
]

BASE_CONFIG = {
    "sim_params": {
        "is2d": 'true',
        'use_fresnel_scaling': 'true',
        "N": 16384 * 16384,
        "nx": 16384,
        "ny": 16384,
        "dx": 8.5e-8,
        "z_detector": 4.0,
        "detector_size": 0.85e-3,
        "detector_size_y": 0.85e-3,
        "detector_pixel_size_x": 2e-7,
        "detector_pixel_size_y": 2e-7,
        "chunk_size": 2 * 1024 * 1024 * 1024 // 16,
    },
    "use_disk_vector": False,
    "save_final_u_vectors": False,
    "dtype": "c8",
    "multisource": {
        "type": "points",
        "energy_range": [1, 100001],
        "x_range": [-1e-6, 1e-6],
        "y_range": [-1e-6, 1e-6],
        "z": 0.0,
        "nr_source_points": 1,
        "seed": 1,
        "spectrum": "/mnt/d/rave-sim-main/rave-sim-main/spectrum/spectrum_microX.h5",
    },
    "elements": [
        {
            "type": "sample",
            "z_start": 1.0,
            "pixel_size_x": 5e-8,
            "pixel_size_y": 5e-8,
            "pixel_size_z": 5e-8,
            "grid_path": None,
            "materials": [["H", 0.255]],
            "x_positions": [0],
            "y_positions": [0],
        },
    ],
}

# ─── 中文标签（用于出图） ─────────────────────────────────────────────
LABELS_ZH = ["偶极偏心 (l=1,m=0)", "P2 椭球 (l=2,m=0)", "壁厚不均匀 (l=1,varT)"]


# ══════════════════════════════════════════════════════════════════════
#  阶段一：设置三组仿真
# ══════════════════════════════════════════════════════════════════════
def setup_simulations() -> dict:
    """为三组靶丸分别创建仿真目录，返回 {label: sim_path}"""
    sim_paths = {}
    print("=" * 60)
    print("阶段一：设置三组仿真")
    print("=" * 60)

    for i, sc in enumerate(SHELL_CONFIGS):
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["elements"][0]["grid_path"] = sc["grid_path"]
        print(f"\n[{i + 1}/{len(SHELL_CONFIGS)}] {sc['label']}")
        print(f"  {sc['desc']}")
        sim_path = multisim.setup_simulation(cfg, Path("."), SIMULATIONS_DIR)
        sim_paths[sc["label"]] = sim_path
        print(f"  -> {sim_path}")

    print(f"\n全部 {len(sim_paths)} 组仿真已设置完毕")
    return sim_paths


# ══════════════════════════════════════════════════════════════════════
#  阶段二：运行仿真（GPU）
# ══════════════════════════════════════════════════════════════════════
def run_simulations(sim_paths: dict):
    """依次运行三组仿真，每组约 10 分钟"""
    print("\n" + "=" * 60)
    print("阶段二：运行三组仿真（每组 ~10 分钟）")
    print("=" * 60)

    for i, sc in enumerate(SHELL_CONFIGS):
        label = sc["label"]
        sim_path = sim_paths[label]
        print(f"\n[{i + 1}/{len(SHELL_CONFIGS)}] 运行 {label}: {sc['desc']}")
        nr = int(BASE_CONFIG["multisource"]["nr_source_points"])
        for j in range(nr):
            print(f"  源点 {j + 1}/{nr} ...")
            ret = os.system(
                f"CUDA_VISIBLE_DEVICES=0 "
                f"{RAVE_SIM_DIR / 'fast-wave' / 'build-Release' / 'fastwave'} "
                f"-s {j} {sim_path}"
            )
            if ret != 0:
                print(f"  ⚠ 源点 {j} 返回非零 {ret}")
        print(f"  [{label}] 完成")


# ══════════════════════════════════════════════════════════════════════
#  阶段三：加载结果
# ══════════════════════════════════════════════════════════════════════
def load_results(sim_paths: dict) -> dict:
    """加载三组仿真的波前结果，返回 {label: wavefront_2d}"""
    print("\n" + "=" * 60)
    print("阶段三：加载波前结果")
    print("=" * 60)

    results = {}
    for i, sc in enumerate(SHELL_CONFIGS):
        label = sc["label"]
        sim_path = sim_paths[label]
        print(f"\n[{i + 1}/{len(SHELL_CONFIGS)}] {label}")
        wavefronts = util.load_wavefronts_filtered(sim_path, x_range=(-30e-6, 30e-6))
        print(f"  加载源点数: {len(wavefronts)}")
        wavef = [r[0] for r in wavefronts]
        wf = np.sum(wavef, axis=0)
        print(f"  波前形状: {wf.shape}")
        results[label] = wf

    return results


# ══════════════════════════════════════════════════════════════════════
#  阶段四：出图对比
# ══════════════════════════════════════════════════════════════════════
def plot_results(results: dict):
    """输出三组对比图：2D 衍射图 / 线剖面 / 差异图"""
    print("\n" + "=" * 60)
    print("阶段四：出图对比")
    print("=" * 60)

    sp = BASE_CONFIG["sim_params"]
    detector_x = util.detector_x_vector(sp["detector_size"], sp["detector_pixel_size_x"])
    detector_y = util.detector_y_vector(sp["detector_size_y"], sp["detector_pixel_size_y"])
    extent = [detector_x[0] * 1e3, detector_x[-1] * 1e3,
              detector_y[0] * 1e3, detector_y[-1] * 1e3]

    # ─── 1. 三张并排 2D 衍射图 ────────────────────────────
    print("  图1: 三张并排衍射图 ...")
    wf_list = list(results.values())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, (label, wf) in enumerate(results.items()):
        im = axes[idx].imshow(abs(wf[0]) ** 2, extent=extent,
                               cmap='rainbow', origin='lower')
        axes[idx].set_title(LABELS_ZH[idx], fontsize=12)
        axes[idx].set_xlabel('x (mm)')
        axes[idx].set_ylabel('y (mm)')
        plt.colorbar(im, ax=axes[idx], fraction=0.046)
    plt.suptitle("DT 靶丸菲涅耳衍射成像 — 静态制造缺陷对比 (Fresnel On)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path_2d = SIMULATIONS_DIR / "comparison_3shells_2d.png"
    plt.savefig(path_2d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {path_2d}")

    # ─── 2. 中心行 lineout 对比 ────────────────────────────
    print("  图2: 中心行线剖面对比 ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (label, wf) in enumerate(results.items()):
        center_y = wf.shape[1] // 2
        lineout = abs(wf[0, center_y, :]) ** 2
        ax.plot(detector_x * 1e3, lineout, label=LABELS_ZH[idx], lw=1.5)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title('中心行线剖面对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path_lineout = SIMULATIONS_DIR / "comparison_3shells_lineout.png"
    plt.savefig(path_lineout, dpi=150)
    plt.close()
    print(f"  已保存: {path_lineout}")

    # ─── 3. 两两差异图 ──────────────────────────────────────
    print("  图3: 两两差异图 ...")
    labels = [sc["label"] for sc in SHELL_CONFIGS]
    pairs = [
        (labels[0], labels[1]),
        (labels[0], labels[2]),
        (labels[1], labels[2]),
    ]
    pair_titles = [
        "(偶极偏心) - (P2椭球)",
        "(偶极偏心) - (壁厚不均)",
        "(P2椭球) - (壁厚不均)",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (a, b), title in zip(axes, pairs, pair_titles):
        diff = abs(results[a][0]) ** 2 - abs(results[b][0]) ** 2
        im = ax.imshow(diff, extent=extent, cmap='RdBu', origin='lower')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        plt.colorbar(im, ax=ax, fraction=0.046, label='DI')
    plt.suptitle("两两差异图", fontsize=14)
    plt.tight_layout()
    path_diff = SIMULATIONS_DIR / "comparison_3shells_diff.png"
    plt.savefig(path_diff, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {path_diff}")

    print("\n全部对比图已生成！")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="DT 靶丸菲涅耳衍射成像 — 三组静态制造缺陷对比"
    )
    parser.add_argument("--nosim", action="store_true",
                        help="跳过仿真运行阶段（仅设置 + 出图，用于已有结果时重新绘图）")
    args = parser.parse_args()

    sim_paths = setup_simulations()

    if not args.nosim:
        run_simulations(sim_paths)
    else:
        print("\n(--nosim) 跳过仿真运行阶段")

    results = load_results(sim_paths)
    plot_results(results)

    print("\n全部完成！")


if __name__ == "__main__":
    main()
