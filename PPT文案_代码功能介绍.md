# PPT 文案：RAVE-SIM — 1D/2D X 射线波传播模拟框架

---

## 第 1 页：封面

**标题：** RAVE-SIM — X 射线波传播模拟框架

**副标题：** 一维/二维波场仿真能力详解

**标注：** ETH Zurich | Optics Express (DOI: 10.1364/OE.543500)

---

## 第 2 页：项目概述

**目标：** 模拟相干 X 射线穿过光栅/样品后，在自由空间中传播并到达探测器的完整物理过程。

**物理模型：** 菲涅耳衍射（傍轴近似）+ 物质相互作用

**两套仿真引擎：**

| 引擎 | 实现语言 | 核心特性 |
|------|----------|----------|
| **big-wave** | Python + Rust | 核外计算（磁盘存储），支持任意大规模 1D 波场 |
| **fast-wave** | C++ + CUDA | GPU 加速，支持 1D 和 2D 波场 |

**两套仿真维度：**

| 维度 | 波场 | FFT | 探测器 | 样品 | 状态 |
|------|------|-----|--------|------|------|
| **1D** | `u(x)` | 1D FFT | 线探测器 | 2D (z × x) | 成熟，完整测试 |
| **2D** | `u(x, y)` | 2D FFT | 面探测器 | 3D (z × y × x) | 开发中，fast-wave 独占 |

---

## 第 3 页：1D 仿真原理

**数学基础：一维菲涅耳衍射**

波函数 `u(x, z)` 满足一维傍轴波动方程，在自由空间中传播距离 dz：

```
u(x, z+dz) = IFFT[ FFT[u(x,z)] × H(f_x) ]
```

传递函数：
```
H(f_x) = exp(-2π·i·dz/λ) × exp(π·i·λ·dz·f_x²)
```

**一维波场的物理含义**
- 波场 `u(x)` 表示沿 x 方向的复振幅分布
- 假设在 y 方向均匀（线光栅/线样品的标准假设）
- 所有光学元件（光栅、样品）在 y 方向无限延伸
- 适用于 X 射线光栅干涉仪的标准建模

**球面波初始化（点源）**
```
u(x) = exp(-2π·i·r/λ) / √r,  r = √(x² + z²)
```
- 1/√r 衰减对应于二维空间中的强度线性衰减

---

## 第 4 页：1D 仿真 — 软件架构

```
配置文件 (config.yaml)
     │
     ▼
multisim.setup_simulation()
     │ 生成仿真目录，每个源一个子目录
     │
     ├── big-wave (Python + Rust)
     │     - DiskVector：波场存储在 .npy 文件中
     │     - NumpyVector：波场在内存中（小规模/测试）
     │     - 核外 FFT：四步法，磁盘中转
     │     - 分块处理：逐个 chunk 读/写/修改
     │
     └── fast-wave (C++ / CUDA)
           - 波场驻留在 GPU 显存
           - cuFFT 加速 FFT
           - 全部核函数在 GPU 上运行
           - 是 big-wave 物理模型的精确 GPU 移植
```

**两套引擎等效验证**
- NumpyVector ↔ DiskVector：同一 `Vector` 协议，可互换验证
- big-wave ↔ fast-wave：逐像素对比，确保物理一致性
- 共享同一 YAML 配置格式和目录结构

---

## 第 5 页：1D 仿真 — 数据流与算法

**单源仿真流程（wavesim.py / simulation.cpp）**

```
   Input: 源参数 + 光学元件列表 + 截止角列表
   
   ① 从点源生成初始球面波 u(x)
   ② 1D FFT → 频域 U(f_x)
   ③ 乘 Fresnel 传递函数 + 频率截止
   ④ 1D IFFT → 实空间 u(x)
       ┌───────────────────────────────────┐
   ⑤  │ 对每个光学元件：                   │
   ⑥  │   - 传播到元件起始位置              │
   ⑦  │   - 应用元件（光栅/样品）          │
   ⑧  │   - 传播穿过元件厚度                │
       └───────────────────────────────────┘
   ⑨ 传播到探测器
   ⑩ square_and_downsample → 探测器输出
   
   Output: detected.npy = 一维强度分布
```

**探测器输出**
- `detected.npy` 形状：`(nr_phase_steps, nr_pixels)`
- `nr_pixels = detector_size / detector_pixel_size_x`
- y 方向通过 `detector_pixel_size_y` 因子解析积分

---

## 第 6 页：1D 仿真 — 光学元件

**光栅（Grating）**
- 一维周期结构，在 x 方向交替排列材料 A/B
- 参数：周期、占空比（可线性渐变）、厚度、台阶数
- 每步生成布尔掩膜 `mask(x)` → 分别乘以材料因子
- 相步进：沿 x 方向平移光栅

**包络光栅（EnvGrating）**
- 两个周期 `pitch0 × pitch1` 的乘积
- 用于更复杂的干涉仪配置（如 Talbot-Lau）

**样品（Sample）**
- 2D 像素网格 `grid[z_row, x_col]` = 材料索引
- 逐行应用：将行材料因子插值到仿真网格 → 乘到波场上 → 传播
- `precise_Sample`：额外密度网格，运行时计算 δ+β

**标记元件（SaveAndExit）**
- 保存当前波场并停止，支持分阶段仿真

---

## 第 7 页：1D 仿真 — 频率截止优化

**核心问题：** 波场在传播中会自然展宽。仿真网格太大则计算量过高，太小则截断有效信号。

**解决方案：动态计算每段的最优截止角**

```
源 → [段1] → 元件1 → [段2] → 元件2 → ... → 探测器

段1 截止角 = min(arctan((detector/2 + max_x) / z_total), 能量限制角)
段2 截止角 = min(arctan((detector/2 + 段1展宽) / 剩余距离), 能量限制角)
...
```

**效果**
- 早期段保持窄角，网格小
- 随传播逐段放宽，确保所有可达探测器的射线被保留
- 大幅减少无效计算

**代码入口：** `propagation.py::compute_cutoff_angles()`
**核函数：** `propagate_convolve_step_kernel`（传播 + 截止一步完成）

---

## 第 8 页：1D 仿真的局限

**一维近似的假设**
- 波场在 y 方向均匀 → 无法模拟 y 方向有结构的样品
- 探测器沿 x 方向一维排列 → 无法获取二维图像
- 光学元件在 y 方向无限延伸 → 无法模拟有限尺寸元件

**何时适用**
- 线光栅干涉仪（Talbot-Lau、Talbot 干涉仪）
- 样品在 y 方向变化缓慢或均匀
- 仅关注 x 方向相位衬度
- 快速参数扫描

**何时需要升级到 2D**
- 样品在 x 和 y 方向都有精细结构
- 需要二维投影/CT 重建
- 探测器输出应为二维图像

---

## 第 9 页：2D 仿真原理

**二维菲涅耳衍射**

波函数 `u(x, y, z)`，传播距离 dz：

```
u(x,y,z+dz) = IFFT2D[ FFT2D[u(x,y,z)] × H(f_x, f_y) ]
```

二维传递函数：
```
H(f_x, f_y) = exp(-2π·i·dz/λ) × exp(π·i·λ·dz·(f_x² + f_y²))
```

**与 1D 的关键区别**

| 特性 | 1D | 2D |
|------|-----|-----|
| 波场 | `u(x)` 复数向量 | `u(x, y)` 复数矩阵 |
| 网格 | N 个点 | nx × ny 个点 |
| FFT | 1D FFT | 2D FFT（cuFFT 原生支持） |
| 源位置 | `(x, z)` | `(x, y, z)` |
| 样品 | 2D (z×x) | 3D (z×y×x) |
| 探测器 | 线探测器 | 面探测器 |
| 相步进 | x 方向平移 | x 和 y 方向平移 |

**实现位置：** `fast-wave` 独占，`kernels.cu` 和 `simulation.cpp` 中标记为 "3d"

---

## 第 10 页：2D 仿真 — 软件架构

**配置文件扩展**

```yaml
# config.yaml (2D 模式)
sim_params:
  nx: 4096        # x 方向网格点数
  ny: 4096        # y 方向网格点数
  dx: 1.0e-6      # x 方向网格间距 (m)
  dy: 1.0e-6      # y 方向网格间距 (m)
  detector_size_x: 0.05    # 探测器 x 方向宽度 (m)
  detector_size_y: 0.05    # 探测器 y 方向宽度 (m)
  is2d: true               # 启用 2D 模式
  
source:
  ...
  y: 0.0          # 源在 y 方向的位置
```

**新增/修改的 2D 核函数**

| 核函数 | 功能 |
|--------|------|
| `propagate_analytically_2d` | 二维球面波初始化 |
| `propagate_convolve_step_2d` | 二维 Fresnel 传播 + 截止 |
| `apply_sample_factors_2d` | 二维样品因子插值与应用 |
| `square_and_downsample_2d` | 二维探测器下采样 |

**2D FFT**
- 使用 cuFFT 原生 2D FFT 接口
- `FFT<S>(nx, ny)` 构造函数创建 2D 变换计划

---

## 第 11 页：2D 仿真 — 样品与探测器

**三维样品模型**

样品网格维度：`[z_layers, y_rows, x_cols]`（代码中表示为 `z_len × y_len × x_len`）

```
         ┌──────────────────────────────┐
         │         x (列)               │
         │   ┌───┬───┬───┬───┬───┐      │
   y(行)  │   │M12│M22│M32│M42│M52│      │
         │   ├───┼───┼───┼───┼───┤      │
         │   │M11│M21│M31│M41│M51│      │
         │   └───┴───┴───┴───┴───┘      │
         └──────────────────────────────┘
                     
    每层 z_slice (y × x 平面) 由波场逐层穿透
```

**探测器输出**
- `detected.npy` 形状：`(nr_phase_steps, nr_pixels_y, nr_pixels_x)`
- 二维面探测器，记录每个像素的强度
- 支持相步进：在 x 和/或 y 方向平移样品

**样品参数扩展**
- `x_positions`：x 方向相步进偏移
- `y_positions`：y 方向相步进偏移（2D 新增）
- `pixel_size_x` / `pixel_size_y`：像素横向尺寸

---

## 第 12 页：1D vs 2D 详细对比

| 方面 | 1D 仿真 | 2D 仿真 |
|------|---------|---------|
| **波场表示** | `complex[N]` 一维数组 | `complex[ny][nx]` 二维矩阵 |
| **网格参数** | N, dx | nx, ny, dx, dy |
| **总点数** | N | nx × ny |
| **源** | PointSource(x, z), VectorSource | PointSource(x, y, z) |
| **初始化** | `exp(-2πir/λ)/√r`, r=√(x²+z²) | `exp(-2πir/λ)/r`, r=√(x²+y²+z²) |
| **FFT** | 1D (rustfft / cuFFT 1D) | 2D (cuFFT 2D) |
| **光栅** | Grating, EnvGrating | ❌ 暂不支持 |
| **样品** | Sample (2D grid), precise_Sample | Sample (3D grid) |
| **precise_Sample** | ✅ 支持 | ❌ 暂不支持 |
| **探测器形状** | 一维线阵 | 二维面阵 |
| **探测器输出** | `(steps, nr_pixels)` | `(steps, ny, nx)` |
| **相步进维度** | x 方向 | x 和 y 方向 |
| **核外 (big-wave)** | ✅ 完整支持 | ❌ 不支持 |
| **GPU (fast-wave)** | ✅ 完整支持 | ✅ 支持 (开发中) |
| **可视化** | 一维曲线+历史(时空图) | 二维图像+历史(视频) |

---

## 第 13 页：2D 仿真 — 当前状态与限制

**已完成**
- 二维 Fresnel 传播（`propagate_convolve_step_2d`）
- 二维解析球面波初始化（`propagate_analytically_2d`）
- 三维 Sample 逐层应用（`apply_sample_factors_2d`）
- 二维探测器下采样（`square_and_downsample_2d`）
- 多角度相步进（x + y 方向）
- GPU 2D FFT（cuFFT 原生支持）
- 保存中间波场用于调试

**限制与待完成**
- ❌ 不支持光栅（Grating / EnvGrating）— 仅支持 Sample
- ❌ 不支持 precise_Sample
- ❌ 无核外模式（仅 GPU）
- ❌ 不支持 VectorSource（从文件恢复）
- ⚠️ 含大量调试输出和注释代码（清理中）
- ⚠️ 历史记录功能被注释（`propagate_with_history_2d`）
- ⚠️ 未经过大规模验证测试

---

## 第 14 页：使用场景与选型指南

**选择 1D 仿真**
```
场景                    → 推荐配置
────────────────────────────────────────────────
光栅干涉仪参数扫描       → big-wave 1D, 核外模式
Talbot 效应验证           → fast-wave 1D / big-wave 1D
样品 x 方向结构分析       → big-wave 1D + Sample
大批量源点统计            → multisim + big-wave 1D
分阶段仿真（多分辨率）    → 1D + SaveAndExit + VectorSource
内存受限的超大规模网格    → big-wave 1D DiskVector
```

**选择 2D 仿真**
```
场景                    → 推荐配置
────────────────────────────────────────────────
三维样品的二维投影        → fast-wave 2D + 3D Sample
CT 重建的数据模拟        → fast-wave 2D 多角度
二维探测器效果评估        → fast-wave 2D
样品具有 x/y 方向结构    → fast-wave 2D + 3D Sample
```

**两步法示例**
```
Step 1: 1D 大范围粗扫 → 确定感兴趣区域
Step 2: 加载保存的波前 → 2D 精细模拟
```

---

## 第 15 页：验证与测试

**1D 仿真验证**（充分）
- 单元测试覆盖：`big-wave/test.py`
  - 核外 FFT 正确性
  - 频率截止函数
  - 探测器下采样
  - 光栅掩膜生成 + 相步进
  - 多源仿真完整流程
  - 回归测试
- NumpyVector ↔ DiskVector 一致性
- big-wave ↔ fast-wave 逐像素对比

**2D 仿真验证**（进行中）
- 中间波场保存：`wave_after_source.npy`、`wave_after_sample.npy`、`wave_at_detector.npy`
- 逐层 CUDA 错误检查
- GitHub Issues 跟踪：`#20`、`#26`、`#29`（样品坐标系问题）
- 当前主要验证方法：可视化检查

---

## 第 16 页：总结与展望

**核心能力**
- **1D 仿真**：成熟、完整、经过验证，支持核外和 GPU 两种模式
- **2D 仿真**：正在开发中，实现了核心物理模型，支持三维样品和二维探测器

**代码规模**
| 组件 | 语言 | 核心代码 | 状态 |
|------|------|----------|------|
| big-wave | Python | ~1500 行 | 稳定 |
| big-fourier | Rust | ~400 行 | 稳定 |
| fast-wave 1D | C++/CUDA | ~500 行 | 稳定 |
| fast-wave 2D | C++/CUDA | ~500 行 | 开发中 |

**路线图**
1. 完成 2D 光栅支持
2. 清理 2D 调试代码
3. 添加 2D 单元测试
4. 支持 precise_Sample 的 2D 模式
5. 性能优化与大规模验证
