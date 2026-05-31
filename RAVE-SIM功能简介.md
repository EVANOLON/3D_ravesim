# RAVE-SIM 功能简介

RAVE-SIM（Really big/fast wAVE SIMulation）是 ETH Zurich 开发的 X 射线波传播模拟框架，相关论文发表于 Optics Express（DOI: 10.1364/OE.543500）。该框架用于模拟相干 X 射线从点源发出，穿过光栅、样品等光学元件，在自由空间中传播，最终到达探测器的完整物理过程。

---

## 物理模型

波函数在自由空间中的传播基于**菲涅耳衍射（傍轴近似）**：

```
u(x, z+dz) = IFFT[ FFT[u(x,z)] × H(f) ]
H(f) = exp(-2π·i·dz/λ) × exp(π·i·λ·dz·f²)
```

当波穿过材料时，与物质的相互作用表示为：

```
u_out = u_in × exp(2π·i·t/λ · (δ + iβ))
```

其中 δ 是折射率衰减（相位偏移），β 是吸收指数，两者通过 NIST 数据库查询获得。

从点源出发的初始球面波由下式生成：

```
u(x) = exp(-2π·i·r/λ) / √r ,  r = √(x² + z²)
```

采用 exp(+i·ω·t) 时域符号约定，FFT 为正变换。

---

## 两套仿真引擎

### 1. big-wave（Python + Rust，核外计算引擎）

核心思想是将波向量存储为磁盘上的 .npy 文件，通过分块读写的方式处理任意大规模的数据，突破内存限制。

**核心抽象：Vector 协议**

| 方法 | 功能 |
|------|------|
| `fft()` / `ifft()` | 正向/逆傅里叶变换 |
| `write_chunked()` | 分块写入 |
| `modify_chunked()` | 分块读取-修改-写入 |
| `read_chunked()` | 分块只读遍历 |

**两种实现：**
- **NumpyVector**：波场在内存中，适用于小规模仿真和测试验证
- **DiskVector**：波场在磁盘 .npy 文件中，适用于大规模仿真

核外 FFT 由 Rust 编写的 big-fourier 库提供，采用四步法（Four-step FFT）将大规模 FFT 分解为多次小规模 FFT，整个过程仅需一个分块缓冲区。

### 2. fast-wave（C++ / CUDA，GPU 加速引擎）

big-wave 物理模型的精确 GPU 移植，所有计算驻留在 GPU 显存，无磁盘 I/O。

**技术栈：** C++17 + CUDA + cuFFT + yaml-cpp

**GPU 核函数（fwcuda/kernels.cu）：**

| 核函数 | 功能 |
|--------|------|
| `propagate_analytically_kernel` | 球面波初始化（内部双精度） |
| `propagate_convolve_step_kernel` | Fresnel 传播 + 频率截止 |
| `apply_grating_factors_kernel` | 光栅掩膜生成与材料因子应用 |
| `apply_env_grating_factors_kernel` | 双周期包络光栅 |
| `apply_sample_factors_kernel` | 样品材料因子插值与应用 |
| `scale_kernel` | 复标量乘法 |
| `square_and_downsample_kernel` | 探测器下采样 |
| `analytical_history_row_kernel` | 传播历史生成 |

两套引擎共享同一 YAML 配置格式和仿真目录结构，通过 multisim.setup_simulation() 生成。

---

## 一维（1D）仿真能力

### 基本原理

波场是一维复向量 `u(x)`，假设在 y 方向均匀。这是标准 X 射线光栅干涉仪建模方法，适用于线光栅和线样品。

### 仿真流程

1. 从点源生成初始球面波 `u(x)`
2. FFT 变换到频域 `U(f_x)`
3. 乘 Fresnel 传递函数并应用频率截止
4. IFFT 变换回实空间
5. 依次处理每个光学元件（传播到元件 → 应用元件 → 传播穿过元件）
6. 传播到探测器，通过 square_and_downsample 计算探测器输出

### 支持的光学元件

**Grating（光栅）：** 在 x 方向交替排列材料 A/B 的周期结构。参数包括周期、占空比（可沿 z 线性渐变）、厚度、台阶数。每步生成布尔掩膜并乘以对应材料因子。支持相步进（沿 x 方向平移光栅）。

**EnvGrating（包络光栅）：** 两个周期 pitch0 和 pitch1 的乘积，用于更复杂的干涉仪配置。

**Sample（样品）：** 2D 像素网格 `grid[z_row, x_col]`，每个像素存储材料索引。按 z 方向逐行处理：将行材料因子插值到仿真网格，乘到波场上，传播该行厚度。

**precise_Sample（精确样品）：** 在 Sample 基础上增加独立密度网格，运行时通过 NIST 库逐像素计算 δ+β，实现更精确的密度建模。

**SaveAndExit（标记元件）：** 在指定 z 位置保存当前波场并停止仿真，支持分阶段仿真。

### 1D 探测器输出

输出形状为 `(nr_phase_steps, nr_pixels)` 的一维强度分布，`detector_pixel_size_y` 作为解析积分因子。

### 频率截止优化

仿真动态计算每段的最优截止角：早期段保持窄角减小计算域，随传播逐段放宽确保所有可达探测器的射线被保留。算法入口为 `compute_cutoff_angles()`。

### 1D 仿真的适用场景

适用于线光栅干涉仪（Talbot-Lau、Talbot 干涉仪）、样品在 y 方向均匀或变化缓慢的场景、快速参数扫描等。受限于一维近似，无法模拟 y 方向有结构的样品，也无法输出二维图像。

---

## 二维（2D）仿真能力

### 基本原理

波场是二维复矩阵 `u(x, y)`，使用 2D FFT 进行传播：

```
u(x,y,z+dz) = IFFT2D[ FFT2D[u(x,y,z)] × H(f_x, f_y) ]
H(f_x, f_y) = exp(-2π·i·dz/λ) × exp(π·i·λ·dz·(f_x² + f_y²))
```

### 当前实现状态（开发中，fast-wave 独占）

在 `fast-wave` 中以 "3d" 为标记进行开发，位于 `simulation.cpp` 和 `kernels.cu` 中。

**配置扩展：**

```yaml
sim_params:
  nx: 4096        # x 方向网格点数
  ny: 4096        # y 方向网格点数
  dx: 1.0e-6      # x 方向间距 (m)
  dy: 1.0e-6      # y 方向间距 (m)
  detector_size_x: 0.05
  detector_size_y: 0.05
  is2d: true
```

**已实现的功能：**
- 二维 Fresnel 传播（propagate_convolve_step_2d）
- 二维球面波初始化，支持源的 x 和 y 位置（propagate_analytically_2d）
- 三维 Sample 逐层应用：`grid[z_layers, y_rows, x_cols]`（apply_sample_factors_2d）
- 二维面探测器下采样（square_and_downsample_2d）
- 多角度相步进：x 和 y 方向平移
- cuFFT 原生 2D FFT 支持

**探测器输出形状：** `(nr_phase_steps, nr_pixels_y, nr_pixels_x)`

### 当前的限制

- 不支持 Grating / EnvGrating，仅 Sample 可工作
- 不支持 precise_Sample
- 无核外模式（仅 GPU 显存驻留）
- 不支持 VectorSource（从文件恢复波前）
- 含大量调试输出和注释代码
- 历史记录功能未完全启用
- 未经过大规模验证测试

### 2D 仿真的适用场景

适用于三维样品的二维投影模拟、CT 重建的数据准备、样品在 x 和 y 方向均有精细结构的场景。

---

## 1D 与 2D 仿真对比

| 特性 | 1D 仿真 | 2D 仿真 |
|------|---------|---------|
| 波场 | `u(x)` 复向量 | `u(x,y)` 复矩阵 |
| 网格 | N 个点 | nx × ny 个点 |
| FFT | 1D | 2D |
| 源位置 | (x, z) | (x, y, z) |
| 光栅 | Grating + EnvGrating | 暂不支持 |
| 样品网格 | 2D (z×x) | 3D (z×y×x) |
| 精确样品 | precise_Sample | 暂不支持 |
| 探测器 | 线探测器 | 面探测器 |
| 输出形状 | (steps, nr_pixels) | (steps, ny, nx) |
| 核外(big-wave) | 完整支持 | 不支持 |
| GPU(fast-wave) | 完整支持 | 开发中 |
| 状态 | 成熟，测试充分 | 开发中 |

---

## 多源仿真

`multisim.py` 支持两种多源模式：

- **points 模式**：在指定 x 范围和能量范围内随机生成多个点源，每个源独立运行仿真。可配置光源光谱分布，能量按光谱概率密度采样。
- **vectors 模式**：基于之前保存的波向量继续仿真，支持分阶段、多分辨率策略：先用大范围粗网格跑基础仿真，保存波前，再加载到精细网格继续模拟。

---

## 验证与测试

1D 仿真有完整的单元测试覆盖（big-wave/test.py），包括核外 FFT 正确性验证，频率截止函数，探测器下采样，光栅掩膜生成与相步进，多源仿真完整流程（setup、run、keypoints），DiskVector 初始化和修改，以及回归测试。

NumpyVector 与 DiskVector 实现同一 Vector 协议，可互换使用进行验证。big-wave 与 fast-wave 的结果经过逐像素对比验证，确保物理一致性。

2D 仿真目前通过保存中间波场（wave_after_source、wave_after_sample、wave_at_detector）进行逐段验证，完整的测试套件正在开发中。
