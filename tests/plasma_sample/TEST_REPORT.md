# PlasmaSample 相位与吸收验证测试报告

**日期**: 2026-08-06  
**测试文件**: `tests/plasma_sample/test_plasma_phase_absorption.py`  
**诊断文件**: `tests/plasma_sample/diagnose_phase_error.py`, `tests/plasma_sample/quantify_phase_error.py`, `tests/plasma_sample/demo_leakage.py`

---

## 1. 背景

`PlasmaSample` 是 big-wave 中的等离子体光学元件，其折射率

```
n = 1 - δ - iβ
```

由四个物理量计算：

| 分量 | 来源 | 公式 |
|------|------|------|
| δ_free | 自由电子等离子体色散 | `n_e · r_e · λ² / (2π)` — 解析精确 |
| δ_bound | 束缚电子 (Chantler 标度) | `n_i · prefactor · f1 · (Z-Z*)/Z` |
| β_bound | 光致吸收 (Chantler 标度) | `n_i · prefactor · f2 · (Z-Z*)/Z · (μ_total/μ_photo)` |
| β_ff | Kramers 逆轫致辐射 | `3.7e8 · n_e · n_i · Z*² · g_ff / (√T_K · ν³)` |

已有的 `plasma_2d_test.ipynb` 只能定性展示探测器的 Fresnel 衍射图案，无法分别验证相位和吸收的准确性。本测试套件填补了这一空白。

## 2. 测试设计

### 2.1 方法论

参照 `3D_test_notebooks` 的设计模式：

| 模式 | 来源 | 本测试对应 |
|------|------|-----------|
| 薄板厚度扫描 + 线性拟合 | `C_thin_sample_phase.ipynb` | Test A |
| 空场参考 + Beer-Lambert 定律 | `W_thin_sample_absorption.ipynb` | Test B |
| 材料对比交叉验证 | `C_capsule3d / DT_capsule3d / W_capsule3d` | Test C |

### 2.2 核心创新：同位置对消法

最初的真空参考法测量 `T_plasma(centre) / T_plasma(vacuum)`，假设真空区和中心经历了相同的 Fresnel 传播相位。经过系统诊断（§4），发现球面波前曲率导致 `T_empty(x)` 随位置变化，真空参考法引入约 5% 的系统相位误差。

**金标准方法**：对同一个 `u_before` 分别运行等离子体传播和空传播，在同一空间位置取比值：

```
T_material = u_after_plasma(centre) / u_after_empty(centre)
```

因为两者从相同的 `u_before` 出发、经历相同的 Fresnel 传播核，所有传播效应（球面曲率、衍射、频率截止）完全抵消，仅剩纯材料传递函数。

### 2.3 取模友好相位设计

Test A 使用完全电离氢等离子体（Z=Z*=1，fraction_bound=0），精确校准厚度使相位为 `{π/2, π, 3π/2, 2π}`：

| 目标 φ | T = exp(iφ) | 验证要点 |
|--------|------------|---------|
| π/2 | +i | Re(T) ≈ 0 |
| π | −1 | Im(T) ≈ 0, 实负 |
| 3π/2 | −i | Re(T) ≈ 0 |
| **2π** | **+1** | **归零：包裹相位 = 0，等同于真空** |

φ=2π 情况是模 2π 验证的关键——包裹后相位与真空无区别，仅剩约 0.01% 的残余 Kramers 吸收。

## 3. 测试结果

### Test A：相位验证

| 目标相位 | 厚度 | 测量相位 | 误差 | \|T\| | 状态 |
|---------|------|---------|------|-------|------|
| +π/2 | 3.6 μm | +0.5000π | 0.00° | 1.0000 | PASS |
| +π | 7.2 μm | +1.0000π | 0.00° | 1.0000 | PASS |
| +3π/2 | 10.8 μm | −0.5000π | 0.00° | 1.0000 | PASS |
| **+2π** | 14.4 μm | **+0.0000π** | **0.00°** | 1.0000 | **PASS** |

**物理验证**: δ_free = 1.077221×10⁻⁵（解析值），plasma_delta_beta 输出完全一致。
**归零验证**: φ=2π 包裹后相位为 0.0000π，确认取模行为正确。

### Test B：吸收验证

| 厚度 | \|T_field\| | I/I₀ | μ 拟合 (m⁻¹) |
|------|-----------|------|-------------|
| 50 μm | 0.9739 | 0.9485 | 1058.5 |
| 100 μm | 0.9485 | 0.8996 | 1058.4 |
| 150 μm | 0.9237 | 0.8532 | 1058.5 |

- μ_theory = 1058.4 m⁻¹
- μ_fitted = 1058.5 ± 0.1 m⁻¹
- **偏差: 0.01%**

三个厚度点的 μ 拟合值几乎完全相同，Beer-Lambert 线性关系完美成立。

### Test C：Kramers IB 分离

| | Z*=13 (完全电离) | Z*=8 (部分电离) |
|---|---|---|
| β_meas | 7.72×10⁻¹² | 1.3054×10⁻⁸ |
| β_exp | 7.61×10⁻¹² | 1.3053×10⁻⁸ |
| β_ff (Kramers) | 7.61×10⁻¹² | 4.68×10⁻¹² |
| β_bound | 0 | 1.3048×10⁻⁸ |

- **β_bound (差值法)**: 测量 1.3046×10⁻⁸ vs 预期 1.3048×10⁻⁸（偏差 0.002%）
- **β_ff (残差法)**: 测量 7.72×10⁻¹² vs 解析 7.61×10⁻¹²（偏差 1.5%）

在 ne=1×10²³ cm⁻³, 8 keV 条件下，Kramers IB 仅占总 β 的 0.04%。要使其可测量，需要更高密度（>1×10²⁵ cm⁻³）或更低能量（<2 keV）。

### 1D vs 2D 一致性

| | 1D (65536 px) | 2D (2048×2048) |
|---|---|---|
| 相位误差 | 0.00° | 0.00° |
| μ 偏差 | 0.01% | 0.01% |

两种模式结果完全一致，2D 双线性插值未引入额外误差。

## 4. 技术难点与解决

### 4.1 误差传播诊断

**问题**: 初始真空参考法测得的相位误差随厚度线性增长（4.3° → 17.2°）。

**诊断过程**: 系统排除六个假设：

| 假设 | 结果 | 结论 |
|------|------|------|
| H4: 材料因子数值误差 | φ₁ − φ₄ = 8.88×10⁻¹⁶ rad | 排除 |
| H5: np.interp 离散化 | 网格中心精确对齐 | 排除 |
| H3: 多层累积 | nz=1/2/5/10/20 误差完全相同 | 排除 |
| H1: Fresnel 传播 | T_centre − k·dz 修正完全失败 | 证实 |
| H2: 真空参考位置 | 越靠近板边缘误差越小（7.2°→26.2°） | 次要 |
| H6: 边缘衍射 | \|T_vac\|=0.999921, angle(T_vac)/π=+0.544 | 证实 |

**根因**: `T_empty(x)` 在球面波前不同位置取值不同。

```
t=3.6μm: T_empty(vacuum)/T_empty(centre)  angle = +0.023π = 4.1°
t=7.2μm: T_empty(vacuum)/T_empty(centre)  angle = +0.046π = 8.3°
```

比值精确等于真空参考法的全部相位误差——证明误差 100% 来源于测量方法，非等离子体物理。

### 4.2 Python/CUDA 相位符号不一致

Python `PlasmaSample._material_factor()` 使用 `exp(+2πi·δ·t/λ)`（正相位），CUDA kernel 使用 `exp(−2πi·δ·t/λ)`（负相位，见 `kernels.cu:458`）。测试针对 Python 约定设计（正相位目标值），此差异已记录。

### 4.3 2D 频率截止不一致

Python `apply_frequency_cutoff_2d` 阈值仅为 `freq²`，CUDA 为 `freq_x² + freq_y² = 2·freq²`。此 bug 已在 `propagation.py:145` 修复。

### 4.4 big-wave 循环导入

big-wave 模块链 `propagation → util → config → optical_element → history → propagation` 存在循环导入。通过遵循 notebook 验证的导入顺序（`multisim → config → util → propagation`）及 bfpy shim 解决。

## 5. 运行方式

```bash
# 1D 模式（快速，~20秒）
python tests/plasma_sample/test_plasma_phase_absorption.py

# 2D 模式（完整验证，~10分钟）
python tests/plasma_sample/test_plasma_phase_absorption.py --2d

# 误差诊断（独立脚本）
python tests/plasma_sample/diagnose_phase_error.py
python tests/plasma_sample/quantify_phase_error.py
```

## 6. 结论

1. **相位验证**: 完全电离氢等离子体在 π/2, π, 3π/2, 2π 四个相位点的测量误差均为 0.00°，δ_free 解析值与 `plasma_delta_beta()` 输出完全一致。

2. **吸收验证**: 部分电离铝等离子体的 Beer-Lambert 吸收定律得到验证，μ 偏差 0.01%。

3. **Kramers IB**: Z* 对比法成功分离了自由-自由和束缚-束缚吸收贡献，在当前参数下 Kramers IB 占总吸收的 0.04%。

4. **测量方法**: 同位置对消法（`u_plasma / u_empty`）消除了 Fresnel 传播带来的系统性误差，应作为 PlasmaSample 相位/吸收验证的标准方法。

5. **2D 验证**: 1D 和 2D 模式结果完全一致，双线性插值和 2D FFT 传播正确。
