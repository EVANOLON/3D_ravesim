# CB-BPM 原理、工作流与函数调用链报告

> 生成日期：2026-09-03
> 范围：cone-beam BPM（`use_cone_beam_bpm: true`）在 big-wave（Python 参考实现）与 fast-wave（CUDA 生产实现）中的原理、完整工作流、函数级调用链，以及当前运行结果汇总。

---

## 1. CB-BPM 原理

### 1.1 问题背景
点源锥束 X 射线几何中，样品在 `z_sample`、探测器在 `z_detector`、源在 `z_source`。
直接笛卡尔球面波传播需要对每个横向网格点保存完整的球面相位 `exp(ikr)/r`，
且长距离传播的奈奎斯特采样要求极细网格（`dx` 极小），计算量巨大。

### 1.2 Fresnel 缩放（Fresnel scaling theorem）
对于**薄样品**，锥束几何可映射为等效平面波几何：
- 等效传播距离 `z_eff = z_s·z_d / (z_s + z_d)`（`z_s = z_sample − z_source`，`z_d = z_det − z_sample`）
- 几何放大率 `M = (z_s + z_d) / z_s`
- 探测器有效像素 `ds_eff = ds_phys / M`
- 源被替换为**等效倾斜平面波**（横向线性相位），消除了球面相位带来的奈奎斯特负担

**限制**：仅适用于内部传播可忽略的单个薄样品（`validate_fresnel_mode` 强制 thin 模式只能有 1 个 element）。

### 1.3 CB-BPM（cone-beam Beam Propagation Method / 相似坐标多层 BPM）
对厚样品（如 1.2 mm 壁厚的 DT 靶丸，z 向 1200+ 层），样品**内部**的锥束放大不可忽略，
需要在每一材料切片上用**随 z 变化的横向尺度**做多层传播。

核心思想（相似坐标 / similarity coordinates）：
- 保持固定横向参考网格（不逐层重采样整个波前）
- 每个材料切片在**局部锥束尺度** `m(z)` 下采样：
  ```
  m(z) = (z − z_source) / (z_reference − z_source)
  ```
  其中 `z_reference` 是参考平面（首 element 的 `z_start`）。
- 物理传播区间 `[z0, z1]` 映射到等效（moving-coordinate）传播距离：
  ```
  dz_eff(z0, z1) = (z1 − z0) / (m(z0) · m(z1))
  ```
- 材料透射仍用**物理切片厚度**（`material_factor` 用真实 `pixel_size_z`）
- 探测器几何与 thin-Fresnel 相同（`z_eff`、`ds/M`）

数学上，该相似坐标映射是锥束多层传播的**精确**表示（对一个相似坐标缩放后的参考系），
避免了逐层插值整个波前的开销，是 CB-BPM 的核心正确性论断——
因此论文的 1b 验证（CB-BPM vs 直接笛卡尔球面波）是关键算例。

### 1.4 与 thin-Fresnel 的关系
`SimParams.__post_init__`：`use_cone_beam_bpm=True` **强制** `use_fresnel_scaling=True`。
区别在于传播路径：
| | thin Fresnel | CB-BPM |
|---|---|---|
| 切片坐标采样 | 参考面尺度（`fresnel_transverse_scale=1`） | 每片局部 `m(z)` 尺度 |
| 层间传播 | 物理 `dz` | `dz_eff = dz/(m(z0)·m(z1))` |
| 到探测器 | 单腿 `z_eff` | `effective_final_dz`（含 `M`） |
| 适用 | 单薄样品 | 厚/多层样品 |

---

## 2. 端到端工作流（fast-wave CUDA 生产路径，本会话实际运行）

### 2.1 阶段总览
```
config.yaml + computed.yaml + 50×subconfig.yaml
   │
   ├─[准备] multisim.setup_simulation (Python, 生成目录/子配置)
   │
   └─[运行, 每源一次] fastwave -s <i> <sim_dir>
         main.cpp::main
           └─ parse_config(sim_dir, source_idx)
                ├─ parse_sim_params        (读 use_cone_beam_bpm)
                ├─ parse_optical_elements  (Sample + grid + deltabeta)
                ├─ parse_source            (PointSource)
                └─ compute_fresnel_params  (z_eff, M)
           └─ run_simulation(config, sub_dir, history_dz)
                └─ run_simulation_inner_2d<float|double>
                     ├─ [源] initialize_fresnel_plane_2d → kernel
                     ├─ [样品] apply_sample_2d (z 逐层循环)
                     │     ├─ apply_sample_factors_2d → kernel (m(z) 坐标缩放)
                     │     └─ propagate_2d(dz_eff) → 2D FFT 传播
                     └─ [探测器] square_and_downsample_2d
                           └─ square_and_downsample_2d_kernel
                                (面积加权积分, 本会话修复)
```

### 2.2 准备阶段（Python, multisim.setup_simulation）
调用链（`big-wave/multisim.py`）：
1. `config.resolve_sim_params(dct)` — 解析 `chunk_size`、`memory_budget_gb`、`fft2_backend`、`detector_integrator`
2. `config.parse_sim_params(dct["sim_params"])` → `propagation.SimParams`（`use_cone_beam_bpm` → 强制 fresnel）
3. `config.parse_optical_element(el, config_dir)` → `optical_element.Sample`（mmap 载入 uint32 网格）
4. `validate_fresnel_mode(sim_params, elements, source_type)`：
   - CB-BPM 要求 2D、点源、仅 Sample/PlasmaSample、`source < element < detector`
5. `compute_fresnel_cutoff_geometry(...)` → `(cutoff_x, cutoff_y, M, z_eff)`
   - `M = (z_s+z_d)/z_s`；`z_eff = z_s·z_d/(z_s+z_d)`
6. 生成 50 个 `sub_dcts`（每个含 `source/energy/deltabeta_table/plasma_optics_table`）
   - `load_spectrum` → `generate_energies_from_spectrum`
   - `generate_deltabeta_table(materials, energy)` → `deltabeta_table_to_tuples`
7. 写 `config.yaml` / `computed.yaml`（记录 `fresnel_mode: cone_beam_bpm`、
   `fresnel_geometry.version: cone_beam_similarity_bpm_v1`）/ 50×`0000000i/subconfig.yaml`

### 2.3 运行阶段（fast-wave CUDA，每源一次）
`main.cpp`：
- `parse_config(sim_dir, source_idx)` → `config_parsing.cpp`:
  - `parse_sim_params(node, wl)`：解析 `use_cone_beam_bpm`（`config_parsing.cpp:662`），
    自动置 `use_fresnel_scaling=true`
  - `compute_fresnel_params(z_src, z_sample, z_det, z_eff, M)`（`simulation.hpp:72`）
- `run_simulation` → `run_simulation_inner_2d<S>`（`simulation.cpp:735`）：

**① 源初始化（Fresnel 倾斜平面波）**
```
PointSource → sim_params.use_fresnel_scaling → initialize_fresnel_plane_2d
  → initialize_fresnel_plane_2d_kernel (kernels.cu:653)
     相位 = −2π(x·x_src + y·y_src)/(wl·z_source_to_sample)
     幅度 = 1/z_source_to_sample
```
替换直接球面波 `propagate_analytically_2d`（后者仅 FS_Off 用）。

**② 样品多层传播（CB-BPM 核心）**
```
apply_sample_2d(sample, d_u, d_U, params, fft, cutoff_freq_x, cutoff_freq_y, phase_step)
  → for i in [0, z_len):                      // 1200 层
      slice_z = z_start + i·dz
      coordinate_scale = params.fresnel_transverse_scale(slice_z + 0.5·dz)
                         = (slice_z+0.5dz − z_source)/(z_ref − z_source)
      apply_sample_factors_2d(d_u, params, dz, coordinate_scale, d_sample, …)
          → apply_sample_factors_2d_kernel     // 按 m(z) 缩放的材料坐标插值
      propagate_2d(params, fft, params.effective_slice_dz(slice_z, dz), …)
          → 2D FFT 传播（频率域乘 Fresnel 核 + 频率截止）
```
其中 `effective_slice_dz`（`fwcuda/types.hpp:79`）：
```
if (!use_cone_beam_bpm) return dz;
return dz / (fresnel_transverse_scale(z) * fresnel_transverse_scale(z + dz));
```

**③ 到探测器**
```
prop_dz = effective_final_dz(current_z)
        = (z_det − current_z) / (m(current_z) · M)     // cone-beam
propagate_2d(prop_dz) → square_and_downsample_2d
  → ds_px = detector_pixel_size_x / M  (effective)
  → ds_z  = z_eff
  → square_and_downsample_2d_kernel     // 面积加权积分（修复后）
      per detector pixel: 对覆盖网格单元按重叠面积加权 Σ|u|²·ov_x·ov_y
      out = Σ·dx·dy·cos_angle
```
写 `0000000i/detected.npy`（shape (1, 833, 833) float32）。

### 2.4 big-wave Python 参考路径（同一数学，独立实现）
`multisim.run_single_simulation(sim_dir, source_idx, scratch, …)`：
```
config.load(config.yaml) → config.parse_sim_params → SimParams
sim_params.configure_fresnel_detector(z_source, z_sample)   // multisim.py:290
     → 计算 fresnel_source_z / fresnel_reference_z / M / z_eff / effective_detector_geometry
config.parse_optical_element(...) → Sample
wavesim.run_simulation(params, source, elements, cutoff_angles, u, U, …)
  → source.propagate_to(z_start, ...)      // fresnel: 平面波初始化 (source.py:62)
  → for el in elements:
      el.apply(u, U, params, cutoff_freq, 0, history)   // optical_element.py
        → Sample._apply_2d
            for rowidx in z_len:
              coordinate_scale = fresnel_transverse_scale(z+0.5dz)   // 每层局部尺度
              modifier: 材料坐标 x·m(z) 采样 deltabeta → material_factor(dz物理)
              propagate_2d(..., effective_slice_dz(z, dz), ...)
  → propagate_with_history_2d(effective_final_dz)   // 到探测器
  → square_and_downsample_2d(u, params, z_det)
      → detector_integrator:
           legacy_fastwave → _legacy_fastwave_stream   (兼容旧核语义)
           area_v1         → _area_v1_stream           (面积积分, 与 fast-wave 修复后一致)
```
- big-wave 提供 `detector_integrator: area_v1`（`propagation._area_v1_stream`），
  与本次 fast-wave 核修复（面积加权）同语义——是正确性的交叉参照。

---

## 3. 关键函数-文件索引（函数调用一级速查）

### big-wave（Python）
| 函数 | 文件 | 作用 |
|---|---|---|
| `SimParams` / `__post_init__` | propagation.py:27 | 参数解析；CB-BPM 强制 fresnel |
| `SimParams.configure_fresnel_detector` | propagation.py:112 | 计算 z_eff/M/source/reference z |
| `SimParams.fresnel_transverse_scale` | propagation.py:135 | m(z) 局部尺度 |
| `SimParams.effective_slice_dz` | propagation.py:147 | dz/(m(z0)m(z1)) |
| `SimParams.effective_final_dz` | propagation.py:156 | 到探测器等效距离 |
| `SimParams.effective_detector_geometry` | propagation.py:170 | ds/M、z_eff 探测器几何 |
| `propagate_2d` | propagation.py:394 | 2D Fresnel 传播（行/列 FFT） |
| `apply_frequency_cutoff_2d` | propagation.py:296 | 圆频率截止 |
| `_area_v1_stream` | propagation.py:993 | 面积积分探测器（参考实现） |
| `run_simulation` | wavesim.py:89 | 单源主循环 |
| `Sample._apply_2d` | optical_element.py:502 | 厚样品逐层 CB-BPM |
| `material_factor` | optical_element.py | 切片透射 exp 因子 |
| `setup_simulation` | multisim.py:683 | 目录/子配置生成 |
| `run_single_simulation` | multisim.py:228 | 单源执行（big-wave 引擎） |
| `validate_fresnel_mode` | multisim.py:578 | CB-BPM 几何合法性 |
| `compute_fresnel_cutoff_geometry` | multisim.py:610 | 截止角 + M/z_eff |
| `fresnel_mode_name` | multisim.py:570 | thin / cone_beam_bpm / off |

### fast-wave（CUDA）
| 函数 | 文件 | 作用 |
|---|---|---|
| `parse_sim_params` | config_parsing.cpp:641 | 读 `use_cone_beam_bpm` |
| `compute_fresnel_params` | simulation.hpp:72 | z_eff/M |
| `initialize_fresnel_plane_2d(_kernel)` | simulation.cpp / kernels.cu:653 | 倾斜平面波源 |
| `SimParams::fresnel_transverse_scale` | fwcuda/types.hpp:72 | m(z) |
| `SimParams::effective_slice_dz` | fwcuda/types.hpp:79 | dz/(m0·m1) |
| `SimParams::effective_final_dz` | fwcuda/types.hpp:84 | 探测器等效距离 |
| `apply_sample_2d` | simulation.cpp:653 | 样品逐层循环 |
| `apply_sample_factors_2d(_kernel)` | kernels.cu | m(z) 缩放材料采样 |
| `propagate_2d` | simulation.cpp | 2D FFT 传播 |
| `square_and_downsample_2d(_kernel)` | kernels.cu:525 | **面积加权积分**（本次修复） |
| `run_simulation_inner_2d` | simulation.cpp:735 | 2D 单源主循环 |

### 本次会话修改
- `kernels.cu::square_and_downsample_2d_kernel`：整数计数 → 面积加权（commit fba9c5d）

---

## 4. 当前 CB-BPM 运行结果整理

### 4.1 四靶丸 CB-BPM（主结果，50 源求和，833×833，count-纹波已修正）
| 靶丸 | mean | min | max |
|---|---|---|---|
| perfect_sphere（参考） | 1.0482e-05 | 9.07e-06 | 1.18e-05 |
| l1_m0（偶极位移） | 1.0482e-05 | 9.08e-06 | 1.18e-05 |
| l2_m0（P2 椭球） | 1.0482e-05 | 9.06e-06 | 1.18e-05 |
| l1_varT（壁厚变化） | 1.0482e-05 | 9.07e-06 | 1.18e-05 |

文件：`output/_agent_runs/plots/cbbpm_corr_<shell>_sum50.npy`

### 4.2 真空场（2a，开放场 I₀，50 源求和）
- `cbbpm_vacuum_sum50.npy`：mean 1.0509e-05，平滑（1.0248e-5–1.0652e-5，<2%）
- 真空归一化透射 mean 0.9975（DT 低 Z 弱吸收，合理）
- 归一化残差（vs perfect）：l1_m0 RMS 0.88%、l2_m0 0.72%、varT 0.86%（% 透射）

### 4.3 薄板验证（1a/1b，面积核重跑后）
1a（CB-BPM vs FS_On，薄极限收敛）：
| 板 | C10(10µm) | C20(20µm) | W1(1µm) | W2 | W5 | 单层 |
|---|---|---|---|---|---|---|
| L2 | 5.7e-5 | 1.6e-4 | 2.0e-5 | 4.7e-5 | 1.7e-4 | ~1.7e-6 |

1b（CB-BPM vs FS_Off 直接球面波）：L2 ≈ 4.0–5.3e-2（修复前 0.28，混叠已除）
- 径向分布/能量分布与 FS_Off 一致；剩余为亚像素边界差异

### 4.4 轴向收敛（1c，l1_varT）
| dz | 2µm(603层) | 4µm(302层) |
|---|---|---|
| L2 vs 1µm(1206层) | 4.5e-3 | 6.9e-3 |

单调收敛 ✓

### 4.5 缺陷定量识别（2b，方位角傅里叶）
| 缺陷 | 残差 m1 | 残差 m2 | m2/m1 | 判读 |
|---|---|---|---|---|
| l1_m0 | 9.3e-8 | 7.1e-9 | 0.076 | 偶极（位移）✓ |
| l2_m0 | 5.8e-9 | 3.9e-8 | 6.74 | P2（椭球）✓ |
| varT | 3.4e-7 | 3.0e-8 | 0.089 | 位移型 ✓ |

数据：`cbbpm_2b_azimuthal.json`

### 4.6 性能（3a，4096² CB-BPM vs FS_On，同口径实测）

测量条件：RTX 5070 Ti Laptop（12227 MiB）、driver 573.22、c8（complex64）、
perfect_sphere（Nz=1200，1206³ 网格）、fastwave 面积核（2026-09-03 实测）、
显存峰值由运行时 nvidia-smi 采样（0.3 s 间隔）。

| 指标 | FS_On（thin Fresnel） | CB-BPM（cone-beam BPM） | 差异 |
|---|---|---|---|
| 网格 / Nz | 4096², dx=500 nm / 1200 | 同左 | — |
| 单源计算时间 | 51.9 s | 52.6 s | +1.3% |
| 单源 wall 时间（含 6.9 GB 网格上传） | 115.2 s | 113.2 s | −1.7% |
| 每层平均时间 | 43.2 ms/layer | 43.8 ms/layer | +1.4% |
| 峰值显存 | 8477 MiB（8.3 GB） | 8425 MiB（8.2 GB） | −0.6% |
| 50 源总耗时 | 72.7 min（2026-06-27） | 76–169 min（122–250 s/源，含断点） | — |
| GPU / 精度 / 驱动 | RTX 5070 Ti Laptop / c8 / 573.22 | 同左 | — |

**CPU–GPU 交叉验证**：big-wave（area_v1 积分器）vs fastwave（面积核），
算例 l1_varT dz=4µm（302 层 4096²，源 0）：
- 相对 L2 = **8.9×10⁻⁵**，相关系数 0.99997
- CPU（big-wave）单源 567.5 s vs GPU（fastwave）单源 ~11 s（同算例，GPU 快 ~50×）

**关键结论**：CB-BPM 相对 thin-Fresnel **计算时间仅 +1.4%、显存相同**
——厚样品锥束正确性的获得无性能代价；CPU/GPU 独立实现面积积分器一致
（L2 ~1e-4），同时交叉验证了 fastwave 面积核修复的正确性。

数据：`output/_agent_runs/plots/cbbpm_perf.json`

### 4.7 图
- `cbbpm_corrected_four_shell.png` / `cbbpm_corrected_residuals.png`（四靶丸）
- `cbbpm_normalized_transmission.png`（真空归一化）
- `cbbpm_summary_all.png`（综合）
- `http://127.0.0.1:8811/<file>.png` 可内嵌

### 4.8 待办（未完成）
- 1d 横向采样收敛（2048² vs 4096²，已暂缓，评审 A1 建议项）
- （3a 已补测完成）
