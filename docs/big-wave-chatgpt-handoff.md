# Big-Wave 2D：供 ChatGPT 使用的项目说明包

> 文档用途：将本文件上传到另一台设备的 ChatGPT，使其在没有当前聊天记录、没有本机
> 工作区访问权限的情况下，正确理解 big-wave 2D 的功能、架构、验证状态和边界。
>
> 事实基线日期：2026-08-19  
> 当前阶段：P0–P5 核心验收完成；P6 与 R2 尚未完成。

## 1. 给 ChatGPT 的阅读规则

回答 big-wave 相关问题时，请遵守以下规则：

1. 将“已实现”“已测试通过”“计划实现”严格区分。
2. 本文的完成度限定在 **fast-wave 已覆盖的二维功能范围**，不能理解为已经覆盖全部
   一维功能。
3. `big-wave-2d-updated-plan.md` 中的未来任务属于计划，不能仅凭计划文档宣称已经实现。
4. 数值、性能和内存结论只适用于文中注明的配置与测试环境，不能直接外推为所有设备的
   性能保证。
5. 若同时提供源代码，以源代码和最新测试结果为最终事实；发现与本文不一致时，应明确指出
   差异，而不是自行调和。
6. 不要宣称 P6、生产 capsule/真实 Plasma 网格 R2 验收或 phase-stepping 检查点已经完成。

## 2. 项目定位

RAVE-SIM 包含 fast-wave 和 big-wave 两类波动光学计算路径。big-wave 的目标是通过磁盘波场、
瓦片化局部算子和核外 FFT，在有限内存中执行大规模计算。

big-wave 2D 当前使用原有一维引擎的对象模型，但将波场按二维 row-major 网格解释：

```text
logical shape = (ny, nx)
physical NPY shape = (nx * ny,)
flat_index = iy * nx + ix
N = nx * ny
```

当前准确定位是：

> 在 fast-wave 已覆盖的二维功能范围内，big-wave 2D 的传播、Sample/PlasmaSample、FFT2、
> detector 和 History 已实现内存有界处理，并完成 16384² 真空、单 slice Sample、四 slice
> PlasmaSample 测试。剩余风险主要是 DSH 运行工程、生产数据全量验收和多相位恢复语义。

## 3. 已实现功能

| 功能 | 状态 | 说明 |
|---|---|---|
| 二维 PointSource | 已实现并测试 | 支持 `(x,y,z)` 点源及球面波 |
| NumpyVector FFT2 | 已实现 | SciPy 内存路径，用于小规模参考 |
| DiskVector FFT2/IFFT2 | 已实现并测试 | bfpy 事务式核外 FFT2，支持 complex64/complex128 |
| 二维 Fresnel 传播 | 已实现并测试 | 按完整行 tile 处理 |
| 二维 frequency cutoff | 已实现并测试 | 二维圆形截止及边界测试 |
| 二维 Sample | 已实现并测试 | 输入网格 mmap，tile 内双线性插值 |
| 二维 PlasmaSample | 已实现并测试 | 多网格 mmap，tile 内向量化 δ/β 计算 |
| `legacy_fastwave` detector | 已实现并测试 | 默认；保持 fast-wave 的中心、截断和 count-map 语义 |
| `area_v1` detector | 已实现并测试，可选 | 面积重叠积分；当前不是默认积分器 |
| History v2 | 已实现并测试 | canonical `(z,y,x)`；支持 HDF5、ROI、下采样和旧格式迁移 |
| 运行检查点 | 已实现并测试 | 元件边界及 Sample/PlasmaSample slice 边界恢复 |
| 多源 y | 已实现并测试 | 贯通配置、Nyquist 预检、子配置和 provenance |
| 资源预检 | 已实现首阶段能力 | 检查 RAM、scratch、事务文件及 backend/vector 组合 |
| 进度与取消接口 | 引擎侧已具备 | FFT2、History 等路径可上报进度和响应取消 |

## 4. 尚未实现或尚未完成验收

| 项目 | 当前边界 |
|---|---|
| P6 DSH 运行工程 | 持久化 job、阶段 ETA、进程组取消、并发限流、磁盘治理仍待完成 |
| 生产数据 R2 验收 | 尚未用生产 capsule 和真实 Plasma 网格完成全量验收 |
| phase stepping + checkpoint | 当前明确拒绝同时启用，尚未定义多相位事务恢复语义 |
| VectorSource 2D | 当前范围外；fast-wave 2D 也未正确覆盖 |
| Grating/EnvGrating 2D | 当前范围外 |
| precise_Sample 2D | 当前范围外 |
| 通用性能 SLA | 尚未建立；现有时间仅是测试设备上的软基线 |

因此，“P0–P5 完成”不能表述为“已经达到一维版本的全部工程完成度”。

## 5. 当前架构

```text
用户配置
  -> multisim.setup_simulation
  -> 校验 nx / ny / N、资源预算和 backend
  -> config.yaml + computed.yaml
  -> run_single_simulation
  -> NumpyVector 或 DiskVector
  -> wavesim 二维调度
      -> 按行 tile 的传播和 frequency cutoff
      -> mmap Sample / vectorized PlasmaSample
      -> History v2（RAM 或 HDF5）
      -> 运行检查点
  -> 流式二维 detector
  -> atomic detected.npy + detector_metadata.yaml
```

### 5.1 向量后端

- `NumpyVector` 使用 SciPy 内存 FFT2，适合小规模参考。
- `DiskVector` 使用 bfpy 核外 FFT2；不再回退到 `np.load + scipy.fft2` 的整幅内存路径。
- 二维 DiskVector 默认解析为 `bfpy_ooc` 后端。

### 5.2 核外 FFT2

主要阶段：

```text
row_fft
-> transpose_to_scratch
-> column_fft
-> transpose_to_output
-> fsync + atomic rename
```

FFT2 的输入保持只读，中间 pass 通过 manifest 记录。未完成的 pass 不登记；已登记产物损坏时
回退到前一安全 pass。正式输出仅在完整成功后原子替换。

FFT2 内存复杂度主要由单行 FFT scratch 和预算受控的转置 tile 决定，而不是把完整二维复数
波场加载到 RAM。

### 5.3 Sample 与 PlasmaSample

- 材料和等离子体输入网格以只读 mmap 打开。
- 每个 z slice 再按波场 y tile 处理。
- Sample 在 tile 内完成双线性插值和复数材料因子计算。
- PlasmaSample 在 tile 内向量化计算 δ/β，并缓存元素/能量常数。

### 5.4 流式 detector

两个积分器均按波场行 tile 累积，不创建完整 float64 强度图、二维前缀和或
`ny × detector_nx` 中间数组。

- 默认 `legacy_fastwave`：用于兼容 fast-wave 的离散计数语义。
- 可选 `area_v1`：使用波场网格单元与 detector pixel 的几何重叠面积。

输出较小时存 RAM；预测超过内存预算时使用临时 NPY memmap，并通过事务式提交生成
`detected.npy`。

### 5.5 History 与恢复

- 二维 History 的标准轴序是 `(z,y,x)`，格式版本为 2。
- 大 History 使用可扩展 HDF5 dataset；可在写入前应用 ROI、下采样和帧数上限。
- 旧 `(y,x,z)` 文件可由兼容读取器识别和迁移。
- 运行检查点保存空间域/频域波场、当前位置、元件和 slice 索引、阶段及配置指纹。
- Snapshot 是 phase stepping 数据，不等同于运行检查点。

## 6. 关键配置契约

典型配置：

```yaml
runtime:
  big_wave:
    memory_budget_gb: 6.0
    chunk_size: auto
    fft2_backend: bfpy_ooc
    detector_integrator: legacy_fastwave

use_disk_vector: true
sim_params:
  N: 268435456
  nx: 16384
  ny: 16384
```

关键规则：

- `N` 必须等于 `nx * ny`。
- 当前 setup 要求 `nx`、`ny` 为正数和 2 的幂。
- `chunk_size` 必须对应完整行；`auto` 会解析为完整行数。
- `bfpy_ooc` 必须搭配 DiskVector。
- detector 积分器默认是 `legacy_fastwave`，不能把 P5 大规模收敛测试使用的 `area_v1`
  误认为全局默认值。
- feasibility 会检查内存、临时磁盘空间及 backend/vector 组合。

## 7. 测试与验收状态

### 7.1 自动回归

| 阶段 | 测试数 | 状态 |
|---|---:|---|
| P0 配置与资源契约 | 17 | 通过 |
| P1 局部算子瓦片化 | 11 | 通过 |
| P2 核外 FFT2 | 11 | 通过 |
| P3 流式 detector | 8 | 通过 |
| P4 History/兼容/检查点 | 11 | 通过 |
| P5 科学与跨引擎验证 | 10 | 通过 |
| 合计 | **68/68** | **通过** |

原有 `big-wave/test.py` 为 26/28。两项失败是既有的边界计数和旧 golden 数据问题，不是
P0–P5 新增回归，但仍不应对外表述为“全项目所有测试通过”。

### 7.2 科学与跨引擎验证

P5 覆盖点源球面波、平面/倾斜平面波、均匀薄层、Beer–Lambert 吸收、二维 cutoff、矩孔、
薄钨片、空心胶囊、Plasma 1D/2D 中心线，以及 big-wave/fast-wave detector 交叉验证。

偏移点源 detector 的 big-wave/fast-wave 相对 L2：

```text
2.27808137e-7
```

对应测试容差为 `2e-4`。

### 7.3 16384² 大规模结果

测试条件：complex64、`bfpy_ooc`、512 MiB 算子预算、tile rows 32、WSL ext4 scratch、
固定 3.2768 mm × 3.2768 mm 物理域。

| 场景 | wall time | 峰值 RSS | 进程 swap | 累计写入量 |
|---|---:|---:|---:|---:|
| 真空 16384² | 159.12 s | 626.7 MiB | 0 | 20.22 GiB |
| Sample 16384²，1 slice | 199.60 s | 627.8 MiB | 0 | 55.01 GiB |
| PlasmaSample 16384²，4 slices | 692.15 s | 660.7 MiB | 0 | 121.91 GiB |

这三项均未发生 OOM kill。结果证明该配置下主路径内存有界，但不代表任意配置、任意网格和
任意存储设备都不会 OOM。

8192² 到 16384² 的 Sample/Plasma 中心线相对 L2 为约 3.76%，detector 总能量相对差约
`7.2e-5`，随网格细化下降。3.76% 是当前设备和场景的可复验基线，不是已经冻结的通用
科学容差。

## 8. 已知风险与运行注意事项

1. Plasma 16384² 单次累计写入约 122 GiB，磁盘空间、NVMe 写入量和并发任务必须受控。
2. 在 Windows `/mnt/d` 路径上曾出现大事务 `.part` 文件创建失败；生产 OOC scratch 当前
   建议放在 WSL 原生 ext4/NVMe，直到完成 DrvFS 专项复测。
3. 16384² 性能强烈依赖 CPU、存储和 WSL 配置，不能把上述 wall time 当成跨机器 SLA。
4. checkpoint 当前不能与 phase stepping 同时启用。
5. 大规模 Plasma 验收使用均匀四 slice 负载，不等同于真实生产流体网格的科学验收。

## 9. 代码和证据索引

若同时上传完整仓库，优先查看：

```text
docs/big-wave-2d-current-architecture.md
docs/big-wave-2d-updated-plan.md
tests/big_wave_2d/test_p0.py
tests/big_wave_2d/test_p1.py
tests/big_wave_2d/test_p2.py
tests/big_wave_2d/test_p3.py
tests/big_wave_2d/test_p4.py
tests/big_wave_2d/test_p5.py
tests/big_wave_2d/P4_TEST_REPORT.md
tests/big_wave_2d/P5_TEST_REPORT.md
tests/big_wave_2d/p5_benchmark.py
tests/big_wave_2d/p5_results/
```

事实来源优先级：

```text
最新源代码和最新测试产物
> P4/P5 验收报告
> current architecture
> updated plan
> 本交接摘要
> 旧聊天中的自然语言结论
```

## 10. 推荐的首次提问

上传本文件后，可向另一台设备的 ChatGPT 发送：

```text
请把附件《Big-Wave 2D：供 ChatGPT 使用的项目说明包》作为当前事实基线。
先用不超过 10 条要点复述 big-wave 2D 的已实现功能、验证状态和剩余边界。
后续回答必须区分“已实现”“已测试通过”和“计划中”，不得把 P6、R2 生产数据验收
或 phase-stepping 检查点描述为已经完成。如果需要判断代码细节，请明确说明需要补充
哪些源文件或测试证据。
```

