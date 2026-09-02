# big-wave 2D P3 验收记录

日期：2026-08-19

## 结论

P3 已完成。`square_and_downsample_2d` 不再创建整场 float64 强度、二维前缀和或
`meshgrid`。默认 `legacy_fastwave` 保留 fast-wave CUDA 的像素中心与整数截断边界；
可选 `area_v1` 使用可分离的网格单元/探测器像素重叠面积。

两种积分器均按完整行 tile 读取 `Vector`，只保留当前行强度、一维 x 前缀和、活动
detector 行和输出。余弦几何因子融合到逐行累加，避免第二次全输出扫描。

## 实现范围

- `detector_integrator` 支持 `legacy_fastwave`（默认）和 `area_v1`；
- detector 输出超过 `memory_budget_gb` 时自动使用临时 NPY memmap；
- 新建 memmap 利用稀疏零页，不预触碰完整输出；
- 单相位 memmap 通过等长 NPY header 改写和原子重命名提交为 `detected.npy`；
- 多相位/跨设备情况使用固定 8 MiB 顺序 I/O 回退；
- 非整数像素/网格比、零计数或零覆盖输出 `RuntimeWarning`；
- 每个 source 保存 `detector_metadata.yaml`，包含积分器、版本、有效几何、输出存储和
  count-map/coverage 统计；
- Fresnel 模式记录并使用 `M`、`z_eff` 和缩放后的 detector pixel size，输出形状仍由
  物理 detector pixel size 决定，与 fast-wave 一致；
- detector progress/cancel 已接入 `run_single_simulation`；
- feasibility 的 detector RAM 模型已改为 P3 流式模型。

## 自动测试

`tests/big_wave_2d/test_p3.py` 覆盖：

1. `legacy_fastwave` 对照直接 CUDA 语义循环；
2. 禁止整场强度和二维前缀和分配；
3. 非整数像素比与零计数告警/metadata；
4. `area_v1` 对照暴力重叠面积积分；
5. 非整数比例下均匀场的面积守恒；
6. 输出 memmap、正式文件提交和临时文件清理；
7. Fresnel 有效几何；
8. progress/cancel、配置、feasibility 和 DiskVector 端到端运行。

| 测试 | 结果 |
|---|---:|
| P0 配置/资源门 | 17/17 通过 |
| P1 局部算子 | 11/11 通过 |
| P2 OOC FFT2 | 11/11 通过 |
| P3 流式 detector | 8/8 通过 |
| `test_2d_simulation.py` | 4/4 通过 |

原有 `big-wave/test.py` 为 26/28；仍是 P2 前已存在的两个失败：
`TestMultisim.test_load_wavefronts` 的边界计数和 `TestSnapshot.test` 的旧 golden snapshot。
独立 `test_plasma_sample.py` 默认参数在传播前触发既有 Nyquist 检查，未进入 P3 detector。

## 独立进程 RSS 探针

命令入口：`tests/big_wave_2d/p3_benchmark.py`。输入由父进程写盘，子进程同时采样
`/proc/self/status`，测量 detector 和正式结果提交全过程。

共同参数：c8、`tile_rows=32`、detector 预算 32 MiB、detector pixel 与网格间距相同。

| 网格 | 积分器 | 输入 | 输出 | 存储 | detector | 全流程峰值 RSS | detector + 提交耗时 |
|---|---|---:|---:|---|---:|---:|---:|
| 1024² | legacy | 8 MiB | 8 MiB | RAM | 0.054 s | 121.0 MiB | 0.087 s |
| 4096² | legacy | 128 MiB | 128 MiB | memmap | 1.322 s | 107.3 MiB | 1.330 s |
| 4096² | area_v1 | 128 MiB | 128 MiB | memmap | 1.568 s | 107.6 MiB | 1.575 s |

4096² 输入和输出各扩大到 128 MiB 后，进程 RSS 未随两幅文件总大小增长；两个积分器
在相同预算下的峰值基本一致。探针只用于验证内存复杂度，耗时不代表生产磁盘性能。

## 剩余边界

P3 解除了完整 detector 帧的整场强度/前缀和峰值。尚未执行 16384² 完整成像验收，
原因是 History 仍把多帧保存在 Python list，phase stepping 多输出的生命周期和 canonical
轴序也属于 P4。下一发布门是 P4，而不是继续修改 detector 数值语义。
