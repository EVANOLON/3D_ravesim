# big-wave 2D P5 验收记录

日期：2026-08-19

## 结论

P5 核心发布门已执行完成：解析物理/成像/跨引擎测试 10/10，P0–P5 聚合回归 68/68，
4096²、8192²、16384² 三档真空、单 slice Sample 和四 slice PlasmaSample 全部完成。
16384² 三个用例的进程峰值 RSS 为 626.7–660.7 MiB，均远低于 6 GiB 门槛，进程
`VmSwap=0`，没有再发生 OOM kill。

本结论是“P5 核心门可验收”，不是“已达到 1D 全部工程完成度”：生产 capsule/真实流体网格
全量数据和 P6 DSH 运行工程仍未验收，首次性能数据也只能作为软基线。

## 测试环境与统一物理域

- WSL 目标内存上限：13 GiB（运行时可见约 12.67 GiB）；
- dtype：complex64；OOC backend：`bfpy_ooc`；
- FFT/算子预算：512 MiB；tile rows：32；
- scratch：WSL 原生 ext4 `/tmp`；
- 固定模拟视场：3.2768 mm × 3.2768 mm；
- 三档 dx：0.8 µm、0.4 µm、0.2 µm；
- 固定 cutoff frequency：250000 m⁻¹；
- detector：0.4 mm × 0.4 mm，pixel 0.8 µm，输出 500×500；
- 大规模收敛使用 `area_v1`，比较同一 detector 物理坐标上的中心 256 像素。

固定物理视场、截止频率、样品尺寸和 detector 坐标很重要。早期随 N 同时扩大视场的探索性
结果已删除，不能用于分辨率收敛判断。

## P5 自动测试

`tests/big_wave_2d/test_p5.py` 覆盖：

1. 偏移点源球面波 `1/r`、圆周相位和 x/y 峰值位置；
2. 平面波与倾斜平面波传递函数；
3. 均匀薄层公共相位、圆周相位误差和 Beer–Lambert 吸收；
4. 二维圆形 cutoff 的轴向/对角边界；
5. 矩孔 Fresnel 积分解析对照；
6. 带 x/y 偏移的实际 W 材料薄片对照显式透射场；
7. 空心胶囊 x/y 对称性以及与实心球的可区分性；
8. PlasmaSample 1D 截面与 2D 中心线一致性；
9. 偏移点源下 big-wave/fast-wave 同网格交叉验证。
10. 收敛比较器拒绝不同物理视场、cutoff、detector 几何或积分器的结果。

big-wave/fast-wave detector 图像相对 L2 为 `2.27808137e-7`，通过 `2e-4` 容差。

## 大规模端到端结果

父进程时间包含 worker 启动、完整成像和结果提交。I/O 是 `/proc/<pid>/io` 的累计写字节，
包含 OOC FFT 的事务文件放大。

| 场景 | 网格 | wall time | 峰值 RSS | 进程 swap | 写入量 |
|---|---:|---:|---:|---:|---:|
| 真空 | 4096² | 5.40 s | 363.6 MiB | 0 | 1.25 GiB |
| 真空 | 8192² | 16.72 s | 622.1 MiB | 0 | 5.00 GiB |
| 真空 | 16384² | 159.12 s | 626.7 MiB | 0 | 20.22 GiB |
| Sample，1 slice | 4096² | 24.64 s | 364.1 MiB | 0 | 3.00 GiB |
| Sample，1 slice | 8192² | 44.08 s | 622.3 MiB | 0 | 12.58 GiB |
| Sample，1 slice | 16384² | 199.60 s | 627.8 MiB | 0 | 55.01 GiB |
| PlasmaSample，4 slices | 4096² | 38.46 s | 376.6 MiB | 0 | 6.75 GiB |
| PlasmaSample，4 slices | 8192² | 218.15 s | 642.0 MiB | 0 | 27.50 GiB |
| PlasmaSample，4 slices | 16384² | 692.15 s | 660.7 MiB | 0 | 121.91 GiB |

Plasma 16384² 期间系统级 swap 使用量变化 0.25 MiB，但被监控子进程的 `VmSwap` 峰值仍为
0；其余用例系统级变化为 0。

## 固定域收敛

| 场景 | 网格细化 | 中心线相对 L2 | detector 总能量相对差 | peak 相对差 |
|---|---:|---:|---:|---:|
| 真空 | 4096→8192 | 3.5598e-4 | 1.6632e-8 | 5.8741e-4 |
| 真空 | 8192→16384 | 1.1559e-4 | 2.6964e-8 | 5.3056e-5 |
| Sample | 4096→8192 | 1.8739e-1 | 3.9737e-4 | 2.2304e-1 |
| Sample | 8192→16384 | 3.7608e-2 | 7.2233e-5 | 4.8053e-2 |
| PlasmaSample | 4096→8192 | 1.8739e-1 | 3.9724e-4 | 2.2304e-1 |
| PlasmaSample | 8192→16384 | 3.7608e-2 | 7.2108e-5 | 4.8053e-2 |

三类场景均呈网格细化收敛；Sample/Plasma 的中心线误差约下降 5 倍，总能量误差下降到
`7.3e-5` 以下。原计划没有预先冻结成像中心线的硬阈值，因此 `3.76%` 应作为本机首个
可复验基线，而不能事后包装成通用科学容差。建议在下一次正式发布前由应用负责人确认
中心线/ROI 容差，并将其固化为 CI 门。

## 回归结果

| 测试门 | 结果 |
|---|---:|
| P0 | 17/17 |
| P1 | 11/11 |
| P2 | 11/11 |
| P3 | 8/8 |
| P4 | 11/11 |
| P5 | 10/10 |
| 合计 | **68/68** |
| `test_2d_simulation.py --no-plot` | 4/4 |
| 原 `big-wave/test.py` | 26/28（两项既有失败） |

## 发现并修复的问题

- 将仅大小写不同的 `u.npy`/`U.npy` 改为 `u.npy`/`spectrum.npy`，避免 Windows/DrvFS
  路径冲突；检查点 generation 文件也使用 UUID，不再依赖大小写区分；
- 基准比较固定物理视场与样品尺寸，避免把视场变化误报为数值不收敛；
- 基准输出加入固定中心 256 像素指纹、dx/FOV/cutoff、RSS/swap/I/O 和阶段时间；
- 在 `/mnt/d` 上曾出现大事务 `.part` 文件创建失败，而相同任务在 WSL ext4 成功。
  在完成改名后的 DrvFS 专项复测前，生产 OOC scratch 应放在 WSL 原生 ext4/NVMe。

## 尚未关闭的发布风险

- 尚未用生产级 capsule/真实辐射流体网格做全量 16384² 验收；当前胶囊是小规模结构测试，
  大规模 Plasma 是均匀四 slice 性能/稳定性负载；
- 16384² Plasma 单次累计写入约 122 GiB，NVMe 寿命、磁盘余量和并发限流必须由 P6 管理；
- 691 s 是首次环境观察值，不是跨机器硬性能 SLA；
- P6 的持久化 job、ETA、进程组取消、错误分类和安全清理尚待实施；
- checkpoint 尚未支持 phase stepping 的多相位事务恢复。

## 复验入口

- 物理/跨引擎：`tests/big_wave_2d/test_p5.py`
- 大规模基准：`tests/big_wave_2d/p5_benchmark.py`
- 原始证据：`tests/big_wave_2d/p5_results/*_convergence_*.json`
