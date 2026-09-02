# big-wave 2D P4 验收记录

日期：2026-08-19

## 结论

P4 已完成，独立测试 11/11、P0–P4 聚合测试 58/58 通过。二维 History 已从完整帧
Python list 收口为带版本的 canonical `(z, y, x)` 数据模型，并支持预算触发的 HDF5
流式写入、ROI、下采样、帧数上限和取消。多源 `y` 坐标已贯通配置、子任务与 provenance；
运行检查点可以在元件边界和 Sample/PlasmaSample slice 边界恢复。

## 实现范围

### 多源 y

- `multisource.y_range` 缺省为 `[0, 0]`，非法范围在 setup 阶段拒绝；
- 每个 `subconfig.yaml` 写入 `source.y`；
- `computed.yaml` 的 `source_points` 同时记录 x/y/energy；
- 二维 Nyquist 预检分别检查 x 和 y；
- 每个 source 使用隔离的输出目录和检查点目录。

### History v2

- 2D canonical 轴序为 `(z, y, x)`，1D 保持原有 `(x, z)`；
- metadata 写入 `format_version=2` 和明确的 `axis_order`；
- 小数据可驻留内存，超过预算自动写入可扩展 HDF5 dataset；
- 支持 `roi=(y0,y1,x0,x1)`、`downsample=(y,x)`、`max_frames`；
- 支持 progress/cancel，解析点源 History 不再创建整场 `meshgrid`；
- `history_x/history_y` 采用 detector 输出的居中物理坐标；
- 读取器兼容旧 `(y,x,z)`，迁移器显式转置且不覆盖源文件。

### 运行检查点

- `RunCheckpoint` 保存 `u`、频域场、当前 z、element/slice/phase 索引、FFT 有效状态；
- generation 文件与 manifest 均事务提交，manifest 保存 SHA-256；
- 恢复时校验配置 fingerprint、文件结构和 checksum；
- Sample/PlasmaSample 每完成一个 slice 即可提交恢复点；
- Snapshot 与 checkpoint 使用不同结构。当前明确拒绝“checkpoint + phase stepping”组合，
  防止把相位快照误解释为运行恢复点；
- 磁盘波场使用 `u.npy` 与 `spectrum.npy`，避免大小写不敏感文件系统把 `u.npy/U.npy`
  识别为同一路径。

## 自动测试

`tests/big_wave_2d/test_p4.py` 覆盖 11 项：

1. 多源 y 写入各 source 子配置与 computed metadata；
2. 缺省 y 范围保持旧配置兼容；
3. 非法 y 范围拒绝；
4. 小规模 History canonical 轴序；
5. HDF5 流式、ROI、下采样和版本化读取；
6. 旧 `(y,x,z)` History 显式迁移；
7. `max_frames` 与 cancel 硬门；
8. 完整运行写入居中坐标和 canonical shape；
9. 检查点双向量 checksum/restore；
10. Sample/Plasma slice 恢复不重放已完成 source/slice；
11. checkpoint 与 phase-stepping Snapshot 冲突拒绝。

| 测试门 | 结果 |
|---|---:|
| P0 配置/资源 | 17/17 |
| P1 局部算子 | 11/11 |
| P2 OOC FFT2 | 11/11 |
| P3 流式 detector | 8/8 |
| P4 History/兼容/恢复 | 11/11 |
| 合计 | **58/58** |

原 `big-wave/test.py` 为 26/28；失败仍是 P2/P3 前已经存在的
`TestMultisim.test_load_wavefronts` 边界计数和 `TestSnapshot.test` 旧 golden 数据，不是 P4
新增回归。

## 验收边界

- checkpoint 当前不与 phase stepping 并用；需要先定义多相位事务和恢复语义；
- checkpoint 的 slice 级恢复只适用于 2D Sample/PlasmaSample，其他元件按元件边界恢复；
- History HDF5 解决的是内存边界，长期归档、压缩策略和跨版本数据治理仍属于运行工程；
- P4 不包含 fast-wave 当前也不支持的 2D Grating、VectorSource 和 precise_Sample。

