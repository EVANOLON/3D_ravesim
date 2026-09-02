# big-wave 2D P1 验收记录

日期：2026-08-19  
环境：WSL，`/home/taylor/anaconda3/envs/rave-sim/bin/python`  
范围：二维局部算子行瓦片化、Sample/PlasmaSample mmap 与 tile 读取、Plasma 向量化。P2 OOC FFT2 不在本次验收范围内。

## 自动测试

```text
python tests/big_wave_2d/test_p1.py -v
Ran 11 tests: OK
```

覆盖：

- `tile_rows=1/3/4/8` 的结果不变性；
- c8/c16 点源解析传播；
- cutoff 和 Fresnel 传播；
- Sample 双线性插值及越界真空语义；
- Sample/PlasmaSample 的只读 `np.memmap`；
- Plasma 向量公式与逐像素标量公式对照；
- 非完整行 chunk 拒绝；
- P1 algorithm metadata。

回归矩阵：

| 测试 | 结果 | 说明 |
|---|---:|---|
| Python `py_compile` | 通过 | P1 相关实现与测试文件 |
| `tests/big_wave_2d/test_p1.py` | 11/11 通过 | P1 专项验收 |
| `tests/big_wave_2d/test_p0.py` | 17/17 通过 | P0 契约未被破坏 |
| `big-wave/test_2d_simulation.py` | 4/4 通过 | 既有 2D 数值与配置回归 |
| `big-wave/test.py` | 26/28 通过 | 两个失败与修改前基线一致：区间边界计数、旧 snapshot 金值 |

`tests/plasma_sample/test_plasma_phase_absorption.py --2d` 固定使用 2048×2048
整场并执行多轮 FFT2；运行超过 3 分钟仍未完成，因此按有界测试策略停止。这个结果不计为
P1 通过，也不计为 P1 回归失败：P1 专项已覆盖 PlasmaSample 的瓦片局部算子，完整大场 FFT2
仍需 P2 的 OOC backend 才能形成有效的大样例验收。

测试内的临时内存与 Plasma 性能结果：

| 项目 | 结果 |
|---|---:|
| 解析点源，8 行 tile，tracemalloc peak | 0.47 MiB |
| 解析点源，128 行 tile，tracemalloc peak | 5.03 MiB |
| Plasma 96×96 vectorized | 0.0009 s |
| Plasma 96×96 scalar loop | 0.0153 s |
| Plasma 向量化加速 | 16.3× |

## 独立进程 RSS 探针

命令模板：

```text
python tests/big_wave_2d/p1_benchmark.py --operator <analytic|sample|plasma> --nx 1024 --ny 1024 --tile-rows <8|128>
```

探针仅执行被测局部算子，并禁用后续 FFT，以免 P2 之前的整场 SciPy FFT2 掩盖局部算子的内存特征。

| 算子 | tile_rows | peak RSS | wall time | checksum 是否一致 |
|---|---:|---:|---:|---|
| analytic | 8 | 102.289 MiB | 0.088436 s | 是 |
| analytic | 128 | 108.438 MiB | 0.139607 s | 是 |
| Sample | 8 | 107.570 MiB | 0.239591 s | 是 |
| Sample | 128 | 125.426 MiB | 0.265918 s | 是 |
| PlasmaSample | 8 | 119.500 MiB | 0.561850 s | 是 |
| PlasmaSample | 128 | 138.879 MiB | 0.649471 s | 是 |

结论：临时内存随 `tile_rows × nx` 增长，而不再按完整 `ny × nx` 分配；不同 tile_rows 的数值 checksum 一致。大规模完整仿真仍受 P2 OOC FFT2 和 P3 流式 detector 限制。
