# big-wave 2D P2 验收记录

日期：2026-08-19  
环境：WSL，`/home/taylor/anaconda3/envs/rave-sim/bin/python`，bfpy 0.3.0 release build  
范围：事务式核外 FFT2/IFFT2、DiskVector 接入、进度/取消、pass 级恢复与资源校验。

## 实现结论

P2 的 FFT2 路径不再执行 `np.load + scipy.fft2`。`DiskVector` 调用 Rust bfpy，按以下
四个 pass 处理扁平 NPY 波场：

1. `row_fft`：逐行读取、FFT，写 `outfile.part`；
2. `transpose_to_scratch`：按内存预算分块转置；
3. `column_fft`：在转置布局中逐行执行原矩阵列 FFT，写独立 column part；
4. `transpose_to_output`：转置回原布局，校验和 fsync 后原子替换正式输出。

每个完整 pass 写入原子 manifest。中断中的 pass 从头重做；只有 manifest 已登记且
NPY header、dtype、shape、文件长度均有效的 pass 才会复用。输入文件只读，正式输出
在全部 pass 完成前不会变化。

## 自动测试

```text
cargo test -p big-fourier
22 passed

python tests/big_wave_2d/test_p2.py -v
11 passed
```

P2 Python 测试覆盖：

- c8/c16；
- 方形、2 的幂非方形，以及非整 tile 边缘尺寸；
- FFT2/IFFT2 与 SciPy 数值对照；
- 输入不变、已有正式输出的事务保护；
- 一维和二维 NPY shape、dtype、文件长度校验；
- 普通路径、符号链接/规范路径和 hardlink 冲突保护；
- Python progress callback 和 cooperative cancel token；
- 损坏的已登记 scratch 不复用；
- 分别在 row FFT、第一次转置、column FFT、最终转置结束点强制终止子进程，重启后恢复并得到正确输出；
- `DiskVector` 禁止整场 `np.load`；
- DiskVector 与 NumpyVector 传播结果对照；
- `setup_simulation → run_single_simulation → detected.npy` 小场端到端运行；
- P2 metadata 和 feasibility backend gate。

回归矩阵：

| 测试 | 结果 |
|---|---:|
| P0 契约测试 | 17/17 通过 |
| P1 局部算子测试 | 11/11 通过 |
| P2 OOC FFT2 测试 | 11/11 通过 |
| 原有 2D 脚本 | 4/4 通过 |
| 原有 `big-wave/test.py` | 26/28；两个修改前历史失败未增加 |

## 独立进程内存探针

命令：

```text
python tests/big_wave_2d/p2_benchmark.py \
  --nx 4096 --ny 4096 --dtype c8 --memory-budget-mib 8
```

探针在每个 pass 的完成回调内读取实时 `VmRSS`，确保转置 buffer 尚未释放。不能直接把
子进程的 `ru_maxrss` 当作结论，因为 Linux fork/exec 会保留输入生成进程的历史高水位。

| shape | dtype | 输入波场 | FFT2 tile 预算 | pass 内峰值 VmRSS | wall time |
|---|---:|---:|---:|---:|---:|
| 1024×1024 | c8 | 8 MiB | 8 MiB | 44.145 MiB | 0.187 s |
| 4096×4096 | c8 | 128 MiB | 8 MiB | 43.926 MiB | 1.693 s |

字段放大 16 倍时，FFT2 pass 内峰值 RSS 没有随整场大小增长；两次运行都完成全部 pass，
输出文件长度正确且 scratch/manifest 清理完成。

## 尚未解除的端到端限制

P2 只解除 FFT2 的整场 RAM 峰值。当前 `legacy_fastwave` 二维 detector 仍构造完整强度和
二维前缀和，History 也仍保留完整帧。因此：

- 可以用 `bfpy_ooc + DiskVector` 执行内存有界 FFT2；
- 尚不能据此宣称 16384×16384 完整成像链路可稳定运行；
- 下一发布门仍是 P3 流式 detector，随后才适合做完整大样例验收。
