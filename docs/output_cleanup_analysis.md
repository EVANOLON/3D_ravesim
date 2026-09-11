# output/ 目录整理与清理分析报告

> 生成时间：2026-09-04（基于当时磁盘实际内容逐文件扫描，未删除任何文件）
> 所有路径相对于仓库根 `/mnt/d/rave-sim-main/rave-sim-main`。`output/` 已被 `.gitignore` 忽略，清理不影响版本库。

## 1. 总体情况（整理）

**总量：~505 GB，71,184 个文件**，其中 `.npy` 占 495 GB（98%）。

| 顶层区域 | 大小 | 内容 |
|---|---|---|
| `output/2025/`（07–12 月） | 6.8 GB | 2025 年的历史算例（每算例一个 `<时间戳>/` 目录） |
| `output/2026/`（01–08 月） | 310 GB | 按月归档的正式算例，最大头：06 月 122 GB、05 月 80 GB、07 月 53 GB |
| `output/_agent_runs/` | 149 GB | 自动化（agent）运行的拷贝/中间产物，**大量与 2026/06–07 归档重复** |
| `output/hdc_3layer_1um/` | 3.6 GB | 单层/多层 HDC 研究的散装网格与分析脚本 |
| `output/thin_w_absorption_1um/` | 1.2 GB | 薄壁吸收研究 |
| `output/output1`、`shockwave_1d_run`、`_demo_run` | 各 ~8 MB | 早期演示算例 |
| `output/hdc3_bigwave`、`hdc3_shot`、`hdc3_slice_fw` | 0 | **空目录** |
| `output/` 顶层散文件 | ~537 MB | 6 张分析 PNG、2 个 268 MB 的 `keypoint_00_*.npy`、0 字节的 `ppt.pptx` |

算例目录内部结构：`config.yaml` + `computed.yaml` + 输入网格（多 GB 的 `*_grid.npy` / `*_shell*.npy`）+ `000000NN/`（每帧的 `detected.npy` + `subconfig.yaml`）。部分 2026/04–05 算例还有 `wave_*.npy`（2.15 GB × ~10 个/算例）及 debug 文本导出。

## 2. 可清理项（按收益与风险分级）

### 🟢 T0 — 纯垃圾（收益 ~0.5 MB，无风险）
- `output/hdc3_bigwave`、`output/hdc3_shot`、`output/hdc3_slice_fw`：三个空目录。
- `output/ppt.pptx`：0 字节空文件。

### 🟢 T1 — debug 文本转储（收益 ~8.7 GB，无风险，npy 结果不受影响）
`output/2026/04/20260430_005657333092/00000000/wave_real_imag_2.txt`（**7.49 GB**）、`wave_real_imag_3.txt`（0.24 GB）、`wave_real_imag_4.txt`（0.30 GB）、`detected.txt`(0)；
`output/2026/05/20260517_023258291382/00000000/wave_real_imag_1.txt`（0.79 GB）、`detected.txt`（0.23 GB）。
这些是把波形写成文本的调试残留（同目录另有完整 npy），可安全删除。`wave_*.npy`（各 2.15 GB）是中间态结果，保留与否单独决策。

### 🟢 T2 — 内容级验证过的冗余整目录拷贝（收益 ~24.8 GB，已逐文件比对）
agent 运行机制会把算例整体复制为 `xxx__agentrun_<ts>`，下表右侧均为左侧目录的**子集/逐字节相同**拷贝（仅个别帧 `00000000/detected.npy` 因中途写入不同）：

| 保留 | 可删 | 收益 |
|---|---|---|
| `_agent_runs/june_l1_repro`（7.34 GB） | `_agent_runs/june_l1_repro__agentrun_20260903170606` | 7.34 GB |
| `_agent_runs/l1varT_dz2um_prep`（3.65 GB） | `_agent_runs/l1varT_dz2um_prep__agentrun_20260902165024` | 3.51 GB |
| `_agent_runs/vacuum_field_prep`（7.05 GB） | `_agent_runs/vacuum_field_prep__agentrun_20260902150426` | 6.91 GB |
| `2026/06/20260627_194236291561`（归档） | `_agent_runs/20260627_194236291561__agentrun_20260903150238` | 7.05 GB（唯一差异为 1 帧 detected，删除前建议抽查该帧是否有用） |

另外 C10 族小算例（各 ~0.2–0.3 GB）在 `thin_family_prep/C10__*` 与 `_agent_runs/C10__*__agentrun_*` 间互为拷贝（内容 4/5 文件相同），可任留一组，再省 ~0.8 GB。

### 🟡 T3 — 无损去重：相同内容的 npy 只留 1 份物理文件，其余改硬链接（收益 **~350 GB**，零数据丢失）
全库按（文件大小 + 头部/尾部 64 KB 指纹）聚类：≥1 MB 的 4448 个 npy 中，**2097 个（417 GB）可归入 673 个“同内容”组**。每组保留 1 份、其余替换为硬链接（已在 /mnt/d 上验证 `ln` 可用），可回收 **~349.9 GB**，所有目录路径、文件名、config 引用均不变。

典型大头（同内容出现在多个算例目录中，多为同一输入网格随算例重复复制，或同一算例的 agent 拷贝）：

| 同内容网格 | 份数 | 单份大小 |
|---|---|---|
| `500um_100um_shell_l1_m0_eps002.npy`（含 rot90 同名同内容） | 13 | 7.34 GB |
| `500um_100um_perfect_shell.npy` | 12 | 6.91 GB |
| `100um_half_hollow_millisphere*_grid.npy` | 11 | 4.01 GB |
| `500um_100um_shell_l1_varT_er002_et005.npy` | 5 | 7.02 GB |
| `thin_w_plate_5um/2um/1um_200um.npy` | 4 / 4 / 7 | 6.4 / 2.56 / 1.28 GB |
| `ne_grid.npy` 等 1D 网格 | 16 | 0.54 GB |
| `double_ring_911_844_803_3d.npy` | 13 | 0.33 GB |
| `thin_c_plate_10um/20um_25um.npy` | 10 / 4 | 0.20 / 0.40 GB |
| `ne/ni/te/zstar_2d_grid.npy` | 各 6 | 0.10 GB |

说明：硬链接去重按“内容指纹”聚类，理论上存在头部/尾部相同而中部不同的极小误判风险；对明显的整体拷贝与输入网格族基本为 0。若需绝对稳妥，可仅对跨目录整体拷贝（config 哈希一致）的组执行。
另外 `wave_before_sample/after_source/...npy`（2.15 GB × ~10/算例）中也存在跨算例完全相同的多份（同参数序列的确定性中间态），去重同样无损。

### 🟠 T4 — 需人工决策的历史/散落数据（合计 ~54 GB+）
- `output/2025/` 全部 **6.8 GB**：2025-07 起的历史算例。若确认已无引用价值可整树删除，否则保留。
- `output/` 顶层 `keypoint_00_0.npy`、`keypoint_00_1.npy`（2026-03-28，各 268 MB）：全库无同尺寸副本，是当时提取的中间文件；如流程不再需要可删。
- 顶层 6 张 PNG（`residual_*.png`、`four_shell_comparison.png`，~8.9 MB）：分析成图，可移入 `_agent_runs/plots/` 或删除。
- `output/output1`(8.9 MB)、`shockwave_1d_run`(7.8 MB)、`_demo_run`(7.8 MB)：早期演示/冒烟算例。
- `_agent_runs` 中 2026/06 系重复族（`june_l1_repro`、`l1_m0_eps002*`、`l1_varT*`、`l2_m0_eps002*`、`perfect_sphere*` 等 ~73 GB）：内容与 `2026/06/` 归档算例高度重合（相同 config + 相同输入网格，多为同一算例的续跑/裁剪），若以 `2026/06/` 为权威归档，可整体移除（需先确认各族的最终帧已包含在保留目录中）。

## 3. 建议执行顺序
1. T0 + T1：清空文件与 8.7 GB 文本转储（零风险）。
2. T2：删除 4 组内容级验证过的 `__agentrun_` 拷贝，回收 ~24.8 GB。
3. T3：全库 npy 硬链接去重，回收 ~350 GB（无损、可随时撤销——撤销即重新复制）。
4. T4：确认后处理 2025 树、根目录散文件与 `_agent_runs` 大族。

> 复核方法备忘：同内容判定 = `(size, md5(前64KB), md5(后64KB))` 一致；T2 目录级还逐文件比对过。全部操作可在 `output/` 内完成，`.gitignore` 已忽略该目录。

---

## 4. 执行结果（2026-09-04，用户已确认 T0+T1+T2+T3）

| 步骤 | 内容 | 结果 |
|---|---|---|
| **T0** | 删除空目录 `hdc3_bigwave/hdc3_shot/hdc3_slice_fw`、0 字节 `ppt.pptx` | ✅ 完成 |
| **T1** | 删除 debug 文本转储（`wave_real_imag_*.txt` ×4 + `detected.txt` ×2，共 ~8.7 GB） | ✅ 完成 |
| **T2** | 删除 4 组冗余 `__agentrun_` 拷贝（`june_l1_repro__agentrun_*`、`l1varT_dz2um_prep__agentrun_*`、`vacuum_field_prep__agentrun_*`、`20260627_194236291561__agentrun_*`） | ✅ 完成（删除前已抽查：被删目录中差异帧 `00000000/detected.npy` 为 ~1e-13 噪声帧，保留目录中为有效信号 ~1e-7，无有价值数据丢失） |
| **T3** | npy 硬链接去重：内容指纹 = ≤64 MB 全量 md5；>64 MB 用（size + 头部/25%/50%/75%/尾部 64 KB 采样）。686 个同内容组，1,351 处替换为硬链接，回收 **304.9 GB**，0 失败；事后抽查 1,351 对均同 inode | ✅ 完成 |

**T3 关键设计修正**：早期两窗口（头+尾）调查曾把 `vacuum_field_prep` 的 `perfect_shell.npy` 误并入其它 10 份（同为 6.91 GB、头尾一致）——多窗口指纹正确识别出它内容不同（`433c7148` ≠ `6aa5c070`）而未做链接。因此正式执行采用了 5 窗口采样，规避了"网格零边距导致头尾相同但中部不同"的误判风险。执行后对 June–July 跨月同名列的大文件另做全量 md5 抽验（结果见下文状态）。

### 清理后状态
- `output/` 总量：**~505 GB → ~170–215 GB**（逻辑回收 **~338 GB** = T1 8.7 + T2 24.8 + T3 304.9；`du` 按 inode 去重统计且在 WSL/drvfs 上有波动，多次实测 181–215 GB）。
- `_agent_runs`：149 GB → ~38 GB（T2 删除 + T3 去重）。
- `2026/` 月归档：310 GB → ~137 GB（主要为 June/May 输入网格多副本去重）。
- 全部路径、文件名、`config.yaml` 引用保持不变；被链接文件仍可通过原路径正常访问。抽验：3 组跨目录/跨月大网格全量 md5 一致（June l1_m0、June↔July perfect_shell、July↔agent thin_w）。

### 遗留（T4，未执行，需人工决策）
- `output/2025/`（6.8 GB 历史算例）、根目录 `keypoint_00_*.npy`（536 MB）、顶层分析 PNG、`output1/shockwave_1d_run/_demo_run` 演示目录、`_agent_runs` 中与 `2026/06/` 归档高度重合的 perfect_sphere/l1_m0/l2_m0/varT 大族（~70 GB，含各自独有的中间帧，删除前需确认各帧在别处有备份）。
