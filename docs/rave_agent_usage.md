# RAVE-SIM Agent 使用文档

> 最后更新：2026-08-15
> 本文档说明如何在 DSH 中使用 RAVE-SIM 仿真 Agent（GPU 仿真工具 + 物理/硬件可行性预检），以及重启后如何启用。

---

## 0. 协作约定（重要）

**热更新后必须提醒固化**：每次通过动态插件完成热更新（`cordis_define` 追加版本 / `cordis_run update`）后，
Agent 必须主动询问用户是否需要固化，固化清单：

1. 工具逻辑同步进 `rave_agent/rave-sim-tools/lib/index.js`（重启后 preset 加载新版）
2. 动态插件源码更新 `rave_agent/dynamic-plugin/host.js` + `client.js`（如需内嵌查看器）
3. `git commit`（可选但推荐）

用户回复"不固化"时，热更新成果仅存于当前进程，重启后丢失（需用户知晓）。

---

## 1. 简介

RAVE-SIM Agent 为 DSH 会话提供 5 个工具，覆盖 **X 射线波传播仿真的执行环节**：

| 工具 | 功能 | 状态 |
|---|---|---|
| `rave_config_validate` | 物理/逻辑校验（7 项检查） | ✅ 可用 |
| `rave_feasibility_check` | 物理校验 + 硬件可行性（显存/磁盘/时长） | ✅ 可用 |
| `rave_sim_run` | 前置门 → 白名单复制 → GPU 后台运行 fastwave | ✅ 可用 |
| `rave_sim_status` | 轮询后台 job（状态/日志/退出码） | ✅ 可用 |
| `rave_result_summary` | 读取 npy 结果统计（shape/dtype/min/max/mean） | ✅ 可用 |
| `rave_result_plot` | 渲染 npy 为 PNG（1D 曲线 / 2D 热图自动识别） | ✅ 可用 |

**设计原则**：
- 仿真执行前**强制**物理 + 硬件可行性检查（不通过拒绝启动）
- 仿真目录**复制后运行**，绝不修改原 sim 目录
- 复制目标**只允许** `output/` 目录（白名单，路径规范化防绕过）
- 仿真在**后台**运行，工具立即返回 `job_id`，用 `rave_sim_status` 轮询

---

## 2. 快速开始（重启后启用）

### 2.1 重启 DSH

```powershell
# Windows PowerShell 中
wsl --shutdown
```

然后重新打开终端启动 `dsh web`，浏览器访问 `http://127.0.0.1:3080`。

### 2.2 新建会话并选择 RAVE-SIM 预设

1. 在 dsh web 界面**新建会话**
2. 预设/模式选择器中选 **`RAVE-SIM`**（基于"标准模式"全能力 + 仿真引导器）
3. 会话第一轮 agent **自动调用 `rave_plugin_activate`** 激活完整动态插件：
   - **首次**：会出现 Run 卡片，**批准一次**（Client 半边）→ 6 工具 + 内嵌查看器生效
   - **后续**：插件已激活则跳过

> 激活前仅有引导工具（`rave_plugin_activate` / `rave_plugin_status`）；激活后获得全部 6 个业务工具。

### 2.3 验证就绪

对会话中的 agent 说：

> 运行 `rave_feasibility_check` 检查 `/mnt/d/rave-sim-main/rave-sim-main/output/2026/08/20260814_231146680024` 是否可跑

或直接用自然语言：

> 用 GPU 跑一遍 shockwave 1D 仿真，先做可行性检查，结果放 output 下

---

## 3. 工具详细说明

### 3.1 `rave_config_validate` — 物理/逻辑校验

**输入**：`sim_dir`（仿真目录，含 config.yaml / computed.yaml）

**7 项检查**：

| 检查 | 说明 |
|---|---|
| `config_parse` | YAML 与 sim_params 可解析 |
| `power_of_two` | N（1D）/ nx、ny（2D）为 2 的幂 |
| `fov` | 仿真窗口覆盖探测器（N·dx ≥ detector_size） |
| `z_layout` | 元件 z 单调、不重叠、不越过探测器 |
| `cutoff_angles` | computed.yaml 中截止角单调递增且 ∈ [0, π/2] |
| `nyquist` | dx 满足 Nyquist 采样（按能量上限最严波长） |
| `phase_steps` | 多元件相步进数一致 |

**输出**：`{ ok, checks: [{name, ok, detail}], N, dx, energy_range, ... }`

### 3.2 `rave_feasibility_check` — 物理 + 硬件可行性

**输入**：`sim_dir`、`engine`（默认 `fast-wave`；`big-wave` 为 CPU/核外模式）

**额外输出**：
- `gpu.est_vram_gb`：显存估算（波场 u+U+工作区 ≈ 点数 × 8 B × 3）
- `gpu.free_gb`：nvidia-smi 实测可用显存（按 85% 可用计算余量）
- `disk`：核外模式磁盘估算（`use_disk_vector=true` 时）
- `est_time_min`：时长粗估（按 2.68e8 点 / ~3 min 标定，2D 加倍）
- `feasible` + `blocked_by`：是否可执行、被哪项阻塞

### 3.3 `rave_sim_run` — 执行仿真（带前置门）

**输入**：
| 参数 | 必填 | 说明 |
|---|---|---|
| `sim_dir` | ✅ | 仿真目录绝对路径 |
| `source_idx` | — | 源点索引，默认 0 |
| `fastwave_path` | — | 默认 `fast-wave/build-Release/fastwave` |
| `target_dir` | — | **必须**在 `output/` 内；默认 `output/_agent_runs/<名字>__agentrun_<时间戳>` |
| `skip_check` | — | `true` 时跳过可行性门（仅确认已知合法时使用） |

**执行顺序**：
1. 白名单检查（target_dir 必须在 `output/` 下）
2. **可行性门**（物理 + 显存，不通过 → 拒绝，返回 `blocked_by` 与失败项）
3. GPU 自检（nvidia-smi）
4. 复制仿真目录到独立副本
5. 后台启动 fastwave，返回 `job_id`

**输出**：`{ job_id, run_dir, gpu, pid }`

### 3.4 `rave_sim_status` — 轮询

**输入**：`job_id` → **输出**：`{ status: running|done|failed, elapsed_s, exit_code, detected, log_tail }`

### 3.5 `rave_result_summary` — 读结果

**输入**：`.npy` 文件路径 → **输出**：`{ shape, dtype, min, max, mean, head }`（只返回统计，安全读取大数组）

### 3.6 `rave_result_plot` — 结果可视化

**输入**：`path`（npy 路径）或 `sim_dir`（自动取 `00000000/detected.npy`）

**两种展示方式**：
1. **对话窗口内嵌**（动态插件形态，需重新激活，见 `rave_agent/dynamic-plugin/README.md`）：工具卡片直接显示 1D 曲线 / 2D 热图
2. **新窗口 / 文件查看**（所有形态）：PNG 落盘 `output/_agent_runs/plots/<name>_<ts>.png`，返回路径可随时打开

**输出**：`{ png_path, shape, dtype, kind: "1d"|"2d", min, max, mean }`

---

## 4. 端到端工作流示例

```
用户：验证 8.9 keV 冲击波样品的 1D 仿真
Agent：
  ① rave_config_validate(sim_dir)      → 7 项检查全过
  ② rave_feasibility_check(sim_dir)    → est 6.0 GB < free 10.1 GB，feasible
  ③ rave_sim_run(sim_dir)              → job_id rv1（output/_agent_runs/ 下复制）
  ④ rave_sim_status(rv1) × n           → done, exit 0（约 3 分钟）
  ⑤ rave_result_summary(detected.npy)  → (1,179), min/max/mean
```

---

## 5. 安全边界与护栏

| 层 | 规则 |
|---|---|
| 物理门 | `rave_sim_run` 强制先跑可行性检查，失败即拒绝（`skip_check` 需显式） |
| 显存门 | 估算显存 > 85% 可用显存 → `gpu_vram` 阻塞 |
| 复制白名单 | 复制目标只允许 `output/` 下（规范化路径，`..`/`//` 绕过无效） |
| 原数据保护 | 仿真一律在副本上运行，原 sim 目录分毫不动 |
| 后台隔离 | job 在插件内存注册表管理；插件停止/更新会终止未完成任务 |
| git 审计 | 全部代码已提交：`792cc74`（基线）→ `d3e0e88`（工具）→ `f4e6906`（白名单） |

---

## 6. 架构与文件位置

```
rave_agent/                                   # 项目内（已 git 跟踪）
├── validate_sim.py                           # 校验脚本（物理 + 硬件检查，输出 JSON）
└── rave-sim-tools/                           # 持久化 Cordis 插件（5 工具）
    ├── package.json
    └── lib/index.js                          # 工具实现（ESM，注入 subprocess/fs/tools）

~/.dsh/.agent-presets/rave-sim/               # Agent 预设（重启后选此）
├── preset.yml                                # 名称：RAVE-SIM
└── agent.cordis.yml                          # standard 全能力 + rave-sim-tools 行
```

**两种形态（同一套工具逻辑）**：
- **静态引导器**（`rave_agent/rave-sim-tools/`，RAVE-SIM preset 自带）：只注册 `rave_plugin_activate` / `rave_plugin_status`——重启后新会话第一轮自动激活完整插件（persona 引导）
- **动态插件本体**（`rave_agent/dynamic-plugin/`：host.js + client.js）：6 业务工具 + 内嵌查看器，由引导器定义并运行；进程内热更新迭代（`cordis_define` + `cordis_run`）

**依赖环境**：
- Python 3.11 conda 环境 `rave-sim`（含 bfpy、numpy、ruamel.yaml）
- fastwave 二进制：`fast-wave/build-Release/fastwave`（与源码同步，Aug 7 构建）
- GPU：RTX 5070 Ti（12 GB）——插件子进程直接访问 `/dev/dxg`，无需沙箱升级

---

## 7. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `target_dir rejected` | 复制目标不在 `output/` 下 | 显式传 `output/` 内的 target_dir，或省略用默认 |
| `feasibility gate REJECTED` | 物理/显存检查未过 | 看 `blocked_by`：`power_of_two`→N 改 2 的幂；`nyquist`→减小 dx；`gpu_vram`→降 N 或换 c8 |
| `GPU unavailable` | 沙箱 /dev 隔离 | 该工具在 DSH Host 进程内运行，正常可达；若不可达检查 `/usr/lib/wsl/lib/nvidia-smi` 是否存在 |
| `unknown job_id` | 插件重启（update/stop/DSH 重启）丢了内存注册表 | 任务已无法轮询；检查 run_dir 下的 detected.npy 是否生成 |
| 新会话没有工具 | 预设未选 RAVE-SIM | 重新新建会话选择 RAVE-SIM；检查 `~/.dsh/.agent-presets/rave-sim/` 存在 |
| 仿真中途被杀 | 插件 update/stop 触发 job 终止 | 确认无更新操作后再跑长任务 |

---

## 8. 更新与维护（SOP）

### 8.1 更新机制速查

| 组件 | 加载方式 | 修改后何时生效 |
|---|---|---|
| `validate_sim.py` | 每次工具调用新起 python 进程执行 | **改完立即生效**（无需重启） |
| `rave-sim-tools/lib/index.js` | preset 按文件路径引用，会话启动时加载 | 重启/新会话生效；当前会话不热更新 |
| `agent.cordis.yml`（预设） | 会话启动时 mount | 重启/新会话生效 |
| 动态插件（当前会话 `rave-1`） | 进程内 immutable package | 必须 `cordis_define` 新版本 + `cordis_run update` |

### 8.2 按变更类型的操作流程

**场景 A：改校验逻辑（`validate_sim.py`）—— 零重启**

```bash
# 编辑 rave_agent/validate_sim.py
/home/taylor/anaconda3/envs/rave-sim/bin/python rave_agent/validate_sim.py --validate <sim_dir>   # 独立验证
git add rave_agent/validate_sim.py && git commit -m "feat: ..."
```
当前会话与重启后均立即生效。

**场景 B：改工具逻辑（`rave-sim-tools/lib/index.js`）**

```bash
# 编辑 lib/index.js
node --input-type=module -e "await import('file:///mnt/d/rave-sim-main/rave-sim-main/rave_agent/rave-sim-tools/lib/index.js')"  # 语法验证
git add rave_agent/rave-sim-tools/ && git commit -m "feat: ..."
# 重启 DSH → 新会话（RAVE-SIM preset）自动加载新版
```

**场景 C：新增工具** — 在 `lib/index.js` 加 `tools.register({...})`，按场景 B 验证提交；若需要预设配置再改 `agent.cordis.yml`；重启后新会话自动多出工具。

**场景 D：当前会话立即用新版（同步动态插件）**

动态插件为不可变 package，需追加版本：
```
1. cordis_define（kind: existing, pluginId: rave-1）→ 得到新 packageId
2. cordis_run（mode: update, packageId: 新id）→ 新 Run 激活
```
⚠️ update 会终止旧 Run 的进行中 job，长任务先等跑完。

### 8.3 更新后验证清单

| 层级 | 方法 |
|---|---|
| JS 语法 | `node --input-type=module -e "await import('file://...lib/index.js')"` |
| 校验逻辑 | 直接跑 `validate_sim.py --validate/--feasibility <sim>` |
| 预设可挂载 | 新会话选 RAVE-SIM 直接试（或临时插件调 `standingKeyFor`） |
| 端到端 | `rave_sim_run` 跑小 sim（或 `skip_check` 快速验证） |

---

## 9. 已知限制与后续

- 当前覆盖**执行环节**（5 工具）；全流程（材料查询/网格生成/配置构建/物理验证/报告）见 `docs/rave_agent_mcp_design.md`（v1）与 `docs/rave_agent_mcp_design_DSH_调整.md`（v2）
- job 注册表在内存：DSH 重启后无法恢复进行中的任务（可接受：仿真写入 detected.npy 即算完成，副本可复查）
- `rave_sim_run` 目前仅支持 fast-wave（GPU）；big-wave（CPU/核外）执行工具待扩展
- 2D 模式（`is2d`/`nx,ny`）的校验已支持（FOV/Nyquist/显存按 nx×ny 估算），但需在 2D 仿真上补充实测回归
