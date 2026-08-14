# RAVE-SIM 工具型 Agent（MCP）详细设计

> 版本：v1.0（设计稿，暂不实现）
> 目标：以 MCP 工具型 Agent 驱动 X 射线全流程仿真工作流
> 前提：不改动 big-wave / fast-wave 核心物理代码，只做薄封装与编排

---

## 1. 目标与非目标

### 1.1 目标

- 用自然语言驱动 X 射线全流程仿真：
  需求 → 参数设计 → 样品网格 → 配置生成 → 仿真（1D/2D、多源、光谱）→ 后处理 → 物理验证 → 报告
- 以 MCP 工具（Tools）形式暴露仿真能力，Agent（LLM）通过工具调用编排
- 可复现：每次运行留下配置快照、决策日志、量化指标

### 1.2 非目标（本期不做）

- 不修改 big-wave / fast-wave 物理内核（仅封装现有 API/CLI）
- 不做多用户任务队列与资源调度（单机单 Agent）
- 不补齐引擎缺失能力（2D 光栅、2D precise_Sample、2D 核外模式等）

---

## 2. 总体架构

```
┌───────────────────────────────────────────────────┐
│  MCP Host（Claude Desktop / DSH / 自研前端）       │
│   └─ Agent（LLM + 系统提示 + knowledge 知识库）    │
└────────────────────────┬──────────────────────────┘
                         │ MCP 协议（JSON-RPC，stdio 或 streamable-http）
┌────────────────────────▼──────────────────────────┐
│  rave-sim-mcp server（Python 3.10+，FastMCP）      │
│  ├─ tools/      14 个工具（薄封装现有 API/CLI）     │
│  ├─ jobs/       后台任务注册表（仿真进程管理）       │
│  ├─ state/      session 状态（JSON）               │
│  └─ knowledge/  规则与常量（选型表、约束、容差）     │
└────────┬───────────────────────┬──────────────────┘
         │ Python API 调用          │ subprocess + CUDA
┌────────▼─────────┐   ┌───────────▼───────────┐
│ big-wave         │   │ fast-wave             │
│ (multisim.py     │   │ (fastwave 可执行文件,  │
│  config.py 等)   │   │  GPU 加速)             │
└──────────────────┘   └───────────────────────┘
         │ 读写                     │ 读写
         ▼                         ▼
┌───────────────────────────────────────────────────┐
│  仿真目录（事实源）：年/月/时间戳/源索引/           │
│  config.yaml · computed.yaml · subconfig.yaml     │
│  detected.npy · u_XXXX.npy · keypoints · history  │
└───────────────────────────────────────────────────┘
```

**部署说明**

- MCP server 必须运行在装有 GPU、已编译 fast-wave、已配置 Python 环境（big-fourier 经 `maturin develop` 编译）的机器上
- Agent 本体（LLM）可在任意位置，通过 MCP 协议远程调用
- 仿真目录结构复用现有约定，不另起炉灶

---

## 3. 工具详细规格（14 个工具，6 组）

命名空间统一为 `rave.*`。

### 组 1：材料与物理

**1. rave.material.query**

| 项 | 内容 |
|---|---|
| input | `{element: "W"|"Al"|…, energy_eV: float, density_g_cm3?: float}` |
| 实现 | 封装 `nist_lookup`（Chantler 散射因子，SQLite `xrayref.db`） |
| output | `{delta, beta, mu_cm, attenuation_length_um}` |

**2. rave.plasma.query**

| 项 | 内容 |
|---|---|
| input | `{ne_cm3, ni_cm3, Te_eV, Zstar, Z, energy_eV}`（单点）或网格路径（批量） |
| 实现 | 封装 `big-wave/plasma.py::plasma_delta_beta()` |
| output | `{delta, beta, delta_free, delta_bound, beta_bound, beta_ff, atlen_cm}` |

### 组 2：样品与配置

**3. rave.grid.generate**

| 项 | 内容 |
|---|---|
| input | `{kind: "sphere"\|"hollow_sphere"\|"hollow_sphere_with_void"\|"non_spherical_shell"\|"thin_plate"\|"resolution_chart"\|"double_ring"\|"custom", dims: {nx, ny?, nz}, scale: {dx, dy?, dz}, material_id: int, density?: float, output_path}` |
| 实现 | 封装 `grid/grid_generation.py`、`grid/3d-spherical-shell-generation.py`、`grid/3d-non-spherical-shell-generation.py`、`grid/generate_thin_plate_grids.py` |
| output | `{path, shape, dtype, material_ids, stats}` |
| 备注 | 统一输出 uint32（对齐仓库 grid_type_unify_to_uint32 约定） |

**4. rave.config.build**

| 项 | 内容 |
|---|---|
| input | 结构化参数（JSON 化的 config.yaml）：sim_params、source、multisource、elements、dtype、engine 偏好 |
| 实现 | schema 校验后序列化为 YAML；元素类型白名单：`grating` / `env_grating` / `sample` / `precise_sample` / `plasma_sample` / `save_and_exit` |
| output | `{config_path, warnings}` |

**5. rave.config.validate**

| 项 | 内容 |
|---|---|
| input | `{config_path}` |
| 实现 | 复刻 `check_validity()` 规则（z 单调、相步进数跨元件一致、探测器参数为正、元件不越过探测器）+ `config.parse` 全量试解析 + 1D/2D 模式一致性检查 |
| output | `{ok, errors[], warnings[]}` |

**6. rave.feasibility.check**

| 项 | 内容 |
|---|---|
| input | `{config_path, engine, gpu_mem_gb?, disk_gb?}` |
| 实现 | Nyquist 检查（`propagation.grid_density_check` / `max_dx`）；N 为 2 的幂检查（big-fourier `split_sizes` 断言）；显存估算（1D：u+U+快照+探测器；2D：nx·ny·sizeof(complex)·(2~3)）；big-wave 磁盘估算；时长粗估 |
| output | `{ok, checks: [{name, ok, detail}], est_vram_gb, est_disk_gb, est_time_min}` |

### 组 3：执行

**7. rave.sim.setup**

| 项 | 内容 |
|---|---|
| input | `{config_path, save_dir, engine_hint?}` |
| 实现 | 调用 `multisim.setup_simulation(dct, config_dir, save_dir)`（内部 `setup_sim_dir` 生成时间戳目录、写 `computed.yaml`、逐源 `subconfig.yaml`） |
| output | `{sim_dir, nr_sources, cutoff_angles, source_points, energy_range}` |

**8. rave.sim.run**

| 项 | 内容 |
|---|---|
| input | `{sim_dir, source_idx?: int\|list, engine: "big-wave"\|"fast-wave", gpu_id?: int, history_dz?: float, save_keypoints?: bool, timeout_s?}` |
| 实现 | big-wave：subprocess `python main.py <config>` / `python multisim.py`，细粒度可直调 `multisim.run_single_simulation`；fast-wave：subprocess `<build>/fastwave <sim_dir> -s <idx> [--history_dz …]`，env `CUDA_VISIBLE_DEVICES`；多源展开为多个后台 job（天然并行） |
| output | `{job_ids: [{source_idx, job_id}], sim_dir}` |

**9. rave.sim.status / rave.sim.kill**

| 项 | 内容 |
|---|---|
| input | `{job_id}` / `{job_ids}` |
| 实现 | jobs 注册表（JSON 持久化）+ spdlog 日志尾部解析 |
| output | `{status: running\|done\|failed\|killed, progress?, log_tail, exit_code?, elapsed_s}` |

### 组 4：结果

**10. rave.artifacts.inspect**

| 项 | 内容 |
|---|---|
| input | `{sim_dir}` |
| 实现 | 解析目录树、`computed.yaml`、`subconfig.yaml` |
| output | 文件清单（shape/dtype/size）、关键元数据（截止角、能量、相步进数） |

**11. rave.results.read**

| 项 | 内容 |
|---|---|
| input | `{path, mode: "summary"\|"slice"\|"stats", slice?: {…}}` |
| 实现 | numpy memmap 读取（DiskVector 大文件安全）；**默认只返回 shape/dtype/min/max/mean/std/分位数**；slice 模式降采样（≤1024 元素/像素）；可生成缩略图 PNG |
| 设计原则 | **禁止将全量 16k×16k 数组送入 LLM 上下文** |
| output | `{shape, dtype, stats, slice?, thumbnail_path?}` |

**12. rave.results.plot**

| 项 | 内容 |
|---|---|
| input | `{sim_dir, kind: "detector_1d"\|"detector_2d"\|"history_xt"\|"phase_stepping"\|"compare", …}` |
| 实现 | 封装 `big-wave/visualize.py` + matplotlib；输出 PNG 路径 |
| output | `{png_path}` |

### 组 5：验证

**13. rave.validate.physics**

| 项 | 内容 |
|---|---|
| input | `{sim_dir, kind: "beer_lambert"\|"phase_slope"\|"both", element?, thicknesses_um?}` |
| 实现 | 复刻 `results_analysis_zh.md` 方法：吸收——空场 I₀ 与样品 I 比值 → μ 最小二乘拟合 → 对比理论 μ=4πβ/λ；相位——`u_after/u_before` 复数除法 + 左右真空参考法 → Δφ 斜率拟合 |
| 判定阈值 | 吸收 μ 偏差 < 5%；相位斜率偏差 < 10%（仓库实测基线 1.84% / 0.04%） |
| output | `{metric, sim_value, theory_value, deviation_pct, pass, table}` |

**14. rave.experiment.compare**

| 项 | 内容 |
|---|---|
| input | `{sim_output, exp_output, format: "txt_cols"\|…, normalization: "max"\|"integral"}` |
| 实现 | 读取 `post-processing/` 的 outputactual/outputsimu 文本列格式，插值对齐后计算 RMSE / R² / 峰值比 |
| output | `{rmse, r2, peak_ratio, png_path}` |

---

## 4. 状态模型

`session.json`（server 维护，与 sim 目录并存）：

```json
{
  "session_id": "uuid",
  "created_at": "ISO8601",
  "intent": "自然语言需求原文",
  "decisions": [
    {"step": "engine_select", "choice": "fast-wave", "reason": "2D 面探测器需求"},
    {"step": "grid_design", "choice": "nx=4096, dx=1e-6", "reason": "Nyquist 上限 2.1e-6"}
  ],
  "sims": [
    {"sim_dir": "…", "source_idx": 0, "job_id": "…", "status": "done",
     "metrics": {"mu_dev_pct": 1.84}, "config_sha": "…"}
  ],
  "artifacts": {"plots": ["…"], "reports": ["…"]}
}
```

**规则**：sim 目录永远是事实源，session.json 只是索引；"继续仿真"（vectors 模式）一律通过 `computed.yaml` + `u_XXXX.npy` 完成，不另存中间状态。

---

## 5. Agent 系统提示与知识库

系统提示包含四块（随 server 提供，位于 `rave_agent/knowledge/`）：

1. **SOP（8 段工作流）**：需求解析 → 参数设计与可行性 → 材料/样品 → 配置生成校验 → 执行 → 后处理可视化 → 物理验证 → 迭代报告；每段标注"必用工具"
2. **选型决策表**：引擎 × 维度 × 元件能力矩阵（含 2D 限制清单：仅 Sample/PlasmaSample、无光栅/precise_Sample/核外）
3. **硬约束清单**：N 为 2 的幂（big-fourier 要求）；dx 满足 Nyquist；相步进数跨元件一致；z 单调不越探测器；1D `1/√r` vs 2D `1/r`；材料因子含 `(δ+iβ)` 共轭
4. **故障字典**（报错信息 → 诊断 → 修复动作）：

| 报错/现象 | 诊断 | 修复 |
|---|---|---|
| `Nyquist not satisfied, dx too large` | 初始解析传播采样不足 | 减小 dx 或增大 N（feasibility.check 给上限） |
| `Grid sampled to coarsly …` | 高能波无法覆盖探测器 | dx ≤ 2/(sin θ/λ_max) |
| CUDA OOM | 显存超限 | 降 N / 降 nx·ny / c8 精度 / 换 GPU |
| `Number of phase steps is not consistent` | 元件相步进数不一致 | 统一 x_positions 长度 |
| `History is full` | 历史容量不足 | 减小 history_dz 或加大容量 |
| `Last optical element ends at z …` | 元件越过探测器 | 调整 z_start/厚度 |

---

## 6. 护栏与资源管理

| 项 | 设计 |
|---|---|
| 预算上限 | server 级配置：max_N（1D）、max_nx·ny（2D）、max_sources、max_parallel_jobs、gpu_mem_limit_gb、disk_limit_gb |
| 强制预检 | `rave.sim.run` 内部强制先执行 `feasibility.check`，失败即拒绝运行 |
| 后台任务 | 仿真一律后台 job；`status` 轮询；超时 kill；失败返回 `log_tail` 而非盲目重试 |
| 并行 | 多源 fan-out（每源独立 job），受 max_parallel_jobs 限制 |
| 权限 | server 需工作区写权限 + GPU 访问；只读工具（inspect/read/plot）可降权运行 |

---

## 7. MCP Server 技术选型与项目结构

- **语言**：Python 3.10+
- **MCP 框架**：`mcp`（FastMCP）；传输支持 stdio（Claude Desktop/DSH）与 streamable-http（自研前端）
- **进程管理**：`asyncio.create_subprocess_exec`
- **npy 读取**：numpy memmap（大文件安全）

```
rave_agent/
├── pyproject.toml
├── server.py            # FastMCP 入口，注册 14 工具
├── tools/
│   ├── material.py  plasma.py  grid.py  config.py  feasibility.py
│   ├── sim.py  artifacts.py  results.py  validate.py  experiment.py
├── jobs/
│   ├── registry.py      # job 注册表（JSON 持久化 + 日志 tail）
│   └── runners.py       # big-wave / fast-wave 进程封装
├── state/session.py     # session.json 读写
├── knowledge/           # 选型表/约束/容差/故障字典（YAML，注入 prompt）
└── tests/               # 每工具 pytest（含负例：Nyquist 报错路径）
```

---

## 8. 验证闭环细节

- **Beer-Lambert 工具**：读空场与样品 `detected.npy` 中心 ROI → I/I₀ → ln(I₀/I)=μt 最小二乘 → 与理论 μ=4πβ/λ 对比
- **相位工具**：需要 `save_debug_wavefields`（或 `u_XXXX.npy`）→ 复数除法逐像素归一化 → 左右真空参考法 → Δφ 斜率拟合（注意：球面波前曲率 ~10⁵ rad 量级，必须归一化）
- **超阈值处置**：Agent 不得静默通过，应检查能量/厚度/材料/网格配置并给出诊断

---

## 9. 端到端场景示例（验收用例）

**场景 A（1D 吸收验证，P2 验收）**
用户："验证 10keV X 射线穿过 2μm 钨板的吸收，与 Beer-Lambert 对比"
Agent 路径：`material.query(W,10keV)` → `grid.generate(thin_plate)` → `config.build(1D)` → `config.validate` + `feasibility.check` → `sim.setup` → `sim.run(fast-wave)` → `results.read` → `validate.physics` → 报告（期望 μ 偏差 < 5%）

**场景 B（2D ICF 胶囊投影，P4 验收）**
用户："模拟 8keV 点源对 500μm/100μm 空心球壳的 2D 投影，探测器 0.85mm@200nm"
Agent 路径：`feasibility.check`（16k×16k 显存估算）→ `grid.generate(hollow_sphere 3D)` → `config.build(is2d)` → fast-wave 运行 → `results.plot(detector_2d)` + 对称性验证

**场景 C（多源光谱，P4 验收）**
用户："用 spectrum_microX.h5 谱，100 个源点，1D Talbot-Lau 干涉仪"
Agent 路径：`config.build(multisource points + spectrum)` → `sim.setup` → 100 jobs fan-out（max_parallel=8）→ 汇总统计

---

## 10. 实施计划（共约 5–8 个工作日）

| 阶段 | 交付 | 验收标准 |
|---|---|---|
| P0 脚手架（0.5d） | server.py + jobs/registry + 2 个探路工具（material.query、sim.run fast-wave） | 手工跑通 W 板 1μm 单源 |
| P1 工具集（2d） | 全部 14 工具 + pytest | 单工具测试覆盖，含负例路径 |
| P2 Agent 接入（1d） | knowledge/ + 系统提示 + session 状态 | 场景 A 自然语言全自动通过 |
| P3 验证闭环（1d） | validate.physics + experiment.compare + 报告生成 | 自动复现 μ 偏差 <5%、相位 <10% |
| P4 并行与 2D（1–2d） | 多源 fan-out、状态监控、2D/等离子体场景 | 场景 B、C 通过 |

---

## 11. 风险与对策

| 风险 | 对策 |
|---|---|
| 大数组撑爆 LLM 上下文 | results.read 强制 summary/slice/缩略图，工具层限制返回字节数 |
| 配置错误信息不友好 | config.validate 前置 + 故障字典映射人话诊断 |
| GPU 抢占 / 长任务挂死 | 每 job 显存预算 + 超时 kill + 日志诊断 |
| 2D 引擎限制被误用 | knowledge 选型表 + config.validate 内置引擎×元件能力检查 |
| DiskVector 磁盘爆满 | feasibility 磁盘估算 + 运行后清理 scratch |

---

## 12. 开放问题（待决策）

1. MCP Host 选型：Claude Desktop / DSH 前端 / 自研？
2. server 与 GPU 是否同机（同机直连 vs 异机需远程执行层）？
3. 报告生成是否对齐现有 `PPT文案_代码功能介绍.md` / Word 模板？
4. 实验对比数据格式是否固定为 `post-processing/` 的 txt 双列格式？
5. 是否需要把本设计中的 14 工具进一步精简为 P0 最小集（6 个）先行验证？
