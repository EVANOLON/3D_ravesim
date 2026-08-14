# RAVE-SIM Agent 设计修订（v2）：基于 DSH + 本地 DS 模型的决策调整

> 依据的用户决策：① MCP Host 采用 DSH + DS 基本框架；② server 与 GPU 同机（后续考虑扩展）；③ 报告采用自拟分析模板；④ 实验对比数据待定。
> 本文是对 `docs/rave_agent_mcp_design.md`（v1）的修订与细化；冲突处以本文为准。
> 全部 DSH 事实均实测自本机 DSH rc.6（`@deepseek-ai/dsh` 0.1.0-rc.6）及其插件文档。

---

## 1. 总体架构调整：Host 具体化为 DSH

v1 把 MCP Host 写成抽象层，现明确三件套：

- **Host = DSH**（`dsh web`，Web GUI 默认 `127.0.0.1:3080`，暴露 `DSH_WEB_URL`）
- **Agent 模型 = 本地部署的 DeepSeek 模型**：经 vLLM 等 OpenAI 兼容服务暴露，DSH 的 `@deepseek-ai/dsh-llm-deepseek` 适配器 `baseURL` 指向本地端点
- **rave-sim-mcp server = 独立 MCP 进程**（Python FastMCP，stdio），经 DSH 的 `@deepseek-ai/dsh-mcp-client` 插件桥接

注册方式（写入 profile 或 `$DSH_HOME` 的 `cordis.patch.yml`）：

```yaml
- id: mcp-rave
  name: '@deepseek-ai/dsh-mcp-client'
  config:
    serverName: rave
    transport: stdio
    command: python
    args: ['-m', 'rave_agent.server']
    cwd: /path/to/rave-sim
    env:
      CUDA_VISIBLE_DEVICES: '0'
    toolCallTimeoutMs: 30000   # 只覆盖“提交类”工具；长仿真走 job 异步
    failOnStartupError: true
```

模型侧看到的工具名形如 `mcp__rave__material_query`。

---

## 2. 工具规格修订（逐条，重要）

### 2.1 工具命名：`rave.x.y` → snake_case raw name

- DSH 将公开名规范化为 DeepSeek 函数名约定（≤64 字符、`[A-Za-z0-9_-]`）；`.` 等字符会被替换并追加确定性 12 位 hash，名字难读。
- 修订：raw name 全用 snake_case——`material_query`、`plasma_query`、`grid_generate`、`config_build`、`config_validate`、`feasibility_check`、`sim_setup`、`sim_run`、`sim_status`、`sim_kill`、`artifacts_inspect`、`results_read`、`results_plot`、`validate_physics`、`experiment_compare`、`report_render`（新增）。
- 命名空间由 `serverName: rave` 承担；serverName 一经确定不再改（工具名是 `(serverName, rawName)` 的纯函数，改名会整体替换工具集合、浪费 KV cache 前缀）。

### 2.2 长任务与 `toolCallTimeoutMs`

- DSH MCP 调用默认超时 **60 秒**，而仿真动辄分钟~小时级。
- 修订（v1 已隐含，现在写死）：`sim_run` 只做**提交**并同步返回 job_id（<30s），后续一律 `sim_status` 轮询；任何 MCP 工具调用都不得阻塞等待仿真结束。`toolCallTimeoutMs: 30000` 只保护提交类工具；`sim_status` 读日志尾部同样受此约束。

### 2.3 图片通道：MCP image 块不能给模型看图

- DSH MCP 客户端会把图片块在模型上下文中替换为占位符（原生渲染有损），模型**看不到** PNG 内容。
- 修订：`results_plot` / `results_read` 的产出定位为"给用户看 + 给模型数值"：
  a) PNG 落盘到 sim/输出目录，返回文件路径，用户在 DSH Web GUI 文件面板查看；
  b) 模型判断依据数值特征（min/max/mean/分位、切片数值曲线、对称性/峰值比指标），而非图片。
- 若将来需要模型真正读图，应走 DSH 原生附件（attachments/image limits）而非 MCP 通道，需单独评估。

### 2.4 `outputSchema` 用 DSH 支持的保守词汇子集

- DSH 对已声明的 MCP `outputSchema` 用 harness 支持词汇校验；不支持的词汇回退为 `JsonValue`（不报错但失去结构化）。
- 修订：工具输出 schema 只使用保守词汇（object 根 + `type/properties/required/additionalProperties/items/enum/const`），不用 `pattern/format` 等高级词汇。

### 2.5 护栏执行点必须在 server 内（同机部署下最关键的一条）

- DSH Agent 同时拥有原生 bash/read/write 工具，**它可以直接调 fastwave/multisim 绕过 MCP 护栏**；同机部署使这条路径更短（bash 就在仿真机上）。
- 修订：
  - feasibility 强制预检、预算上限、job 注册表、目录白名单全部在 rave_agent server 内实现；server 配置 `allowed_roots`（workspace + sim 输出目录），写路径越界即拒绝；
  - knowledge/SOP 明确写死：**仿真语义操作必须经 `mcp__rave__*` 工具，不得用 bash 直调 CLI**。

### 2.6 新增工具 `report_render`（第 16 个）

- 决策"报告自拟分析模板"落地：
  - `knowledge/report_template.md`：自拟模板（概述 / 实验设计 / 结果表 / 结论 / 文件产出，风格对齐 `results_analysis_zh.md`）；
  - `report_render`：输入 sim_dir + 指标 JSON，渲染报告写入 sim 目录，返回路径；图引用一律相对路径。

### 2.7 实验对比待定 → `experiment_compare` 改适配器模式

- 本期实现"接口 + 格式适配层 + 空驱动"；内部标准格式定为 `{x, sim, exp}` 三列（CSV/JSON），`format` 参数预留。
- 数据格式确定后只补驱动、不动接口；验收用例（场景 A/B/C）不依赖此工具。

### 2.8 多源并行 fan-out 载体

- 修订：由 rave_agent server 的 **job 注册表**统一管理（`sim_run` 接受 `source_idx` 列表 → 展开 N 个 job，受 `max_parallel_jobs` 限制）。
- 不用 DSH 原生 subagent/workflow 直接并行跑仿真——避免绕过 GPU 预算。DSH 的 goal 机制可用于"长任务会话管理"（如一次大型参数扫描作为一个 goal），执行仍走 MCP job。

### 2.9 配置热更新与开发效率

- DSH MCP 插件支持 HMR：编辑 cordis 配置触发断开+重连，无需重启 dsh；`serverName` 不变则工具名稳定（KV cache 友好）。
- 注意重连预算（默认 `maxAttempts: 10`、指数退避上限 30s）：server 崩溃循环会被注销工具，开发时留意日志中的 `reconnecting` / `disabled-loss`。

---

## 3. 模型接入层调整（本地 DS 模型）

- **首选** `@deepseek-ai/dsh-llm-deepseek`（路由 `deepseek-official`，DeepSeek chat-completions 协议）：
  ```yaml
  # cordis.patch.yml（或 settings.yaml 的 llm-deepseek: 分节，可热改）
  - id: llm-deepseek
    name: '@deepseek-ai/dsh-llm-deepseek'
    config:
      baseURL: http://127.0.0.1:8000/v1     # 本地 vLLM OpenAI 兼容端点
      apiKeyEnv: LOCAL_LLM_KEY              # 本地无鉴权也给“合法格式”占位密钥
      models:
        - id: <本地模型id>
          name: Local-DeepSeek
          contextWindow: <本地上下文>
          maxTokens: <本地输出上限>
  ```
  - 注意：DSH 会校验密钥格式（无效格式直接 `INVALID_CREDENTIAL`），占位密钥须形如合法 token；`agent-default-model`（dsh-base 默认 `deepseek-official/deepseek-v4-flash`）需改为本地组合。
- **备用**：`@deepseek-ai/dsh-llm-pi-ai` 的 OpenAI 兼容路由（省略 `apiKeyEnv` 即未认证状态）。
- **必过兼容清单**（详见部署报告第 4 节）：流式 SSE（适配器只支持流式）、`tool_calls`、含工具调用轮次的 `reasoning_content` 回传、顶层 `reasoning_effort` 字段兼容性、`finish_reason`、`usage`。

---

## 4. 同机部署布局与 GPU 预算

```
rave-sim/                 # 仓库根
└── rave_agent/           # MCP server 源码（tools/jobs/state/knowledge）
~/.dsh/                   # DSH 用户数据
├── profiles/web/         # web profile（cordis.patch.yml 追加 mcp-rave 行）
├── settings.yaml         # 用户设置（llm-deepseek 分节可热改 baseURL/models）
└── .credentials.yaml     # 占位密钥
simulations/…             # 仿真输出（setup_simulation 生成）
```

- 传输：**stdio** 即可；streamable-http 留作将来多机扩展（runner 接口保持抽象、transport 可切换，设计不变）。
- **GPU 争用是新风险**：本地模型常驻显存 + fast-wave 仿真峰值显存共享同一 GPU（本机实测环境为 RTX 5070 Ti Laptop，16GB）。修订：
  - server 预算同时计入"模型常驻占用"与"仿真峰值"，`feasibility_check` 增加模型占用参数；
  - 优先量化模型降低常驻占用；必要时模型与仿真按时间段错峰，或模型走 CPU+GPU offload；
  - DSH 本身纯 CPU/Node，无 GPU 冲突。

---

## 5. 原计划保持不变的部分

P0–P4 分期、工具功能语义（除 2.x 修订项）、状态模型 `session.json`、知识库四块（SOP/选型表/硬约束/故障字典）、验证闭环阈值（μ<5%、相位<10%）、场景 A/B/C 验收——均沿用 v1。

---

## 6. 修订差异清单（v1 → v2）

| # | v1 | v2 |
|---|---|---|
| Host | 抽象 MCP Host | DSH（`dsh web`，127.0.0.1:3080） |
| 工具名 | `rave.material.query` | `mcp__rave__material_query`（raw name 为 snake_case） |
| 超时 | 未明确 | 提交类 30s；仿真一律 job 异步 + status 轮询 |
| 图片 | 返回 PNG 路径 | PNG 落盘 + 数值摘要；模型不读图（MCP 图片被占位） |
| outputSchema | 未约束 | DSH 保守词汇子集 |
| 护栏 | server 内 | server 内 + SOP 禁止 bash 直调 CLI（同机关键项） |
| 报告 | 未定 | 新增 `report_render` + `knowledge/report_template.md` |
| 实验对比 | 固定格式 | 接口 + 格式适配层 + 空驱动，格式待定 |
| fan-out | 未定载体 | server job 注册表，不经 DSH subagent |
| 模型 | 未定 | 本地 vLLM + `dsh-llm-deepseek`（`dsh-llm-pi-ai` 备用） |
| 传输 | stdio / http 二选 | stdio（同机）；http 留待扩展 |
| GPU 预算 | 只算仿真 | 叠加模型常驻占用与错峰策略 |
