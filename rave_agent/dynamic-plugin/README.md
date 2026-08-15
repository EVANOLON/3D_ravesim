# 动态插件（内嵌查看器版）说明

`rave_agent/dynamic-plugin/` 保存「对话窗口内嵌结果查看器」版本的插件源码（Host+Client 完整版）。
这是 RAVE-SIM 全部 6 个业务工具 + 内嵌 plot 查看器的**本体**。

## 文件说明

| 文件 | 内容 |
|---|---|
| `host.js` | Host 半边：7 个工具（validate / feasibility / run / status / summary / plot / grid plot）+ `rave-plot-png` RPC |
| `client.js` | Client 半边：`tool.call.toolview` 注册 `rave_result_plot` / `rave_grid_plot` 卡片，对话窗口内嵌显示 PNG |

## 对话内嵌图片/网格（markdown 方式）

除对话卡片外，所有绘图工具还会返回一个 markdown 可嵌入的 `url`
（`http://127.0.0.1:8811/<file>.png`）。Agent 在回复里用
`![标题](url)` 即可直接把图片/网格渲染在对话流中。

- 图床服务：`rave_agent/plot_server.py`（自愈：端口被占则复用，未启动则拉起，
  服务 `output/_agent_runs/plots/`）。bootstrap 工具 `rave_plot_server` 无需激活，
  会话第一轮即可调用（persona 已指示每会话启动时 ensure 一次）。
- 结果图：`rave_result_plot`（调用 `rave_agent/plot_result.py`）。
- 网格图：`rave_grid_plot`（调用 `rave_agent/grid_plot.py`，读取
  config.yaml `elements[].grid_path`，材料索引 → 密度 g/cm³ 渲染，2D 网格按
  (z, x) 约定标注 µm 坐标轴）。

> 预设同步：RAVE-SIM agent preset 的 persona（`~/.dsh/.agent-presets/rave-sim/agent.cordis.yml`）
> 也包含本功能指令（启动时 `rave_plot_server ensure`、结果用 `![标题](url)` 内嵌）。
> 该文件在仓库外（DSH 本机配置），仓库内备份在 `rave_agent/preset/`（含恢复说明）；
> 修改 preset 后请同步 `cp ~/.dsh/.agent-presets/rave-sim/*.yml rave_agent/preset/` 并 git 提交。

## 如何激活（推荐：引导器自动激活）

RAVE-SIM preset 自带静态引导器（`rave_agent/rave-sim-tools/`），注册两个工具：

- **`rave_plugin_activate`**：读取本目录 host.js/client.js → 经 `dynamicCordisRunner` 定义并启动插件
  （首次需在 Run 卡片批准一次；重复调用自动复用已有插件）
- **`rave_plugin_status`**：查询插件是否激活

**新会话第一轮 agent 会自动调用 `rave_plugin_activate`**（persona 指令）；也可以手动对 agent 说"激活 RAVE-SIM 插件"。

## 手动激活（备用，无需 preset 引导器）

在有 cordis 工具的会话（如 cordis preset）中：

1. 读取 host.js / client.js 内容
2. `cordis_define`：kind "new"、idPrefix "rave"、code.host = host.js 函数体、code.client = client.js 函数体
3. `cordis_run`（mode "run"）→ 批准 Run 卡片

## 保持同步

- 修改工具逻辑后：当前会话用 `cordis_define`（existing）+ `cordis_run`（update）热更新；
  按协作约定提醒用户是否需要固化（同步本目录源码 + git 提交）。
- 静态引导器只负责激活，不含业务逻辑——业务逻辑的唯一事实源是本目录 + `rave_agent/validate_sim.py` / `plot_result.py`。
