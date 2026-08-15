# 动态插件（内嵌查看器版）说明

`rave_agent/dynamic-plugin/` 保存「对话窗口内嵌结果查看器」版本的插件源码（Host+Client 完整版）。
这是 RAVE-SIM 全部 6 个业务工具 + 内嵌 plot 查看器的**本体**。

## 文件说明

| 文件 | 内容 |
|---|---|
| `host.js` | Host 半边：6 个工具（validate / feasibility / run / status / summary / plot）+ `rave-plot-png` RPC |
| `client.js` | Client 半边：`tool.call.toolview` 注册 `rave_result_plot` 卡片，对话窗口内嵌显示 PNG |

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
