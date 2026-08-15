# 动态插件重新激活指南（重启 DSH 后）

`rave_agent/dynamic-plugin/` 保存了「对话窗口内嵌结果查看器」版本（Host+Client 完整版）的插件源码。
DSH 重启后动态插件定义会丢失，按以下步骤重新激活（约 1 分钟）。

## 文件说明

| 文件 | 内容 |
|---|---|
| `host.js` | Host 半边：6 个工具（validate / feasibility / run / status / summary / plot）+ `rave-plot-png` RPC |
| `client.js` | Client 半边：`tool.call.toolview` 注册 `rave_result_plot` 卡片，对话窗口内嵌显示 PNG |

## 激活步骤

在新会话中对 agent 说（或直接粘贴）：

1. **定义插件**：让 agent 读取 `rave_agent/dynamic-plugin/host.js` 与 `client.js` 的内容，
   调用 `cordis_define`：
   - `plugin.kind: "new"`，`idPrefix: "rave"`
   - `code.host` = host.js 内容（函数体，不含 `export`/`import`）
   - `code.client` = client.js 内容（同上）
   - name / purpose 自定
2. **运行插件**：用返回的 `pluginId`/`packageId` 调 `cordis_run`（mode: `run`）。
   Client 半边需要你在界面批准（Run 卡片上允许）。
3. **验证**：对 agent 说
   `运行 rave_result_plot，sim_dir 用 output/2026/08/20260814_231146680024`
   —— 工具卡片应内嵌显示 1D 曲线，PNG 同时落盘
   `output/_agent_runs/plots/`。

## 保持同步

- **host.js 与持久版静态插件** `rave_agent/rave-sim-tools/lib/index.js` 逻辑同源：
  静态版用 `ctx.tools.register`（无 RPC、无 Client），动态版用 `harness`（有 RPC + Client）。
  修改工具逻辑时两处同步更新。
- 日常使用推荐 **RAVE-SIM preset**（新会话自动加载 6 工具，无内嵌查看器）；
  动态插件仅在需要「对话窗口内嵌图片」时激活。
