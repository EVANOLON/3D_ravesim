# DSH 离线部署包（方案 A：整目录打包）

- **包内容**：DeepSeek Harness（DSH）`@deepseek-ai/dsh` **0.1.0-rc.6** 完整安装，
  含全量 `node_modules` 依赖、启动器 `node_modules/.bin/dsh` 与随附 profile 模板。
- **打包平台**：Linux x86_64（原生插件按 linux-x64 编译），验证环境 Node.js v24.15.0。

---

## 1. 目标机要求

| 项 | 要求 |
|---|---|
| OS/架构 | **x86_64 Linux**（与打包机一致；Windows/ARM 不能直接用本包） |
| Node.js | 建议 20/22/24 LTS（本包在 v24.15.0 下验证） |
| 端口 | 默认 GUI `127.0.0.1:3080` |
| 磁盘 | 解压后约 360 MB |

---

## 2. 安装步骤

```bash
# 1) 上传并解压（解压到任意目录，如 /opt）
tar -xzf dsh-offline-0.1.0-rc.6-linux-x64.tar.gz -C /opt

# 2) 冒烟测试（一次性问答，无 GUI）
/opt/dsh-offline-0.1.0-rc.6/node_modules/.bin/dsh --profile headless "1+1"

# 3) 启动 Web GUI（默认 127.0.0.1:3080）
/opt/dsh-offline-0.1.0-rc.6/node_modules/.bin/dsh web
```

可选：把 `…/node_modules/.bin` 加入 `PATH`，之后直接 `dsh web`。

用户数据默认存于 `~/.dsh`（可用 `DSH_HOME` 环境变量指定其他目录）；
首次 `dsh web` 会从随附模板自动初始化 web profile。

---

## 3. 从早期 v1 版本升级的注意事项（重要）

1. **先备份**：`cp -a ~/.dsh ~/.dsh.bak.$(date +%s)`，并保留旧版 dsh 安装目录。
2. **配置格式可能漂移**：本包为 0.1.0-rc.6，与早期 v1 的 `cordis.patch.yml` / `dsh.profile`
   格式不保证兼容。**建议让新版本全新初始化 profile**，再把旧配置按新格式人工核对后逐条迁移
   （不要直接整体拷贝旧 profile 目录）。
3. 迁移重点条目（本部署场景下）：
   - `@deepseek-ai/dsh-llm-deepseek`：`baseURL` 指向本地模型端点、`models` 目录、`apiKeyEnv`
   - `@deepseek-ai/dsh-agent-default-model`：`provider` / `model` 改为本地组合
   - `@deepseek-ai/dsh-mcp-client`：`serverName: rave` 的 stdio 条目（rave_agent MCP server）
   - 密钥迁移到 `$DSH_HOME/.credentials.yaml`（注意 DSH 会校验密钥格式，本地无鉴权服务
     也需 `sk-local-*` 形式的占位密钥）
4. 模型端点相关热配置可放在 `$DSH_HOME/settings.yaml` 的 `llm-deepseek:` 分节（外部编辑热发布，
   免重启生效）。

---

## 4. 接入本地模型（DeepSeek 系列，vLLM 示例）

```bash
# 模型服务（另一终端/后台）
vllm serve ./weights --served-model-name local-ds --port 8000

# DSH 侧：$DSH_HOME/cordis.patch.yml 追加/覆盖
#   - id: llm-deepseek
#     name: '@deepseek-ai/dsh-llm-deepseek'
#     config:
#       baseURL: http://127.0.0.1:8000/v1
#       apiKeyEnv: LOCAL_LLM_KEY          # .credentials.yaml 给 sk-local-* 占位密钥
#       models: [{id: local-ds, name: Local-DeepSeek, contextWindow: 131072, maxTokens: 8192}]
#   - id: agent-default-model
#     name: '@deepseek-ai/dsh-agent-default-model'
#     config: {provider: deepseek-official, model: local-ds}
```

兼容必查：流式 SSE、`tool_calls`、含工具调用轮次的 `reasoning_content` 回传、
顶层 `reasoning_effort`（不兼容时配 `thinking: disabled`）。冒烟：

```bash
dsh --profile headless "请用一句话回答：1+1=?"
```

---

## 5. 校验

解压后建议核对 sha256（见包外同名 `.sha256` 文件）：
```bash
sha256sum -c dsh-offline-0.1.0-rc.6-linux-x64.tar.gz.sha256
```
