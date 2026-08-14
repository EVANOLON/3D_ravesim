# DSH 内网部署与本地模型接入报告

> 版本：v1.0
> 目标：把 DSH（DeepSeek Harness）打包成可上传到内部电脑、并接入本地部署模型（DeepSeek 系列）的离线版本
> 事实依据：实测本机 DSH 0.1.0-rc.6（`@deepseek-ai/dsh` 及 90+ 插件包）的包结构、启动机制、配置层与 LLM/MCP 接口

---

## 0. 结论摘要

1. DSH 本质是 **npm 包集合 + `dsh` CLI 启动器**，运行时全部依赖都在 `node_modules` 里，前端 dist 已随包发布——**可以整目录打包离线搬运**，无需源码构建。
2. 本地模型接入不需要改 DSH 代码：把 `@deepseek-ai/dsh-llm-deepseek` 适配器的 `baseURL` 指向本地 OpenAI 兼容服务（vLLM 推荐），并在 `models` 目录里登记本地模型 id 即可。
3. 推荐组合：**开发/快速验证用整目录 tar 包；生产用 Docker 镜像**；模型权重经外网机下载后随移动介质搬运，vLLM 加载。
4. 三个必查兼容点：流式 SSE、工具调用、思考模式（`reasoning_content`/`reasoning_effort`）——直接影响 Agent 功能完整度。

---

## 1. DSH 运行机制速览（内网部署须知的事实）

### 1.1 组成与启动

| 项 | 事实 |
|---|---|
| 安装形态 | npm 包 `@deepseek-ai/dsh`（bin: `dsh`）+ 约 90 个 `@deepseek-ai/*` 插件包；`npx`/全局安装后运行 |
| 版本 | 当前 rc.6（部署时建议 pin 死版本） |
| 入口 | `dsh web`（浏览器 GUI，默认 `127.0.0.1:3080`）、`dsh --profile headless "prompt"`（一次性问答，冒烟测试利器）、`dsh --profile <name> --port …` |
| 用户数据 | 单根主目录：`$DSH_HOME`（未设置则 `~/.dsh`）；profiles、settings、credentials、会话日志都在其下 |
| 运行要求 | 现代 Node.js（ESM + 全局 fetch；包内未声明 engines，建议 Node 20/22 LTS，打包前用目标版本冒烟）；含少量原生二进制插件（如 `node-addon-landlock-run-linux-x64`），**必须与目标机 OS/架构一致** |

### 1.2 配置层（决定了离线版怎么预置）

配置树按顺序叠加：bundle patch → profile 的 `cordis.patch.yml` → `$DSH_HOME/cordis.patch.yml` → `--patch`。首次 `dsh web` 会从随附模板自动初始化 web profile。

- **MCP 服务器**：在 cordis 配置中为每个 server 加一个 `@deepseek-ai/dsh-mcp-client` 插件实例（stdio：`command/args/env/cwd`；或 streamable-http：`url/headers`），工具以 `mcp__<serverName>__<rawName>` 暴露。
- **模型端点**：`@deepseek-ai/dsh-llm-deepseek` 行（`baseURL`、`apiKeyEnv`、`models` 目录、`thinking`、`maxTokens`）；`dsh-llm-pi-ai` 提供 OpenAI 兼容备用路由。
- **热配置**：`$DSH_HOME/settings.yaml` 的 `llm-deepseek:` 分节可免重启覆盖 baseURL/目录等（外部编辑热发布）。
- **凭据**：四层来源——启动环境 > `$DSH_HOME/.credentials.yaml` > 项目 `.env` > `$DSH_HOME/.env`；配置只写环境变量名（`apiKeyEnv`）不写明文；**密钥格式会被校验**。
- **默认模型**：dsh-base 把 `agent-default-model` 设为 `deepseek-official/deepseek-v4-flash`，内网必须改为本地组合。

---

## 2. 离线打包方案（三选一）

### 方案 A：整目录打包（推荐起步，1 天内可跑通）

步骤：
1. 在联网机执行 `npx --yes @deepseek-ai/dsh@0.1.0-rc.6 --help`（生成完整 checkout，含全部依赖）；
2. 定位 npx 缓存目录（本机示例 `/home/taylor/.npm/_npx/<hash>/`），连同内部 `node_modules` 一起打 tar：
   ```bash
   tar -C <npx缓存父目录> -cf dsh-offline.tar <checkout目录名>
   # 或压缩: zstd -T0 / gzip
   ```
3. 内部机安装 Node.js（或随包携带同版本绿色 Node），解压后运行：
   ```bash
   <解压目录>/node_modules/.bin/dsh web
   ```
4. 预置配置：把 `cordis.patch.yml`（含 llm-deepseek 本地端点和 mcp-rave 行）放进包内模板或写好文档，首启后拷入 `$DSH_HOME/`。

- 优点：零网络、无需构建、最快。缺点：体积大（node_modules 数百 MB~GB 级）；原生插件绑定 OS/架构（linux-x64 包不能直接给 Windows 用，需在对应平台各自打包）。

### 方案 B：内网 npm 镜像（适合有内网基础设施的团队）

1. 内网部署 verdaccio / Nexus 镜像；
2. 外网机 `npm pack` 全部 `@deepseek-ai/*` 依赖（或直接从 lock 文件提取 tarball URL 批量下载）；
3. 导入内网镜像，内部机 `npm install --registry http://内网镜像` 安装；
4. 后续升级走同一流程（pin 版本号）。

- 优点：版本管理规范、可增量更新。缺点：依赖数量多（90+ 包），需要维护镜像。

### 方案 C：Docker 镜像（推荐生产）

```dockerfile
FROM node:22-bookworm-slim
COPY dsh-checkout /opt/dsh
ENV DSH_HOME=/data/dsh-home
ENV PATH=/opt/dsh/node_modules/.bin:$PATH
EXPOSE 3080
CMD ["dsh", "web"]
```
外网 `docker build` + `docker save -o dsh-internal.tar`，内部 `docker load` 后 `docker run -v /data:/data -p 3080:3080`。

- 优点：跨机一致、可版本化、随容器带 DSH_HOME。缺点：需要内部机有 Docker；注意原生插件仍按 `linux/x64` 打。

**共同前提**：内部机与打包机 OS/架构一致（推荐 x86_64 Linux）；Node 版本一致；打包后先在联网机做一次 `dsh --profile headless "1+1"` 冒烟再断网交付。

---

## 3. 本地模型部署与 DSH 接入

### 3.1 服务选型

| 方案 | 适用 | 说明 |
|---|---|---|
| **vLLM（推荐）** | 有 GPU、追求协议完整 | OpenAI 兼容 `/v1`，DeepSeek 模型官方支持；SSE 流式、tool_calls、thinking 回传完整 |
| SGLang | 备选 | 同上能力，按团队熟悉度选 |
| Ollama + GGUF | 显存受限/CPU 部署 | 量化权重省显存，但协议兼容面较窄（工具调用/思考支持依赖模板，需逐项验证） |

### 3.2 模型权重搬运流程

1. 外网机 `hf download deepseek-ai/<模型> --local-dir ./weights`（或下载 GGUF 量化版）；
2. 校验 sha256 后经移动硬盘/内网摆渡到内部机；
3. 内部机启动 vLLM：
   ```bash
   vllm serve ./weights --served-model-name local-ds --port 8000 \
     --max-model-len <按显存定> --gpu-memory-utilization 0.9
   ```
4. 磁盘规划：671B 级完整模型 BF16 约 1.3TB、FP8 约 700GB，多卡才能跑；单卡/小显存选蒸馏版（如 R1-Distill-Qwen-14B/32B）或 GGUF 量化版。**模型需具备 tool calling 与 thinking 能力**（DeepSeek 系列模板支持）。

### 3.3 DSH 侧配置（改两处）

```yaml
# $DSH_HOME/cordis.patch.yml 或 profile 的 cordis.patch.yml

# ① 模型端点：指向本地 vLLM
- id: llm-deepseek
  name: '@deepseek-ai/dsh-llm-deepseek'
  config:
    baseURL: http://127.0.0.1:8000/v1
    apiKeyEnv: LOCAL_LLM_KEY          # 见下方占位密钥说明
    thinking: enabled                 # 若本地模型/vLLM 不支持思考则改 disabled
    models:
      - id: local-ds
        name: Local-DeepSeek
        contextWindow: 131072         # 按部署实际值
        maxTokens: 8192               # 必须 ≤ 本地输出上限

# ② 默认模型指向本地组合
- id: agent-default-model
  name: '@deepseek-ai/dsh-agent-default-model'
  config:
    provider: deepseek-official
    model: local-ds
```

占位密钥：本地 vLLM 通常无鉴权，但 DSH 会校验密钥格式——在 `$DSH_HOME/.credentials.yaml` 写入 `LOCAL_LLM_KEY: sk-local-<随机串>`（形如合法 token），或启动时 `LOCAL_LLM_KEY=sk-local-xxx dsh web`。若用 `dsh-llm-pi-ai` 的 OpenAI 兼容路由则可省略 `apiKeyEnv`（未认证状态）。

### 3.4 兼容性验证清单（逐项必过）

| # | 检查项 | 方法 |
|---|---|---|
| 1 | 流式 SSE | DSH 适配器只支持流式；`curl -N` 打 `/v1/chat/completions` 带 `"stream":true` |
| 2 | 工具调用 `tool_calls` | 发带 tools 参数的请求，验证回传与 `finish_reason: tool_calls` |
| 3 | 思考回传 | 含工具调用的轮次，适配器会回传 `reasoning_content`——vLLM 的 DeepSeek 模板默认处理；验证多轮工具往返不报错 |
| 4 | 顶层 `reasoning_effort` | 新 vLLM 对 DeepSeek 模型支持；旧版可能忽略或拒绝未知字段。拒绝时：DSH 配 `thinking: disabled` 或仅用 off 强度 |
| 5 | `finish_reason` / `usage` | 确认 `stop` 与 usage 字段正常（DSH 依赖 usage 计量） |
| 6 | 模型目录 | `models` 列表 id 与 `--served-model-name` 一致 |

冒烟命令（无需 GUI）：
```bash
dsh --profile headless "请用一句话回答：1+1=?"     # 模型连通性
dsh --profile headless "列出你当前可用的工具"       # 应出现 mcp__rave__*（装了 rave MCP 后）
```

---

## 4. 与 RAVE-SIM Agent 集成的部署包清单

内网交付三件套 + 一键脚本：

```
rave-sim-internal/
├── dsh-offline.tar            # 方案A产物（或 dsh-internal.tar 镜像）
├── rave-sim/                  # 仓库（含 rave_agent/ MCP server 源码）
│   └── rave_agent/            # FastMCP server：tools/jobs/state/knowledge
├── wheels/                    # pip 离线轮子：mcp、numpy、yaml、pydantic…（外网 pip download）
├── weights/                   # 本地模型权重（外网 hf download）
└── start_local.sh
```

`start_local.sh` 职责：检查 nvidia-smi → 启动 vLLM → 等待 `/v1/models` 就绪 → 启动 `dsh web` → 打印 GUI URL 与冒烟命令。DSH 内通过 `cordis.patch.yml` 的 `mcp-rave` 行（stdio，`command: python -m rave_agent.server`）自动拉起仿真工具。

---

## 5. 分阶段验证步骤

| 阶段 | 验证 | 通过标准 |
|---|---|---|
| V1 平台 | 解压/镜像加载，`dsh web` 起在 3080 | GUI 可访问 |
| V2 模型 | headless 冒烟 "1+1" | 正常回答；无 INVALID_CREDENTIAL/TRANSPORT 报错 |
| V3 工具 | 工具列表出现 `mcp__rave__*` | 16 个工具齐全、schema 正常 |
| V4 仿真 | 场景 A（W 板吸收验证）自然语言全流程 | μ 偏差 < 5% |
| V5 报告 | `report_render` 产出 report.md | 模板字段填充完整、图引用可打开 |

---

## 6. 风险与注意事项

| 风险 | 对策 |
|---|---|
| 原生插件/平台绑定 | 打包机与内部机同 OS/架构（x86_64 Linux）；Windows 需在 Windows 上重新打包 |
| Node 版本不匹配 | 固定 Node 20/22 LTS，随包提供绿色 Node 或明确版本要求 |
| 离线 pip 依赖遗漏 | 外网 `pip download -r rave_agent/requirements.txt --dest wheels/`，内部 `pip install --no-index --find-links wheels/` |
| 模型协议不兼容（4.4 清单） | 逐项过清单；不兼容项降级（thinking: disabled）并在交付文档标注能力损失 |
| 模型与仿真争显存（同机） | 量化模型降常驻；`feasibility_check` 计入模型占用；必要时错峰运行 |
| 占位密钥格式 | DSH 校验密钥格式，用 `sk-local-*` 形占位；不用纯数字/空串 |
| 版本漂移 | pin `@deepseek-ai/dsh@0.1.0-rc.6` 与全部依赖版本；升级走相同打包流程 |
| 许可合规 | DSH MIT；DeepSeek 模型权重按其各自许可审核后再入内网；npm 依赖 license 随包附带 |
| GUI 仅绑 127.0.0.1 | `dsh web` 拒绝 `--host 0.0.0.0`；内网多机访问需反向代理或后续官方方案（同机部署不受影响） |

---

## 7. 遗留问题（待确认）

1. 内部机 OS/架构与 GPU 型号、显存（决定打包目标与模型尺寸）
2. 内部机是否有 Docker / 内网 npm 镜像（决定方案 A/B/C）
3. 目标 DeepSeek 模型规格（完整版/蒸馏版/量化版）与权重来源许可
4. 是否需要多用户并发使用 DSH（影响端口暴露与反代方案）
