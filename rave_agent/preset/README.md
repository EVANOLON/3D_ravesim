# RAVE-SIM agent preset（备份）

本目录是 DSH agent preset 的**仓库内备份**，与 `~/.dsh/.agent-presets/rave-sim/` 保持一致。

## 文件

| 文件 | 说明 |
|---|---|
| `preset.yml` | preset 元数据（名称 / 描述） |
| `agent.cordis.yml` | agent-plane 组合：persona（含 GPU 运行链、可行性门禁、内嵌图片/网格展示指令）+ 标准工具集 + `rave-sim-tools` bootstrap |

## 恢复 / 安装

```bash
mkdir -p ~/.dsh/.agent-presets/rave-sim
cp rave_agent/preset/preset.yml rave_agent/preset/agent.cordis.yml ~/.dsh/.agent-presets/rave-sim/
```

## 同步约定

修改 preset 后请同时更新本目录（`cp ~/.dsh/.agent-presets/rave-sim/*.yml rave_agent/preset/`）并 git 提交，
避免机器迁移 / 重装后丢失。业务逻辑（工具实现）的单一事实源是
`rave_agent/rave-sim-tools/` 与 `rave_agent/dynamic-plugin/`，本目录只备份预设壳。
