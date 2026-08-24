# multi1d-bridge-tools

DSH 插件：把 **Multi1D++ 辐射流体力学 ASCII 输出** 桥接成 **RAVE-SIM 网格输入**
（`PlasmaSample` 四网格 / `precise_Sample`）以及一份可运行的 `config.yaml`。

与 `rave-sim-tools` 同构：它是一个**静态引导包**，直接经 `ctx.tools.register` 注册 4 个
面向模型的工具；业务逻辑全部在 Python 桥接模块中（本目录只负责起子进程并回传 JSON）。

## 工具

| 工具 | 作用 |
|------|------|
| `multi1d_load` | 加载 Multi1D++ `.scalars.dat`/`.center.dat`，返回 nt/ncell/群数、时间/坐标/密度/温度范围、材料数、可用时间步 |
| `multi1d_build_grids` | 把某时间步的 1D 剖面转成 RAVE-SIM 网格并存 `.npy`（side-on/face-on；face-on 另有冷固体 `material/density_grid.npy`），返回每网格 shape 与 ne/te/zstar 范围 |
| `multi1d_gen_config` | 在 build 基础上再写一份 `config.yaml`（1D/2D、8–10 keV 默认源、混合 `plasma_sample`+`precise_sample` 元素），并用 RAVE-SIM 自身 `config.load` 校验可回读 |
| `multi1d_grid_plot` | 把生成的 ne/te/zstar（或材料/密度）网格渲染成 PNG，经 `http://127.0.0.1:8811` 内嵌到对话 |

## 文件

- `package.json` / `lib/index.js` — 插件本体（注册 4 工具）
- `rave_agent/multi1d_bridge.py` — 桥接 CLI（`load` / `build_grids` / `gen_config` 子命令）
- `rave_agent/multi1d_grid_plot.py` — 网格渲染器（输出 markdown-embeddable `url`）

## 依赖

- Python `/home/taylor/anaconda3/envs/rave-sim/bin/python`（含 numpy/scipy/ruamel.yaml）
- 桥接模块在 `/mnt/d/rave-sim-main/Multi1D++Portable20241128/`（multi1d_loader / plasma_grid_builder / rave_config_gen）
- RAVE-SIM `big-wave/` 与 `nist_lookup/`（`gen_config` 回读校验用）

## 启用

在 `~/.dsh/.agent-presets/rave-sim/agent.cordis.yml` 中，`rave-sim-tools` 行之后加入：

```yaml
- id: multi1d-bridge-tools
  name: '/mnt/d/rave-sim-main/rave-sim-main/rave_agent/multi1d-bridge-tools/lib/index.js'
```

仓库侧备份已同步到 `rave_agent/preset/agent.cordis.yml`。按其同步约定把备份拷回
`~/.dsh/.agent-presets/rave-sim/*.yml` 即可（`cp rave_agent/preset/agent.cordis.yml ~/.dsh/.agent-presets/rave-sim/`）。

> 注意：DSH 静态包的工具目录在**会话启动时**从 preset 构建。本工具需在**下一个会话**才可调用；
> 当前会话内可用本章末尾的直接 Python 命令验证同一逻辑（本文已全部执行通过）。

## 用法示例

```bash
# 1) 查看算例概况
python rave_agent/multi1d_bridge.py load '{"case_path": "/path/to/251222"}'

# 2) 生成网格（核心）
python rave_agent/multi1d_bridge.py build_grids '{"case_path": "...", "timestep": 330, "output_dir": ".../t330", "geometry": "side-on", "nx": 256}'

# 3) 生成 config.yaml（并校验回读）
python rave_agent/multi1d_bridge.py gen_config '{"case_path": "...", "timestep": 330, "output_dir": ".../cfg", "dimension": "1d", "N": 8192}'

# 4) 渲染网格图（内嵌）
python rave_agent/multi1d_grid_plot.py --grid_dir ".../cfg" --fields ne,te,zstar
```

## 验证

已在真实算例 `251222`（334 时间步 × 599 网格，材料 251 种）上跑通全部 4 步，
并用 RAVE-SIM 的 `config.load` + `parse_optical_element` 确认生成的
`plasma_sample` 元素与四网格可被解析（`ne.shape=(1,256)`、Z=6、厚度 1e-4 m）。
