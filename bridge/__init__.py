"""
Multi1D++ → RAVE-SIM 工作流桥接包

将 Multi1D++ 辐射流体力学输出的 1D 等离子体剖面转换为 RAVE-SIM
波动光学仿真所需的 PlasmaSample / precise_Sample 网格输入，
自动化配置生成和批量运行。

模块:
  multi1d_loader      — Multi1D++ ASCII 输出解析
  plasma_grid_builder — 1D 剖面 → 2D/3D 混合网格转换
  rave_config_gen     — RAVE-SIM YAML 配置生成
  run_workflow        — 批量编排命令行入口
  result_aggregator   — 结果汇总与诊断

使用:
  python -m bridge.run_workflow <case_file> --timesteps 0:10:344 --mode 1d
"""

__version__ = "0.1.0"
__all__ = [
    "multi1d_loader",
    "plasma_grid_builder",
    "rave_config_gen",
    "run_workflow",
    "result_aggregator",
]
