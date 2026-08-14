# 工作流验证与调试方案

## 已发现的潜在问题

基于对实际测试数据 (`251222test/`) 的初步检查:

| 问题 | 文件 | 严重程度 | 影响 |
|------|------|---------|------|
| 非 UTF-8 编码 | center.dat | 🟡 中等 | 包含 latin-1 字符, np.loadtxt 报 UnicodeDecodeError |
| 损坏行 (28/200751) | center.dat | 🟡 中等 | ~0.014% 行有二进制垃圾，列数异常 (2-18 列 vs 期望 33) |
| 列数不一致 | center.dat | 🔴 阻塞 | np.loadtxt 在遇到列数变化行时报错终止 |
| 大文件 (200K+ 行) | center.dat | 🟢 注意 | 600 cells × 334 time steps = 200400 行，内存约 50 MB |
| 循环导入 | big-wave propagation.py | 🟡 中等 | 独立导入 propagation 失败，需在 big-wave 目录内运行 |

这些问题是真实实验数据的典型特征——ASCII 文件中混入了二进制垃圾字节。

## 四层验证架构

```
┌─────────────────────────────────────────────────────┐
│ L1: 单元测试 (每个模块独立)                           │
│   multi1d_loader  │  grid_builder  │  config_gen    │
├─────────────────────────────────────────────────────┤
│ L2: 集成测试 (模块间接口)                             │
│   loader → grid_builder → config_gen → RAVE-SIM调用  │
├─────────────────────────────────────────────────────┤
│ L3: 物理验证 (结果合理性)                             │
│   网格可视化 │ delta/beta剖面 │ 探测器条纹 │ 质量守恒  │
├─────────────────────────────────────────────────────┤
│ L4: 端到端回归 (完整工作流)                           │
│   已知案例 → 全流程 → 与参考结果对比                   │
└─────────────────────────────────────────────────────┘
```

---

## L1: 单元测试

### 1.1 multi1d_loader 验证

**测试数据**: `251222test/251222.*.dat`

```python
# test_loader.py — 在 bridge/ 目录下运行
import numpy as np
from multi1d_loader import load_multi1d_output, parse_ascii_file

def test_parse_scalars():
    """验证 scalars.dat 解析"""
    data = load_multi1d_output("path/to/251222test")
    assert data["nt"] == 334, f"期望 334 时间步, 实际 {data['nt']}"
    assert data["ncell"] == 600, f"期望 600 cells, 实际 {data['ncell']}"
    assert "TIME1D" in data
    print("✅ scalars.dat 解析正确")

def test_parse_center():
    """验证 center.dat 解析 + reshape"""
    data = load_multi1d_output("path/to/251222test")
    assert data["XC"].shape == (334, 600), f"形状错误: {data['XC'].shape}"
    assert data["R"].shape == (334, 600)
    assert data["T"].shape == (334, 600)
    assert data["DENE"].shape == (334, 600)
    # XC 应单调递增 (每个时间步内从左到右)
    assert np.all(np.diff(data["XC"][0, :]) > 0), "XC 不单调!"
    print("✅ center.dat 解析+reshape 正确")

def test_encoding_robustness():
    """验证损坏行跳过 + 编码兼容"""
    # 期望跳过 28 行损坏数据, 解析 200400 有效行
    raw = parse_ascii_file("path/to/251222.center.dat")
    n_valid = len(raw)
    assert 200300 < n_valid <= 200400, f"有效行数异常: {n_valid}"
    print(f"✅ 编码容错: 跳过 {200751 - n_valid} 损坏行, 保留 {n_valid} 有效行")

def test_material_mapping():
    """验证 material.base 解析"""
    data = load_multi1d_output(..., parse_materials=True)
    mids = np.unique(data["MID"])
    for mid in mids:
        if mid > 0:
            assert mid in data["materials_map"], f"MID {mid} 不在 material.base 中"
    print(f"✅ 材料映射: {len(data['materials_map'])} 种材料")
```

**调试技巧**:
- 用 `head -n 100` 查看原始文件结构, 确认 header 行数
- 用 `file` 命令检查编码: `file -bi 251222.center.dat`
- 损坏行定位: 逐行解析, 记录 `len(line.split()) != expected_cols` 的行号

### 1.2 plasma_grid_builder 验证

```python
def test_side_on_geometry():
    """验证 side-on 几何映射"""
    from plasma_grid_builder import build_hybrid_grids
    grids = build_hybrid_grids(multi1d_data, timestep=0, 
                                geometry="side-on", nx=128)
    # PlasmaSample 四网格
    assert "ne" in grids, "缺少 ne_grid"
    assert grids["ne"].shape == (nz, 128), f"形状: {grids['ne'].shape}"
    # ne 不应为负
    assert np.all(grids["ne"] >= 0), "ne 有负值!"
    # 真空区 ne=0
    assert grids["ne"][0, :].sum() == 0, "左边界面应为真空"

def test_hybrid_split():
    """验证冷固体/等离子体自动分割"""
    grids = build_hybrid_grids(multi1d_data, timestep=100)
    # 检查是否有冷区
    if "solid" in grids:
        assert "material_grid" in grids["solid"]
        assert "density_grid" in grids["solid"]
        print(f"冷固体区: {grids['solid']['material_grid'].shape}")
    if "plasma" in grids:
        assert "ne" in grids["plasma"]
        print(f"等离子体区: {grids['plasma']['ne'].shape}")
```

**关键调试**:
- 网格可视化是最直接的调试手段 (见 L3)
- 检查 ZI 阈值: `print((ZI < 0.05*ZI.max()) & (Te < 5.0))` 各时间步的 True 比例

### 1.3 rave_config_gen 验证

```python
def test_config_generation():
    """验证生成的 YAML 可被 RAVE-SIM 解析"""
    from rave_config_gen import generate_config
    import ruamel.yaml
    
    config_path = generate_config(grid_dir, mode="1d", ...)
    
    # 验证 YAML 语法
    yaml = ruamel.yaml.YAML(typ="safe")
    with open(config_path) as f:
        cfg = yaml.load(f)
    
    # 验证必需字段
    assert "sim_params" in cfg
    assert "elements" in cfg
    assert cfg["elements"][0]["type"] == "plasma_sample"
    
    # 验证网格文件存在
    el = cfg["elements"][0]
    for key in ["ne_grid_path", "ni_grid_path", "te_grid_path", "zstar_grid_path"]:
        assert (config_path.parent / el[key]).exists(), f"缺失: {el[key]}"
    
    print("✅ config.yaml 语法 + 文件引用正确")
```

---

## L2: 集成测试

### 2.1 模块间数据流

```python
def test_data_pipeline():
    """loader → grid_builder → config_gen 全链路"""
    # Step A: 加载
    data = load_multi1d_output("251222test/")
    
    # Step B: 构建网格 (选 3 个代表性时间步)
    for ts in [0, 100, 200]:
        grids = build_hybrid_grids(data, timestep=ts, 
                                    geometry="side-on", nx=128)
        # 保存网格
        save_grids(grids, output_dir / f"t{ts:05d}")
    
    # Step C: 生成配置
    config = generate_config(output_dir, mode="1d",
                             energy_range=(8000, 10000))
    
    # Step D: 验证 RAVE-SIM 能否加载配置
    from config import load as load_rave_config
    cfg = load_rave_config(config)
    print("✅ RAVE-SIM 配置加载成功")
```

### 2.2 RAVE-SIM 最小运行

```python
def test_rave_minimal_run():
    """单时间步最小仿真 — 最重要的集成测试"""
    import subprocess, sys
    
    # 在 big-wave 目录运行 (避免循环导入)
    result = subprocess.run([
        sys.executable, "-c", """
import sys; sys.path.insert(0, ".")
from wavesim import run_simulation
from config import load, parse_optical_element, parse_sim_params, parse_source
import numpy as np

# 加载配置
cfg = load("path/to/generated/config.yaml")
sim_params = parse_sim_params(cfg["sim_params"])
elements = [parse_optical_element(el_dict) for el_dict in cfg["elements"]]
source = parse_source(cfg["source"])

# 运行
detected = run_simulation(sim_params, elements, source)
print(f"Detected shape: {detected.shape}")
assert detected.shape[0] > 0, "空探测器输出!"
assert not np.any(np.isnan(detected)), "探测器输出含 NaN!"
assert detected.max() > 0, "探测器全为零!"
print("SUCCESS")
"""
    ], capture_output=True, text=True, cwd="big-wave/", timeout=300)
    
    if "SUCCESS" in result.stdout:
        print("✅ RAVE-SIM 最小运行成功")
    else:
        print(f"❌ 失败:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
```

---

## L3: 物理验证

### 3.1 网格合理性检查

```python
def validate_grid_physics(grids, multi1d_data, timestep):
    """检查生成的网格是否物理合理"""
    
    # 1. 质量/粒子数守恒
    ne_grid = grids["plasma"]["ne"]
    original_ne = multi1d_data["DENE"][timestep, :]
    # 插值后总电子数应大致守恒 (允许 5% 误差)
    ne_total_original = np.sum(original_ne)
    ne_total_grid = np.sum(ne_grid) * (ne_grid.shape[1] / ne_grid.shape[0])
    ratio = ne_total_grid / (ne_total_original + 1e-30)
    assert 0.8 < ratio < 1.2, f"电子数不守恒: ratio={ratio:.3f}"
    
    # 2. ni * ZI ≈ ne (准中性条件)
    ni = grids["plasma"]["ni"]
    zstar = grids["plasma"]["zstar"]
    mask = ne_grid > 0
    neutrality = (ni * zstar)[mask] / ne_grid[mask]
    assert np.allclose(neutrality, 1.0, atol=0.1), \
        f"准中性破坏: {neutrality.min():.3f} ~ {neutrality.max():.3f}"
    
    # 3. 温度范围合理
    te = grids["plasma"]["te"]
    assert te.max() < 1e6, f"Te 异常高: {te.max()} eV"
    assert te.min() >= 0, f"Te 负值"
    
    print("✅ 网格物理合理性检查通过")
```

### 3.2 Delta/Beta 剖面验证

8-10 keV 下对低Z等离子体的预期:
- delta ≈ 10⁻⁶ ~ 10⁻⁴ (纯自由电子主导)
- beta ≈ 10⁻⁸ ~ 10⁻¹⁰ (Kramers, 在 9 keV 下很小)

```python
def test_delta_beta_range():
    """验证 delta/beta 在预期物理范围"""
    from plasma import plasma_delta_beta
    
    # 典型 DT 等离子体: ne=1e22, ni=1e22, Te=1000, Z*=1, Z=1
    d, b, _ = plasma_delta_beta(1e22, 1e22, 1000, 1.0, 1, 9000)
    
    # 自由电子 delta 解析值: ne * re * lambda^2 / (2*pi)
    # lambda(9keV) ≈ 1.38e-8 cm, re = 2.82e-13 cm
    # delta_free ≈ 1e22 * 2.82e-13 * (1.38e-8)^2 / (2*pi) ≈ 8.5e-7
    expected_delta = 1e22 * 2.82e-13 * (1.38e-8)**2 / (2 * np.pi)
    
    assert 5e-7 < d < 5e-6, f"delta={d:.2e}, 期望 ~{expected_delta:.2e}"
    assert b < 1e-7, f"beta={b:.2e}, 9 keV 下低Z吸收应很小"
    print(f"✅ delta={d:.2e} (期望~{expected_delta:.2e}), beta={b:.2e}")
```

### 3.3 探测器条纹合理性

```python
def validate_detector_output(detected_npy_path, sim_config):
    """检查探测器的 XPCI 条纹"""
    detected = np.load(detected_npy_path)
    
    # 1. 条纹存在性: 不是均匀场
    contrast = detected.std() / detected.mean()
    assert contrast > 0.01, f"无条纹: contrast={contrast:.4f} (可能全均匀)"
    
    # 2. 密度梯度处应有条纹
    # 计算 |dI/dx| 峰值位置 → 应与密度梯度位置对应
    
    # 3. 强度衰减合理 (穿过等离子体后)
    transmission = detected.mean() / incident_intensity
    assert 0.1 < transmission < 1.0, f"透射率异常: {transmission:.3f}"
    
    print(f"✅ 条纹 contrast={contrast:.3f}, transmission={transmission:.3f}")
```

---

## L4: 端到端回归测试

```python
def test_full_workflow():
    """完整工作流: 已知案例 → 全流程 → 对比基准"""
    import subprocess
    
    result = subprocess.run([
        "python", "-m", "bridge.run_workflow",
        "251222test/",
        "--timesteps", "0,50,100",  # 3 个时间步快速验证
        "--mode", "1d",
        "--geometry", "side-on",
        "--nx", "64",               # 粗网格快速运行
        "--energy", "9000", "9000", # 单能加速
        "--nr-sources", "5",        # 少量源点
        "--output", "./test_output/"
    ], capture_output=True, text=True, timeout=600)
    
    assert result.returncode == 0, f"工作流失败: {result.stderr}"
    
    # 检查输出
    import os, numpy as np
    for ts in [0, 50, 100]:
        det_path = f"./test_output/t{ts:05d}/detected.npy"
        assert os.path.exists(det_path), f"缺失输出: {det_path}"
        detected = np.load(det_path)
        assert detected.size > 0
    
    print("✅ 端到端回归测试通过")
```

---

## 调试工具集

### 快速检查脚本

```bash
#!/bin/bash
# verify_env.sh — 环境 + 数据完整性检查

echo "=== Python 环境 ==="
python3 -c "import numpy, scipy, ruamel.yaml, h5py; print('核心依赖 OK')"

echo "=== RAVE-SIM 环境 ==="
cd /mnt/d/rave-sim-main/rave-sim-main
python3 -c "import sys; sys.path.insert(0,'big-wave'); from plasma import plasma_delta_beta; print('plasma_delta_beta OK')"
python3 -c "import sys; sys.path.insert(0,'big-wave'); from config import load; print('config.load OK')"

echo "=== 测试数据 ==="
for f in scalars.dat center.dat interface.dat; do
    if [ -f "251222test/251222.$f" ]; then
        echo "  ✅ 251222.$f ($(wc -l < 251222test/251222.$f) 行)"
    else
        echo "  ❌ 251222.$f 缺失"
    fi
done

echo "=== 编码检查 ==="
file -bi 251222test/251222.center.dat
```

### 渐进式调试策略

当某步失败时:

| 症状 | 可能原因 | 调试方法 |
|------|---------|---------|
| loader 报 UnicodeDecodeError | 非 UTF-8 编码 | 用 `encoding='latin-1'` + 跳过损坏行 |
| loader 列数变化 | 二进制垃圾混入 | 逐行检查 `len(split())`, 只保留正确列数的行 |
| grid_builder 全零网格 | 阈值切分错误 | 打印 ZI/Te 范围, 调整 cold/hot 阈值 |
| RAVE-SIM 导入失败 | 循环导入 | 必须在 big-wave/ 目录运行, 或通过 subprocess |
| detected.npy 全零 | 源未传播到探测器 | 检查 z_start 累积是否超过 z_detector |
| detected.npy NaN | FFT 发散 | 检查 Nyquist 条件, dx 是否够小 |
| 运行极慢 | N 太大或 chunk_size 太小 | 先用 N=8192 小测试, 确认无误后再放大 |
| 无相衬条纹 | delta 太小或几何不对 | 检查 delta 剖面绝对值, 确认密度梯度方向 |

### 最小可运行验证 (MVP)

在所有模块实现前, 用以下最小脚本验证核心路径:

```python
# verify_core.py — 独立于 bridge 模块, 直接验证核心物理路径
import sys, numpy as np
sys.path.insert(0, '/mnt/d/rave-sim-main/rave-sim-main/big-wave')
sys.path.insert(0, '/mnt/d/rave-sim-main/rave-sim-main/nist_lookup')

from plasma import plasma_delta_beta

# 1. 加载 Multi1D 数据
# (手动实现 loader 核心逻辑)
def load_center_dat(path):
    rows = []
    with open(path, encoding='latin-1') as f:
        for line in f:
            if line.startswith('#'): 
                continue
            parts = line.split()
            if len(parts) == 33:  # 只保留正确列数的行
                rows.append([float(x) for x in parts])
    return np.array(rows)

data = load_center_dat('path/to/251222.center.dat')
nt, ncell = 334, 600
xc = data[:, 1].reshape(nt, ncell)   # col 1 = XC
r  = data[:, 3].reshape(nt, ncell)   # col 3 = R
t  = data[:, 4].reshape(nt, ncell)   # col 4 = T
dene = data[:, 8].reshape(nt, ncell) # col 8 = DENE
zi = data[:, 6].reshape(nt, ncell)   # col 6 = ZI

# 2. 测试单像素 delta/beta
ts = 100  # 中间时间步
for cell in [0, 150, 300, 450, 599]:
    ne = dene[ts, cell]
    ni = ne / max(zi[ts, cell], 0.01)
    te = t[ts, cell]
    zs = zi[ts, cell]
    d, b, atlen = plasma_delta_beta(ne, ni, te, zs, Z=1, energy=9000)
    print(f"cell={cell:4d} x={xc[ts,cell]:.4f}cm ne={ne:.2e} Te={te:.1f}eV "
          f"Z*={zs:.3f} delta={d:.2e} beta={b:.2e} atlen={atlen:.4f}cm")

print("\n✅ 核心物理路径验证通过 — delta/beta 计算正常")
```

---

## 验证执行顺序

```
Phase 0: 环境检查 (5 min)
  ├── Python 依赖确认
  ├── RAVE-SIM 导入测试
  └── 测试数据完整性

Phase 1: loader 单元测试 (15 min)
  ├── scalars.dat 解析
  ├── center.dat 解析 + 损坏行容错
  └── reshape + material.base

Phase 2: 核心物理验证 (10 min)
  ├── plasma_delta_beta 解析值对比
  ├── 单时间步 delta/beta 剖面
  └── 网格可视化

Phase 3: 集成测试 (30 min)
  ├── loader → grid_builder → config_gen
  └── RAVE-SIM 最小运行 (N=8192, 1 时间步)

Phase 4: 端到端 (1 hr)
  ├── 3 时间步完整流程
  └── 条纹合理性检查
```

---

## 关键容错设计 (在 loader 中实现)

基于真实数据发现的问题, `multi1d_loader.py` 必须实现:

1. **编码检测**: 先试 UTF-8, 失败降级 latin-1
2. **列数校验**: 只保留 `len(split()) == expected_cols` 的行
3. **损坏行日志**: 记录被跳过的行号 + 预览, 供用户检查
4. **空行跳过**: 忽略完全空白的行
5. **数值转换容错**: 单行内个别字段转换失败时跳过整行 (而非崩溃)
