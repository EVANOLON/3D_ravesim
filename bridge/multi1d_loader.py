"""
Multi1D++ ASCII 输出数据加载器
================================
纯 Python 实现，无 MATLAB 依赖。

解析 Multi1D++ 的制表符分隔 ASCII 输出文件 (.scalars.dat, .center.dat,
.interface.dat, .group.dat)，将扁平列向量 reshape 为 (nt, ncell) 二维数组。

容错设计 (基于真实数据验证):
  - 自动检测编码 (UTF-8 → latin-1 降级)
  - 跳过列数不匹配的损坏行 (记录日志)
  - 跳过空白行
  - 单字段数值转换失败时跳过整行

用法:
    from multi1d_loader import load_multi1d_output

    data = load_multi1d_output("/path/to/case_output/")
    # 或指定 case 文件前缀:
    data = load_multi1d_output("/path/to/case_output/251222")

    print(data["XC"].shape)   # (nt, ncell)
    print(data["R"].shape)    # (nt, ncell)
    print(data.keys())        # 所有可用变量
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger("multi1d_loader")

# ─── 常量 ────────────────────────────────────────────────────────────────────

# ASCII 文件后缀
SCALARS_SUFFIX = ".scalars.dat"
CENTER_SUFFIX = ".center.dat"
INTERFACE_SUFFIX = ".interface.dat"
GROUP_SUFFIX = ".group.dat"
FINAL_SUFFIX = ".final.dat"

# 编码尝试顺序
_ENCODINGS = ["utf-8", "latin-1", "cp1252"]


# ─── 核心解析函数 ────────────────────────────────────────────────────────────

def _detect_encoding(filepath: Path) -> str:
    """检测文件编码，通过完整读取尝试已知编码。"""
    for enc in _ENCODINGS:
        try:
            with open(filepath, encoding=enc) as f:
                f.read()
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "latin-1"  # 最终降级，不会抛异常


def _parse_header_and_data(
    filepath: Path,
) -> tuple[list[str], np.ndarray, int, int]:
    """
    解析 Multi1D++ ASCII 文件。

    返回:
        col_names: 列名列表
        data:      (nrows, ncols) 数值数组
        n_bad:     跳过的损坏行数
        n_blank:   跳过的空白行数
    """
    # ── 读取整个文件 (尝试多种编码) ──
    raw_text: str = ""
    used_encoding = "latin-1"
    for enc in _ENCODINGS:
        try:
            with open(filepath, encoding=enc) as f:
                raw_text = f.read()
            used_encoding = enc
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        # 最后用 latin-1 强制读取 (不会抛 UnicodeDecodeError)
        with open(filepath, encoding="latin-1") as f:
            raw_text = f.read()
        used_encoding = "latin-1"

    logger.debug(f"文件 {filepath.name}: 编码={used_encoding}")
    raw_lines = raw_text.splitlines(keepends=False)

    # 找列名行 (最后一个以 # 开头且包含列名的行)
    col_line_idx = -1
    for i in range(len(raw_lines) - 1, -1, -1):
        line = raw_lines[i]
        if line.startswith("#") and len(line.split()) > 3:
            # 检查是否像列名行 (含 Time/XC/CMC 等关键词)
            stripped = line.lstrip("#").strip()
            if any(kw in stripped for kw in ("Time", "CMC", "XC", "TIME")):
                col_line_idx = i
                break

    if col_line_idx < 0:
        raise ValueError(f"找不到列名行: {filepath}")

    col_names = raw_lines[col_line_idx].lstrip("#").strip().split()
    ncols_expected = len(col_names)
    logger.debug(f"  列数={ncols_expected}, 列名={col_names[:6]}...")

    # ── 第二遍: 解析数据行 ──
    data_rows: list[list[float]] = []
    n_bad = 0
    n_blank = 0

    for i in range(col_line_idx + 1, len(raw_lines)):
        line = raw_lines[i].strip()
        if not line:
            n_blank += 1
            continue

        parts = line.split()
        if len(parts) != ncols_expected:
            n_bad += 1
            if n_bad <= 5:
                logger.debug(f"  跳过行 {i+1}: 列数={len(parts)} (期望={ncols_expected})")
            continue

        try:
            data_rows.append([float(x) for x in parts])
        except ValueError:
            n_bad += 1
            if n_bad <= 5:
                logger.debug(f"  跳过行 {i+1}: 数值转换失败")
            continue

    if n_bad > 0:
        logger.warning(f"  {filepath.name}: 跳过 {n_bad} 损坏行 ({100*n_bad/(n_bad+len(data_rows)):.3f}%)")
    if n_blank > 0:
        logger.debug(f"  {filepath.name}: 跳过 {n_blank} 空白行")

    data = np.array(data_rows, dtype=np.float64)
    return col_names, data, n_bad, n_blank


# ─── 公开 API ────────────────────────────────────────────────────────────────


def load_multi1d_output(
    case_path: Union[str, Path],
    *,
    parse_materials: bool = True,
    load_interface: bool = False,
    load_groups: bool = False,
    nt: Optional[int] = None,
    ncell: Optional[int] = None,
) -> dict:
    """
    加载 Multi1D++ ASCII 输出。

    参数:
        case_path: case 文件路径、输出目录或 case 名称前缀。
                   自动匹配 .scalars.dat, .center.dat 等后缀。
        parse_materials: 是否解析 matter++/material.base 材料映射。
        load_interface:  是否加载 .interface.dat (较大，默认 False)。
        load_groups:     是否加载 .group.dat (较大，默认 False)。
        nt:              手动指定时间步数 (None=自动从 scalars 推断)。
        ncell:           手动指定网格数 (None=自动从 center 推断)。

    返回:
        dict, 含以下键:
            # 维度
            "nt": int, "ncell": int, "ngroups": int (若加载 group),
            # scalars (shape: (nt,))
            "TIME1D", "ISTEP", "ENLAS", "ENQUE", "ENIE", "ENII",
            "ENKI", "ENRAD", "EYIELD", "TOTALMASS", "TOTALENERGY",
            "TR_Left", "TR_Right", "rhoR", "Centroid", "V_imp",
            "IFAR", "Adiabat", "AblatedMass", "AblationPressure",
            "LaserPowerTotal", ... (全部 66 列),
            # center (shape: (nt, ncell))
            "XC", "R", "T", "TI", "DENE", "ZI", "MID",
            "CMC", "D", "P", "PI", "PT", "EE", "EI",
            "TR_cell", "kei", "ke", "ki", "Index", ... (全部 33 列),
            # interface (shape: (nt, ncell+1)) — 若 load_interface=True
            "X", "V", "CMI", "Splus", "Sminus", "TR", ...,
            # group (shape: (nt, ngroups)) — 若 load_groups=True
            "ILeft", "IRight", ...,
            # 材料
            "materials_map": dict[int, dict]  (若 parse_materials=True)
    """
    case_path = Path(case_path)

    # ── 定位文件 ──
    files = _locate_files(case_path)
    logger.info(f"加载 Multi1D++ 输出: {files['base']}")

    result: dict = {}

    # ── 1. 标量文件 (.scalars.dat) ──
    scalars_path = files["scalars"]
    if scalars_path and scalars_path.exists():
        col_names, sdata, n_bad, _ = _parse_header_and_data(scalars_path)
        result["nt"] = nt if nt else sdata.shape[0]

        # 所有标量作为 1D 数组存储
        for j, name in enumerate(col_names):
            result[name] = sdata[:, j]

        logger.info(f"  scalars: {sdata.shape[0]} 时间步 × {len(col_names)} 变量")
    else:
        if nt is None:
            raise FileNotFoundError(f"找不到 .scalars.dat: {scalars_path}")

    # ── 2. 中心量文件 (.center.dat) ──
    center_path = files["center"]
    if center_path and center_path.exists():
        col_names, cdata, n_bad, _ = _parse_header_and_data(center_path)

        if nt is None:
            nt_auto = result.get("nt")
        else:
            nt_auto = nt
        if nt_auto is None:
            raise ValueError("无法确定 nt，请手动指定")

        ncell_auto = ncell if ncell else cdata.shape[0] // nt_auto
        n_valid = nt_auto * ncell_auto

        if n_valid > cdata.shape[0]:
            logger.warning(
                f"  center: 期望 {n_valid} 行，实际 {cdata.shape[0]} 行。"
                f"截断到 {cdata.shape[0] // ncell_auto} 时间步"
            )
            nt_auto = cdata.shape[0] // ncell_auto
            n_valid = nt_auto * ncell_auto

        result["nt"] = nt_auto
        result["ncell"] = ncell_auto

        # 截断到有效行数并 reshape
        cdata_valid = cdata[:n_valid, :]
        for j, name in enumerate(col_names):
            result[name] = cdata_valid[:, j].reshape(nt_auto, ncell_auto)

        logger.info(
            f"  center: {nt_auto}×{ncell_auto} 网格 × {len(col_names)} 变量"
            + (f" (跳过 {n_bad} 损坏行)" if n_bad > 0 else "")
        )

    # ── 3. 界面量文件 (.interface.dat) (可选) ──
    if load_interface:
        interface_path = files.get("interface")
        if interface_path and interface_path.exists():
            col_names, idata, n_bad, _ = _parse_header_and_data(interface_path)
            nt_cur = result["nt"]
            ncell_cur = result["ncell"]

            n_valid = nt_cur * (ncell_cur + 1)
            idata_valid = idata[:n_valid, :]
            for j, name in enumerate(col_names):
                result[name] = idata_valid[:, j].reshape(nt_cur, ncell_cur + 1)

            logger.info(
                f"  interface: {nt_cur}×{ncell_cur+1} 网格"
                + (f" (跳过 {n_bad} 损坏行)" if n_bad > 0 else "")
            )

    # ── 4. 群文件 (.group.dat) (可选) ──
    if load_groups:
        group_path = files.get("group")
        if group_path and group_path.exists():
            col_names, gdata, n_bad, _ = _parse_header_and_data(group_path)
            nt_cur = result["nt"]

            # group.dat 每时间步有 ngroups 行
            # 需要从 scalars 或 header 推断 ngroups
            header_info = _read_header_info(group_path)
            ngroups = header_info.get("number of groups", 1)
            n_valid = nt_cur * ngroups
            gdata_valid = gdata[:n_valid, :]
            for j, name in enumerate(col_names):
                result[name] = gdata_valid[:, j].reshape(nt_cur, ngroups)

            result["ngroups"] = ngroups
            logger.info(f"  group: {nt_cur}×{ngroups} 群")

    # ── 5. 材料映射 ──
    if parse_materials:
        mat_base = _find_material_base(case_path, files["base"])
        if mat_base:
            result["materials_map"] = parse_material_base(mat_base)
            logger.info(f"  materials: {len(result['materials_map'])} 种材料")

    # ── 6. 派生便利变量 ──
    _add_derived_variables(result)

    return result


def parse_material_base(filepath: Union[str, Path]) -> dict[int, dict]:
    """
    解析 matter++/material.base 材料索引文件。

    格式是基于关键词的块状结构:
        MATERIAL MID<N>
           A <real>
           Z <real>
           Formula <text>
           Name <text>
           RHO <real>
           Gamma <real>
           EOS <module> <file>
           ...

    返回:
        {MID: {"Z": float, "A": float, "formula": str, "name": str,
               "rho": float, "gamma": float, ...}}
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"material.base 不存在: {filepath}")
        return {}

    materials: dict[int, dict] = {}
    current_mid: Optional[int] = None
    current_entry: dict = {}

    with open(filepath, encoding=_detect_encoding(filepath)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue

            # 去掉行内 REM 注释
            if " REM " in line.upper():
                rem_pos = line.upper().find(" REM ")
                line = line[:rem_pos].strip()

            # 新 material 块
            if line.upper().startswith("MATERIAL "):
                # 保存上一个
                if current_mid is not None:
                    materials[current_mid] = current_entry

                # 解析 MID
                parts = line.split()
                if len(parts) >= 2:
                    mid_str = parts[1]
                    if mid_str.upper().startswith("MID"):
                        mid_str = mid_str[3:]  # 去掉 "MID" 前缀
                    try:
                        current_mid = int(mid_str)
                    except ValueError:
                        current_mid = None
                current_entry = {}
                continue

            # 块内的键值对
            if current_mid is not None:
                parts = line.split(None, 1)  # 最多分两段: keyword + value
                if len(parts) >= 2:
                    keyword = parts[0].upper()
                    value = parts[1].strip()

                    if keyword in ("A", "Z", "GAMMA"):
                        try:
                            current_entry[keyword.lower()] = float(value)
                        except ValueError:
                            pass
                    elif keyword == "FORMULA":
                        current_entry["formula"] = value.strip('"')
                    elif keyword == "NAME":
                        current_entry["name"] = value.strip('"')
                    elif keyword == "RHO":
                        try:
                            current_entry["rho"] = float(value)
                        except ValueError:
                            pass
                    elif keyword == "GAMMAG":
                        try:
                            current_entry["gamma_g"] = float(value)
                        except ValueError:
                            pass
                    # EOS, PLANCK, ROSSELAND 等文件路径
                    elif keyword in ("EOS", "IEOS", "EEOS", "PLANCK", "ROSSELAND",
                                     "EMS", "NONLTE", "ZEFF", "COLDPACITY"):
                        file_parts = value.split()
                        current_entry[keyword.lower()] = file_parts

    # 保存最后一个
    if current_mid is not None:
        materials[current_mid] = current_entry

    return materials


# ─── 内部辅助 ────────────────────────────────────────────────────────────────


def _locate_files(case_path: Path) -> dict:
    """根据 case_path 定位四个输出文件。"""
    if case_path.is_dir():
        # 目录模式: 扫描目录找 *.scalars.dat 等
        base = case_path
        scalars_files = list(case_path.glob(f"*{SCALARS_SUFFIX}"))
        center_files = list(case_path.glob(f"*{CENTER_SUFFIX}"))
        interface_files = list(case_path.glob(f"*{INTERFACE_SUFFIX}"))
        group_files = list(case_path.glob(f"*{GROUP_SUFFIX}"))
    else:
        # 文件前缀模式: case_path 去掉后缀
        base = case_path.parent
        stem = case_path.stem
        # 处理双重后缀 (.case → 去掉 .case)
        if case_path.suffix == ".case":
            stem = Path(case_path.stem).stem if "." in case_path.stem else case_path.stem
            # 实际上 case_path.stem 去掉了最后一个后缀
            base_name = case_path.stem  # e.g. "251222" from "251222.case"
        else:
            base_name = case_path.name
            # 去掉已知后缀
            for sfx in [SCALARS_SUFFIX, CENTER_SUFFIX, ".dat", ".case"]:
                if base_name.endswith(sfx):
                    base_name = base_name[: -len(sfx)]
                    break

        scalars_files = [base / f"{base_name}{SCALARS_SUFFIX}"]
        center_files = [base / f"{base_name}{CENTER_SUFFIX}"]
        interface_files = [base / f"{base_name}{INTERFACE_SUFFIX}"]
        group_files = [base / f"{base_name}{GROUP_SUFFIX}"]

    return {
        "base": base,
        "scalars": scalars_files[0] if scalars_files else None,
        "center": center_files[0] if center_files else None,
        "interface": interface_files[0] if interface_files else None,
        "group": group_files[0] if group_files else None,
    }


def _read_header_info(filepath: Path) -> dict:
    """从注释头读取元信息 (ncell, ngroups, nlayers 等)。"""
    info: dict = {}
    encoding = _detect_encoding(filepath)
    with open(filepath, encoding=encoding) as f:
        for line in f:
            if not line.startswith("#"):
                break
            # 解析 "# key = value" 格式
            m = re.match(r"#\s*([a-zA-Z_\s]+)\s*=\s*(\d+)", line)
            if m:
                key = m.group(1).strip().lower()
                info[key] = int(m.group(2))
    return info


def _split_material_line(line: str) -> list[str]:
    """分割 material.base 行 (处理引号内的空格)。"""
    parts = []
    current = ""
    in_quotes = False
    for ch in line:
        if ch == '"':
            in_quotes = not in_quotes
            current += ch
        elif ch in (" ", "\t") and not in_quotes:
            if current:
                parts.append(current)
                current = ""
        else:
            current += ch
    if current:
        parts.append(current)
    return parts


def _find_material_base(case_path: Path, base_dir: Path) -> Optional[Path]:
    """查找 material.base 文件。"""
    # 从 case 目录向上查找 matter++/material.base
    search_dirs = [case_path if case_path.is_dir() else case_path.parent, base_dir]
    for d in search_dirs:
        for candidate in [
            d / "matter++" / "material.base",
            d.parent / "matter++" / "material.base",
            d / ".." / "matter++" / "material.base",
        ]:
            resolved = candidate.resolve()
            if resolved.exists():
                return resolved
    return None


def _add_derived_variables(result: dict) -> None:
    """添加派生便利变量。"""
    # Time1D (行向量, 1×nt) — 兼容 MATLAB 代码习惯
    if "TIME1D" not in result and "TIME" in result:
        result["TIME1D"] = result["TIME"]

    # 已从 center.dat 获取的 2D 变量无需额外派生
    # 标记哪些列是 cell-centered vector → 在 reader 给出时已 reshape

    # ngroups 从 header 推断
    if "ngroups" not in result:
        result["ngroups"] = 1


# ─── 便捷函数 ────────────────────────────────────────────────────────────────


def get_timestep(data: dict, t: int) -> dict:
    """
    从加载结果中提取单个时间步的 1D 剖面。

    返回:
        {"XC": (ncell,), "R": (ncell,), "T": (ncell,),
         "TI": (ncell,), "DENE": (ncell,), "ZI": (ncell,),
         "MID": (ncell,), "time": float}
    """
    return {
        "XC": data["XC"][t, :],
        "R": data["R"][t, :],
        "T": data["T"][t, :],
        "TI": data["TI"][t, :],
        "DENE": data["DENE"][t, :],
        "ZI": data["ZI"][t, :],
        "MID": data["MID"][t, :] if "MID" in data else np.zeros_like(data["XC"][t, :]),
        "time": float(data["TIME1D"][t]) if "TIME1D" in data else float(t),
    }


def available_timesteps(data: dict) -> range:
    """返回可用时间步范围。"""
    return range(data["nt"])


def summary(data: dict) -> str:
    """生成数据加载摘要。"""
    lines = [
        f"Multi1D++ 输出摘要",
        f"{'─'*40}",
        f"  时间步数:  {data.get('nt', '?')}",
        f"  网格数:    {data.get('ncell', '?')}",
        f"  群数:      {data.get('ngroups', '?')}",
    ]
    if "XC" in data:
        xc = data["XC"]
        lines.append(f"  XC 范围:   [{xc.min():.4f}, {xc.max():.4f}] cm")
    if "R" in data:
        r = data["R"]
        mask = r > 0
        lines.append(f"  密度范围:  [{r[mask].min():.2e}, {r.max():.2e}] g/cm³")
    if "T" in data:
        t = data["T"]
        lines.append(f"  Te 范围:   [{t.min():.2f}, {t.max():.2f}] eV")
    if "materials_map" in data:
        lines.append(f"  材料数:    {len(data['materials_map'])}")
    return "\n".join(lines)
