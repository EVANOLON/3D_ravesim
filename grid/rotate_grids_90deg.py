"""
将三个壳层网格绕 x 轴旋转 90° → 原 z 轴变形变为 y 轴
使 xy 截面能看出差异。

旋转规则：
  x' =  x
  y' = -z
  z' =  y

数组变换 (C-order, shape=[nz, ny, nx])：
  new_arr[iz_new, iy_new, ix] = old_arr[iy_old, nz-1-iz_old, ix]
  → np.transpose(arr, (1, 0, 2))  # 交换 z↔y
  → np.flip(..., axis=1)           # 翻转新 y 轴（原 z 方向）
"""
import numpy as np
import os

GRID_DIR = "D:/rave-sim-main/rave-sim-main/grid"

files = [
    ("偶极偏心 (l=1,m=0)", "500um_100um_shell_l1_m0_eps002.npy"),
    ("P2 椭球 (l=2,m=0)",  "500um_100um_shell_l2_m0_eps002.npy"),
    ("壁厚不均 (l=1,varT)", "500um_100um_shell_l1_varT_er002_et005.npy"),
]

for desc, fname in files:
    in_path = os.path.join(GRID_DIR, fname)
    stem, ext = os.path.splitext(fname)
    out_fname = stem + "_rot90" + ext
    out_path = os.path.join(GRID_DIR, out_fname)

    print(f"\n{'='*60}")
    print(f"[{desc}]")
    print(f"  加载: {fname}  ({os.path.getsize(in_path)/1e9:.1f} GB) ...", flush=True)
    arr = np.load(in_path)
    nz, ny, nx = arr.shape
    print(f"  形状: ({nz}, {ny}, {nx})", flush=True)

    print(f"  旋转中 ...", flush=True)
    # 转置 z↔y 再翻转新 y 轴
    rotated = np.flip(np.transpose(arr, (1, 0, 2)), axis=1)
    del arr  # 释放原数组内存

    print(f"  保存: {out_fname} ...", flush=True)
    np.save(out_path, rotated)
    del rotated

    print(f"  完成  ({os.path.getsize(out_path)/1e9:.1f} GB)", flush=True)

print(f"\n{'='*60}")
print("全部旋转完成！")
