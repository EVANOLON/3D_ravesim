import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress

# 读取数据 （文件 SC_result.txt）
data = np.loadtxt('SC_result.txt', delimiter=',')  # (241, 200)
data = data.T
data = np.flip(data, axis=1)
nt, nx = data.shape

# 时空坐标
dt = 0.02          # ns
dx = 1.176         # μm
t = np.arange(nt) * dt
x = np.arange(nx) * dx

# 平滑与波前检测
sigma = 1.0
wave_positions = []

for i in range(nt):
    row = data[i, :]
    if np.std(row) < 0.01:
        continue   # 跳过未受扰动区
    smoothed = gaussian_filter(row, sigma=sigma)
    grad = np.abs(np.gradient(smoothed))
    # 消除边界噪声，取前 180 个点
    grad[:30] = 0
    idx = np.argmax(grad)
    if i*dt < 2.0:
        idxs = np.argsort(grad, axis=0)[-3:]
    else:
        idxs = np.argsort(grad, axis=0)[-10:]
    print(idxs)
    for idx in idxs:
        if idx > 30 and idx < nx-30:
            wave_positions.append((t[i], x[idx]))

wave_positions = np.array(wave_positions)
t_vals, x_vals = wave_positions[:,0], wave_positions[:,1]

# 线性拟合
# slope, intercept, r_value, p_value, std_err = linregress(t_vals, x_vals)
# D_km_per_s = slope / 1000   # 转换为 km/s
# print(f"shockwave_velocity = {D_km_per_s:.2f} km/s")
# print(f"R² = {r_value**2:.4f}")

# 绘图
plt.figure(figsize=(8,6))
plt.plot(t_vals, x_vals, 'o', markersize=3, label='Detected Wavefronts')
# plt.plot(t_vals, intercept + slope*t_vals, 'r-', label=f'Fitted Line D={D_km_per_s:.2f} km/s')
plt.xlabel('Time (ns)')
plt.ylabel('Position (μm)')
plt.legend()
plt.grid(True)
plt.show()