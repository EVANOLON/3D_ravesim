import numpy as np
import matplotlib.pyplot as plt


# # # 创建实数数组
# real_data = np.arange(2, 14, dtype=float)  # 1到12
# # 转换为复数（虚部为0）
# image = real_data.astype(complex)
# image = image.reshape(4, 3)
# f = np.fft.fft2(image)
# f = f.reshape(4, 3)
# array_2d=f
# # print(f)

# import numpy as np
np.random.seed(19260817)
# 创建一个3×4的随机复数数组
array_2d = np.random.rand(4,3) + 1j * np.random.rand(4,3)
print(array_2d.shape[0])
for i in range(array_2d.shape[0]):  # 遍历行
    for j in range(array_2d.shape[1]):  # 遍历列
        element = array_2d[i, j]
        print("{"+f"{element.real:.6f},{element.imag:.6f}"+"},")
print("_________________")
array_2d = np.fft.fft2(array_2d)
for i in range(array_2d.shape[0]):  # 遍历行
    for j in range(array_2d.shape[1]):  # 遍历列
        element = array_2d[i, j]
        print("{"+f"{element.real:.6f},{element.imag:.6f}"+"},")