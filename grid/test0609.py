import logging
import math
from pathlib import Path
import tempfile
import numpy as np
import matplotlib.pyplot as plt

def circle(radius: float, scale_x: float, scale_z: float) -> np.ndarray:
    """
    Generate a circle sample with materials 1 inside and 0 outside the circle
    """

    len_x = int(math.ceil(radius / scale_x * 2))
    print(len_x)
    len_z = int(math.ceil(radius / scale_z * 2))
    print(len_z)
    arr = np.zeros((len_z, len_x), dtype=np.int8)

    for iz in range(len_z):
        z = (iz - len_z / 2) * scale_z
        for ix in range(len_x):
            x = (ix - len_x / 2) * scale_x

            if x * x + z * z <= radius * radius:
                arr[iz, ix] = 1

    return arr

def square(lenx: float, lenz: float, scale_x: float, scale_z: float) -> np.ndarray:
    len_x = int(math.ceil(lenx / scale_x * 2))
    len_z = int(math.ceil(lenz / scale_z * 2))
    arr = np.zeros((len_z, len_x), dtype=np.int8)

    for iz in range(len_z):
        for ix in range(len_x):
            arr[iz, ix] = 1
    return arr
#grid=circle(3e-5, 10 * 1e-6, 10 * 1e-6)
#print(grid)
grid=square(2e-4, 2e-3, 5 * 1e-6, 5 * 1e-6)
print(grid)
np.save("grid0704.npy",grid)
#0704-square(2e-4, 2e-3, 5 * 1e-6, 5 * 1e-6)