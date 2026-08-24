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
    arr = np.zeros((len_z, len_x), dtype=np.uint32)

    for iz in range(len_z):
        z = (iz - len_z / 2) * scale_z
        for ix in range(len_x):
            x = (ix - len_x / 2) * scale_x

            if x * x + z * z <= radius * radius:
                arr[iz, ix] = 1

    return arr

def sphere(radius: float, scale_x: float, scale_y: float, scale_z: float) -> np.ndarray:
    """
    Generate a sphere sample with materials 1 inside and 0 outside the sphere
    """
    len_x = int(math.ceil(radius / scale_x * 2))
    len_y = int(math.ceil(radius / scale_y * 2))
    len_z = int(math.ceil(radius / scale_z * 2))
    arr = np.zeros((len_z, len_y, len_x), dtype=np.uint32)

    for iz in range(len_z):
        z = (iz - len_z / 2) * scale_z
        for iy in range(len_y):
            y = (iy - len_y / 2) * scale_y
            for ix in range(len_x):
                x = (ix - len_x / 2) * scale_x

                if x * x + y * y + z * z <= radius * radius:
                    arr[iz, iy, ix] = 1

    return arr

def hollow_sphere(radius1: float, radius2: float, scale_x: float, scale_y: float, scale_z: float) -> np.ndarray:
    """
    Generate a hollow sphere sample with materials 1 inside and 0 outside the sphere
    """
    len_x = int(math.ceil(radius1 / scale_x * 2))
    len_y = int(math.ceil(radius1 / scale_y * 2))
    len_z = int(math.ceil(radius1 / scale_z * 2))
    arr = np.zeros((len_z, len_y, len_x), dtype=np.uint32)

    for iz in range(len_z):
        z = (iz - len_z / 2) * scale_z
        for iy in range(len_y):
            y = (iy - len_y / 2) * scale_y
            for ix in range(len_x):
                x = (ix - len_x / 2) * scale_x

                if x * x + y * y + z * z <= radius1 * radius1 and x * x + y * y + z * z >= radius2 * radius2:
                    arr[iz, iy, ix] = 1

    return arr

def normalize_void_positions(void_positions):
    if void_positions is None:
        return []
    if isinstance(void_positions, np.ndarray):
        void_positions = void_positions.tolist()
    if isinstance(void_positions, (tuple, list)) and len(void_positions) == 3 and isinstance(void_positions[0], (int, float, np.floating)):
        return [tuple(void_positions)]
    return [tuple(pos) for pos in void_positions]


def add_voids(arr: np.ndarray, void_positions, void_radius: float, scale_x: float, scale_y: float, scale_z: float) -> np.ndarray:
    """Set points inside each void to 0 in an existing 3D grid."""
    void_positions = normalize_void_positions(void_positions)
    if len(void_positions) == 0:
        return arr

    len_z, len_y, len_x = arr.shape
    for iz in range(len_z):
        z = (iz - len_z / 2) * scale_z
        for iy in range(len_y):
            y = (iy - len_y / 2) * scale_y
            for ix in range(len_x):
                x = (ix - len_x / 2) * scale_x
                for void_x, void_y, void_z in void_positions:
                    if (x - void_x) * (x - void_x) + (y - void_y) * (y - void_y) + (z - void_z) * (z - void_z) <= void_radius * void_radius:
                        arr[iz, iy, ix] = 0
                        break
    return arr


def hollow_sphere_with_void(radius1: float, radius2: float, scale_x: float, scale_y: float, scale_z: float, void_number: int, void_positions, void_radius: float) -> np.ndarray:
    """
    Generate a hollow sphere sample with materials 1 inside and 0 outside the sphere, then carve void(s) inside.
    """
    void_positions = normalize_void_positions(void_positions)
    len_x = int(math.ceil(radius1 / scale_x * 2))
    len_y = int(math.ceil(radius1 / scale_y * 2))
    len_z = int(math.ceil(radius1 / scale_z * 2))
    arr = np.zeros((len_z, len_y, len_x), dtype=np.uint32)

    for iz in range(len_z):
        z = (iz - len_z / 2) * scale_z
        for iy in range(len_y):
            y = (iy - len_y / 2) * scale_y
            for ix in range(len_x):
                x = (ix - len_x / 2) * scale_x

                if x * x + y * y + z * z <= radius1 * radius1 and x * x + y * y + z * z >= radius2 * radius2:
                    arr[iz, iy, ix] = 1

    if void_number > 0:
        arr = add_voids(arr, void_positions[:void_number], void_radius, scale_x, scale_y, scale_z)

    return arr

def non_circle(radius: float, scale_x: float, scale_z: float) -> np.ndarray:
    """
    Generate a circle sample with materials 1 inside and 0 outside the circle
    """

    len_x = int(math.ceil(radius / scale_x * 2))
    print(len_x)
    len_z = int(math.ceil(radius / scale_z * 2))
    print(len_z)
    arr = np.zeros((len_z, len_x), dtype=np.uint32)

    for iz in range(len_z):
        z = (iz - len_z / 2) * scale_z
        for ix in range(len_x):
            x = (ix - len_x / 2) * scale_x

            if x * x + z * z <= radius * radius:
                arr[iz, ix] = 0

    return arr

def double_square(lenx1: float, lenx2: float, lenz: float, scale_x: float, scale_z: float) -> np.ndarray:
    a1=square(lenx1,lenz,scale_x,scale_z,1)
    a2=square(lenx2,lenz,scale_x,scale_z,2)
    a3 = np.concatenate((a1, a2), axis=1)
    #plt.imshow(a3)
    #plt.show()
    return a3

def square(lenx: float, lenz: float, scale_x: float, scale_z: float,id: int) -> np.ndarray:
    len_x = int(math.ceil(lenx / scale_x * 2))
    len_z = int(math.ceil(lenz / scale_z * 2))
    arr = np.zeros((len_z, len_x), dtype=np.uint32)

    for iz in range(len_z):
        for ix in range(len_x):
            arr[iz, ix] = id
    return arr

def square1(lenx: float, lenz: float, scale_x: float, scale_z: float,id: int) -> np.ndarray:
    print(lenx / scale_x)
    len_x = int(np.round(lenx / scale_x))
    len_z = int(np.round(lenz / scale_z))
    arr = np.zeros((len_z, len_x), dtype=np.uint32)

    for iz in range(len_z):
        for ix in range(len_x):
            arr[iz, ix] = id
    return arr
#grid=circle(3e-5, 10 * 1e-6, 10 * 1e-6)
#print(grid)
#grid=square(1e-6, 5e-6, 1 * 1e-6, 1 * 1e-6)
#print(grid)
#np.save("square_grid_1_5_1_1.npy",grid)
#0704-square(2e-4, 2e-3, 5 * 1e-6, 5 * 1e-6)

def hollowcircle(radius1: float, radius2: float, scale_x: float, scale_z: float) -> np.ndarray:
    len_x1=int(math.ceil(radius1 / scale_x * 2))
    len_z1=int(math.ceil(radius1 / scale_z * 2))
    arr = np.zeros((len_z1, len_x1), dtype=np.uint32)
    for iz in range(len_z1):
        z = (iz - len_z1 / 2) * scale_z
        for ix in range(len_x1):
            x = (ix - len_x1 / 2) * scale_x

            if x * x + z * z <= radius1 * radius1 and x * x + z * z >= radius2 * radius2:
                arr[iz, ix] = 1
            elif  x * x + z * z <= radius2 * radius2:
                arr[iz, ix] = 0
    return arr

def double_ring(radius1: float, radius2: float, radius3: float, scale_x: float, scale_z: float) -> np.ndarray:
    len_x1=int(math.ceil(radius1 / scale_x * 2))
    len_z1=int(math.ceil(radius1 / scale_z * 2))
    arr = np.zeros((len_z1, len_x1), dtype=np.uint32)
    for iz in range(len_z1):
        z = (iz - len_z1 / 2) * scale_z
        for ix in range(len_x1):
            x = (ix - len_x1 / 2) * scale_x

            if x * x + z * z <= radius1 * radius1 and x * x + z * z >= radius2 * radius2:
                arr[iz, ix] = 1
            elif  x * x + z * z <= radius2 * radius2 and x * x + z * z >= radius3 * radius3:
                arr[iz, ix] = 2
            elif x * x + z * z <= radius3 * radius3:
                arr[iz, ix] = 3
            else:
                arr[iz, ix] = 0
    return arr

def triple_ring(radius1: float, radius2: float, radius3: float, radius4: float, scale_x: float, scale_z: float) -> np.ndarray:
    len_x1=int(math.ceil(radius1 / scale_x * 2))
    len_z1=int(math.ceil(radius1 / scale_z * 2))
    arr = np.zeros((len_z1, len_x1), dtype=np.uint32)
    for iz in range(len_z1):
        z = (iz - len_z1 / 2) * scale_z
        for ix in range(len_x1):
            x = (ix - len_x1 / 2) * scale_x

            if x * x + z * z <= radius1 * radius1 and x * x + z * z >= radius2 * radius2:
                arr[iz, ix] = 1
            elif  x * x + z * z <= radius2 * radius2 and x * x + z * z >= radius3 * radius3:
                arr[iz, ix] = 2
            elif  x * x + z * z <= radius3 * radius3 and x * x + z * z >= radius4 * radius4:
                arr[iz, ix] = 3
            else:
                arr[iz, ix] = 0
    return arr
def defect_generation(arr,scale_x: float ,scale_z: float ,position_x: float ,position_z: float ,radiusd: float ):
    for iz in range(int((position_z-radiusd)/scale_z),int((position_z+radiusd)/scale_z)):
        z = iz * scale_z - position_z
        for ix in range( int((position_x-radiusd)/scale_x),int((position_x+radiusd)/scale_x)):
            x = ix * scale_x - position_x

            if x * x + z * z <= radiusd * radiusd:
                arr[iz, ix] = 1
    return arr

def double_ring_d (radius1: float, radius2: float, radius3: float, scale_x: float, scale_z: float, density1: float, density2: float, density3: float) -> np.ndarray:
    len_x1=int(math.ceil(radius1 / scale_x * 2))
    len_z1=int(math.ceil(radius1 / scale_z * 2))
    arr = np.zeros((len_z1, len_x1), dtype=np.float32)
    for iz in range(len_z1):
        z = (iz - len_z1 / 2) * scale_z
        for ix in range(len_x1):
            x = (ix - len_x1 / 2) * scale_x

            if x * x + z * z <= radius1 * radius1 and x * x + z * z >= radius2 * radius2:
                arr[iz, ix] = density1
            elif  x * x + z * z <= radius2 * radius2 and x * x + z * z >= radius3 * radius3:
                arr[iz, ix] = density2
            elif x * x + z * z <= radius3 * radius3:
                arr[iz, ix] = density3
            else:
                arr[iz, ix] = 0.0
    return arr
def square_d(lenx: float, lenz: float, scale_x: float, scale_z: float,density: float) -> np.ndarray:
    len_x = int(math.ceil(lenx / scale_x * 2))
    len_z = int(math.ceil(lenz / scale_z * 2))
    arr = np.zeros((len_z, len_x), dtype=np.float32)

    for iz in range(len_z):
        for ix in range(len_x):
            arr[iz, ix] = density
    #print(arr)
    arr=arr.astype(np.float32)
    return arr
def defect_generation_d(arr,scale_x: float ,scale_z: float ,position_x: float ,position_z: float ,radiusd: float, density: float):
    for iz in range(int((position_z-radiusd)/scale_z),int((position_z+radiusd)/scale_z)):
        z = iz * scale_z - position_z
        for ix in range( int((position_x-radiusd)/scale_x),int((position_x+radiusd)/scale_x)):
            x = ix * scale_x - position_x

            if x * x + z * z <= radiusd * radiusd:
                arr[iz, ix] = density
    return arr

# grid=double_ring(911e-6, 844e-6, 803e-6, 1 * 1e-7, 1 * 1e-7)
# grid=defect_generation(grid, 1 * 1e-7, 1 * 1e-7, 80e-6 , 911e-6 , 5e-6)
# grid=defect_generation(grid, 1 * 1e-7, 1 * 1e-7, 911e-6 , 70e-6 , 1e-6)
# grid=defect_generation(grid, 1 * 1e-7, 1 * 1e-7, 316e-6 , 316e-6 , 3e-6)
# np.save("m_grid0922.npy",grid)
#grid=defect_generation(grid, 1 * 1e-6, 1 * 1e-6, 1600e-6 , 470e-6 , 3e-6)

# d_grid=double_ring_d(911e-6, 844e-6, 803e-6, 1 * 1e-7, 1 * 1e-7, 3.23, 0.25, 0.44e-3)
# print(d_grid)
# d_grid=defect_generation_d(d_grid, 1 * 1e-7, 1 * 1e-7, 80e-6 , 911e-6 , 5e-6, 3.23)
# d_grid=defect_generation_d(d_grid, 1 * 1e-7, 1 * 1e-7, 911e-6 , 70e-6 , 1e-6, 3.23)
# d_grid=defect_generation_d(d_grid, 1 * 1e-7, 1 * 1e-7, 316e-6 , 316e-6 , 3e-6, 3.23)
# np.save("d_grid0922.npy",d_grid)
# plt.imshow(d_grid)
# plt.xlabel("x_length/um")
# plt.ylabel("z_length/um")
# plt.show()
# grid=double_square(6e-6,6.5e-6,1e-3,1e-7,1e-5)
# grid=grid.T
# np.save("shockwave_CH.npy",grid)
import numpy as np
from scipy.ndimage import rotate
# m_grid1=square(1e-3, 2e-3, 1 * 1e-6, 1 * 1e-6,1)
# m_grid2=square(1e-3, 2e-3, 1 * 1e-6, 1 * 1e-6,0)
# m_grid=np.hstack((m_grid1,m_grid2))
m_grid=square(2e-3, 2e-3, 10 * 1e-7, 10 * 1e-7,1)

# 原始二维数组（例如一个 3x3 的矩阵）

# 旋转角度（度数）
angle = -0.5  # 任意角度

# 旋转数组，reshape=True 表示自动扩展尺寸以包含所有原始数据
# mode='constant', cval=0 表示超出原数组范围的部分填充 0
rotated = rotate(m_grid, angle, reshape=True, mode='constant', cval=0)

# print("原始数组：\n", m_grid)
# print("旋转后的数组：\n", rotated)
# # rotated=rotated.T
# plt.imshow(rotated)
# plt.colorbar()
# plt.show()
# np.save("m_grid_square_2_2_1_1__-1.npy",rotated)
# # d_grid=square_d(3e-3, 1e-3, 5 * 1e-6, 5 * 1e-6, 1.3)
# # np.save("d_grid_square250928.npy",d_grid)



def resolution_generation_04(resolution: float):
    petfilm=square1(resolution*9, 25e-6, 1 * 1e-7, 0.5 * 1e-7,3)
    sinfilm=square1(resolution*9, 0.2e-6, 1 * 1e-7, 0.5 * 1e-7,2)
    solid=square1(resolution, 0.65e-6, 1 * 1e-7, 0.5 * 1e-7,1)
    void=square1(resolution, 0.65e-6, 1 * 1e-7, 0.5 * 1e-7,2)
    sibase=square1(resolution*9, 15e-6, 1 * 1e-7, 0.5 * 1e-7,4)
    grating=square1(resolution*9, 0.2e-6, 1 * 1e-7, 0.5 * 1e-7,2)
    grating=np.hstack((void,void,solid,void,solid,void,solid,void,void))
    # print(grating.shape)
    # print(petfilm.shape)
    total=np.vstack((petfilm,sinfilm,grating,sinfilm,sibase))
    plt.imshow(total)
    plt.show()
    return total
    #W 19.35 SiN 3.44 PET 1.35 Si 2.33
def resolution_generation_05B(resolution: float):
    petfilm=square1(resolution*9, 50e-6, 1 * 1e-7, 0.5 * 1e-7,2)
    solid=square1(resolution, 1e-6, 1 * 1e-7, 0.5 * 1e-7,1)
    void=square1(resolution, 1e-6, 1 * 1e-7, 0.5 * 1e-7,0)
    sibase=square1(resolution*9, 200e-6, 1 * 1e-7, 0.5 * 1e-7,3)
    grating=np.hstack((void,void,solid,void,solid,void,solid,void,void))
    # print(grating.shape)
    # print(petfilm.shape)
    total=np.vstack((petfilm,grating,sibase))
    plt.imshow(total)
    plt.show()
    return total
    #Au 19.32 ["Au", 19.32],["C10H8O4",1.35],["Si",2.33]],
def five_line(resolution: float):
    line=circle(resolution/2,1e-7,1e-7)
    void=non_circle(resolution/2,1e-7,1e-7)
    line_pair=np.hstack((line,void,line,void,line,void,line,void,line))
    plt.imshow(line_pair)
    plt.show()
    return line_pair
def stair2(depth1: float):
    void=square1(1e-3,1e-3,1e-6,1e-6,0)
    solid=square1(1e-3,1e-3,1e-6,1e-6,1)
    base=np.hstack((void,solid,solid))
    level=np.hstack((void,void,solid))
    total=np.vstack((base,level))
    plt.imshow(total)
    plt.show()
    return total
def stair(n:int, depth:float):
    sx=1e-6
    sz=1e-6
    void=square1(depth,depth,sx,sz,0)
    solid=square1(depth,depth,sx,sz,1)
    total=square1(depth*(n+1),depth,sx,sz,0)
    for i in range(1,n+1):
        base=square1(depth,depth,sx,sz,0)
        print(i)
        for j in range(1,n+1):
            print(j)
            if i+j>=n+1 :
                base=np.hstack((base,solid))
            else :
                base=np.hstack((base,void))
        total=np.vstack((total,base))
    plt.imshow(total)
    plt.show()
    return total
# grid=resolution_generation_05B(5e-6)
# grid=three_line(5e-6)
# grid=stair(4,5e-4)
# grid=np.hstack((square1(2e-4,2e-4,2e-7,2e-7,1),square1(1.2*1e-4,2e-4,2e-7,2e-7,2),square1(0.2*1e-4,2e-4,2e-7,2e-7,3),square1(0.6*1e-4,2e-4,2e-7,2e-7,4)))
# grid=np.hstack((square1(2e-4,2e-4,2e-7,2e-7,1),square1(1.2*1e-4,2e-4,2e-7,2e-7,2),square1(0.2*1e-4,2e-4,2e-7,2e-7,3),square1(0.6*1e-4,2e-4,2e-7,2e-7,4)))
# output_species_final = np.zeros((1000, 2000), dtype=np.uint32)
# for i in range(1000):    
#     for j in range(2000):
#         output_species_final[i,j]=grid[0][j]
# # output_species_final=output_species_final.T
# np.set_printoptions(threshold=np.inf)
# file=open('shockwave_sample_preview.txt','w')
# file.write(str(grid))
# file.close()
# #[['SiO2',2.65],['C8H8',1.06],['C8H8',3.79],['C8H8',0.79]]
# np.save("D:/rave-sim-main/rave-sim-main/grid/shockwave_same_process_sample_80.npy",grid)
vp = (1e-4, 1e-4, np.sqrt(4e-4**2 - 2 * 1e-4**2))
grid = hollow_sphere_with_void(5e-4, 4e-4, 1e-6, 1e-6, 1e-6, 1, [vp], 1e-6)
np.save("D:/rave-sim-main/rave-sim-main/grid/100um_half_hollow_millisphere_with_1e-6_with_void_grid.npy", grid)