import math
import numpy as np


def circle(radius: float, scale_x: float, scale_z: float) -> np.ndarray:
    """Generate a 2D circle: material 1 inside, 0 outside."""
    len_x = int(math.ceil(radius / scale_x * 2))
    len_z = int(math.ceil(radius / scale_z * 2))
    arr = np.zeros((len_z, len_x), dtype=np.uint32)

    for iz in range(len_z):
        z = (iz - len_z / 2) * scale_z
        for ix in range(len_x):
            x = (ix - len_x / 2) * scale_x
            if x * x + z * z <= radius * radius:
                arr[iz, ix] = 1
    return arr


def non_circle(radius: float, scale_x: float, scale_z: float) -> np.ndarray:
    """Generate an inverted 2D circle: material 0 inside, 1 outside."""
    len_x = int(math.ceil(radius / scale_x * 2))
    len_z = int(math.ceil(radius / scale_z * 2))
    arr = np.ones((len_z, len_x), dtype=np.uint32)

    for iz in range(len_z):
        z = (iz - len_z / 2) * scale_z
        for ix in range(len_x):
            x = (ix - len_x / 2) * scale_x
            if x * x + z * z <= radius * radius:
                arr[iz, ix] = 0
    return arr


def square(lenx: float, lenz: float, scale_x: float, scale_z: float, id: int) -> np.ndarray:
    """Generate a 2D rectangle filled with material id."""
    len_x = int(math.ceil(lenx / scale_x * 2))
    len_z = int(math.ceil(lenz / scale_z * 2))
    arr = np.full((len_z, len_x), id, dtype=np.uint32)
    return arr


def square1(lenx: float, lenz: float, scale_x: float, scale_z: float, id: int) -> np.ndarray:
    """Generate a 2D rectangle filled with material id (using round)."""
    len_x = int(np.round(lenx / scale_x))
    len_z = int(np.round(lenz / scale_z))
    arr = np.full((len_z, len_x), id, dtype=np.uint32)
    return arr


def double_square(lenx1: float, lenx2: float, lenz: float, scale_x: float, scale_z: float) -> np.ndarray:
    """Horizontally concatenate two rectangles."""
    a1 = square(lenx1, lenz, scale_x, scale_z, 1)
    a2 = square(lenx2, lenz, scale_x, scale_z, 2)
    return np.concatenate((a1, a2), axis=1)


def hollowcircle(radius1: float, radius2: float, scale_x: float, scale_z: float) -> np.ndarray:
    """Generate a 2D hollow circle (ring): ring region = 1."""
    len_x = int(math.ceil(radius1 / scale_x * 2))
    len_z = int(math.ceil(radius1 / scale_z * 2))
    arr = np.zeros((len_z, len_x), dtype=np.uint32)

    for iz in range(len_z):
        z = (iz - len_z / 2) * scale_z
        for ix in range(len_x):
            x = (ix - len_x / 2) * scale_x
            r2 = x * x + z * z
            if r2 <= radius1 * radius1 and r2 >= radius2 * radius2:
                arr[iz, ix] = 1
    return arr


def double_ring(radius1: float, radius2: float, radius3: float, scale_x: float, scale_z: float) -> np.ndarray:
    """Generate a 2D double ring: materials 1, 2, 3 / 0 from outer to inner."""
    len_x = int(math.ceil(radius1 / scale_x * 2))
    len_z = int(math.ceil(radius1 / scale_z * 2))
    arr = np.zeros((len_z, len_x), dtype=np.uint32)

    for iz in range(len_z):
        z = (iz - len_z / 2) * scale_z
        for ix in range(len_x):
            x = (ix - len_x / 2) * scale_x
            r2 = x * x + z * z
            if r2 <= radius1 * radius1 and r2 >= radius2 * radius2:
                arr[iz, ix] = 1
            elif r2 <= radius2 * radius2 and r2 >= radius3 * radius3:
                arr[iz, ix] = 2
            elif r2 <= radius3 * radius3:
                arr[iz, ix] = 3
    return arr


def triple_ring(radius1: float, radius2: float, radius3: float, radius4: float, scale_x: float, scale_z: float) -> np.ndarray:
    """Generate a 2D triple ring: materials 1, 2, 3 / 0 from outer to inner."""
    len_x = int(math.ceil(radius1 / scale_x * 2))
    len_z = int(math.ceil(radius1 / scale_z * 2))
    arr = np.zeros((len_z, len_x), dtype=np.uint32)

    for iz in range(len_z):
        z = (iz - len_z / 2) * scale_z
        for ix in range(len_x):
            x = (ix - len_x / 2) * scale_x
            r2 = x * x + z * z
            if r2 <= radius1 * radius1 and r2 >= radius2 * radius2:
                arr[iz, ix] = 1
            elif r2 <= radius2 * radius2 and r2 >= radius3 * radius3:
                arr[iz, ix] = 2
            elif r2 <= radius3 * radius3 and r2 >= radius4 * radius4:
                arr[iz, ix] = 3
    return arr


def double_ring_d(radius1: float, radius2: float, radius3: float, scale_x: float, scale_z: float,
                  val1: int, val2: int, val3: int) -> np.ndarray:
    """Generate a 2D double ring with custom int32 values per ring."""
    len_x = int(math.ceil(radius1 / scale_x * 2))
    len_z = int(math.ceil(radius1 / scale_z * 2))
    arr = np.zeros((len_z, len_x), dtype=np.uint32)

    for iz in range(len_z):
        z = (iz - len_z / 2) * scale_z
        for ix in range(len_x):
            x = (ix - len_x / 2) * scale_x
            r2 = x * x + z * z
            if r2 <= radius1 * radius1 and r2 >= radius2 * radius2:
                arr[iz, ix] = val1
            elif r2 <= radius2 * radius2 and r2 >= radius3 * radius3:
                arr[iz, ix] = val2
            elif r2 <= radius3 * radius3:
                arr[iz, ix] = val3
    return arr


def square_d(lenx: float, lenz: float, scale_x: float, scale_z: float, value: int) -> np.ndarray:
    """Generate a 2D rectangle filled with a custom int32 value."""
    len_x = int(math.ceil(lenx / scale_x * 2))
    len_z = int(math.ceil(lenz / scale_z * 2))
    return np.full((len_z, len_x), value, dtype=np.uint32)


def defect_generation(arr: np.ndarray, scale_x: float, scale_z: float,
                      position_x: float, position_z: float, radiusd: float) -> np.ndarray:
    """Insert a circular defect (material 1) into an existing 2D grid."""
    for iz in range(int((position_z - radiusd) / scale_z), int((position_z + radiusd) / scale_z)):
        z = iz * scale_z - position_z
        for ix in range(int((position_x - radiusd) / scale_x), int((position_x + radiusd) / scale_x)):
            x = ix * scale_x - position_x
            if x * x + z * z <= radiusd * radiusd:
                arr[iz, ix] = 1
    return arr


def defect_generation_d(arr: np.ndarray, scale_x: float, scale_z: float,
                        position_x: float, position_z: float, radiusd: float, value: int) -> np.ndarray:
    """Insert a circular defect with custom int32 value into an existing 2D grid."""
    for iz in range(int((position_z - radiusd) / scale_z), int((position_z + radiusd) / scale_z)):
        z = iz * scale_z - position_z
        for ix in range(int((position_x - radiusd) / scale_x), int((position_x + radiusd) / scale_x)):
            x = ix * scale_x - position_x
            if x * x + z * z <= radiusd * radiusd:
                arr[iz, ix] = value
    return arr


def sphere(radius: float, scale_x: float, scale_y: float, scale_z: float) -> np.ndarray:
    """Generate a 3D sphere: material 1 inside, 0 outside."""
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


def hollow_sphere(radius1: float, radius2: float,
                  scale_x: float, scale_y: float, scale_z: float) -> np.ndarray:
    """Generate a 3D hollow sphere: shell region = 1."""
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
                r2 = x * x + y * y + z * z
                if r2 <= radius1 * radius1 and r2 >= radius2 * radius2:
                    arr[iz, iy, ix] = 1
    return arr


def normalize_void_positions(void_positions):
    """Normalize void position input into a list of tuples."""
    if void_positions is None:
        return []
    if isinstance(void_positions, np.ndarray):
        void_positions = void_positions.tolist()
    if isinstance(void_positions, (tuple, list)) and len(void_positions) == 3 and isinstance(void_positions[0], (int, float, np.floating)):
        return [tuple(void_positions)]
    return [tuple(pos) for pos in void_positions]


def add_voids(arr: np.ndarray, void_positions, void_radius: float,
              scale_x: float, scale_y: float, scale_z: float) -> np.ndarray:
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
                    if (x - void_x) ** 2 + (y - void_y) ** 2 + (z - void_z) ** 2 <= void_radius * void_radius:
                        arr[iz, iy, ix] = 0
                        break
    return arr


def hollow_sphere_with_void(radius1: float, radius2: float,
                            scale_x: float, scale_y: float, scale_z: float,
                            void_number: int, void_positions, void_radius: float) -> np.ndarray:
    """Generate a hollow sphere then carve void(s) inside."""
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
                r2 = x * x + y * y + z * z
                if r2 <= radius1 * radius1 and r2 >= radius2 * radius2:
                    arr[iz, iy, ix] = 1

    if void_number > 0:
        arr = add_voids(arr, void_positions[:void_number], void_radius, scale_x, scale_y, scale_z)
    return arr


def five_line(resolution: float) -> np.ndarray:
    """Generate a five-line resolution test pattern."""
    line = circle(resolution / 2, 1e-7, 1e-7)
    void = non_circle(resolution / 2, 1e-7, 1e-7)
    line_pair = np.hstack((line, void, line, void, line, void, line, void, line))
    return line_pair


def resolution_generation_04(resolution: float) -> np.ndarray:
    """Resolution test pattern 04: multilayer film stack."""
    petfilm = square1(resolution * 9, 25e-6, 1e-7, 0.5e-7, 3)
    sinfilm = square1(resolution * 9, 0.2e-6, 1e-7, 0.5e-7, 2)
    solid = square1(resolution, 0.65e-6, 1e-7, 0.5e-7, 1)
    void = square1(resolution, 0.65e-6, 1e-7, 0.5e-7, 2)
    sibase = square1(resolution * 9, 15e-6, 1e-7, 0.5e-7, 4)
    grating = square1(resolution * 9, 0.2e-6, 1e-7, 0.5e-7, 2)
    grating = np.hstack((void, void, solid, void, solid, void, solid, void, void))
    total = np.vstack((petfilm, sinfilm, grating, sinfilm, sibase))
    return total


def resolution_generation_05B(resolution: float) -> np.ndarray:
    """Resolution test pattern 05B: multilayer film stack."""
    petfilm = square1(resolution * 9, 50e-6, 1e-7, 0.5e-7, 2)
    solid = square1(resolution, 1e-6, 1e-7, 0.5e-7, 1)
    void = square1(resolution, 1e-6, 1e-7, 0.5e-7, 0)
    sibase = square1(resolution * 9, 200e-6, 1e-7, 0.5e-7, 3)
    grating = np.hstack((void, void, solid, void, solid, void, solid, void, void))
    total = np.vstack((petfilm, grating, sibase))
    return total


def stair2(depth1: float) -> np.ndarray:
    """Generate a 2-level stair structure."""
    void = square1(1e-3, 1e-3, 1e-6, 1e-6, 0)
    solid = square1(1e-3, 1e-3, 1e-6, 1e-6, 1)
    base = np.hstack((void, solid, solid))
    level = np.hstack((void, void, solid))
    total = np.vstack((base, level))
    return total


def stair(n: int, depth: float) -> np.ndarray:
    """Generate an n-level stair structure."""
    sx = 1e-6
    sz = 1e-6
    void = square1(depth, depth, sx, sz, 0)
    solid = square1(depth, depth, sx, sz, 1)
    total = square1(depth * (n + 1), depth, sx, sz, 0)

    for i in range(1, n + 1):
        base = square1(depth, depth, sx, sz, 0)
        for j in range(1, n + 1):
            if i + j >= n + 1:
                base = np.hstack((base, solid))
            else:
                base = np.hstack((base, void))
        total = np.vstack((total, base))
    return total
