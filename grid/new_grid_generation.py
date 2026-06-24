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


def _sph_harm(l: int, m: int, theta: float, phi: float) -> float:
    """Evaluate real spherical harmonic Y_l^m(theta, phi) without scipy.

    Uses the associated Legendre polynomial recurrence for l <= 4.
    For higher l, prefer scipy.special.sph_harm.

    Parameters
    ----------
    l : int
        Degree (l >= 0).
    m : int
        Order (|m| <= l).
    theta : float
        Polar angle from z-axis [0, pi].
    phi : float
        Azimuthal angle [0, 2pi).
    """
    from math import sqrt, cos, sin, pi

    N = {
        (0, 0): sqrt(1.0 / (4 * pi)),
        (1, -1): sqrt(3.0 / (8 * pi)),
        (1, 0):  sqrt(3.0 / (4 * pi)),
        (1, 1):  sqrt(3.0 / (8 * pi)),
        (2, -2): sqrt(15.0 / (32 * pi)),
        (2, -1): sqrt(15.0 / (8 * pi)),
        (2, 0):  sqrt(5.0 / (16 * pi)),
        (2, 1):  sqrt(15.0 / (8 * pi)),
        (2, 2):  sqrt(15.0 / (32 * pi)),
        (3, -3): sqrt(35.0 / (64 * pi)),
        (3, -2): sqrt(105.0 / (32 * pi)),
        (3, -1): sqrt(21.0 / (64 * pi)),
        (3, 0):  sqrt(7.0 / (16 * pi)),
        (3, 1):  sqrt(21.0 / (64 * pi)),
        (3, 2):  sqrt(105.0 / (32 * pi)),
        (3, 3):  sqrt(35.0 / (64 * pi)),
        (4, -4): sqrt(315.0 / (512 * pi)),
        (4, -3): sqrt(315.0 / (64 * pi)),
        (4, -2): sqrt(45.0 / (128 * pi)),
        (4, -1): sqrt(45.0 / (64 * pi)),
        (4, 0):  sqrt(81.0 / (256 * pi)),
        (4, 1):  sqrt(45.0 / (64 * pi)),
        (4, 2):  sqrt(45.0 / (128 * pi)),
        (4, 3):  sqrt(315.0 / (64 * pi)),
        (4, 4):  sqrt(315.0 / (512 * pi)),
    }

    x = cos(theta)
    s = sin(theta)

    if l == 0:
        P = 1.0
    elif l == 1:
        if m == -1: P = 0.5 * s
        elif m == 0: P = x
        elif m == 1: P = -s
        else: raise ValueError(f"Invalid m={m} for l={l}")
    elif l == 2:
        s2 = s * s
        if m == -2: P = 0.25 * s2
        elif m == -1: P = 0.5 * x * s
        elif m == 0:  P = 0.5 * (3 * x * x - 1)
        elif m == 1:  P = -3 * x * s
        elif m == 2:  P = 3 * s2
        else: raise ValueError(f"Invalid m={m} for l={l}")
    elif l == 3:
        x2 = x * x; s2 = s * s
        if m == -3: P = (1.0 / 8) * s2 * s
        elif m == -2: P = (1.0 / 4) * x * s2
        elif m == -1: P = (1.0 / 8) * s * (5 * x2 - 1)
        elif m == 0:  P = 0.5 * x * (5 * x2 - 3)
        elif m == 1:  P = -(3.0 / 2) * s * (5 * x2 - 1)
        elif m == 2:  P = 15 * x * s2
        elif m == 3:  P = -15 * s2 * s
        else: raise ValueError(f"Invalid m={m} for l={l}")
    elif l == 4:
        x2 = x * x; s2 = s * s
        if m == -4: P = (3.0 / 16) * s2 * s2
        elif m == -3: P = (3.0 / 8) * x * s2 * s
        elif m == -2: P = (3.0 / 8) * s2 * (7 * x2 - 1)
        elif m == -1: P = (3.0 / 8) * x * s * (7 * x2 - 3)
        elif m == 0:  P = (1.0 / 8) * (35 * x2 * x2 - 30 * x2 + 3)
        elif m == 1:  P = -(5.0 / 2) * s * (7 * x2 * x - 3 * x)
        elif m == 2:  P = (15.0 / 2) * s2 * (7 * x2 - 1)
        elif m == 3:  P = -105 * x * s2 * s
        elif m == 4:  P = 105 * s2 * s2
        else: raise ValueError(f"Invalid m={m} for l={l}")
    else:
        raise ValueError(f"l={l} > 4 not supported without scipy; "
                         "install scipy and use scipy.special.sph_harm")

    norm = N.get((l, m))
    if norm is None:
        raise ValueError(f"No pre-computed norm for l={l}, m={m}; l <= 4 only")

    if m == 0:
        return norm * P
    elif m > 0:
        return norm * P * cos(m * phi) * sqrt(2.0)
    else:
        return norm * P * sin(abs(m) * phi) * sqrt(2.0)


def non_spherical_solid(base_radius: float, l: int, m: int, epsilon: float,
                        scale_x: float, scale_y: float, scale_z: float,
                        dtype: np.dtype = np.uint32) -> np.ndarray:
    """Generate a 3D solid bounded by a spherical-harmonic surface.

    The surface radius at each angle is:
        r(theta, phi) = base_radius * (1 + eps * Y_l^m(theta, phi))

    Voxels inside r(theta, phi) are set to 1, outside to 0.
    """
    max_radius = base_radius * (1 + abs(epsilon))
    len_x = int(math.ceil(max_radius / scale_x * 2))
    len_y = int(math.ceil(max_radius / scale_y * 2))
    len_z = int(math.ceil(max_radius / scale_z * 2))
    arr = np.zeros((len_z, len_y, len_x), dtype=dtype)

    cx, cy, cz = len_x / 2, len_y / 2, len_z / 2

    for iz in range(len_z):
        z = (iz - cz) * scale_z
        for iy in range(len_y):
            y = (iy - cy) * scale_y
            for ix in range(len_x):
                x = (ix - cx) * scale_x
                r = math.sqrt(x * x + y * y + z * z)
                if r == 0:
                    if base_radius > 0:
                        arr[iz, iy, ix] = 1
                    continue

                theta = math.acos(z / r)
                phi = math.atan2(y, x)

                r_surface = base_radius * (1 + epsilon * _sph_harm(l, m, theta, phi))
                if r <= r_surface:
                    arr[iz, iy, ix] = 1
    return arr


def non_spherical_shell(base_radius: float, thickness: float, l: int, m: int, epsilon: float,
                        scale_x: float, scale_y: float, scale_z: float,
                        dtype: np.dtype = np.uint32) -> np.ndarray:
    """Generate a 3D shell bounded by inner/outer spherical-harmonic surfaces.

    Inner surface: r(theta, phi) = base_radius * (1 + eps * Y_l^m(theta, phi))
    Outer surface: inner + thickness (constant along radial direction; the shell
    has uniform angular thickness).

    Voxels between the two surfaces are set to 1, outside to 0.
    """
    max_radius = (base_radius + thickness) * (1 + abs(epsilon))
    len_x = int(math.ceil(max_radius / scale_x * 2))
    len_y = int(math.ceil(max_radius / scale_y * 2))
    len_z = int(math.ceil(max_radius / scale_z * 2))
    arr = np.zeros((len_z, len_y, len_x), dtype=dtype)

    cx, cy, cz = len_x / 2, len_y / 2, len_z / 2

    for iz in range(len_z):
        z = (iz - cz) * scale_z
        for iy in range(len_y):
            y = (iy - cy) * scale_y
            for ix in range(len_x):
                x = (ix - cx) * scale_x
                r = math.sqrt(x * x + y * y + z * z)
                if r == 0:
                    continue

                theta = math.acos(z / r)
                phi = math.atan2(y, x)

                r_inner = base_radius * (1 + epsilon * _sph_harm(l, m, theta, phi))
                r_outer = r_inner + thickness

                if r_inner <= r <= r_outer:
                    arr[iz, iy, ix] = 1
    return arr


def non_spherical_shell_variable_thickness(
    base_radius: float, thickness0: float, l: int, m: int,
    epsilon_r: float, epsilon_t: float,
    scale_x: float, scale_y: float, scale_z: float,
    dtype: np.dtype = np.uint32
) -> np.ndarray:
    """Generate a 3D shell where both the mean radius AND thickness vary
    independently with spherical harmonics.

    Inner surface: r_in(theta, phi) = base_radius * (1 + eps_r * Y_l^m(theta, phi))
    Outer surface: r_out(theta, phi) = (base_radius + thickness0) * (1 + eps_t * Y_l^m(theta, phi))

    This allows independent control of the outer shape (eps_t) and inner
    cavity shape (eps_r), including pure offset, pure squash, or both.
    """
    r_out_mean = base_radius + thickness0
    max_r_in = base_radius * (1 + abs(epsilon_r))
    max_r_out = r_out_mean * (1 + abs(epsilon_t))
    max_radius = max(max_r_in, max_r_out)

    len_x = int(math.ceil(max_radius / scale_x * 2))
    len_y = int(math.ceil(max_radius / scale_y * 2))
    len_z = int(math.ceil(max_radius / scale_z * 2))
    arr = np.zeros((len_z, len_y, len_x), dtype=dtype)

    cx, cy, cz = len_x / 2, len_y / 2, len_z / 2

    for iz in range(len_z):
        z = (iz - cz) * scale_z
        for iy in range(len_y):
            y = (iy - cy) * scale_y
            for ix in range(len_x):
                x = (ix - cx) * scale_x
                r = math.sqrt(x * x + y * y + z * z)
                if r == 0:
                    continue

                theta = math.acos(z / r)
                phi = math.atan2(y, x)

                Y = _sph_harm(l, m, theta, phi)

                r_in = base_radius * (1 + epsilon_r * Y)
                r_out = r_out_mean * (1 + epsilon_t * Y)

                if r_in < r_out:
                    if r_in <= r <= r_out:
                        arr[iz, iy, ix] = 1
                else:
                    pass
    return arr


def _sph_harm_2d_analytic(l: int, phi: float) -> float:
    """Real Fourier-mode basis on the circle: cos(l*phi) / sin(l*phi).

    For 2D the 'spherical harmonics' reduce to Fourier modes on S^1.
    The convention here uses cos(m*phi) for m >= 0 and sin(|m|*phi) for m < 0,
    normalised such that integral Y^2 dphi = 1.
    """
    from math import sqrt, cos, sin, pi
    if l > 0:
        return cos(l * phi) * sqrt(2.0 / pi)
    elif l < 0:
        return sin(abs(l) * phi) * sqrt(2.0 / pi)
    else:
        return sqrt(1.0 / pi)
