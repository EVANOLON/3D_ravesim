// Copyright (c) 2024, ETH =Zurich

#include "kernels.hpp"
#include "wrappers.hpp"
#include <stdio.h>
template <typename S>
__global__ void propagate_analytically_kernel(DevComplex<S> *d_u, SimParams params, double x_source,
                                              double z) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= params.N)
        return;

    // This part is really sensitive to float precision so we use
    // doubles here.
    const double x = static_cast<double>(i - params.N / 2) * params.dx - x_source;
    const double r = sqrt(x * x + z * z);
    // formula correct
    // const double phase = r * -2.0f * M_PI / params.wl;
    const double phase = r * 2.0f * M_PI / params.wl;


    const double inv_sqrt_r = 1.0 / sqrt(r);
    d_u[i].x = static_cast<S>(cos(phase) * inv_sqrt_r);
    d_u[i].y = static_cast<S>(sin(phase) * inv_sqrt_r);
}

template <typename S>
void propagate_analytically(DevComplex<S> *d_u, const SimParams &params, double x_source,
                            double z) {
    propagate_analytically_kernel<S><<<(params.N + 63) / 64, 64>>>(d_u, params, x_source, z);
}

template <typename S> [[nodiscard]] __host__ __device__ S fftfreq(int i, int N, S dx) {
    const int shifted = i - N * (i >= N / 2);
    return static_cast<S>(shifted) / (static_cast<S>(N) * dx);
}

template <typename S>
[[nodiscard]] __device__ DevComplex<S> complex_mult(DevComplex<S> a, DevComplex<S> b) {
    return DevComplex<S>{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

template <typename S>
__global__ void propagate_convolve_step_kernel(DevComplex<S> *d_U, SimParams params, double dz,
                                               double cutoff_freq) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= params.N)
        return;

    const double ff = fftfreq<double>(i, params.N, params.dx);

    // Only perform the calculation for the entries that don't get zeroed out
    // in the frequency cutoff.
    if (abs(ff) <= cutoff_freq) {
        //formula correct
        //const double angle = M_PI * dz * (-2 / params.wl + params.wl * ff * ff);
        const double angle = M_PI * dz * (2 / params.wl - params.wl * ff * ff);
        d_U[i] = complex_mult<S>(
            d_U[i], DevComplex<S>{static_cast<S>(cos(angle)), static_cast<S>(sin(angle))});

    } else {
        d_U[i] = DevComplex<S>{0.f, 0.f};
    }
}

template <typename S>
void propagate_convolve_step(DevComplex<S> *d_U, const SimParams &params, double dz,
                             double cutoff_freq) {
    propagate_convolve_step_kernel<S><<<(params.N + 63) / 64, 64>>>(d_U, params, dz, cutoff_freq);
}

//3d start

// 角度归约函数（GPU和CPU通用）
__device__ __host__ float reduce_angle(float angle) {
    const float two_pi = 2.0f * M_PI;
    // 归约到[-2π, 2π]
    angle = fmodf(angle, two_pi);
    // 进一步归约到[-π, π]
    if (angle > M_PI) {
        angle -= two_pi;
    } else if (angle < -M_PI) {
        angle += two_pi;
    }
    return angle;
}
__device__ __host__ double reduce_angle(double angle) {
    const double two_pi = 2.0 * M_PI;
    // 归约到[-2π, 2π]
    angle = fmod(angle, two_pi);
    // 进一步归约到[-π, π]
    if (angle > M_PI) {
        angle -= two_pi;
    } else if (angle < -M_PI) {
        angle += two_pi;
    } 
    return angle;
}


// Add this once at file scope to catch layout mismatches at compile time:
static_assert(sizeof(std::optional<int>)    == sizeof(int)    + alignof(int),    "optional<int> layout assumption violated");
static_assert(sizeof(std::optional<double>) == sizeof(double) + alignof(double), "optional<double> layout assumption violated");
// 2D传播卷积核函数 - 傍轴近似（Fresnel）
template <typename S>
__global__ void propagate_analytically_2d_kernel(DevComplex<S> *d_u, 
                                                int nx, int ny, double dx, double dy, double wl,
                                                double x_source, double y_source, double z) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    const int idx = iy * nx + ix;
    const double x = static_cast<double>(ix - nx / 2) * dx - x_source;
    const double y = static_cast<double>(iy - ny / 2) * dy - y_source;
    const double r = sqrt(x * x + y * y + z * z);
    //formula correct
    //const double phase = r * -2.0 * M_PI / wl;
    const double phase = r * 2.0 * M_PI / wl;
    const double inv_r = 1.0 / r;
    double angle = fmod(phase, 2.0 * M_PI);
    if (angle > M_PI) angle -= 2.0 * M_PI;
    else if (angle < -M_PI) angle += 2.0 * M_PI;
    d_u[idx].x = static_cast<S>(cos(angle) * inv_r);
    d_u[idx].y = static_cast<S>(sin(angle) * inv_r);
}

// 2D解析传播包装函数
template <typename S>
void propagate_analytically_2d(DevComplex<S> *d_u, const SimParams &params, 
                               double x_source, double y_source, double z) {
    // 在主机端解包可选值
    const int nx = params.nx;
    const int ny = params.ny;
    const double dx = params.dx;
    const double dy = params.dy;
    const double wl = params.wl;

    int actual_nx = (nx > 0) ? nx : params.N;
    int actual_ny = (ny > 0) ? ny : 1;

    dim3 blockDim(16, 16);
    dim3 gridDim((actual_nx + blockDim.x - 1) / blockDim.x,
                 (actual_ny + blockDim.y - 1) / blockDim.y);

    propagate_analytically_2d_kernel<S><<<gridDim, blockDim>>>(
        d_u, actual_nx, actual_ny, dx, dy, wl, x_source, y_source, z);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in propagate_analytically_2d: %s\n", cudaGetErrorString(err));
    }
}
template <typename S> 
[[nodiscard]] __host__ __device__ S fftfreq_2d_x(int i, int N, S dx) {
    const int shifted = i - N * (i >= N / 2);
    return static_cast<S>(shifted) / (static_cast<S>(N) * dx);
}

template <typename S> 
[[nodiscard]] __host__ __device__ S fftfreq_2d_y(int i, int N, S dy) {
    const int shifted = i - N * (i >= N / 2);
    return static_cast<S>(shifted) / (static_cast<S>(N) * dy);
}
template <typename S>
__global__ void propagate_convolve_step_2d_kernel(DevComplex<S> *d_U, SimParams params, double dz,
                                                  double cutoff_freq_x, double cutoff_freq_y) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 使用helper函数获取值
    const int nx = params.nx;
    const int ny = params.ny;
    const double dx = params.dx;
    const double dy = params.dy;
    
    if (ix >= nx || iy >= ny) return;
    
    const int idx = iy * nx + ix;
    
    // 计算2D频率
    const double kx = fftfreq_2d_x<double>(ix, nx, dx);
    const double ky = fftfreq_2d_y<double>(iy, ny, dy);
    
    // 圆形截止，半径取欧几里得范数以覆盖矩形探测器角落
    if (kx * kx + ky * ky <= cutoff_freq_x * cutoff_freq_x + cutoff_freq_y * cutoff_freq_y) {
        // 2D傍轴近似相位, formula correct
        double angle = M_PI * dz * (2 / params.wl - params.wl * (kx * kx + ky * ky));
        angle = reduce_angle(angle);
        d_U[idx] = complex_mult<S>(
            d_U[idx], 
            DevComplex<S>{static_cast<S>(cos(angle)), static_cast<S>(sin(angle))}
        );
    } else {
        d_U[idx] = DevComplex<S>{0.f, 0.f};
    }
}
template <typename S>
void propagate_convolve_step_2d(DevComplex<S> *d_U, const SimParams &params, double dz,
                             double cutoff_freq_x, double cutoff_freq_y) {
    // fprintf(stderr, "[HOST] ENTER propagate_convolve_step_2d\n");
    fflush(stderr);
    dim3 blockDim(16, 16);
    const int nx = params.nx;
    const int ny = params.ny;
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y);
    propagate_convolve_step_2d_kernel<S><<<gridDim, blockDim>>>(
        d_U, params, dz, cutoff_freq_x, cutoff_freq_y);
    // fprintf(stderr, "[HOST] Exit propagate_convolve_step_2d\n");
    fflush(stderr);
}
// 2D样本因子应用核函数
template <typename S>
__global__ void apply_sample_factors_2d_kernel(DevComplex<S> *d_u, SimParams params, double dz,
                                              uint32_t *d_sample, double pixel_size_x, double pixel_size_y,
                                              std::size_t x_len, std::size_t y_len, 
                                              DevComplex<double> *d_deltabetas,
                                              int z_slice_index, double x_position, double y_position) {
    
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 使用helper函数获取值
    const int nx = params.nx;
    const int ny = params.ny;
    const double dx = params.dx;
    const double dy = params.dy;  // 默认使用dx
    
    if (ix >= nx || iy >= ny) return;
    
    const int idx = iy * nx + ix;
    
    // 计算物理位置（考虑样本偏移和中心位置）
    const double x = static_cast<double>(ix - nx / 2) * dx + x_position + pixel_size_x * x_len * 0.5;
    const double y = static_cast<double>(iy - ny / 2) * dy + y_position + pixel_size_y * y_len * 0.5;
    
    // 转换为样本网格索引（连续坐标）
    const double x_index = x / pixel_size_x;
    const double y_index = y / pixel_size_y;
    
    // 超出网格范围 -> 不处理（视为真空，波前不变）
    if (x_index < 0.0 || x_index >= static_cast<double>(x_len) - 1.0 ||
        y_index < 0.0 || y_index >= static_cast<double>(y_len) - 1.0) {
        return;
    }
    double x_clamped = x_index;
    double y_clamped = y_index;

    // 取整和小数部分（用于双线性插值）
    const std::size_t x_floor = static_cast<std::size_t>(x_clamped);
    const std::size_t y_floor = static_cast<std::size_t>(y_clamped);
    const double x_frac = x_clamped - static_cast<double>(x_floor);
    const double y_frac = y_clamped - static_cast<double>(y_floor);

    // 计算3D样本索引（z_slice_index是当前z层）
    const std::size_t slice_offset = z_slice_index * y_len * x_len;
    
    // 获取四个角点的材料索引
    const int32_t sample_idx_00 = d_sample[slice_offset + y_floor * x_len + x_floor];
    const int32_t sample_idx_01 = d_sample[slice_offset + y_floor * x_len + x_floor + 1];
    const int32_t sample_idx_10 = d_sample[slice_offset + (y_floor + 1) * x_len + x_floor];
    const int32_t sample_idx_11 = d_sample[slice_offset + (y_floor + 1) * x_len + x_floor + 1];
    
    // 从d_deltabetas获取对应的deltabeta值（复数）
    const DevComplex<double> db_00 = d_deltabetas[sample_idx_00];
    const DevComplex<double> db_01 = d_deltabetas[sample_idx_01];
    const DevComplex<double> db_10 = d_deltabetas[sample_idx_10];
    const DevComplex<double> db_11 = d_deltabetas[sample_idx_11];
    
    // 双线性插值计算deltabeta
    DevComplex<double> interpolated_db;
    
    // 在x方向线性插值
    DevComplex<double> db_y0, db_y1;
    db_y0.x = db_00.x * (1.0 - x_frac) + db_01.x * x_frac;
    db_y0.y = db_00.y * (1.0 - x_frac) + db_01.y * x_frac;
    
    db_y1.x = db_10.x * (1.0 - x_frac) + db_11.x * x_frac;
    db_y1.y = db_10.y * (1.0 - x_frac) + db_11.y * x_frac;
    
    // 在y方向线性插值
    interpolated_db.x = db_y0.x * (1.0 - y_frac) + db_y1.x * y_frac;
    interpolated_db.y = db_y0.y * (1.0 - y_frac) + db_y1.y * y_frac;
    
    // 计算相位变化指数 formula correct
    // const DevComplex<double> exponent =
    //     complex_mult<double>(interpolated_db, 
    //                         DevComplex<double>{0, 2.0 * M_PI * dz / params.wl});
    const double atomfactor = 2.0 * M_PI * dz / params.wl;
    DevComplex<double> exponent;
    exponent.x = -atomfactor * interpolated_db.y;   // 衰减项：-2πβ dz/λ
    exponent.y = -atomfactor * interpolated_db.x;   // 相移项：-2πδ dz/λ
    
    // 应用材料效应：exp(实部) * exp(i*虚部)
    const double exp_r = exp(exponent.x);
    const double angle = exponent.y;
    
    // 转换为输出类型并应用
    const DevComplex<S> factor{
        static_cast<S>(exp_r * cos(angle)),
        static_cast<S>(exp_r * sin(angle))
    };
    
    d_u[idx] = complex_mult<S>(d_u[idx], factor);
}

// 2D样本因子应用包装函数
template <typename S>
void apply_sample_factors_2d(DevComplex<S> *d_u, const SimParams &params, double dz,
                            uint32_t *d_sample, double pixel_size_x, double pixel_size_y,
                            std::size_t x_len, std::size_t y_len, DevComplex<double> *d_deltabetas,
                            int z_slice_index, double x_position, double y_position) {
    // fprintf(stderr, "[HOST] ENTER apply_sample_factors_2d: z_slice=%d\n", z_slice_index);
    // fflush(stderr);
    // std::cerr << "[apply_sample_factors_2d] ENTER" << std::endl;
    // write(2, "[apply_sample_factors_2d] ENTER\n", 30);                            
    // 打印所有传入参数
    // fprintf(stderr, "  d_u=%p, d_sample=%p, d_deltabetas=%p\n", (void*)d_u, (void*)d_sample, (void*)d_deltabetas);
    // fprintf(stderr, "  pixel_size_x=%g, pixel_size_y=%g, dz=%g\n", pixel_size_x, pixel_size_y, dz);
    // fprintf(stderr, "  x_len=%zu, y_len=%zu, x_pos=%g, y_pos=%g\n", x_len, y_len, x_position, y_position);
    // fflush(stderr);

    // 获取并打印网格尺寸
    const int nx = params.nx;
    const int ny = params.ny;
    // fprintf(stderr, "  nx=%d, ny=%d, dx=%g, dy=%g\n", nx, ny, params.dx, params.dy);
    // fflush(stderr);

    // 计算线程块配置并打印
    int actual_nx = (nx > 0) ? nx : params.N;
    int actual_ny = (ny > 0) ? ny : 1;
    dim3 blockDim(16, 16);
    dim3 gridDim((actual_nx + blockDim.x - 1) / blockDim.x,
                 (actual_ny + blockDim.y - 1) / blockDim.y);
    // fprintf(stderr, "  gridDim=(%d,%d), blockDim=(%d,%d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    // fflush(stderr);

    // 在内核启动前检查所有指针非空
    if (!d_u || !d_sample || !d_deltabetas) {
        fprintf(stderr, "  ERROR: null pointer detected!\n");
        fflush(stderr);
        return;
    }

    // fprintf(stderr, "  Launching kernel...\n");
    // fflush(stderr);

    apply_sample_factors_2d_kernel<S><<<gridDim, blockDim>>>(
        d_u, params, dz, d_sample, pixel_size_x, pixel_size_y,
        x_len, y_len, d_deltabetas, z_slice_index, x_position, y_position);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "  Kernel launch error: %s\n", cudaGetErrorString(err));
        fflush(stderr);
        return;
    }

    // 同步并检查执行错误
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "  Kernel execution error: %s\n", cudaGetErrorString(err));
        fflush(stderr);
        return;
    }

    // fprintf(stderr, "[HOST] EXIT apply_sample_factors_2d\n");
    // fflush(stderr);
    // // 获取网格尺寸
    // const int nx = params.nx;
    // const int ny = params.ny;
    
    // // 如果nx或ny为0，尝试使用N（向后兼容）
    // int actual_nx = (nx > 0) ? nx : params.N;
    // int actual_ny = (ny > 0) ? ny : 1;  // 默认1D情况
    
    // // 配置2D线程块
    // const int BLOCK_SIZE_X = 16;
    // const int BLOCK_SIZE_Y = 16;
    // dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    // dim3 gridDim((actual_nx + blockDim.x - 1) / blockDim.x,
    //              (actual_ny + blockDim.y - 1) / blockDim.y);
    // printf("[apply_sample_factors_2d] Launching kernel with grid=(%d,%d) block=(%d,%d)\n",
    //        gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    // // 启动核函数
    // apply_sample_factors_2d_kernel<S><<<gridDim, blockDim>>>(
    //     d_u, params, dz, d_sample, pixel_size_x, pixel_size_y,
    //     x_len, y_len, d_deltabetas, z_slice_index, x_position, y_position);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("[apply_sample_factors_2d] Kernel launch error: %s\n", cudaGetErrorString(err));
    //     throw std::runtime_error("Kernel launch failed");
    // }

    // err = cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //     printf("[apply_sample_factors_2d] Kernel execution error: %s\n", cudaGetErrorString(err));
    //     throw std::runtime_error("Kernel execution failed");
    // }

    // printf("[apply_sample_factors_2d] EXIT\n");
}

//plasma_sample_begin
template <typename S>
__global__ void apply_plasma_sample_factors_2d_kernel(
    DevComplex<S> *d_u, SimParams params, double dz,
    DevComplex<double> *d_deltabeta_grid,
    double pixel_size_x, double pixel_size_y,
    std::size_t x_len, std::size_t y_len,
    int z_slice_index, double x_position, double y_position)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int nx = params.nx;
    const int ny = params.ny;
    const double dx = params.dx;
    const double dy = params.dy;

    if (ix >= nx || iy >= ny) return;
    const int idx = iy * nx + ix;

    const double x = static_cast<double>(ix - nx / 2) * dx + x_position + pixel_size_x * x_len * 0.5;
    const double x_index = x / pixel_size_x;

    double x_clamped = x_index;
    if (x_clamped < 0.0) x_clamped = 0.0;
    if (x_clamped >= static_cast<double>(x_len) - 1.0) x_clamped = static_cast<double>(x_len) - 2.0;

    const std::size_t x_floor = static_cast<std::size_t>(x_clamped);
    const double x_frac = x_clamped - static_cast<double>(x_floor);

    // y-direction: handle 1D mode (y_len == 1) separately to avoid
    // division by zero when pixel_size_y == 0.
    double y_frac;
    std::size_t y_floor;
    if (y_len > 1) {
        const double y = static_cast<double>(iy - ny / 2) * dy + y_position + pixel_size_y * y_len * 0.5;
        const double y_index = y / pixel_size_y;
        double y_clamped = y_index;
        if (y_clamped < 0.0) y_clamped = 0.0;
        if (y_clamped >= static_cast<double>(y_len) - 1.0) y_clamped = static_cast<double>(y_len) - 2.0;
        y_floor = static_cast<std::size_t>(y_clamped);
        y_frac = y_clamped - static_cast<double>(y_floor);
    } else {
        y_floor = 0;
        y_frac = 0.0;
    }

    const std::size_t slice_offset = z_slice_index * y_len * x_len;

    const DevComplex<double> db_00 = d_deltabeta_grid[slice_offset + y_floor * x_len + x_floor];
    const DevComplex<double> db_01 = d_deltabeta_grid[slice_offset + y_floor * x_len + x_floor + 1];
    DevComplex<double> interpolated_db;

    if (y_len > 1) {
        const DevComplex<double> db_10 = d_deltabeta_grid[slice_offset + (y_floor + 1) * x_len + x_floor];
        const DevComplex<double> db_11 = d_deltabeta_grid[slice_offset + (y_floor + 1) * x_len + x_floor + 1];
        DevComplex<double> db_y0, db_y1;
        db_y0.x = db_00.x * (1.0 - x_frac) + db_01.x * x_frac;
        db_y0.y = db_00.y * (1.0 - x_frac) + db_01.y * x_frac;
        db_y1.x = db_10.x * (1.0 - x_frac) + db_11.x * x_frac;
        db_y1.y = db_10.y * (1.0 - x_frac) + db_11.y * x_frac;
        interpolated_db.x = db_y0.x * (1.0 - y_frac) + db_y1.x * y_frac;
        interpolated_db.y = db_y0.y * (1.0 - y_frac) + db_y1.y * y_frac;
    } else {
        // 1D mode: only x-interpolation, no y-interpolation
        interpolated_db.x = db_00.x * (1.0 - x_frac) + db_01.x * x_frac;
        interpolated_db.y = db_00.y * (1.0 - x_frac) + db_01.y * x_frac;
    }

    const double atomfactor = 2.0 * M_PI * dz / params.wl;
    DevComplex<double> exponent;
    exponent.x = -atomfactor * interpolated_db.y;
    exponent.y = -atomfactor * interpolated_db.x;

    const double exp_r = exp(exponent.x);
    const double angle = exponent.y;
    const DevComplex<S> factor{
        static_cast<S>(exp_r * cos(angle)),
        static_cast<S>(exp_r * sin(angle))
    };
    d_u[idx] = complex_mult<S>(d_u[idx], factor);
}

template <typename S>
void apply_plasma_sample_factors_2d(DevComplex<S> *d_u, const SimParams &params, double dz,
                                     DevComplex<double> *d_deltabeta_grid,
                                     double pixel_size_x, double pixel_size_y,
                                     std::size_t x_len, std::size_t y_len,
                                     int z_slice_index, double x_position, double y_position) {
    const int nx = params.nx;
    const int ny = params.ny;
    int actual_nx = (nx > 0) ? nx : params.N;
    int actual_ny = (ny > 0) ? ny : 1;
    dim3 blockDim(16, 16);
    dim3 gridDim((actual_nx + blockDim.x - 1) / blockDim.x,
                 (actual_ny + blockDim.y - 1) / blockDim.y);

    apply_plasma_sample_factors_2d_kernel<S><<<gridDim, blockDim>>>(
        d_u, params, dz, d_deltabeta_grid,
        pixel_size_x, pixel_size_y, x_len, y_len,
        z_slice_index, x_position, y_position);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "apply_plasma_sample_factors_2d kernel error: %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "apply_plasma_sample_factors_2d sync error: %s\n", cudaGetErrorString(err));
    }
}
//plasma_sample_end

template <typename S>
__global__ void square_and_downsample_2d_kernel(DevComplex<S> *d_u, int nx, int ny, S *d_out, 
                                                int outsize_x, int outsize_y,
                                                double detector_pixel_size_x,
                                                double detector_pixel_size_y,
                                                double current_z, double dx, double dy) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix >= outsize_x || iy >= outsize_y) return;
    
    // 计算探测器像素在物理空间中的位置
    const double x_det = static_cast<double>(ix - outsize_x / 2) * detector_pixel_size_x;
    const double y_det = static_cast<double>(iy - outsize_y / 2) * detector_pixel_size_y;
    
    // 计算探测器像素对应的采样场区域边界
    const double x_min = x_det - detector_pixel_size_x * 0.5;
    const double x_max = x_det + detector_pixel_size_x * 0.5;
    const double y_min = y_det - detector_pixel_size_y * 0.5;
    const double y_max = y_det + detector_pixel_size_y * 0.5;
    
    // 转换为采样场网格索引
    const int i_min = max(static_cast<int>(x_min / dx) + nx / 2, 0);
    const int i_max = min(static_cast<int>(x_max / dx) + nx / 2, nx);
    const int j_min = max(static_cast<int>(y_min / dy) + ny / 2, 0);
    const int j_max = min(static_cast<int>(y_max / dy) + ny / 2, ny);
    
    // 对区域内所有采样点进行强度累加
    double sum = 0.0;
    for (int j = j_min; j < j_max; ++j) {
        for (int i = i_min; i < i_max; ++i) {
            const int idx = j * nx + i;
            sum += d_u[idx].x * d_u[idx].x + d_u[idx].y * d_u[idx].y;
        }
    }
    
    // 计算几何因子（球面波衰减和倾斜因子）
    const double r = sqrt(x_det * x_det + y_det * y_det + current_z * current_z);
    const double angle_xy = sqrt(x_det * x_det + y_det * y_det) / current_z;
    const double cos_angle = 1.0 / sqrt(1.0 + angle_xy * angle_xy);  // cos(atan(angle_xy))
    
    // 应用几何校正并存储结果
    const int out_idx = iy * outsize_x + ix;
    d_out[out_idx] = static_cast<S>(sum * dx * dy * cos_angle);
}
template <typename S>
void square_and_downsample_2d(DevComplex<S> *d_u, int nx, int ny, S *d_out, 
                              int outsize_x, int outsize_y,
                              double detector_pixel_size_x, double detector_pixel_size_y,
                              double current_z, double dx, double dy) {
    // 配置2D线程块
    const int BLOCK_SIZE_X = 16;
    const int BLOCK_SIZE_Y = 16;
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim((outsize_x + blockDim.x - 1) / blockDim.x,
                 (outsize_y + blockDim.y - 1) / blockDim.y);
    
    // 启动核函数
    square_and_downsample_2d_kernel<S><<<gridDim, blockDim>>>(
        d_u, nx, ny, d_out, outsize_x, outsize_y,
        detector_pixel_size_x, detector_pixel_size_y,
        current_z, dx, dy);
    
    // 检查错误
    check_cuda_result("square_and_downsample_2d kernel", cudaPeekAtLastError());
}

// 2D uniform initialization kernel (for Fresnel scaling)
template <typename S>
__global__ void initialize_uniform_2d_kernel(DevComplex<S> *d_u, int nx, int ny,
                                              DevComplex<S> value) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    d_u[iy * nx + ix] = value;
}

template <typename S>
void initialize_uniform_2d(DevComplex<S> *d_u, int nx, int ny, Complex<S> value) {
    constexpr int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((nx + block_size - 1) / block_size,
                 (ny + block_size - 1) / block_size);
    DevComplex<S> dev_val = c2dc(value);
    initialize_uniform_2d_kernel<S><<<gridDim, blockDim>>>(d_u, nx, ny, dev_val);
    check_cuda_result("initialize_uniform_2d", cudaPeekAtLastError());
}
//3d end

template <typename S>
__global__ void apply_grating_factors_kernel(DevComplex<S> *d_u, SimParams params,
                                             DevComplex<S> factor_a, DevComplex<S> factor_b,
                                             double pitch, double dc, double x_position) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= params.N)
        return;

    // todo: gratings seem to be one off from python version
    const double x = static_cast<double>(i - params.N / 2) * params.dx + x_position;

    const double t = x / pitch + dc * 0.5f;
    const double phase = t - floor(t);
    const DevComplex<S> factor = phase < dc ? factor_a : factor_b;
    d_u[i] = complex_mult<S>(d_u[i], factor);
}

template <typename S>
void apply_grating_factors(DevComplex<S> *d_u, const SimParams &params, Complex<S> factor_a,
                           Complex<S> factor_b, double pitch, double dc, double x_position) {
    apply_grating_factors_kernel<S>
        <<<params.N / 64, 64>>>(d_u, params, c2dc(factor_a), c2dc(factor_b), pitch, dc, x_position);
}

template <typename S>
__global__ void apply_env_grating_factors_kernel(DevComplex<S> *d_u, SimParams params,
                                             DevComplex<S> factor_a, DevComplex<S> factor_b,
                                             double pitch0, double pitch1, double dc0, 
                                             double dc1, double x_position) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= params.N)
        return;

    const double x = static_cast<double>(i - params.N / 2) * params.dx + x_position;

    const double t0 = x / pitch0 + dc0 * 0.5f;
    const double t1 = x / pitch1;
    const double phase0 = t0 - floor(t0);
    const double phase1 = t1 - floor(t1);
    const DevComplex<S> factor = (phase0 < dc0 && phase1 < dc1) ? factor_a : factor_b;
    d_u[i] = complex_mult<S>(d_u[i], factor);
}

template <typename S>
void apply_env_grating_factors(DevComplex<S> *d_u, const SimParams &params, Complex<S> factor_a,
                           Complex<S> factor_b, double pitch0, double pitch1, 
                           double dc0, double dc1, double x_position) {
    apply_env_grating_factors_kernel<S><<<params.N / 64, 64>>>(d_u, params, c2dc(factor_a),
                                                        c2dc(factor_b), pitch0, pitch1, dc0, dc1, x_position);
}

template <typename S>
__global__ void apply_sample_factors_kernel(DevComplex<S> *d_u, SimParams params, double dz,
                                            uint32_t *d_sample, double pixel_size_x,
                                            std::size_t x_len, DevComplex<double> *d_deltabetas,
                                            int z_slice_index, double x_position) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= params.N)
        return;

    const double x =
        static_cast<double>(i - params.N / 2) * params.dx + x_position + pixel_size_x * x_len * 0.5;
    const double x_index = x / pixel_size_x;
    const std::size_t x_floor = static_cast<std::size_t>(x_index);
    const double x_frac = x_index - static_cast<double>(x_floor);

    if (x_index < 0 || x_floor + 1 >= x_len) {
        return;
    }

    const DevComplex<double> db_lower = d_deltabetas[d_sample[x_len * z_slice_index + x_floor]];
    const DevComplex<double> db_upper = d_deltabetas[d_sample[x_len * z_slice_index + x_floor + 1]];

    DevComplex<double> interpolated;
    interpolated.x = db_lower.x * (1 - x_frac) + db_upper.x * x_frac;
    interpolated.y = db_lower.y * (1 - x_frac) + db_upper.y * x_frac;

    // formula correct 
    // const DevComplex<double> exponent =
    //     complex_mult<double>(interpolated, DevComplex<double>{0, 2.0 * M_PI * dz / params.wl});
    const double factor = 2.0 * M_PI * dz / params.wl;
    DevComplex<double> exponent;
    exponent.x = -factor * interpolated.y;   // 衰减项：-2πβ dz/λ
    exponent.y = -factor * interpolated.x;   // 相移项：-2πδ dz/λ
    const double exp_r = exp(exponent.x);
    const double angle = exponent.y;
    d_u[i] = complex_mult<S>(d_u[i], DevComplex<S>{static_cast<float>(exp_r * cos(angle)),
                                                   static_cast<float>(exp_r * sin(angle))});
}

template <typename S>
void apply_sample_factors(DevComplex<S> *d_u, const SimParams &params, double dz, uint32_t *d_sample,
                          double pixel_size_x, std::size_t x_len, DevComplex<double> *d_deltabetas,
                          int z_slice_index, double x_position) {
    apply_sample_factors_kernel<S><<<params.N / 64, 64>>>(
        d_u, params, dz, d_sample, pixel_size_x, x_len, d_deltabetas, z_slice_index, x_position);
}

//precise_sample_update_begin
template <typename S>
__global__ void apply_precise_sample_factors_kernel(DevComplex<S> *d_u, SimParams params, double dz,
                                                   float *d_density_grid, uint32_t *d_material_grid,
                                                   DevComplex<double> *d_deltabeta_grid,
                                                   double pixel_size_x, std::size_t x_len,
                                                   int z_slice_index, double x_position) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= params.N) return;

    // 计算物理位置
    const double x = static_cast<double>(i - params.N / 2) * params.dx + x_position + pixel_size_x * x_len * 0.5;
    const double x_index = x / pixel_size_x;
    const std::size_t x_floor = static_cast<std::size_t>(x_index);
    const double x_frac = x_index - static_cast<double>(x_floor);

    if (x_index < 0 || x_floor + 1 >= x_len) return;

    // 直接从预先计算的deltabeta网格中获取值（线性插值）
    const std::size_t base_idx = z_slice_index * x_len;
    
    const DevComplex<double> db_lower = d_deltabeta_grid[base_idx + x_floor];
    const DevComplex<double> db_upper = d_deltabeta_grid[base_idx + x_floor + 1];
    
    // 线性插值deltabeta
    DevComplex<double> interpolated_db;
    interpolated_db.x = db_lower.x * (1.0 - x_frac) + db_upper.x * x_frac;
    interpolated_db.y = db_lower.y * (1.0 - x_frac) + db_upper.y * x_frac;

    // 调试输出：只对第一个线程
    // if (i == 0 && threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("DEBUG: z_slice=%d, x_floor=%zu, x_frac=%f\n", z_slice_index, x_floor, x_frac);
    //     printf("DEBUG: db_lower=(%e, %e), db_upper=(%e, %e)\n", 
    //            db_lower.x, db_lower.y, db_upper.x, db_upper.y);
    //     printf("DEBUG: interpolated_db=(%e, %e)\n", interpolated_db.x, interpolated_db.y);
    // }

    // 计算相位变化 - 直接使用预先计算的deltabeta formula correct
    const double factor = 2.0 * M_PI * dz / params.wl;
    DevComplex<double> exponent;
    exponent.x = -factor * interpolated_db.y;   // 衰减项：-2πβ dz/λ
    exponent.y = -factor * interpolated_db.x;   // 相移项：-2πδ dz/λ
    
    // 应用材料效应
    const double exp_r = exp(exponent.x);
    const double angle = exponent.y;
    
    // 调试输出
    // if (i == 0 && threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("DEBUG: exp_r=%f, angle=%f, factor=(%f, %f)\n", 
    //            exp_r, angle, static_cast<float>(exp_r * cos(angle)), 
    //            static_cast<float>(exp_r * sin(angle)));
    // }
    
    d_u[i] = complex_mult<S>(d_u[i], 
        DevComplex<S>{static_cast<S>(exp_r * cos(angle)),
                      static_cast<S>(exp_r * sin(angle))});
}

template <typename S>
void apply_precise_sample_factors(DevComplex<S> *d_u, const SimParams &params, double dz,
                                 float *d_density_grid, uint32_t *d_material_grid,
                                 DevComplex<double> *d_deltabeta_grid,
                                 double pixel_size_x, std::size_t x_len,
                                 int z_slice_index, double x_position) {
    apply_precise_sample_factors_kernel<S><<<params.N / 64, 64>>>(
        d_u, params, dz, d_density_grid, d_material_grid, d_deltabeta_grid,
        pixel_size_x, x_len, z_slice_index, x_position);
}

// 修改模板实例化

//precise_sample_update_end

template <typename S>
__global__ void scale_kernel(DevComplex<S> *d_u, DevComplex<S> scale, const int N) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;

    d_u[i] = complex_mult<S>(d_u[i], scale);
}

template <typename S> void scale(DevComplex<S> *d_u, Complex<S> scale, const int N) {
    scale_kernel<S><<<(N + 63) / 64, 64>>>(d_u, c2dc(scale), N);
    check_cuda_result("scale kernel", cudaPeekAtLastError());
}

template <typename S>
__global__ void square_and_downsample_kernel(DevComplex<S> *d_u, int N, S *d_out, int outsize,
                                             double detector_pixel_size_x,
                                             double detector_pixel_size_y, double current_z,
                                             double dx) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= outsize)
        return;

    const double min_x = static_cast<double>(i - outsize / 2) * detector_pixel_size_x;
    const double max_x = static_cast<double>(i + 1 - outsize / 2) * detector_pixel_size_x;
    const int min_idx = max(static_cast<int>(min_x / dx) + N / 2, 0);
    const int max_idx = min(static_cast<int>(max_x / dx) + N / 2, N);

    double sum = 0;
    for (int j = min_idx; j < max_idx; ++j) {
        // This is not optimal because global memory access won't be coalesced.
        // The advantage is that the code is much simpler: the kernel is pretty
        // much independent of block size and we don't need shared memory or
        // atomics.
        sum += d_u[j].x * d_u[j].x + d_u[j].y * d_u[j].y;
    }
    double r = sqrt(min_x * min_x + current_z * current_z);
    double angle = atan(min_x / current_z);
    d_out[i] = sum * dx * detector_pixel_size_y * cos(angle) / r;
}

template <typename S>
void square_and_downsample(DevComplex<S> *d_u, int N, S *d_out, int outsize,
                           double detector_pixel_size_x, double detector_pixel_size_y,
                           double current_z, double dx) {
    square_and_downsample_kernel<S><<<(N + 63) / 64, 64>>>(
        d_u, N, d_out, outsize, detector_pixel_size_x, detector_pixel_size_y, current_z, dx);
    check_cuda_result("square_and_downsample kernel", cudaPeekAtLastError());
}

template <typename S>
__global__ void analytical_history_row_kernel(S *row, int nr_detector_pixels,
                                              double detector_pixel_size_x,
                                              double detector_pixel_size_y, double x_source,
                                              double z, double cutoff_gratient) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nr_detector_pixels)
        return;
    const double x =
        static_cast<double>(i - nr_detector_pixels / 2) * detector_pixel_size_x - x_source;
    if (abs(x) > cutoff_gratient * z)
        return;

    const double angle = atan(x / z);
    const double r2 = x * x + z * z;
    const double intensity = detector_pixel_size_x * detector_pixel_size_y * cos(angle) / r2;
    row[i] = static_cast<S>(intensity);
}

template <typename S>
void analytical_history_row(S *row, int nr_detector_pixels, double detector_pixel_size_x,
                            double detector_pixel_size_y, double x_source, double z,
                            double cutoff_gratient) {
    analytical_history_row_kernel<S><<<(nr_detector_pixels + 63) / 64, 64>>>(
        row, nr_detector_pixels, detector_pixel_size_x, detector_pixel_size_y, x_source, z,
        cutoff_gratient);
    check_cuda_result("analytical_history_row kernel", cudaPeekAtLastError());
}

// This is a bit awkward.. usually the simpler solution would be to just
// move the function definitions to the header so that we don't have to
// write down the template instantiations, but we want to make the cuda
// kernel calls invisible to non-cuda-cpp files so we can't put the wrapper
// bodies in a header that is included in a cpp file.

// explicit instantiations for float
template void propagate_analytically<float>(DevComplex<float> *, const SimParams &, double, double);
template void propagate_convolve_step<float>(DevComplex<float> *, const SimParams &, double,
                                             double);
template void apply_grating_factors<float>(DevComplex<float> *, const SimParams &,
                                           Complex<float>, Complex<float>, double, double, double);
template void apply_env_grating_factors<float>(DevComplex<float> *, const SimParams &,
                                           Complex<float>, Complex<float>, double, double, double, double, double);
template void apply_sample_factors<float>(DevComplex<float> *, const SimParams &, double, uint32_t *,
                                          double, std::size_t, DevComplex<double> *, int, double);
template void scale<float>(DevComplex<float> *, Complex<float>, const int);
template void square_and_downsample<float>(DevComplex<float> *, int, float *, int, double, double,
                                           double, double);
template void analytical_history_row<float>(float *, int, double, double, double, double, double);


// explicit instantiations for double
template void propagate_analytically<double>(DevComplex<double> *, const SimParams &, double,
                                             double);
template __host__ __device__ double fftfreq<double>(int, int, double);
template void propagate_convolve_step<double>(DevComplex<double> *, const SimParams &, double,
                                              double);
template void apply_grating_factors<double>(DevComplex<double> *, const SimParams &,
                                            Complex<double>, Complex<double>, double, double,
                                            double);
template void apply_env_grating_factors<double>(DevComplex<double> *, const SimParams &,
                                            Complex<double>, Complex<double>, double, double,
                                            double, double, double);
template void apply_sample_factors<double>(DevComplex<double> *u, const SimParams &, double,
                                           uint32_t *, double, std::size_t, DevComplex<double> *, int,
                                           double);
//precise_sample_update_begin
template void apply_precise_sample_factors<float>(DevComplex<float> *, const SimParams &, double,
                                                 float *, uint32_t *, DevComplex<double> *,
                                                 double, std::size_t, int, double);

template void apply_precise_sample_factors<double>(DevComplex<double> *, const SimParams &, double,
                                                  float *, uint32_t *, DevComplex<double> *,
                                                  double, std::size_t, int, double);
// template void apply_precise_sample_factors<double>(DevComplex<double> *, const SimParams &, double,
//                                                   float *, uint32_t *, double, std::size_t, // 改为float
//                                                   DevComplex<double> *, int, double);     // 删除d_density_levels参数
//precise_sample_update_end

//3d start
template void propagate_analytically_2d<float>(DevComplex<float> *, const SimParams &, double, double, double);
template void propagate_convolve_step_2d<float>(DevComplex<float> *, const SimParams &, double, double, double);
template void apply_sample_factors_2d<float>(DevComplex<float> *, const SimParams &, double,
                                             uint32_t *, double, double, std::size_t, std::size_t,
                                             DevComplex<double> *, int, double, double);
template void square_and_downsample_2d<float>(DevComplex<float> *, int, int, float *, 
                                             int, int, double, double, double, double, double);

template void propagate_analytically_2d<double>(DevComplex<double> *, const SimParams &, double, double, double);
template void propagate_convolve_step_2d<double>(DevComplex<double> *, const SimParams &, double, double, double);
template void apply_sample_factors_2d<double>(DevComplex<double> *, const SimParams &, double,
                                              uint32_t *, double, double, std::size_t, std::size_t,
                                              DevComplex<double> *, int, double, double);
template void square_and_downsample_2d<double>(DevComplex<double> *, int, int, double *,
                                              int, int, double, double, double, double, double);

template void initialize_uniform_2d<float>(DevComplex<float> *, int, int, Complex<float>);
template void initialize_uniform_2d<double>(DevComplex<double> *, int, int, Complex<double>);

// plasma_sample
template void apply_plasma_sample_factors_2d<float>(DevComplex<float> *, const SimParams &, double,
    DevComplex<double> *, double, double, std::size_t, std::size_t, int, double, double);
template void apply_plasma_sample_factors_2d<double>(DevComplex<double> *, const SimParams &, double,
    DevComplex<double> *, double, double, std::size_t, std::size_t, int, double, double);
//3d end

template void scale<double>(DevComplex<double> *, Complex<double>, const int);
template void square_and_downsample<double>(DevComplex<double> *, int, double *, int, double,
                                            double, double, double);
template void analytical_history_row<double>(double *, int, double, double, double, double, double);
