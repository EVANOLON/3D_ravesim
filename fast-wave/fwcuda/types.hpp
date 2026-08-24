// Copyright (c) 2024, ETH Zurich

#ifndef _FAST_WAVE_TYPES_HPP
#define _FAST_WAVE_TYPES_HPP

#include <cstdint>
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <optional>

// We differentiate between the complex type used on the GPU vs the one used on the CPU
//
// std::copmlex has the advantage that it overloads the + and * operators, but to access
// the components we need to use .real() and .imag() which are not compiled as `__host__ __device__`
// so we can't use them from CUDA.
//
// The two complex types have the same layout so we can memcopy between them without problems.

/// Complex numbers on the CPU
template <typename S> using Complex = std::complex<S>;

template <typename S> struct DevComplexHelper {
    using type = void;
};
template <> struct DevComplexHelper<double> {
    using type = cuDoubleComplex;
};
template <> struct DevComplexHelper<float> {
    using type = cufftComplex;
};

/// Complex numbers on the GPU
template <typename S> using DevComplex = typename DevComplexHelper<S>::type;

static_assert(sizeof(Complex<float>) == sizeof(DevComplex<float>));
static_assert(sizeof(Complex<double>) == sizeof(DevComplex<double>));

struct SimParams {
    // 1D参数（保持不变）
    int N;                      // 总点数
    double dx;                  // x方向像素间距
    double z_detector;          // 探测器z位置
    double detector_size;       // 探测器尺寸（x方向）
    double detector_pixel_size_x;  // 探测器像素尺寸（x方向）
    double detector_pixel_size_y;  // 探测器像素尺寸（y方向）- 保持原样
    double wl;                  // 波长
    bool is2d;

    // 2D扩展（新增，可选）
    // std::optional<int> nx;      // x方向网格点数（可选）
    // std::optional<int> ny;      // y方向网格点数（可选）
    // std::optional<double> dy;   // y方向像素间距（可选）
    // std::optional<double> detector_size_x;  // 探测器x方向尺寸（可选）
    // std::optional<double> detector_size_y;  // 探测器y方向尺寸（可选）
    int nx = 0;
    int ny = 1;
    double dy = 0.0;
    double detector_size_x = 0.0;
    double detector_size_y = 0.0;

    // Fresnel scaling (for short source-sample distance)
    bool use_fresnel_scaling = false;
    double z_eff = 0.0;          // effective propagation distance
    double magnification = 1.0;  // geometric magnification M

    // 维度标志（内部计算，不暴露给外部）
    bool is_2d() const { return is2d && ny > 1; }

    
    // 获取有效参数（向后兼容）
    int get_nx() const { return is2d ? nx : N; }
    int get_ny() const { return is2d ? ny : 1; }
    double get_dy() const { return is2d ? dy : dx; }
    double get_detector_size_x() const { return is2d ? detector_size_x : detector_size; }
    double get_detector_size_y() const { return is2d ? detector_size_y : detector_pixel_size_y; }
    
    // 获取总点数（考虑2D）
     int get_total_points() const {
        return is_2d() ? nx * ny : N;
    }
    
    // 验证参数
    bool validate() const {
        if (N <= 0) { std::cerr << "Error: N must be positive\n"; return false; }
        if (dx <= 0) { std::cerr << "Error: dx must be positive\n"; return false; }
        if (wl <= 0) { std::cerr << "Error: wavelength must be positive\n"; return false; }

        if (is2d) {
            if (nx <= 0) { std::cerr << "Error: nx must be positive\n"; return false; }
            if (ny <= 0) { std::cerr << "Error: ny must be positive\n"; return false; }
            if (dy <= 0) { std::cerr << "Error: dy must be positive\n"; return false; }
        }
        return true;
    }
    
    // 调试输出
    void print_info(const std::string& prefix = "") const {
        std::cout << prefix << "Simulation Parameters:" << std::endl;
        std::cout << prefix << "  Dimensions: " << (is_2d() ? "2D" : "1D") << std::endl;
        std::cout << prefix << "  Total points (N): " << N << std::endl;
        
        if (is_2d()) {
            std::cout << prefix << "  Grid: nx=" << nx << ", ny=" << ny << "\n";
            std::cout << prefix << "  Pixel size: dx=" << dx << ", dy=" << dy << "\n";
        } else {
            std::cout << prefix << "  Grid: N=" << N << " (1D)\n";
            std::cout << prefix << "  Pixel size: dx=" << dx << "\n";
        }
        
        std::cout << prefix << "  Detector pixel: " << detector_pixel_size_x * 1e6 
                  << " μm x " << detector_pixel_size_y * 1e6 << " μm" << std::endl;
        std::cout << prefix << "  Wavelength: " << wl * 1e9 << " nm" << std::endl;
        std::cout << prefix << "  Detector z: " << z_detector * 1e3 << " mm" << std::endl;
    }
};

// 向后兼容的辅助函数
namespace simparams_legacy {
    // 用于从旧的1D代码创建SimParams
    inline SimParams create_1d(int N_val, double dx_val, double wl_val,
                               double z_detector_val = 0,
                               double detector_size_val = 0,
                               double detector_pixel_size_x_val = 0,
                               double detector_pixel_size_y_val = 1.0) {
        SimParams params;
        params.N = N_val;
        params.dx = dx_val;
        params.wl = wl_val;
        params.z_detector = z_detector_val;
        
        if (detector_size_val > 0) params.detector_size = detector_size_val;
        else params.detector_size = N_val * dx_val;
        
        if (detector_pixel_size_x_val > 0) params.detector_pixel_size_x = detector_pixel_size_x_val;
        else params.detector_pixel_size_x = dx_val;
        
        params.detector_pixel_size_y = detector_pixel_size_y_val;
        
        return params;
    }
    
    // 用于创建2D SimParams（扩展接口）
    inline SimParams create_2d(int nx_val, int ny_val, 
                               double dx_val, double dy_val,
                               double wl_val,
                               double z_detector_val = 0,
                               double detector_size_x_val = 0,
                               double detector_size_y_val = 0,
                               double detector_pixel_size_x_val = 0,
                               double detector_pixel_size_y_val = 0) {
        SimParams params;
        
        // 设置1D参数（保持兼容性）
        params.N = nx_val * ny_val;  // 总点数
        params.dx = dx_val;
        params.wl = wl_val;
        params.z_detector = z_detector_val;
        
        // 设置2D扩展参数
        params.nx = nx_val;
        params.ny = ny_val;
        params.dy = dy_val;
        
        // 探测器参数
        if (detector_size_x_val > 0) {
            params.detector_size = detector_size_x_val;  // 向后兼容
            params.detector_size_x = detector_size_x_val;
        } else {
            params.detector_size = nx_val * dx_val;
            params.detector_size_x = nx_val * dx_val;
        }
        
        if (detector_size_y_val > 0) {
            params.detector_size_y = detector_size_y_val;
        } else {
            params.detector_size_y = ny_val * dy_val;
        }
        
        if (detector_pixel_size_x_val > 0) {
            params.detector_pixel_size_x = detector_pixel_size_x_val;
        } else {
            params.detector_pixel_size_x = dx_val;
        }
        
        if (detector_pixel_size_y_val > 0) {
            params.detector_pixel_size_y = detector_pixel_size_y_val;
        } else {
            params.detector_pixel_size_y = dy_val;
        }
        
        return params;
    }
}

/// convert a cpu complex to a gpu complex
template <typename S> [[nodiscard]] inline DevComplex<S> c2dc(Complex<S> c) {
    return DevComplex<S>{c.real(), c.imag()};
}

inline void check_cuda_result(const char *operation, cudaError res) {
    if (res != cudaSuccess) {
        const auto msg = cudaGetErrorString(res);
        std::clog << "cuda error in operation " << operation << ": " << msg << "\n";
        exit(1);
    }
}

inline void check_cufft_result(const char *operation, cufftResult res) {
    if (res != CUFFT_SUCCESS) {
        std::clog << "cuFFT error in operation " << operation << ": " << res << '\n';
        exit(1);
    }
}

#endif // _FAST_WAVE_TYPES_HPP
