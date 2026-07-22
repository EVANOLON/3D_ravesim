// simple_test_2d.cu - 最简单的2D传播测试
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 复数结构体
template<typename T>
struct Complex {
    T x, y;
};

// 简化的SimParams结构体
struct SimParams {
    int nx, ny;
    double dx, dy;
    double wl;
    
    SimParams(int x=256, int y=256, double dx_val=1e-6, double dy_val=1e-6, double wavelength=500e-9)
        : nx(x), ny(y), dx(dx_val), dy(dy_val), wl(wavelength) {}
};

// 设备函数：计算2D频率
template<typename T>
__device__ T fftfreq_2d_x(int i, int N, T dx) {
    int shifted = i - N * (i >= N / 2);
    return static_cast<T>(shifted) / (static_cast<T>(N) * dx);
}

template<typename T>
__device__ T fftfreq_2d_y(int i, int N, T dy) {
    int shifted = i - N * (i >= N / 2);
    return static_cast<T>(shifted) / (static_cast<T>(N) * dy);
}

// 设备函数：复数乘法
template<typename T>
__device__ Complex<T> complex_mult(Complex<T> a, Complex<T> b) {
    return Complex<T>{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

// 核心：2D传播核函数
template<typename T>
__global__ void propagate_2d_kernel(Complex<T>* d_U, SimParams params, double dz, 
                                    double cutoff_freq_x, double cutoff_freq_y) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix >= params.nx || iy >= params.ny) return;
    
    int idx = iy * params.nx + ix;
    
    // 计算空间频率
    double kx = fftfreq_2d_x<double>(ix, params.nx, params.dx);
    double ky = fftfreq_2d_y<double>(iy, params.ny, params.dy);
    
    // 频率幅值
    double freq_radius = sqrt(kx * kx + ky * ky);
    double max_cutoff = max(cutoff_freq_x, cutoff_freq_y);
    
    if (freq_radius <= max_cutoff) {
        // 傍轴近似相位
        double angle = M_PI * dz * (-2.0 / params.wl + params.wl * (kx * kx + ky * ky));
        
        d_U[idx] = complex_mult<T>(d_U[idx], 
            Complex<T>{static_cast<T>(cos(angle)), static_cast<T>(sin(angle))});
    } else {
        d_U[idx] = Complex<T>{0, 0};  // 超出截止频率置零
    }
}

// 测试1：基本功能测试
bool test_basic_functionality() {
    std::cout << "1. 基本功能测试..." << std::endl;
    
    SimParams params(64, 64);  // 小尺寸便于测试
    int total_size = params.nx * params.ny;
    
    // 分配设备内存
    Complex<float>* d_U;
    CUDA_CHECK(cudaMalloc(&d_U, total_size * sizeof(Complex<float>)));
    
    // 创建测试数据：平面波（全1）
    Complex<float>* h_U = new Complex<float>[total_size];
    for (int i = 0; i < total_size; i++) {
        h_U[i] = {1.0f, 0.0f};
    }
    
    // 复制到设备
    CUDA_CHECK(cudaMemcpy(d_U, h_U, total_size * sizeof(Complex<float>), cudaMemcpyHostToDevice));
    
    // 配置线程块
    dim3 block(16, 16);
    dim3 grid((params.nx + block.x - 1) / block.x, 
              (params.ny + block.y - 1) / block.y);
    
    // 执行传播
    propagate_2d_kernel<float><<<grid, block>>>(d_U, params, 1e-3, 1e6, 1e6);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 读回结果
    Complex<float>* h_result = new Complex<float>[total_size];
    CUDA_CHECK(cudaMemcpy(h_result, d_U, total_size * sizeof(Complex<float>), cudaMemcpyDeviceToHost));
    
    // 验证：检查振幅是否接近1（平面波传播振幅不变）
    bool passed = true;
    for (int i = 0; i < total_size; i++) {
        float amp = sqrt(h_result[i].x * h_result[i].x + h_result[i].y * h_result[i].y);
        if (fabs(amp - 1.0f) > 0.01f) {  // 1%容差
            std::cout << "  警告：振幅变化过大: " << amp << " at index " << i << std::endl;
            passed = false;
        }
        if (isnan(h_result[i].x) || isnan(h_result[i].y)) {
            std::cout << "  错误：发现NaN值" << std::endl;
            passed = false;
        }
    }
    
    // 清理
    delete[] h_U;
    delete[] h_result;
    CUDA_CHECK(cudaFree(d_U));
    
    if (passed) {
        std::cout << "  ✅ 基本功能测试通过" << std::endl;
    } else {
        std::cout << "  ❌ 基本功能测试失败" << std::endl;
    }
    return passed;
}

// 测试2：截止频率测试
bool test_cutoff_frequency() {
    std::cout << "2. 截止频率测试..." << std::endl;
    
    SimParams params(32, 32);
    int total_size = params.nx * params.ny;
    
    Complex<float>* d_U;
    CUDA_CHECK(cudaMalloc(&d_U, total_size * sizeof(Complex<float>)));
    
    // 创建高频测试图案（棋盘格）
    Complex<float>* h_U = new Complex<float>[total_size];
    for (int y = 0; y < params.ny; y++) {
        for (int x = 0; x < params.nx; x++) {
            int idx = y * params.nx + x;
            // 高频信号：每个像素交替
            float val = ((x + y) % 2 == 0) ? 1.0f : -1.0f;
            h_U[idx] = {val, 0.0f};
        }
    }
    
    CUDA_CHECK(cudaMemcpy(d_U, h_U, total_size * sizeof(Complex<float>), cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid((params.nx + block.x - 1) / block.x, 
              (params.ny + block.y - 1) / block.y);
    
    // 使用极低的截止频率（应该滤掉几乎所有高频）
    double low_cutoff = 1e3;  // 1kHz
    propagate_2d_kernel<float><<<grid, block>>>(d_U, params, 1e-3, low_cutoff, low_cutoff);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    Complex<float>* h_result = new Complex<float>[total_size];
    CUDA_CHECK(cudaMemcpy(h_result, d_U, total_size * sizeof(Complex<float>), cudaMemcpyDeviceToHost));
    
    // 验证：高频信号应被置零
    bool passed = true;
    int zero_count = 0;
    for (int i = 0; i < total_size; i++) {
        float amp = sqrt(h_result[i].x * h_result[i].x + h_result[i].y * h_result[i].y);
        if (amp < 1e-6f) {
            zero_count++;
        }
    }
    
    float zero_percent = 100.0f * zero_count / total_size;
    std::cout << "  置零比例: " << zero_percent << "%" << std::endl;
    
    if (zero_percent > 95.0f) {
        std::cout << "  ✅ 截止频率测试通过" << std::endl;
    } else {
        std::cout << "  ❌ 截止频率测试失败：高频未充分滤除" << std::endl;
        passed = false;
    }
    
    delete[] h_U;
    delete[] h_result;
    CUDA_CHECK(cudaFree(d_U));
    
    return passed;
}

// 测试3：性能测试
bool test_performance() {
    std::cout << "3. 性能测试..." << std::endl;
    
    // 测试不同尺寸
    int sizes[] = {128, 256, 512};
    bool passed = true;
    
    for (int size : sizes) {
        SimParams params(size, size);
        int total_size = params.nx * params.ny;
        
        Complex<float>* d_U;
        CUDA_CHECK(cudaMalloc(&d_U, total_size * sizeof(Complex<float>)));
        
        dim3 block(16, 16);
        dim3 grid((params.nx + block.x - 1) / block.x, 
                  (params.ny + block.y - 1) / block.y);
        
        // 预热
        propagate_2d_kernel<float><<<grid, block>>>(d_U, params, 1e-3, 1e6, 1e6);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 计时
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        const int iterations = 100;
        CUDA_CHECK(cudaEventRecord(start));
        
        for (int i = 0; i < iterations; i++) {
            propagate_2d_kernel<float><<<grid, block>>>(d_U, params, 1e-3, 1e6, 1e6);
        }
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        float avg_ms = milliseconds / iterations;
        float data_size_gb = total_size * sizeof(Complex<float>) * 2 / (1024.0f * 1024.0f * 1024.0f);
        float throughput = data_size_gb / (avg_ms * 1e-3);
        
        std::cout << "  " << size << "x" << size << ": " 
                  << avg_ms << " ms, " << throughput << " GB/s" << std::endl;
        
        // 检查性能
        if (size == 512 && avg_ms > 5.0f) {
            std::cout << "  ⚠️  512x512性能偏慢" << std::endl;
        }
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_U));
    }
    
    std::cout << "  ✅ 性能测试完成" << std::endl;
    return passed;
}

// 测试4：验证正确性（与CPU计算比较）
bool test_correctness_vs_cpu() {
    std::cout << "4. 正确性验证（与CPU计算比较）..." << std::endl;
    
    SimParams params(16, 16);  // 小尺寸便于CPU验证
    int total_size = params.nx * params.ny;
    
    // CPU计算参考结果
    Complex<float>* h_cpu_result = new Complex<float>[total_size];
    for (int y = 0; y < params.ny; y++) {
        for (int x = 0; x < params.nx; x++) {
            int idx = y * params.nx + x;
            h_cpu_result[idx] = {1.0f, 0.0f};  // 平面波输入
            
            double kx = (x - params.nx/2) / (static_cast<double>(params.nx) * params.dx);
            double ky = (y - params.ny/2) / (static_cast<double>(params.ny) * params.dy);
            
            double freq_radius = sqrt(kx * kx + ky * ky);
            double cutoff = 1e6;
            double dz = 1e-3;
            
            if (freq_radius <= cutoff) {
                double angle = M_PI * dz * (-2.0 / params.wl + params.wl * (kx * kx + ky * ky));
                float real_part = cos(angle);
                float imag_part = sin(angle);
                
                // CPU复数乘法
                float new_real = h_cpu_result[idx].x * real_part - h_cpu_result[idx].y * imag_part;
                float new_imag = h_cpu_result[idx].x * imag_part + h_cpu_result[idx].y * real_part;
                
                h_cpu_result[idx] = {new_real, new_imag};
            } else {
                h_cpu_result[idx] = {0, 0};
            }
        }
    }
    
    // GPU计算
    Complex<float>* d_U;
    CUDA_CHECK(cudaMalloc(&d_U, total_size * sizeof(Complex<float>)));
    
    Complex<float>* h_U = new Complex<float>[total_size];
    for (int i = 0; i < total_size; i++) {
        h_U[i] = {1.0f, 0.0f};
    }
    
    CUDA_CHECK(cudaMemcpy(d_U, h_U, total_size * sizeof(Complex<float>), cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid((params.nx + block.x - 1) / block.x, 
              (params.ny + block.y - 1) / block.y);
    
    propagate_2d_kernel<float><<<grid, block>>>(d_U, params, 1e-3, 1e6, 1e6);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    Complex<float>* h_gpu_result = new Complex<float>[total_size];
    CUDA_CHECK(cudaMemcpy(h_gpu_result, d_U, total_size * sizeof(Complex<float>), cudaMemcpyDeviceToHost));
    
    // 比较结果
    bool passed = true;
    float max_error = 0.0f;
    for (int i = 0; i < total_size; i++) {
        float error_real = fabs(h_gpu_result[i].x - h_cpu_result[i].x);
        float error_imag = fabs(h_gpu_result[i].y - h_cpu_result[i].y);
        max_error = fmax(max_error, fmax(error_real, error_imag));
        
        if (error_real > 1e-5f || error_imag > 1e-5f) {
            passed = false;
        }
    }
    
    std::cout << "  最大误差: " << max_error << std::endl;
    
    if (passed) {
        std::cout << "  ✅ 正确性验证通过" << std::endl;
    } else {
        std::cout << "  ❌ 正确性验证失败" << std::endl;
    }
    
    delete[] h_cpu_result;
    delete[] h_U;
    delete[] h_gpu_result;
    CUDA_CHECK(cudaFree(d_U));
    
    return passed;
}

// 主函数
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   2D传播核函数简化测试" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 打印CUDA设备信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "设备: " << prop.name << std::endl;
    std::cout << "CUDA能力: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    bool all_passed = true;
    
    // 运行测试
    all_passed &= test_basic_functionality();
    all_passed &= test_cutoff_frequency();
    all_passed &= test_performance();
    all_passed &= test_correctness_vs_cpu();
    
    std::cout << "\n========================================" << std::endl;
    if (all_passed) {
        std::cout << "✅ 所有测试通过！" << std::endl;
    } else {
        std::cout << "❌ 部分测试失败！" << std::endl;
    }
    std::cout << "========================================" << std::endl;
    
    return all_passed ? 0 : 1;
}