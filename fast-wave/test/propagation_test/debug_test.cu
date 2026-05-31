#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

struct Complex { 
    float x, y; 
    __host__ __device__ Complex(float r=0, float i=0) : x(r), y(i) {}
};

struct SimParams {
    int nx, ny;
    float dx, dy, wl;
    
    __host__ __device__ SimParams() 
        : nx(64), ny(64), dx(1e-6f), dy(1e-6f), wl(500e-9f) {}
    
    __host__ __device__ SimParams(int x, int y, float dx_val, float dy_val, float wavelength)
        : nx(x), ny(y), dx(dx_val), dy(dy_val), wl(wavelength) {}
};

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

// 修正的频率计算
__device__ __host__ float calculate_k(int i, int N, float d) {
    // 正确的频率计算：返回角空间频率（rad/m），而不是Hz
    // 公式：k = 2π * (i - N/2) / (N * d)
    float k = 2.0f * M_PI * (i - N/2) / (N * d);
    return k;
}

// CPU计算（使用修正公式）
void cpu_propagation(Complex* data, const SimParams& params, float dz, float max_k) {
    for (int y = 0; y < params.ny; y++) {
        for (int x = 0; x < params.nx; x++) {
            int idx = y * params.nx + x;
            
            // 使用修正的频率计算
            float kx = calculate_k(x, params.nx, params.dx);
            float ky = calculate_k(y, params.ny, params.dy);
            
            float k_mag_sq = kx * kx + ky * ky;
            
            if (sqrtf(k_mag_sq) <= max_k) {
                // 正确的傍轴近似相位公式
                float k0 = 2.0f * M_PI / params.wl;  // 波数
                float angle = dz * (k0 - 0.5f * k_mag_sq / k0);
                
                // 角度归约
                angle = reduce_angle(angle);
                
                float cos_angle = cosf(angle);
                float sin_angle = sinf(angle);
                
                // 复数乘法
                float new_x = data[idx].x * cos_angle - data[idx].y * sin_angle;
                float new_y = data[idx].x * sin_angle + data[idx].y * cos_angle;
                
                data[idx] = Complex(new_x, new_y);
            } else {
                data[idx] = Complex(0.0f, 0.0f);
            }
        }
    }
}

// GPU核函数（使用修正公式）
__global__ void gpu_propagation(Complex* data, SimParams params, float dz, float max_k) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix >= params.nx || iy >= params.ny) return;
    
    int idx = iy * params.nx + ix;
    
    // 使用修正的频率计算
    float kx = calculate_k(ix, params.nx, params.dx);
    float ky = calculate_k(iy, params.ny, params.dy);
    
    float k_mag_sq = kx * kx + ky * ky;
    float k_mag = sqrtf(k_mag_sq);
    
    if (k_mag <= max_k) {
        // 正确的傍轴近似相位公式
        float k0 = 2.0f * M_PI / params.wl;  // 波数
        float angle = dz * (k0 - 0.5f * k_mag_sq / k0);
        
        // 角度归约
        angle = reduce_angle(angle);
        
        float cos_angle = __cosf(angle);
        float sin_angle = __sinf(angle);
        
        // 复数乘法
        float new_x = data[idx].x * cos_angle - data[idx].y * sin_angle;
        float new_y = data[idx].x * sin_angle + data[idx].y * cos_angle;
        
        data[idx] = Complex(new_x, new_y);
    } else {
        data[idx] = Complex(0.0f, 0.0f);
    }
}

int main() {
    std::cout << "=== 最终修正测试 ===" << std::endl;
    
    // 使用更合理的参数
    SimParams params(32, 32, 5e-6f, 5e-6f, 500e-9f);  // 增加dx, dy到5微米
    const int total = params.nx * params.ny;
    const float dz = 1e-4f;  // 减少传播距离到0.1mm
    const float max_k = 2.0f * M_PI * 1e6f;  // 最大角空间频率
    
    std::cout << "参数设置:" << std::endl;
    std::cout << "  网格: " << params.nx << "x" << params.ny << std::endl;
    std::cout << "  采样: dx=" << params.dx*1e6 << "um, dy=" << params.dy*1e6 << "um" << std::endl;
    std::cout << "  波长: " << params.wl*1e9 << "nm" << std::endl;
    std::cout << "  传播距离: " << dz*1e3 << "mm" << std::endl;
    std::cout << "  波数k0: " << 2.0f*M_PI/params.wl << " rad/m" << std::endl;
    
    // 分配内存
    Complex* d_data;
    Complex* h_cpu = new Complex[total];
    Complex* h_gpu = new Complex[total];
    
    CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(Complex)));
    
    // 初始化：高斯光束
    int center_x = params.nx / 2;
    int center_y = params.ny / 2;
    float sigma = params.nx * params.dx * 0.2f;  // 光束半径
    
    for (int y = 0; y < params.ny; y++) {
        for (int x = 0; x < params.nx; x++) {
            int idx = y * params.nx + x;
            float dx = (x - center_x) * params.dx;
            float dy = (y - center_y) * params.dy;
            float r2 = dx*dx + dy*dy;
            float amplitude = expf(-r2 / (2.0f * sigma * sigma));
            h_cpu[idx] = Complex(amplitude, 0.0f);
        }
    }
    
    // CPU计算
    Complex* cpu_result = new Complex[total];
    for (int i = 0; i < total; i++) {
        cpu_result[i] = h_cpu[i];
    }
    cpu_propagation(cpu_result, params, dz, max_k);
    
    // GPU计算
    CUDA_CHECK(cudaMemcpy(d_data, h_cpu, total * sizeof(Complex), cudaMemcpyHostToDevice));
    
    dim3 block(8, 8);
    dim3 grid((params.nx + block.x - 1) / block.x, 
              (params.ny + block.y - 1) / block.y);
    
    gpu_propagation<<<grid, block>>>(d_data, params, dz, max_k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_gpu, d_data, total * sizeof(Complex), cudaMemcpyDeviceToHost));
    
    // 分析结果
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int error_count = 0;
    
    for (int i = 0; i < total; i++) {
        float error_real = fabsf(h_gpu[i].x - cpu_result[i].x);
        float error_imag = fabsf(h_gpu[i].y - cpu_result[i].y);
        float error = sqrtf(error_real * error_real + error_imag * error_imag);
        
        avg_error += error;
        max_error = fmaxf(max_error, error);
        
        if (error > 1e-4f) {
            error_count++;
        }
    }
    
    avg_error /= total;
    
    std::cout << "\n=== 误差分析 ===" << std::endl;
    std::cout << "平均误差: " << avg_error << std::endl;
    std::cout << "最大误差: " << max_error << std::endl;
    std::cout << "误差 > 1e-4 的点数: " << error_count << "/" << total << std::endl;
    
    // 检查几个关键点
    std::cout << "\n=== 关键点检查 ===" << std::endl;
    
    int test_points[5] = {0, params.nx/2, params.nx*params.ny/2, 3*params.nx*params.ny/4, total-1};
    
    for (int i = 0; i < 5; i++) {
        int idx = test_points[i];
        int x = idx % params.nx;
        int y = idx / params.nx;
        
        // 计算该点的频率
        float kx = calculate_k(x, params.nx, params.dx);
        float ky = calculate_k(y, params.ny, params.dy);
        float k_mag = sqrtf(kx*kx + ky*ky);
        float k0 = 2.0f * M_PI / params.wl;
        
        std::cout << "\n点[" << x << "," << y << "]:" << std::endl;
        std::cout << "  频率: kx=" << kx << " rad/m, ky=" << ky << " rad/m" << std::endl;
        std::cout << "  频率比: k/k0 = " << k_mag/k0 << std::endl;
        std::cout << "  CPU: (" << cpu_result[idx].x << ", " << cpu_result[idx].y << ")" << std::endl;
        std::cout << "  GPU: (" << h_gpu[idx].x << ", " << h_gpu[idx].y << ")" << std::endl;
        
        float error_real = fabsf(h_gpu[idx].x - cpu_result[idx].x);
        float error_imag = fabsf(h_gpu[idx].y - cpu_result[idx].y);
        float error = sqrtf(error_real*error_real + error_imag*error_imag);
        std::cout << "  误差: " << error << std::endl;
    }
    
    // 验证中心点（kx=0, ky=0）的计算
    std::cout << "\n=== 中心点(0,0)详细验证 ===" << std::endl;
    int center_idx = 0;
    
    float kx_center = calculate_k(0, params.nx, params.dx);
    float ky_center = calculate_k(0, params.ny, params.dy);
    float k0 = 2.0f * M_PI / params.wl;
    float angle_center = dz * (k0 - 0.5f * (kx_center*kx_center + ky_center*ky_center) / k0);
    
    std::cout << "中心点频率: kx=" << kx_center << ", ky=" << ky_center << std::endl;
    std::cout << "波数k0: " << k0 << " rad/m" << std::endl;
    std::cout << "原始角度: " << angle_center << " rad" << std::endl;
    std::cout << "归约后角度: " << reduce_angle(angle_center) << " rad" << std::endl;
    std::cout << "sin(归约后): " << sinf(reduce_angle(angle_center)) << std::endl;
    std::cout << "cos(归约后): " << cosf(reduce_angle(angle_center)) << std::endl;
    
    // 判断测试结果
    bool test_passed = (max_error < 1e-3f);
    
    if (test_passed) {
        std::cout << "\n✅ 测试通过！GPU与CPU结果一致" << std::endl;
    } else {
        std::cout << "\n⚠️  测试存在较大误差" << std::endl;
        
        if (max_error > 0.1f) {
            std::cout << "可能的问题：" << std::endl;
            std::cout << "1. 频率计算单位错误" << std::endl;
            std::cout << "2. 相位公式错误" << std::endl;
            std::cout << "3. 超出了傍轴近似适用范围" << std::endl;
        }
    }
    
    // 清理
    delete[] h_cpu;
    delete[] h_gpu;
    delete[] cpu_result;
    CUDA_CHECK(cudaFree(d_data));
    
    return test_passed ? 0 : 1;
}