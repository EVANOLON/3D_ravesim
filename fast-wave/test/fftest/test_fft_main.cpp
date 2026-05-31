// fast-wave/test/test_fft_main.cpp
#include "simulation.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

void print_help() {
    std::cout << "=== FastWave 2D FFT Validation Tool ===\n\n"
              << "Usage: test_2d_fft [options]\n\n"
              << "Options:\n"
              << "  --help               Show this help message\n"
              << "  --basic              Run basic tests (default)\n"
              << "  --all                Run comprehensive test suite\n"
              << "  --benchmark          Run performance benchmarks\n"
              << "  --size NX NY         Test specific grid size\n"
              << "  --precision FLOAT|DOUBLE  Test specific precision\n"
              << "  --list-sizes         List all test sizes\n\n"
              << "Examples:\n"
              << "  ./test_2d_fft                     # Run default tests\n"
              << "  ./test_2d_fft --size 256 256     # Test 256x256 grid\n"
              << "  ./test_2d_fft --all --benchmark  # Run all tests with benchmarks\n"
              << std::endl;
}

void list_test_sizes() {
    std::cout << "Standard test sizes:\n";
    std::cout << "  Small:    8x8, 16x16, 32x32\n";
    std::cout << "  Medium:   64x64, 128x128, 256x256\n";
    std::cout << "  Large:    512x512, 1024x1024\n";
    std::cout << "  Non-power: 100x100, 300x200\n";
}

int main(int argc, char* argv[]) {
    try {
        // 设置spdlog日志级别
        spdlog::set_level(spdlog::level::info);
        
        // 解析命令行参数
        bool run_basic = true;
        bool run_comprehensive = false;
        bool run_benchmark = false;
        bool custom_size = false;
        size_t custom_nx = 64;
        size_t custom_ny = 64;
        std::string precision = "both"; // "float", "double", "both"
        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg == "--help" || arg == "-h") {
                print_help();
                return 0;
            } else if (arg == "--list-sizes") {
                list_test_sizes();
                return 0;
            } else if (arg == "--basic") {
                run_basic = true;
                run_comprehensive = false;
            } else if (arg == "--all") {
                run_comprehensive = true;
                run_basic = false;
            } else if (arg == "--benchmark") {
                run_benchmark = true;
            } else if (arg == "--size" && i + 2 < argc) {
                custom_size = true;
                custom_nx = std::stoul(argv[++i]);
                custom_ny = std::stoul(argv[++i]);
            } else if (arg == "--precision" && i + 1 < argc) {
                precision = argv[++i];
            } else {
                std::cerr << "Unknown option: " << arg << std::endl;
                print_help();
                return 1;
            }
        }
        
        std::cout << "=========================================\n";
        std::cout << "   FastWave 2D FFT Validation Tests\n";
        std::cout << "=========================================\n\n";
        
#ifdef ENABLE_2D_FFT_TEST
        // 检查CUDA设备
        int device_count;
        cudaGetDeviceCount(&device_count);
        
        if (device_count == 0) {
            std::cerr << "Error: No CUDA-capable devices found!" << std::endl;
            return 1;
        }
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        std::cout << "CUDA Device: " << prop.name << "\n";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "Total Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "=========================================\n\n";
        
        // 根据参数执行测试
        if (custom_size) {
            std::cout << "Testing custom size: " << custom_nx << " x " << custom_ny << "\n\n";
            
            if (precision == "float" || precision == "both") {
                std::cout << "[Single Precision - float]\n";
                test_2d_fft_correctness<float>(custom_nx, custom_ny, true);
                std::cout << std::endl;
            }
            
            if (precision == "double" || precision == "both") {
                std::cout << "[Double Precision - double]\n";
                test_2d_fft_correctness<double>(custom_nx, custom_ny, true);
                std::cout << std::endl;
            }
        } 
        else if (run_comprehensive) {
            std::cout << "Running comprehensive test suite...\n\n";
            test_fft_functionality();
        }
        else if (run_benchmark) {
            std::cout << "Running performance benchmarks...\n\n";
            
            // 基准测试的尺寸
            std::vector<std::pair<size_t, size_t>> benchmark_sizes = {
                {128, 128}, {256, 256}, {512, 512}, {1024, 1024}
            };
            
            run_fft_comprehensive_tests(benchmark_sizes, true); // 使用双精度
        }
        else {
            // 默认基本测试
            std::cout << "Running basic tests...\n\n";
            
            // 测试几个典型尺寸
            std::vector<std::pair<size_t, size_t>> basic_sizes = {
                {32, 32}, {64, 64}, {128, 128}
            };
            
            run_fft_comprehensive_tests(basic_sizes, false); // 使用单精度
            
            std::cout << "\nBasic tests completed.\n";
        }
        
        std::cout << "\n=========================================\n";
        std::cout << "        All tests completed!\n";
        std::cout << "=========================================\n";
        
        return 0;
#else
        std::cerr << "Error: 2D FFT testing is not enabled in this build.\n";
        std::cerr << "Recompile with -DENABLE_2D_FFT_TEST=ON\n";
        return 1;
#endif
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred!" << std::endl;
        return 1;
    }
}