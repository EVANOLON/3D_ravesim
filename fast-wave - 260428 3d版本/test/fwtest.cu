#include <boost/ut.hpp>

#include <config_parsing.hpp>
#include <fft.hpp>
#include <kernels.hpp>
#include <wrappers.hpp>

#include <source_location>

template <typename S, typename F> void run_gpu_test(std::vector<Complex<S>> &data, F f) {
    std::size_t nr_bytes = sizeof(Complex<S>) * data.size();

    DevComplex<S> *d_data;
    check_cuda_result("malloc", cudaMalloc((void **)&d_data, nr_bytes));

    check_cuda_result("memcpy h2d",
                      cudaMemcpy(d_data, data.data(), nr_bytes, cudaMemcpyHostToDevice));

    f(d_data);

    check_cuda_result("memcpy d2h",
                      cudaMemcpy(data.data(), d_data, nr_bytes, cudaMemcpyDeviceToHost));

    check_cuda_result("sync", cudaDeviceSynchronize());
}

using source_location = boost::ext::ut::v1_1_8::reflection::source_location;

void expect_close(const double &a, const double &b,
                  const source_location &sl = source_location::current()) {
    using namespace boost::ut;
    expect(std::abs(a - b) < 1e-5, sl);
}

void expect_close(const Complex<double> &a, const Complex<double> &b,
                  const source_location &sl = source_location::current()) {
    using namespace boost::ut;
    expect_close(a.real(), b.real(), sl);
    expect_close(a.imag(), b.imag(), sl);
}

void expect_close(const std::vector<double> &a, const std::vector<double> &b,
                  const source_location &sl = source_location::current()) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        expect_close(a[i], b[i], sl);
    }
}

void expect_close(const std::vector<Complex<double>> &a, const std::vector<Complex<double>> &b,
                  const source_location &sl = source_location::current()) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        expect_close(a[i], b[i], sl);
    }
}

int main() {
    using namespace boost::ut;

    "fftfreq"_test = [] {
        constexpr int N = 10;
        std::vector<double> arr(N);
        for (int i = 0; i < N; ++i) {
            arr[i] = fftfreq<double>(i, N, 1.0f);
        }
        const std::vector<double> expected{0.0f,  0.1f,  0.2f,  0.3f,  0.4f,
                                           -0.5f, -0.4f, -0.3f, -0.2f, -0.1f};

        expect_close(arr, expected);
    };

    "grating"_test = [] {
        using C = Complex<double>;
        std::vector<C> data(16, C{1., 0.});

        run_gpu_test(data, [](DevComplex<double> *d_data) {
            const SimParams params{16, 1.0, 1.77, 0., 0., 0., 0.};
            apply_grating_factors_kernel<double>
                <<<1, 32>>>(d_data, params, DevComplex<double>{2., 0.}, DevComplex<double>{0., 0.},
                            6.0, 0.5, 0.0);
        });

        std::vector<C> expected = {C{0., 0.}, C{2., 0.}, C{2., 0.}, C{2., 0.}, C{0., 0.}, C{0., 0.},
                                   C{0., 0.}, C{2., 0.}, C{2., 0.}, C{2., 0.}, C{0., 0.}, C{0., 0.},
                                   C{0., 0.}, C{2., 0.}, C{2., 0.}, C{2., 0.}};

        expect(data == expected);
    };

    "fft"_test = [] {
        // Require the normalization factor to be 1 in the forward direction
        // and 1/n in the inverse direction

        const std::size_t N = 16;
        std::vector<Complex<double>> data(N, Complex<double>{1., 0.});
        const std::vector<Complex<double>> orig = data;

        const FFT<double> fft(N);

        run_gpu_test(data, [&](DevComplex<double> *d_data) { fft.forward(d_data, d_data); });
        check_cuda_result("forward", cudaPeekAtLastError());

        expect(data[0] == Complex<double>{16., 0.});
        for (std::size_t i = 1; i < N; ++i) {
            expect(data[i] == Complex<double>{0., 0.});
        }

        run_gpu_test(data, [&](DevComplex<double> *d_data) { fft.inverse(d_data, d_data); });
        expect_close(data, orig);
    };

    "fft2"_test = [] {
        // compare to np.fft.fft

        const std::size_t N = 8;

        std::vector<Complex<double>> data{{0., 0.}, {0., 0.}, {1., 0.}, {2., 1.},
                                          {3., 2.}, {2., 0.}, {1., 0.}, {0., 0.}};

        std::vector<Complex<double>> expected{
            {9., 3.}, {-5.12132034, -2.70710678}, {0., 2.}, {0.53553391, -1.29289322},
            {1., 1.}, {-0.87867966, -1.29289322}, {2., 2.}, {-6.53553391, -2.70710678}};
        const FFT<double> fft(N);

        run_gpu_test(data, [&](DevComplex<double> *d_data) { fft.forward(d_data, d_data); });

        expect_close(data, expected);
    };

    "fft2d_simple"_test = [] {
        // 2D FFT test with a simple 3x3 all-ones matrix
        // All-ones matrix should have DFT with only DC component non-zero
        
        const std::size_t nx = 3;
        const std::size_t ny = 3;
        const std::size_t N = nx * ny;
        
        std::vector<Complex<double>> data(N, Complex<double>{1., 0.});
        
        const FFT<double> fft2d(nx, ny);
        
        // Forward transform
        run_gpu_test(data, [&](DevComplex<double> *d_data) { 
            fft2d.forward(d_data, d_data); 
        });
        
        // Expected: DC component = 9, all others = 0
        std::vector<Complex<double>> expected(N, Complex<double>{0., 0.});
        expected[0] = Complex<double>{9., 0.}; // DC component
        
        expect_close(data, expected);
        
        // Inverse transform should recover original
        run_gpu_test(data, [&](DevComplex<double> *d_data) { 
            fft2d.inverse(d_data, d_data); 
        });
        
        std::vector<Complex<double>> orig(N, Complex<double>{1., 0.});
        expect_close(data, orig);
    };

    "fft2d_complex"_test = [] {
            // 2D FFT test with a more complex 4x4 matrix
            // Compare with numpy.fft.fft2 result
            
            const std::size_t nx = 4;
            const std::size_t ny = 4;
            const std::size_t N = nx * ny;
            
            // Create a 4x4 test matrix (row-major order)
            std::vector<Complex<double>> data = {
                {1., 0.}, {2., 0.}, {3., 0.}, {4., 0.},
                {5., 0.}, {6., 0.}, {7., 0.}, {8., 0.},
                {9., 0.}, {10., 0.}, {11., 0.}, {12., 0.},
                {13., 0.}, {14., 0.}, {15., 0.}, {16., 0.}
            };
            
            const FFT<double> fft2d(nx, ny);
            
            // Forward transform
            run_gpu_test(data, [&](DevComplex<double> *d_data) { 
                fft2d.forward(d_data, d_data); 
            });
            
            // Expected result from numpy.fft.fft2
            // These values were computed with Python:
            // import numpy as np
            // data = np.arange(1, 17).reshape(4, 4)
            // result = np.fft.fft2(data)
            std::vector<Complex<double>> expected = {
                {136., 0.}, {-8., 8.}, {-8., 0.}, {-8., -8.},
                {-32., 32.}, {0., 0.}, {0., 0.}, {0., 0.},
                {-32., 0.}, {0., 0.}, {0., 0.}, {0., 0.},
                {-32., -32.}, {0., 0.}, {0., 0.}, {0., 0.}
            };
            
            expect_close(data, expected);
            
            // Inverse transform should recover original
            std::vector<Complex<double>> data_copy = data; // Save transformed data
            run_gpu_test(data, [&](DevComplex<double> *d_data) { 
                fft2d.inverse(d_data, d_data); 
            });
            
            std::vector<Complex<double>> orig = {
                {1., 0.}, {2., 0.}, {3., 0.}, {4., 0.},
                {5., 0.}, {6., 0.}, {7., 0.}, {8., 0.},
                {9., 0.}, {10., 0.}, {11., 0.}, {12., 0.},
                {13., 0.}, {14., 0.}, {15., 0.}, {16., 0.}
            };
            
            expect_close(data, orig);
        };

    "fft2d_non_square"_test = [] {
        // Test non-square 2D FFT (3x4 matrix)
        
        const std::size_t nx = 3;
        const std::size_t ny = 4;
        const std::size_t N = nx * ny;
        
        // Create a 3x4 test matrix (row-major order)
        std::vector<Complex<double>> data = {
            {0.446839,0.195642},
            {0.431379,0.852912},
            {0.720925,0.469890},
            {0.517833,0.433425},
            {0.339297,0.574297},
            {0.428627,0.188463},
            {0.107013,0.751237},
            {0.217230,0.524362},
            {0.609100,0.064641},
            {0.133202,0.808804},
            {0.826633,0.169460},
            {0.455897,0.436411}
        };
        
        // 保存原始数据用于比较
        std::vector<Complex<double>> original = data;
        
        const FFT<double> fft2d(nx, ny);
        
        std::cout << "=== 2D FFT Non-Square Test Debug ===" << std::endl;
        std::cout << "nx = " << nx << ", ny = " << ny << ", N = " << N << std::endl;
        std::cout << "FFT is 2D: " << fft2d.is_2d_transform() << std::endl;
        std::cout << "Original data (row-major):" << std::endl;
        for (std::size_t i = 0; i < ny; ++i) {
            for (std::size_t j = 0; j < nx; ++j) {
                std::cout << "(" << data[i * nx + j].real() << ", " << data[i * nx + j].imag() << ") ";
            }
            std::cout << std::endl;
        }
        
        // Forward transform
        run_gpu_test(data, [&](DevComplex<double> *d_data) { 
            fft2d.forward(d_data, d_data); 
        });
        
        std::cout << "\nAfter forward transform:" << std::endl;
        for (std::size_t i = 0; i < ny; ++i) {
            for (std::size_t j = 0; j < nx; ++j) {
                std::cout << "(" << data[i * nx + j].real() << ", " << data[i * nx + j].imag() << ") ";
            }
            std::cout << std::endl;
        }
        
        // Expected result computed with numpy.fft.fft2
        // These values were computed with Python:
        // import numpy as np
        // data = np.arange(1, 13).reshape(4, 3)  # Note: reshape is (ny, nx)
        // result = np.fft.fft2(data)
        std::vector<Complex<double>> expected = {
            {5.233976,5.469543},
            {0.023136,0.895308},
            {-1.642451,0.202470},
            {0.447310,0.308180},
            {0.055020,-2.218372},
            {-0.608990,-0.910486},
            {-0.169005,0.247824},
            {0.565607,0.266894},
            {-0.688155,-1.400767},
            {0.884290,0.048229},
            {0.165811,0.196153},
            {1.095514,-0.757275}
        };
        
        std::cout << "\nExpected result:" << std::endl;
        for (std::size_t i = 0; i < ny; ++i) {
            for (std::size_t j = 0; j < nx; ++j) {
                std::cout << "(" << expected[i * nx + j].real() << ", " << expected[i * nx + j].imag() << ") ";
            }
            std::cout << std::endl;
        }
        
        // 计算差异
        std::cout << "\nDifferences:" << std::endl;
        for (std::size_t i = 0; i < N; ++i) {
            double diff_real = std::abs(data[i].real() - expected[i].real());
            double diff_imag = std::abs(data[i].imag() - expected[i].imag());
            if (diff_real > 1e-7 || diff_imag > 1e-7) {
                std::cout << "Element " << i << " (" << i/nx << "," << i%nx << "): ";
                std::cout << "actual (" << data[i].real() << ", " << data[i].imag() << ") vs ";
                std::cout << "expected (" << expected[i].real() << ", " << expected[i].imag() << ") ";
                std::cout << "diff = (" << diff_real << ", " << diff_imag << ")" << std::endl;
            }
        }
        
        expect_close(data, expected);
        
        // Test inverse transform
        std::vector<Complex<double>> forward_result = data; // 保存正向变换结果
        run_gpu_test(data, [&](DevComplex<double> *d_data) { 
            fft2d.inverse(d_data, d_data); 
        });
        
        std::cout << "\nAfter inverse transform (should recover original):" << std::endl;
        for (std::size_t i = 0; i < ny; ++i) {
            for (std::size_t j = 0; j < nx; ++j) {
                std::cout << "(" << data[i * nx + j].real() << ", " << data[i * nx + j].imag() << ") ";
            }
            std::cout << std::endl;
        }
        
        // 检查逆变换是否恢复原始数据
        std::cout << "\nInverse transform differences from original:" << std::endl;
        for (std::size_t i = 0; i < N; ++i) {
            double diff_real = std::abs(data[i].real() - original[i].real());
            double diff_imag = std::abs(data[i].imag() - original[i].imag());
            if (diff_real > 1e-7 || diff_imag > 1e-7) {
                std::cout << "Element " << i << ": diff = (" << diff_real << ", " << diff_imag << ")" << std::endl;
            }
        }
        
        expect_close(data, original);
    };

    "config_parsing_grating"_test = [] {
        // We don't care about rounding errors so we use simple values here.
        // Don't use this as an example of a sensible grating.

        const std::string grating_yaml = "type: grating\n"
                                         "pitch: 0.125\n"
                                         "dc: [0.5, 0.75]\n"
                                         "z_start: 0.25\n"
                                         "thickness: 1.25\n"
                                         "nr_steps: 1\n"
                                         "x_positions: [0.0]\n"
                                         "substrate_thickness: 0.0625\n"
                                         "mat_a: [\"Si\", 2.0]\n"
                                         "mat_b: [\"Au\", 3.0]\n"
                                         "mat_substrate: [\"Si\", 4.0]\n";

        const Material si2{"Si", 2.0};
        const Material au3{"Au", 3.0};
        const Material si4{"Si", 4.0};

        const DeltabetaTable db_table{{si2, Complex<double>{11., 0.}},
                                      {au3, Complex<double>{12., 0.}},
                                      {si4, Complex<double>{13., 0.}}};

        const auto grating_node = YAML::Load(grating_yaml);
        const auto grating = parse_grating(grating_node, db_table);

        expect(grating.pitch == 0.125);
        expect(grating.dc[0] == 0.5);
        expect(grating.dc[1] == 0.75);
        expect(grating.z_start == 0.25);
        expect(grating.thickness == 1.25);
        expect(grating.nr_steps == 1);
        expect(grating.x_positions.size() == 1);
        expect(grating.x_positions[0] == 0.0);
        expect(grating.substrate_thickness == 0.0625);
        expect(grating.deltabeta_a == Complex<double>{11., 0.});
        expect(grating.deltabeta_b == Complex<double>{12., 0.});
        expect(grating.deltabeta_substrate == Complex<double>{13., 0.});

        const auto optical_element_ptr =
            parse_optical_element(grating_node, db_table, fs::path("/asdf-config-dir"));
        const auto grating_ptr = static_cast<Grating *>(optical_element_ptr.get());
        expect(grating_ptr->pitch == 0.125);
    };

    "config_parsing_dbtable"_test = [] {
        const YAML::Node db_table_node =
            YAML::Load("- [[\"Si\", 2.34], [2.2888e-7, 2.5847e-10]]\n"
                       "- [[\"Au\", 19.32], [1.4981e-6, 3.6174e-8]]\n");

        const DeltabetaTable db_table = parse_deltabeta_table(db_table_node);

        expect(db_table.size() == 2);
        expect(db_table[0].first.name == "Si");
        expect(db_table[1].first.name == "Au");

        expect_close(db_table[0].first.density, 2.34);
        expect_close(db_table[1].first.density, 19.32);

        expect_close(db_table[0].second, Complex<double>{2.2888e-7, 2.5847e-10});
        expect_close(db_table[1].second, Complex<double>{1.4981e-6, 3.6174e-8});
    };

    "config_parsing_simparams"_test = [] {
        const YAML::Node sim_params_node = YAML::Load("sim_params:\n"
                                                      "    N: 8192\n"
                                                      "    dx: 8e-10\n"
                                                      "    z_detector: 1.77\n"
                                                      "    detector_size: 5e-4\n"
                                                      "    detector_pixel_size_x: 1e-5\n"
                                                      "    detector_pixel_size_y: 1.0\n"
                                                      "    chunk_size: 10469376\n")["sim_params"];

        const SimParams sim_params = parse_sim_params(sim_params_node, 3.0);
        expect(sim_params.N == 8192);
        expect_close(sim_params.dx, 8e-10);
        expect_close(sim_params.z_detector, 1.77);
        expect_close(sim_params.detector_size, 5e-4);
        expect_close(sim_params.detector_pixel_size_x, 1e-5);
        expect_close(sim_params.detector_pixel_size_y, 1.0);
    };
}