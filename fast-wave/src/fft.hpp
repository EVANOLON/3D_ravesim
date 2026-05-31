// Copyright (c) 2024, ETH Zurich

#ifndef _FAST_WAVE_FFT_HPP
#define _FAST_WAVE_FFT_HPP

#include <types.hpp>

template<typename S>
class FFT {
    cufftHandle fft_plan;

    std::size_t N;          // 一维FFT: 点数; 二维FFT: nx * ny
    
    //2d start
    std::size_t nx, ny;     // 二维FFT的尺寸 (nx=0表示一维)
    bool is_2d;             // 标记是否为二维FFT
    //2d end

    FFT(const FFT &) = delete;
    FFT &operator=(const FFT &) = delete;

  public:
    explicit FFT(std::size_t N);
    //2d start
    // 二维FFT构造函数
    FFT(std::size_t nx, std::size_t ny);
    //2d end

    ~FFT();

    void forward(DevComplex<S> *d_u, DevComplex<S> *d_U) const;

    void inverse(DevComplex<S> *d_U, DevComplex<S> *d_u) const;

    //2d start
    bool is_2d_transform() const { return is_2d; }
    std::size_t get_nx() const { return nx; }
    std::size_t get_ny() const { return ny; }
    std::size_t get_n() const { return N; }
    //2d end
};

#endif // _FAST_WAVE_FFT_HPP
