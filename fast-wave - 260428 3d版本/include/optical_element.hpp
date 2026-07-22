// Copyright (c) 2024, ETH Zurich

#ifndef _FAST_WAVE_OPTICAL_ELEMENT_HPP
#define _FAST_WAVE_OPTICAL_ELEMENT_HPP

#include <array>
#include <stdint.h>
#include <vector>

#include <types.hpp>

struct Material {
    std::string name;
    double density;
};

enum class OpticalElementType {
    Grating,
    EnvGrating,
    Sample,
    PreciseSample,
    PlasmaSample,
};

struct OpticalElement {
    double z_start;
    std::vector<double> x_positions;
    std::vector<double> y_positions;
    OpticalElementType type;

    /// Z distance from the start of the element to the end of the element
    virtual double total_thickness() const = 0;

    /// How many history entries will be added for this element?
    virtual std::size_t nr_history_entries() const = 0;
};

struct Grating : public OpticalElement {
    double thickness;
    double pitch;
    std::array<double, 2> dc;
    int nr_steps;
    double substrate_thickness;
    Complex<double> deltabeta_a;
    Complex<double> deltabeta_b;
    Complex<double> deltabeta_substrate;

    double total_thickness() const override { return thickness + substrate_thickness; }

    std::size_t nr_history_entries() const override { return nr_steps; }
};

struct EnvGrating : public OpticalElement {
    double thickness;
    double pitch0;
    double pitch1;
    std::array<double, 2> dc0;
    std::array<double, 2> dc1;
    int nr_steps;
    double substrate_thickness;
    Complex<double> deltabeta_a;
    Complex<double> deltabeta_b;
    Complex<double> deltabeta_substrate;

    double total_thickness() const override { return thickness + substrate_thickness; }

    std::size_t nr_history_entries() const override { return nr_steps; }
};

struct Sample : public OpticalElement {
    std::vector<uint32_t> grid; // linearized: constant z entries are contiguous
    std::size_t x_len;
    std::size_t y_len;           // 添加 y 方向尺寸
    std::size_t z_len;
    double pixel_size_x;                     // in metres
    double pixel_size_y;        // 添加 y 方向像素尺寸
    double pixel_size_z;                     // in metres
    std::vector<Complex<double>> deltabetas; // including vacuum as first entry

    double total_thickness() const override { return pixel_size_z * z_len; }
    std::size_t nr_history_entries() const override { return z_len; }
};

//precise_sample_update_begin
struct PreciseSample : public OpticalElement {
    std::vector<float> density_grid;         // 浮点密度网格
    std::vector<uint32_t> material_grid;      // 材料网格，存储材料类型
    std::vector<Complex<double>> deltabeta_grid; // 新增：预先计算的deltabeta网格
    std::size_t x_len;                       // x方向尺寸
    std::size_t z_len;                       // z方向尺寸  
    double pixel_size_x;                     // x方向像素尺寸(米)
    double pixel_size_z;                     // z方向像素尺寸(米)
    
    // 删除material_deltabetas成员变量
    
    double total_thickness() const override { return pixel_size_z * z_len; }
    std::size_t nr_history_entries() const override { return z_len; }
};
//precise_sample_update_end

//plasma_sample_begin
struct PlasmaSample : public OpticalElement {
    std::vector<float> ne_grid;                  // electron density [cm^-3]
    std::vector<float> ni_grid;                  // ion density [cm^-3]
    std::vector<float> te_grid;                  // electron temperature [eV]
    std::vector<float> zstar_grid;               // average ionisation state
    int Z;                                       // atomic number

    // Pre-computed deltabeta (set at parse time)
    std::vector<Complex<double>> deltabeta_grid;

    std::size_t x_len, y_len, z_len;
    double pixel_size_x, pixel_size_y, pixel_size_z;

    double total_thickness() const override { return pixel_size_z * z_len; }
    std::size_t nr_history_entries() const override { return z_len; }
};
//plasma_sample_end

#endif // _FAST_WAVE_OPTICAL_ELEMENT_HPP
