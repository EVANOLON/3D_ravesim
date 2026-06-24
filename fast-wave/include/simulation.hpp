// Copyright (c) 2024, ETH Zurich

#ifndef _FAST_WAVE_SIMULATION_HPP
#define _FAST_WAVE_SIMULATION_HPP

#include <filesystem>
#include <memory>
#include <optional>
#include <types.hpp>
#include <vector>

#include "optical_element.hpp"
#include <wrappers.hpp>

namespace fs = std::filesystem;

/// planck constant in mˆ2 kg / s
const double h = 6.62607004e-34;
/// Vacuum speed of light in m / s
const double c_0 = 299792458.0;
/// equivalent energy of one electron volt in jules
const double eV_to_joule = 1.602176634e-19;

enum class SourceType {
    Point,
    Vector,
};

struct Source {
    SourceType type;
};

struct PointSource: public Source {
    double x;
    std::optional<double> y;    
    double z;
};

struct VectorSource: public Source {
    fs::path input_path;
    double z;
};

enum class DType {
    C8,
    C16,
};

struct Config {
    SimParams sim_params;
    std::vector<std::unique_ptr<OpticalElement>> optical_elements;
    std::unique_ptr<Source> source;
    DType dtype;
    bool save_final_u_vectors;
    bool save_debug_wavefields;  ///< Save intermediate full wavefields (wave_after_source, wave_before_sample, etc.); disabled by default for large 2D data
    std::vector<double> cutoff_angles;
};

[[nodiscard]] inline Complex<double> material_factor(Complex<double> deltabeta, double thickness,
                                                     double wl) {
    const double factor = 2.0 * M_PI * thickness / wl;
    // deltabeta = (δ, β)
    Complex<double> exponent(-factor * deltabeta.imag(), -factor * deltabeta.real());
    return std::exp(exponent);
}

[[nodiscard]] inline double convert_cutoff_angle_to_frequency(double angle, double wl) {
    return std::sin(angle) / wl;
}

/// Compute Fresnel scaling parameters for short source-sample distance.
inline void compute_fresnel_params(double z_src, double z_sample, double z_det,
                                    double &z_eff, double &M) {
    const double z_s = z_sample - z_src;
    const double z_d = z_det - z_sample;
    if (z_s <= 0.0 || z_d <= 0.0) {
        z_eff = z_d;
        M = 1.0;
        return;
    }
    z_eff = (z_s * z_d) / (z_s + z_d);
    M = (z_s + z_d) / z_s;
}

/// Given an energy in eV, calculate the wavelength in metres or vice versa
[[nodiscard]] inline double convert_energy_wavelength(double energy_or_wavelength) {
    return h * c_0 / (energy_or_wavelength * eV_to_joule);
}

void run_simulation(const Config &config, const std::filesystem::path &sub_dir,
                    std::optional<double> history_dz);

#endif // _FAST_WAVE_SIMULATION_HPP