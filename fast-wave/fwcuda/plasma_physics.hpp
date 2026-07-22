// Copyright (c) 2024, ETH Zurich
//
// Host-side plasma refractive index calculation (free-electron + Kramers).
// Matches big-wave/plasma.py physics.

#ifndef _FAST_WAVE_PLASMA_PHYSICS_HPP
#define _FAST_WAVE_PLASMA_PHYSICS_HPP

#include <cmath>
#include <complex>

namespace plasma_physics {

constexpr double R_ELECTRON_CM = 2.8179403227e-13;
constexpr double H_PLANCK = 6.62607004e-34;
constexpr double C_LIGHT = 299792458.0;
constexpr double EV_TO_JOULE = 1.602176634e-19;
constexpr double PLANCK_HC = H_PLANCK * C_LIGHT;

inline double energy_to_wavelength_cm(double energy_eV) {
    return 1.0e-8 * PLANCK_HC / energy_eV;
}

inline double gaunt_ff(double /*T_e*/, double /*energy*/) { return 1.0; }

inline double kramers_beta_ff(double n_e, double n_i, double Z_star,
                               double T_e, double energy) {
    if (n_e <= 0.0 || n_i <= 0.0 || Z_star <= 0.0 || T_e <= 0.0 || energy <= 0.0)
        return 0.0;
    const double T_K = T_e * 11604.5;
    const double nu = energy / 4.135667e-15;
    const double g_ff = gaunt_ff(T_e, energy);
    const double alpha_ff = 3.7e8 * n_e * n_i * (Z_star * Z_star) * g_ff
                            / (std::sqrt(T_K) * nu * nu * nu);
    const double lamb_cm = energy_to_wavelength_cm(energy);
    return alpha_ff * lamb_cm / (4.0 * M_PI);
}

inline void plasma_delta_beta_host(
    double n_e, double n_i, double T_e, double Z_star, int Z,
    double energy,
    double &delta, double &beta, double &attenuation_length_cm)
{
    const double lamb_cm = energy_to_wavelength_cm(energy);
    const double prefactor = R_ELECTRON_CM * lamb_cm * lamb_cm / (2.0 * M_PI);

    // Free-electron contribution
    const double delta_free = n_e * prefactor;

    // Simplified bound-electron contribution (rough estimate)
    const double fraction_bound = (Z_star < Z) ? (Z - Z_star) / Z : 0.0;
    double delta_bound = 0.0;
    double beta_bound = 0.0;
    if (fraction_bound > 0.0 && n_i > 0.0) {
        delta_bound = n_i * prefactor * static_cast<double>(Z) * fraction_bound * 0.5;
        beta_bound  = n_i * prefactor * std::cbrt(static_cast<double>(Z)) * fraction_bound * 0.1;
    }

    const double beta_ff = kramers_beta_ff(n_e, n_i, Z_star, T_e, energy);

    delta = delta_free + delta_bound;
    beta = beta_bound + beta_ff;

    if (beta > 0.0)
        attenuation_length_cm = lamb_cm / (4.0 * M_PI * beta);
    else
        attenuation_length_cm = INFINITY;
}

} // namespace plasma_physics

#endif // _FAST_WAVE_PLASMA_PHYSICS_HPP
