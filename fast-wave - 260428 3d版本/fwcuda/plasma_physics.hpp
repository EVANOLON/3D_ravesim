// Copyright (c) 2024, ETH Zurich
//
// Host-side plasma refractive index calculation.
// Matches big-wave/plasma.py physics:
//   n = 1 - delta - i*beta
//   delta = n_e * r_e * lambda^2 / (2*pi)          [free-electron]
//   beta  = Kramers inverse bremsstrahlung            [free-free]
//
// Bound-electron Chantler scaling is omitted here (requires nist_lookup
// SQLite database).  For neutral or cold materials use the Sample or
// PreciseSample element types instead.

#ifndef _FAST_WAVE_PLASMA_PHYSICS_HPP
#define _FAST_WAVE_PLASMA_PHYSICS_HPP

#include <cmath>
#include <complex>

namespace plasma_physics {

// Physical constants
constexpr double R_ELECTRON_CM = 2.8179403227e-13;  // classical electron radius [cm]
constexpr double H_PLANCK = 6.62607004e-34;          // J·s
constexpr double C_LIGHT = 299792458.0;              // m/s
constexpr double EV_TO_JOULE = 1.602176634e-19;      // J/eV
constexpr double PLANCK_HC = H_PLANCK * C_LIGHT;     // J·m

/// Compute wavelength in cm from photon energy in eV.
inline double energy_to_wavelength_cm(double energy_eV) {
    return 1.0e-8 * PLANCK_HC / energy_eV;
}

/// Gaunt factor (free-free).  Using g_ff ≈ 1 approximation.
inline double gaunt_ff(double /*T_e*/, double /*energy*/) {
    return 1.0;
}

/// Kramers inverse bremsstrahlung beta (free-free absorption).
/// Rybicki & Lightman Eq. 5.18a.
inline double kramers_beta_ff(double n_e, double n_i, double Z_star,
                               double T_e, double energy) {
    if (n_e <= 0.0 || n_i <= 0.0 || Z_star <= 0.0 || T_e <= 0.0 || energy <= 0.0)
        return 0.0;

    const double T_K = T_e * 11604.5;                // eV → K
    const double nu = energy / 4.135667e-15;         // eV → Hz (h in eV·s)
    const double g_ff = gaunt_ff(T_e, energy);

    // Kramers absorption coefficient [cm^-1]
    const double alpha_ff = 3.7e8 * n_e * n_i * (Z_star * Z_star) * g_ff
                            / (std::sqrt(T_K) * nu * nu * nu);

    const double lamb_cm = energy_to_wavelength_cm(energy);
    return alpha_ff * lamb_cm / (4.0 * M_PI);
}

/// Compute (delta, beta, attenuation_length_cm) for a plasma.
///
/// Returns only the free-electron contribution for delta and the
/// Kramers free-free contribution for beta.  Bound-electron Chantler
/// scaling is omitted (requires nist_lookup SQLite database).
inline void plasma_delta_beta_host(
    double n_e, double n_i, double T_e, double Z_star, int Z,
    double energy,
    double &delta, double &beta, double &attenuation_length_cm)
{
    const double lamb_cm = energy_to_wavelength_cm(energy);
    const double prefactor = R_ELECTRON_CM * lamb_cm * lamb_cm / (2.0 * M_PI);

    // Free-electron contribution (dominant for hot plasma)
    const double delta_free = n_e * prefactor;

    // Bound-electron contribution (simplified: (Z-Z*)/Z scaling)
    // Without nist_lookup database, the Chantler f1/f2 values are not
    // available.  For weakly-ionised plasma this underestimates delta/beta.
    const double fraction_bound = (Z_star < Z) ? (Z - Z_star) / Z : 0.0;
    double delta_bound = 0.0;
    double beta_bound = 0.0;
    if (fraction_bound > 0.0 && n_i > 0.0) {
        // Bound contribution approximated as:
        // delta_bound ≈ n_i * prefactor * Z * fraction_bound
        // beta_bound  ≈ n_i * prefactor * Z^(1/3) * fraction_bound  (rough)
        // These are order-of-magnitude estimates.
        // For accurate values, use the Python big-wave with nist_lookup.
        delta_bound = n_i * prefactor * static_cast<double>(Z) * fraction_bound * 0.5;
        beta_bound = n_i * prefactor * std::cbrt(static_cast<double>(Z)) * fraction_bound * 0.1;
    }

    // Free-free absorption
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
