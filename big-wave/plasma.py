"""
Plasma refractive index (delta, beta) calculation for X-ray wave propagation.

Replaces the neutral-atom Chantler lookup (xray_delta_beta) when the material
is a partially or fully ionized plasma described by n_e, n_i, T_e, Z*.

Physics:
  n = 1 - delta - i*beta

  delta = delta_free + delta_bound
    delta_free = n_e * r_e * lambda^2 / (2*pi)          [analytic]
    delta_bound: Chantler f1 scaled by (Z-Z*)/Z for remaining bound electrons

  beta = beta_bound + beta_ff
    beta_bound: Chantler mu_photo/mu_total scaled by (Z-Z*)/Z
    beta_ff:    Kramers inverse bremsstrahlung  [placeholder — to be added]

References:
  J.D. Jackson, Classical Electrodynamics, 3rd ed., §7.5
  H.A. Kramers, Phil. Mag. 46, 836 (1923)
  B.L. Henke et al., At. Data Nucl. Data Tables 54, 181 (1993)
"""

from math import pi
from typing import Dict, Tuple

try:
    # editable install: nist_lookup maps directly to inner dir
    from nist_lookup.physical_constants import R_ELECTRON_CM, PLANCK_HC  # type: ignore
except ImportError:
    # direct path import
    from nist_lookup.nist_lookup.physical_constants import R_ELECTRON_CM, PLANCK_HC  # type: ignore

# Lazy imports — xraydb requires a compatible SQLite database at runtime.
_Scatterer = None
_xrayDB = None
_scatterer_cache: Dict[Tuple[str, float], object] = {}


def _lazy_import_xraydb():
    """Defer the xraydb import until first use."""
    global _Scatterer, _xrayDB
    if _Scatterer is None:
        from nist_lookup.xraydb_plugin import Scatterer as _S, xrayDB as _X  # type: ignore
        _Scatterer = _S
        _xrayDB = _X


def _get_scatterer(symbol: str, energy: float):
    """Return a cached Scatterer for the given element and energy."""
    _lazy_import_xraydb()
    key = (symbol, energy)
    if key not in _scatterer_cache:
        _scatterer_cache[key] = _Scatterer(symbol, energy)
    return _scatterer_cache[key]


def _gaunt_ff(T_e: float, energy: float) -> float:
    """
    Free-free Gaunt factor.

    Using g_ff ≈ 1 approximation.  The error is modest for the parameter
    ranges of interest.

    For hν/kT ≫ 1 (hard photons):  g_ff → √3/π ≈ 0.55
    For hν/kT ≪ 1 (soft photons):  g_ff → 1

    Reference:
      W.J. Karzas & R. Latter, ApJS 6, 167 (1961), Tables I–III.
    """
    return 1.0


def _kramers_beta_ff(
    n_e: float, n_i: float, Z_star: float, T_e: float, energy: float
) -> float:
    """
    Inverse bremsstrahlung (free-free) absorption beta.

    The free-free absorption coefficient is (Rybicki & Lightman Eq. 5.18a):

        alpha_ff = 3.7e8 * n_e * n_i * Z*^2 * g_ff / (sqrt(T_K) * nu^3)  [cm^-1]

    where T_K is the electron temperature in Kelvin and nu is the photon
    frequency in Hz.  Converting to practical units (T_e, energy in eV):

        T_K = T_e * 11604.5
        nu  = energy / 4.13567e-15

    Then beta_ff = alpha_ff * lambda / (4*pi).

    Parameters
    ----------
    n_e : float
        Electron number density [cm^-3].
    n_i : float
        Ion number density [cm^-3].
    Z_star : float
        Average ionization state.
    T_e : float
        Electron temperature [eV].
    energy : float
        X-ray energy [eV].

    Returns
    -------
    beta_ff : float

    References
    ----------
    G.B. Rybicki & A.P. Lightman, Radiative Processes in Astrophysics,
        Wiley (1979), Eq. 5.18a.
    H.A. Kramers, Phil. Mag. 46, 836 (1923).
    NRL Plasma Formulary (2019), p. 57.
    """
    if n_e <= 0.0 or n_i <= 0.0 or Z_star <= 0.0 or T_e <= 0.0 or energy <= 0.0:
        return 0.0

    T_K = T_e * 11604.5
    nu = energy / 4.135667e-15  # eV -> Hz (h in eV·s)
    g_ff = _gaunt_ff(T_e, energy)

    # Kramers absorption coefficient [cm^-1]
    alpha_ff = (
        3.7e8
        * n_e
        * n_i
        * (Z_star ** 2)
        * g_ff
        / (T_K ** 0.5 * nu ** 3)
    )

    lamb_cm = PLANCK_HC * 1.0e-8 / energy
    return alpha_ff * lamb_cm / (4.0 * pi)


def plasma_delta_beta(
    n_e: float,
    n_i: float,
    T_e: float,
    Z_star: float,
    Z: int,
    energy: float,
) -> Tuple[float, float, float]:
    """
    Compute (delta, beta, attenuation_length) for a plasma.

    Uses the free-electron plasma dispersion relation for delta_free and
    scales the Chantler neutral-atom scattering factors by the bound-electron
    fraction (Z - Z*)/Z for the bound-electron contribution.

    Parameters
    ----------
    n_e : float
        Electron number density [cm^-3].
    n_i : float
        Ion number density [cm^-3].
    T_e : float
        Electron temperature [eV].  Currently unused (reserved for Kramers β).
    Z_star : float
        Average ionization state.  0 = neutral, Z = fully ionized.
    Z : int
        Atomic number (e.g., 13 for Al).
    energy : float
        X-ray photon energy [eV].

    Returns
    -------
    (delta, beta, attenuation_length_cm) : Tuple[float, float, float]
        delta : float
            Refractive index decrement  (n = 1 - delta - i*beta).
        beta : float
            Absorption index.
        attenuation_length_cm : float
            1/e attenuation length [cm] = lambda / (4*pi*beta).
            Returns inf if beta == 0.
    """
    lamb_cm = 1.0e-8 * PLANCK_HC / energy
    prefactor = R_ELECTRON_CM * lamb_cm * lamb_cm / (2.0 * pi)

    # ---- free-electron contribution ----
    delta_free = n_e * prefactor

    # ---- bound-electron contribution (Chantler scaling) ----
    fraction_bound = (Z - Z_star) / Z if Z_star < Z else 0.0

    if fraction_bound > 0.0 and n_i > 0.0:
        # Map Z to element symbol for Chantler lookup
        symbol = {13: "Al"}.get(Z, None)
        if symbol is None:
            _lazy_import_xraydb()
            db = _xrayDB()
            symbol = db.atomic_symbol(Z)
            # Keep the session alive only long enough for the lookup.
            # The Scatterer cache will hold its own references.

        scat = _get_scatterer(symbol, energy)

        # Scale neutral-atom scattering factors by bound-electron fraction.
        # f1 already includes Z (Thomson) + f1' (anomalous).
        # f2 is the imaginary anomalous scattering factor.
        delta_bound = n_i * prefactor * scat.f1 * fraction_bound
        # beta_photo scales with f2; mu_total/mu_photo accounts for scattering
        beta_bound = (
            n_i * prefactor * scat.f2 * fraction_bound * (scat.mu_total / scat.mu_photo)
        )
    else:
        delta_bound = 0.0
        beta_bound = 0.0

    # ---- inverse bremsstrahlung (free-free) ----
    beta_ff = _kramers_beta_ff(n_e, n_i, Z_star, T_e, energy)

    # ---- assemble ----
    delta = delta_free + delta_bound
    beta = beta_bound + beta_ff

    if beta > 0.0:
        atlen_cm = lamb_cm / (4.0 * pi * beta)
    else:
        atlen_cm = float("inf")

    return delta, beta, atlen_cm


def plasma_delta_beta_grid(
    ne_grid,
    ni_grid,
    te_grid,
    zstar_grid,
    Z: int,
    energy: float,
) -> Tuple:
    """
    Vectorized wrapper: compute (delta_grid, beta_grid, atlen_grid) over
    full 2D grids.  Each input grid should be a numpy array of the same shape.

    Returns three numpy arrays of matching shape.
    """
    import numpy as np

    shape = ne_grid.shape
    delta_grid = np.zeros(shape, dtype=np.float64)
    beta_grid = np.zeros(shape, dtype=np.float64)
    atlen_grid = np.zeros(shape, dtype=np.float64)

    it = np.nditer(ne_grid, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        ne = float(ne_grid[idx])
        ni = float(ni_grid[idx])
        te = float(te_grid[idx])
        zs = float(zstar_grid[idx])

        d, b, a = plasma_delta_beta(ne, ni, te, zs, Z, energy)
        delta_grid[idx] = d
        beta_grid[idx] = b
        atlen_grid[idx] = a

        it.iternext()

    return delta_grid, beta_grid, atlen_grid
