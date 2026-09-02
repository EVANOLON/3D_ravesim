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
    beta_ff:    Kramers inverse bremsstrahlung

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
_symbol_cache: Dict[int, str] = {13: "Al"}


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


def _get_element_symbol(Z: int) -> str:
    """Resolve and cache an atomic symbol once per element."""
    if Z not in _symbol_cache:
        _lazy_import_xraydb()
        db = _xrayDB()
        _symbol_cache[Z] = db.atomic_symbol(Z)
    return _symbol_cache[Z]


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
    # Single-element case is a special case of the multi-element formula:
    # f1bar -> f1(Z), f2eff_bar -> f2(Z)*mu_ratio, zbar -> Z, z2bar -> Z^2.
    scat = _get_scatterer(_get_element_symbol(Z), energy)
    f1bar = scat.f1
    f2eff_bar = scat.f2 * (scat.mu_total / scat.mu_photo)
    zbar = float(Z)
    z2bar = float(Z) * float(Z)
    return plasma_delta_beta_multi(
        n_e, n_i, T_e, Z_star, f1bar, f2eff_bar, zbar, z2bar, energy
    )


def plasma_delta_beta_grid(
    ne_grid,
    ni_grid,
    te_grid,
    zstar_grid,
    Z: int,
    energy: float,
) -> Tuple:
    """Vectorized plasma optics for an arbitrary tile of equally shaped grids.

    Chantler constants and the element symbol are scalar cached values.  All
    density/temperature arithmetic is performed by NumPy without per-pixel
    Python calls.
    """
    # Single-element case = multi-element formula with f1bar/f2eff/zbar/z2bar.
    scat = _get_scatterer(_get_element_symbol(Z), energy)
    f1bar = scat.f1
    f2eff_bar = scat.f2 * (scat.mu_total / scat.mu_photo)
    zbar = float(Z)
    z2bar = float(Z) * float(Z)
    return plasma_delta_beta_grid_multi(
        ne_grid, ni_grid, te_grid, zstar_grid, f1bar, f2eff_bar, zbar, z2bar, energy
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-element (compound) plasma optics — 方案 B'
#
# Physics (per-atom-average norm):
#   x_i = nu_i / sum_j nu_j              (atom fraction)
#   zbar    = sum_i x_i Z_i              (mean atomic number)
#   z2bar   = sum_i x_i Z_i^2            (mean-square atomic number)
#   f1bar   = sum_i x_i f1_i             (Chantler f1' + Z, per atom)
#   f2eff   = sum_i x_i f2_i (mu_tot/mu_photo)_i
#   q = clip(Z* / zbar, 0, 1)            (common ionization fraction)
#   delta   = prefactor [n_e + n_i (1-q) f1bar]
#   beta    = prefactor n_i (1-q) f2eff + beta_ff(q, z2bar)
# Kramers free-free uses  n_e * n_i * (q^2 * z2bar)   (sum_i n_i Z_i*^2).
# ═══════════════════════════════════════════════════════════════════════════════


def compound_scattering_factors(formula: str, energy: float) -> Tuple[float, float, float, float]:
    """
    Per-atom-average Chantler coefficients for a compound formula.

    Returns (f1bar, f2eff_bar, zbar, z2bar).
    """
    _lazy_import_xraydb()
    from nist_lookup.chemparser import chemparse

    comp = chemparse(formula)
    total = float(sum(comp.values()))
    if total <= 0:
        raise ValueError(f"empty composition for formula {formula!r}")

    f1bar = 0.0
    f2eff_bar = 0.0
    zbar = 0.0
    z2bar = 0.0
    for symbol, count in comp.items():
        scat = _get_scatterer(symbol, energy)
        x = float(count) / total
        f1bar += x * scat.f1
        f2eff_bar += x * scat.f2 * (scat.mu_total / scat.mu_photo)
        zbar += x * scat.number
        z2bar += x * (scat.number ** 2)
    return f1bar, f2eff_bar, zbar, z2bar


def plasma_optics_table_for_elements(elements_dct, energy: float):
    """
    Build the per-element plasma optics scalar table for a list of config element
    dicts at a given energy.  Non-plasma elements map to None; plasma_sample
    elements map to {formula, f1bar, f2eff_bar, zbar, z2bar}.
    """
    table = []
    for el in elements_dct:
        if el.get("type") != "plasma_sample":
            continue
        formula = el.get("formula")
        if not formula:
            # fall back to a single-element description via the legacy Z field
            z = int(el.get("Z", 0))
            if z <= 0:
                raise ValueError(
                    "plasma_sample requires 'formula' (multi-element) or a valid 'Z'"
                )
            scat = _get_scatterer(_get_element_symbol(z), energy)
            f1bar = scat.f1
            f2eff_bar = scat.f2 * (scat.mu_total / scat.mu_photo)
            zbar = float(z)
            z2bar = float(z) * float(z)
            table.append({
                "formula": None,
                "f1bar": float(f1bar),
                "f2eff_bar": float(f2eff_bar),
                "zbar": float(zbar),
                "z2bar": float(z2bar),
            })
            continue
        f1bar, f2eff_bar, zbar, z2bar = compound_scattering_factors(formula, energy)
        table.append({
            "formula": str(formula),
            "f1bar": float(f1bar),
            "f2eff_bar": float(f2eff_bar),
            "zbar": float(zbar),
            "z2bar": float(z2bar),
        })
    return table


def _kramers_beta_ff_multi(
    n_e: float, n_i: float, q: float, z2bar: float, T_e: float, energy: float
) -> float:
    """Kramers inverse-bremsstrahlung with mean-square charge (sum_i n_i Z_i*^2)."""
    if n_e <= 0.0 or n_i <= 0.0 or q <= 0.0 or z2bar <= 0.0 or T_e <= 0.0 or energy <= 0.0:
        return 0.0

    T_K = T_e * 11604.5
    nu = energy / 4.135667e-15  # eV -> Hz
    g_ff = _gaunt_ff(T_e, energy)
    alpha_ff = (
        3.7e8
        * n_e
        * n_i
        * (q ** 2 * z2bar)
        * g_ff
        / (T_K ** 0.5 * nu ** 3)
    )
    lamb_cm = PLANCK_HC * 1.0e-8 / energy
    return alpha_ff * lamb_cm / (4.0 * pi)


def plasma_delta_beta_multi(
    n_e: float,
    n_i: float,
    T_e: float,
    Z_star: float,
    f1bar: float,
    f2eff_bar: float,
    zbar: float,
    z2bar: float,
    energy: float,
) -> Tuple[float, float, float]:
    """
    Multi-element plasma (delta, beta, attenuation_length_cm).

    n_e, n_i in cm^-3; T_e in eV; Z_star = free electrons per nucleus.
    f1bar/f2eff_bar/zbar/z2bar are the per-atom Chantler coefficients
    (see compound_scattering_factors).
    """
    lamb_cm = 1.0e-8 * PLANCK_HC / energy
    prefactor = R_ELECTRON_CM * lamb_cm * lamb_cm / (2.0 * pi)

    delta_free = n_e * prefactor

    q = (Z_star / zbar) if zbar > 0.0 else 0.0
    q = min(max(q, 0.0), 1.0)  # clip to [0, 1]
    fraction_bound = 1.0 - q

    if fraction_bound > 0.0 and n_i > 0.0:
        delta_bound = n_i * prefactor * f1bar * fraction_bound
        beta_bound = n_i * prefactor * f2eff_bar * fraction_bound
    else:
        delta_bound = 0.0
        beta_bound = 0.0

    beta_ff = _kramers_beta_ff_multi(n_e, n_i, q, z2bar, T_e, energy)

    delta = delta_free + delta_bound
    beta = beta_bound + beta_ff

    if beta > 0.0:
        atlen_cm = lamb_cm / (4.0 * pi * beta)
    else:
        atlen_cm = float("inf")

    return delta, beta, atlen_cm


def plasma_delta_beta_grid_multi(
    ne_grid,
    ni_grid,
    te_grid,
    zstar_grid,
    f1bar: float,
    f2eff_bar: float,
    zbar: float,
    z2bar: float,
    energy: float,
) -> Tuple:
    """Vectorized multi-element plasma optics for a tile of equally shaped grids."""
    import numpy as np

    ne = np.asarray(ne_grid, dtype=np.float64)
    ni = np.asarray(ni_grid, dtype=np.float64)
    te = np.asarray(te_grid, dtype=np.float64)
    zstar = np.asarray(zstar_grid, dtype=np.float64)
    if not (ne.shape == ni.shape == te.shape == zstar.shape):
        raise ValueError("plasma grid tiles must have identical shapes")
    if energy <= 0.0:
        raise ValueError(f"energy must be positive, got {energy}")

    lamb_cm = 1.0e-8 * PLANCK_HC / energy
    prefactor = R_ELECTRON_CM * lamb_cm * lamb_cm / (2.0 * pi)
    delta = ne * prefactor
    beta = np.zeros(ne.shape, dtype=np.float64)

    q = np.where(zbar > 0.0, zstar / zbar, 0.0)
    q = np.clip(q, 0.0, 1.0)
    fraction_bound = 1.0 - q

    bound_mask = (fraction_bound > 0.0) & (ni > 0.0)
    if np.any(bound_mask):
        bound_scale = ni * prefactor * fraction_bound
        delta = delta + np.where(bound_mask, bound_scale * f1bar, 0.0)
        beta += np.where(bound_mask, bound_scale * f2eff_bar, 0.0)

    ff_mask = (ne > 0.0) & (ni > 0.0) & (q > 0.0) & (z2bar > 0.0) & (te > 0.0)
    if np.any(ff_mask):
        nu = energy / 4.135667e-15
        alpha_ff = np.zeros(ne.shape, dtype=np.float64)
        alpha_ff[ff_mask] = (
            3.7e8
            * ne[ff_mask]
            * ni[ff_mask]
            * (np.square(q[ff_mask]) * z2bar)
            / (np.sqrt(te[ff_mask] * 11604.5) * nu ** 3)
        )
        beta += alpha_ff * lamb_cm / (4.0 * pi)

    atlen = np.full(ne.shape, np.inf, dtype=np.float64)
    positive_beta = beta > 0.0
    atlen[positive_beta] = lamb_cm / (4.0 * pi * beta[positive_beta])
    return delta, beta, atlen
