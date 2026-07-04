"""Non-régression de la VolSurface et de la vol locale (Dupire)."""
from __future__ import annotations

import numpy as np
import pytest

from JR_PRICER.market_data.quote import Quote
from JR_PRICER.curves.forward.analytic_forward import AnalyticForwardCurve
from JR_PRICER.curves.vol_smile.volsmile import VolSmile
from JR_PRICER.curves.vol_smile.moneyness_convention import AbsoluteStrike
from JR_PRICER.curves.interpolators_1D.log_linear import LogLinearInterpolator1D
from JR_PRICER.surfaces.vol_surface.volsurface import VolSurface
from JR_PRICER.surfaces.vol_surface.interpolators_2D.bi_cubic import BiCubicInterpolator
from JR_PRICER.surfaces.vol_surface.interpolators_2D.SVI import SVIInterpolator
from JR_PRICER.surfaces.vol_surface.interpolators_2D.SSVI import SSVIInterpolator
from JR_PRICER.surfaces.vol_surface.theta_curve import ThetaCurve
from JR_PRICER.pricing.model.local_vol import LocalVolModel
from tests.helpers import REF_DATE, DC, in_days, flat_discount, flat_vol_surface_underlying


def _smiley_surface(make_interp, with_forward=False):
    """VolSurface non plate (smile en U) en AbsoluteStrike, pour tester la vectorisation."""
    disc = flat_discount(0.03)
    s = Quote(100.0)
    fwd = AnalyticForwardCurve(s, disc, dividend_yield=0.0)
    strikes = [70.0, 85.0, 100.0, 115.0, 130.0]
    maturity_dates = [in_days(d) for d in (91, 182, 365, 730)]
    smiles = []
    for m in maturity_dates:
        vols = [0.25, 0.22, 0.20, 0.225, 0.255]  # smile
        smiles.append(VolSmile(strikes, vols, LogLinearInterpolator1D(), m, AbsoluteStrike()))
    surface = VolSurface(smiles, REF_DATE, DC, make_interp(fwd) if with_forward else make_interp(),
                         forward_curve=fwd)
    surface.calibrate_interpolator()
    return surface


def test_flat_surface_returns_constant_vol():
    und, _ = flat_vol_surface_underlying(sigma=0.2)
    surface = und.vol_provider
    for K in (70.0, 100.0, 130.0):
        for T in (0.3, 0.8, 1.5):
            assert surface.vol(K, T) == pytest.approx(0.2, rel=1e-6)


def test_dupire_local_vol_on_flat_surface_equals_implied():
    """Sur une surface plate, σ_loc(K,t) = σ_impl (Dupire dégénère vers la vol plate)."""
    und, disc = flat_vol_surface_underlying(sigma=0.2)
    surface = und.vol_provider
    fwd = und.forward_curve
    model = LocalVolModel(disc)
    for S in (80.0, 100.0, 120.0):
        for t in (0.25, 0.75, 1.5):
            assert model.sigma_loc(S, t, surface, fwd) == pytest.approx(0.2, rel=1e-4)


def test_dupire_numerator_uses_fixed_forward_moneyness():
    """Régression : le numérateur de Gatheral est ∂_T w à log-moneyness y FIXE.

    Les interpolateurs fournissent ∂_T w à STRIKE K fixe ; `sigma_loc` doit corriger de
    `(r−q)·∂_y w`. Sur une surface à skew avec r≠0 (r=0 rendrait fixe-K = fixe-y ; une
    surface plate annulerait ∂_y w — d'où la non-détection du bug par le test plat), on
    compare à une référence Gatheral indépendante différenciée à y fixe."""
    disc = flat_discount(0.05)                       # r≠0 : indispensable pour exposer le terme
    s = Quote(100.0)
    fwd = AnalyticForwardCurve(s, disc, dividend_yield=0.0)
    strikes = [70.0, 85.0, 100.0, 115.0, 130.0]
    mats = [in_days(d) for d in (91, 182, 365, 730)]
    smiles = [VolSmile(strikes, [0.25, 0.22, 0.20, 0.225, 0.255],   # skew (∂_y w ≠ 0)
                       LogLinearInterpolator1D(), m, AbsoluteStrike()) for m in mats]
    surface = VolSurface(smiles, REF_DATE, DC, SVIInterpolator(fwd), forward_curve=fwd)
    surface.calibrate_interpolator()
    model = LocalVolModel(disc)

    def w_fixed_y(y, T):                             # variance totale à log-moneyness forward FIXE
        K = fwd.forward(T) * np.exp(y)
        return float(surface.sigma(K, T)) ** 2 * T

    T, h = 0.7, 1e-4
    for y in (-0.15, 0.0, 0.12):
        w = w_fixed_y(y, T)
        wy = (w_fixed_y(y + h, T) - w_fixed_y(y - h, T)) / (2 * h)
        wyy = (w_fixed_y(y + h, T) - 2 * w + w_fixed_y(y - h, T)) / h ** 2
        dwT_y = (w_fixed_y(y, T + h) - w_fixed_y(y, T - h)) / (2 * h)   # ∂_T w à y FIXE (référence)
        denom = 1 - y / w * wy + 0.25 * (-0.25 - 1 / w + y ** 2 / w ** 2) * wy ** 2 + 0.5 * wyy
        ref = np.sqrt(dwT_y / denom)
        got = float(model.sigma_loc(fwd.forward(T) * np.exp(y), T, surface, fwd))
        assert got == pytest.approx(ref, rel=3e-3)


# --- Phase 2 : capacité vectorisée (strike scalaire <-> ndarray) -------------

K_QUERY = np.array([72.0, 90.0, 100.0, 110.0, 128.0])
T_QUERY = 0.7


def test_bicubic_vol_scalar_vector_consistency():
    surface = _smiley_surface(BiCubicInterpolator)
    vec = np.asarray(surface.vol(K_QUERY, T_QUERY))
    scal = np.array([surface.vol(float(K), T_QUERY) for K in K_QUERY])
    assert vec.shape == K_QUERY.shape
    assert vec == pytest.approx(scal, rel=1e-10)


def test_bicubic_derivatives_scalar_vector_consistency():
    surface = _smiley_surface(BiCubicInterpolator)
    interp = surface.interpolator
    for fn in (interp.dw_dK, interp.d2w_dK2, interp.dw_dT):
        vec = np.asarray(fn(K_QUERY, T_QUERY))
        scal = np.array([fn(float(K), T_QUERY) for K in K_QUERY])
        assert vec == pytest.approx(scal, rel=1e-9)


def test_svi_vol_and_derivatives_scalar_vector_consistency():
    surface = _smiley_surface(SVIInterpolator, with_forward=True)
    interp = surface.interpolator
    vec = np.asarray(surface.vol(K_QUERY, T_QUERY))
    scal = np.array([surface.vol(float(K), T_QUERY) for K in K_QUERY])
    assert vec == pytest.approx(scal, rel=1e-12)
    for fn in (interp.dw_dK, interp.d2w_dK2, interp.dw_dT):
        vec = np.asarray(fn(K_QUERY, T_QUERY))
        scal = np.array([fn(float(K), T_QUERY) for K in K_QUERY])
        assert vec == pytest.approx(scal, rel=1e-12)


# --- SSVI (Gatheral-Jacquier, φ power-law) ------------------------------------

def _ssvi_smiley_surface():
    """VolSurface non plate calibrée SSVI (smiles 'marché' + ThetaCurve depuis les ATM)."""
    disc = flat_discount(0.03)
    s = Quote(100.0)
    fwd = AnalyticForwardCurve(s, disc, dividend_yield=0.0)
    strikes = [70.0, 85.0, 100.0, 115.0, 130.0]
    maturity_dates = [in_days(d) for d in (91, 182, 365, 730)]
    smiles = [VolSmile(strikes, [0.25, 0.22, 0.20, 0.225, 0.255],
                       LogLinearInterpolator1D(), m, AbsoluteStrike())
              for m in maturity_dates]
    maturities = [DC.year_fraction(REF_DATE, m) for m in maturity_dates]
    theta_curve = ThetaCurve.from_smiles(maturities, smiles, fwd, LogLinearInterpolator1D())
    surface = VolSurface(smiles, REF_DATE, DC, SSVIInterpolator(fwd, theta_curve), forward_curve=fwd)
    surface.calibrate_interpolator()
    return surface


def _ssvi_planted_surface(rho=-0.4, eta=0.8, gamma=0.4, sigma_atm=0.2):
    """Surface SSVI sans arbitrage générée depuis des paramètres plantés (taux nul → F=S)."""
    disc = flat_discount(0.0)
    s = Quote(100.0)
    fwd = AnalyticForwardCurve(s, disc, dividend_yield=0.0)
    strikes = np.array([70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0])
    maturity_dates = [in_days(d) for d in (91, 182, 365, 730)]
    maturities = [DC.year_fraction(REF_DATE, m) for m in maturity_dates]
    theta_pillars = [sigma_atm ** 2 * T for T in maturities]  # σ_ATM cst → θ = σ²·T strict. croissant
    smiles = []
    for T, m, th in zip(maturities, maturity_dates, theta_pillars):
        k = np.log(strikes / fwd.forward(T))
        x = (eta * th ** (-gamma)) * k
        w = 0.5 * th * (1.0 + rho * x + np.sqrt((x + rho) ** 2 + (1.0 - rho ** 2)))
        smiles.append(VolSmile(strikes, np.sqrt(w / T), LogLinearInterpolator1D(), m, AbsoluteStrike()))
    theta_curve = ThetaCurve.from_smiles(maturities, smiles, fwd, LogLinearInterpolator1D())
    surface = VolSurface(smiles, REF_DATE, DC, SSVIInterpolator(fwd, theta_curve), forward_curve=fwd)
    surface.calibrate_interpolator()
    return surface, (rho, eta, gamma), theta_pillars, maturities, fwd


def test_theta_curve_rejects_non_monotone():
    interp = LogLinearInterpolator1D()
    with pytest.raises(ValueError):  # θ décroissant
        ThetaCurve([0.25, 0.5, 1.0], [0.01, 0.008, 0.02], interp)
    with pytest.raises(ValueError):  # maturités non strictement croissantes
        ThetaCurve([0.25, 0.25, 1.0], [0.01, 0.02, 0.03], interp)


def test_ssvi_vol_and_derivatives_scalar_vector_consistency():
    surface = _ssvi_smiley_surface()
    interp = surface.interpolator
    vec = np.asarray(surface.vol(K_QUERY, T_QUERY))
    scal = np.array([surface.vol(float(K), T_QUERY) for K in K_QUERY])
    assert vec == pytest.approx(scal, rel=1e-12)
    for fn in (interp.dw_dK, interp.d2w_dK2, interp.dw_dT):
        vec = np.asarray(fn(K_QUERY, T_QUERY))
        scal = np.array([fn(float(K), T_QUERY) for K in K_QUERY])
        assert vec == pytest.approx(scal, rel=1e-12)


def test_ssvi_analytic_derivatives_match_finite_difference():
    """Les dérivées analytiques de w égalent les différences finies de interpolate."""
    surface, *_ = _ssvi_planted_surface()
    interp = surface.interpolator
    w = lambda K, T: float(interp.interpolate(K, T)) ** 2 * T  # variance totale

    K, T = 100.0, 0.6        # entre piliers (0.499, 1.0), loin d'un kink
    hK, hT = 1e-3 * K, 1e-5

    dw_dK_fd = (w(K + hK, T) - w(K - hK, T)) / (2 * hK)
    d2w_dK2_fd = (w(K + hK, T) - 2 * w(K, T) + w(K - hK, T)) / hK ** 2
    dw_dT_fd = (w(K, T + hT) - w(K, T - hT)) / (2 * hT)

    assert float(interp.dw_dK(K, T)) == pytest.approx(dw_dK_fd, rel=1e-3)
    assert float(interp.d2w_dK2(K, T)) == pytest.approx(d2w_dK2_fd, rel=1e-2)
    assert float(interp.dw_dT(K, T)) == pytest.approx(dw_dT_fd, rel=1e-2, abs=1e-6)


def test_ssvi_butterfly_constraint_satisfied():
    surface = _ssvi_smiley_surface()
    rho, eta, gamma = surface.interpolator._params
    assert eta * (1.0 + abs(rho)) <= 2.0 + 1e-6


def test_ssvi_roundtrip_recovers_planted_surface():
    surface, planted, theta_pillars, maturities, fwd = _ssvi_planted_surface()
    interp = surface.interpolator
    # Aux maturités piliers, θ est exact ⇒ la variance totale reconstruite doit coller.
    for T, th in zip(maturities, theta_pillars):
        for K in (80.0, 100.0, 120.0):
            k = np.log(K / fwd.forward(T))
            rho, eta, gamma = planted
            x = (eta * th ** (-gamma)) * k
            w_true = 0.5 * th * (1.0 + rho * x + np.sqrt((x + rho) ** 2 + (1.0 - rho ** 2)))
            w_fit = float(interp.interpolate(K, T)) ** 2 * T
            assert w_fit == pytest.approx(w_true, rel=1e-4)


def test_dupire_runs_on_ssvi_surface():
    surface, _, _, _, fwd = _ssvi_planted_surface()
    model = LocalVolModel(flat_discount(0.0))
    for S in (85.0, 100.0, 115.0):
        for t in (0.3, 0.7, 1.5):
            sigma = model.sigma_loc(S, t, surface, fwd)
            assert np.isfinite(sigma) and sigma > 0
