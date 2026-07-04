"""Non-régression Heston : constructeur, CF vectorisée (golden) et pricing Fourier."""
from __future__ import annotations

import numpy as np
import pytest

from JR_PRICER.pricing.model.heston import HestonModel
from JR_PRICER.pricing.model.blackscholes import BlackScholesModel
from JR_PRICER.pricing.engine.fourier_carr_madan import FourierCarrMadanEngine
from JR_PRICER.pricing.engine.analytical import AnalyticalEngine
from JR_PRICER.pricing.engine.monte_carlo import MCEngine
from JR_PRICER.pricing.model.discretization.CIR.base import CIRDiscretizationSchemeEuler
from JR_PRICER.market_data.quote import Quote
from JR_PRICER.market_data.underlying import Underlying
from JR_PRICER.curves.forward.analytic_forward import AnalyticForwardCurve
from JR_PRICER.surfaces.vol_surface.volsurface import FlatVol
from JR_PRICER.instruments.derivatives.equity.base import OptionType
from JR_PRICER.instruments.derivatives.equity.european_option import EuropeanOption

from tests.helpers import flat_discount, equity_underlying, REF_DATE, in_days

# Paramètres figés pour la CF golden.
PARAMS = (2.0, 0.04, 0.5, -0.7, 0.04)  # kappa, theta, sigma_v, rho, v0
U = np.linspace(0.0, 20.0, 33)

# Sommes de CF capturées sur la version d'origine (boucle continuous_log).
# Garantissent que la version vectorisée (np.unwrap) est identique.
GOLDEN = {
    ("real", 1.0): 12.17927580861426 + 3.4495472847438777j,
    ("cplx", 1.0): 12.463220224624664 + 7.718954519891621j,
    ("cplx", 2.0): 8.379375798530864 + 6.107419057453985j,
}


def _model():
    return HestonModel(*PARAMS, discount_curve=flat_discount(0.03))


def test_constructor_and_discount():
    disc = flat_discount(0.03)
    m = HestonModel(*PARAMS, discount_curve=disc)
    assert m.kappa == PARAMS[0] and m.v0 == PARAMS[4]
    assert m.discount(1.0) == pytest.approx(disc.discount(1.0))
    # contrat de calibration générique : model_class(*theta, discount_curve=...)
    theta = np.array(PARAMS)
    m2 = HestonModel(*theta, discount_curve=disc)
    assert m2.sigma_v == PARAMS[2]


def test_cf_at_zero_is_one():
    m = _model()
    assert m.characteristic_function(np.array([0.0]), 1.0)[0] == pytest.approx(1.0)


@pytest.mark.parametrize("kind,T", list(GOLDEN))
def test_cf_vectorized_matches_golden(kind, T):
    m = _model()
    grid = U if kind == "real" else U - (1.5 + 1) * 1j
    cf_sum = m.characteristic_function(grid, T).sum()
    assert cf_sum == pytest.approx(GOLDEN[(kind, T)], rel=1e-12)


def _bs_limit_setup():
    disc = flat_discount(0.03)
    sigma = 0.2
    s = Quote(100.0)
    fwd = AnalyticForwardCurve(s, disc, dividend_yield=0.0)
    und = Underlying("EQ", s, fwd, FlatVol(sigma))
    opts = [EuropeanOption(und, K, REF_DATE, in_days(365), OptionType.CALL)
            for K in (90.0, 100.0, 110.0)]
    return disc, sigma, opts


def test_fourier_reduces_to_black_scholes():
    """σ_v → 0 avec v0 = theta = σ² : Heston dégénère vers Black-Scholes(σ)."""
    disc, sigma, opts = _bs_limit_setup()
    heston = HestonModel(1.0, sigma**2, 1e-4, 0.0, sigma**2, discount_curve=disc)
    bs = AnalyticalEngine().price(opts, BlackScholesModel(disc))
    fourier = FourierCarrMadanEngine(alpha=1.5, n=12, eta=0.25).price(opts, heston)
    assert np.asarray(fourier) == pytest.approx(np.asarray(bs), rel=1e-3)


def test_fourier_price_is_grid_stable():
    """Un Carr-Madan correct converge quand la grille se raffine (ne s'effondre pas)."""
    disc, sigma, opts = _bs_limit_setup()
    heston = HestonModel(1.0, sigma**2, 1e-4, 0.0, sigma**2, discount_curve=disc)
    coarse = FourierCarrMadanEngine(alpha=1.5, n=12, eta=0.25).price(opts, heston)
    fine = FourierCarrMadanEngine(alpha=1.5, n=14, eta=0.1).price(opts, heston)
    assert np.asarray(fine) == pytest.approx(np.asarray(coarse), rel=1e-4)


def test_realistic_heston_call_prices_monotone():
    disc, _, opts = _bs_limit_setup()
    heston = HestonModel(2.0, 0.04, 0.5, -0.7, 0.04, discount_curve=disc)
    px = FourierCarrMadanEngine(alpha=1.5, n=14, eta=0.1).price(opts, heston)
    assert all(p > 0 for p in px)
    assert px[0] > px[1] > px[2]  # call décroissant en strike


# --- Monte Carlo (simulate) : seedé, comparé à des références analytiques --------------
SEED = 20240115
N_PATHS = 100_000
MAX_DT = 1.0 / 100  # pas fin : limite le biais de discrétisation Euler full-truncation


def _heston_disc(kappa, theta, sigma_v, rho, v0, disc):
    """HestonModel discrétisé (Euler full-truncation), prêt pour simulate()."""
    m = HestonModel(kappa, theta, sigma_v, rho, v0, discount_curve=disc)
    return m.discretize(CIRDiscretizationSchemeEuler(), MAX_DT)


def test_heston_mc_reduces_to_black_scholes():
    """σ_v → 0, ρ = 0, v0 = theta = σ² : la variance est ~constante → MC Heston = BS."""
    sigma = 0.2
    und, disc = equity_underlying(spot=100.0, sigma=sigma, r=0.03, q=0.0)
    opt = EuropeanOption(und, K=100.0, start_date=REF_DATE,
                         maturity_date=in_days(365), option_type=OptionType.CALL)
    heston = _heston_disc(1.0, sigma**2, 1e-4, 0.0, sigma**2, disc)
    bs = AnalyticalEngine().price([opt], BlackScholesModel(disc))[0]
    mc = MCEngine(N_PATHS, seed=SEED).price([opt], heston)[0]
    assert mc == pytest.approx(bs, rel=0.02)


def test_heston_mc_matches_fourier():
    """Params réalistes : le MC d'un call ATM converge vers le prix Carr-Madan."""
    und, disc = equity_underlying(spot=100.0, sigma=0.2, r=0.03, q=0.0)
    opt = EuropeanOption(und, K=100.0, start_date=REF_DATE,
                         maturity_date=in_days(365), option_type=OptionType.CALL)
    heston = _heston_disc(*PARAMS, disc)
    fourier = FourierCarrMadanEngine(alpha=1.5, n=14, eta=0.1).price([opt], heston)[0]
    mc = MCEngine(N_PATHS, seed=SEED).price([opt], heston)[0]
    assert mc == pytest.approx(fourier, rel=0.03)


def test_heston_mc_forward_is_martingale():
    """Call K=0 → payoff S_T : prix actualisé = df·E[S_T] = spot (q=0). Pas de biais de drift."""
    spot = 100.0
    und, disc = equity_underlying(spot=spot, sigma=0.2, r=0.03, q=0.0)
    opt = EuropeanOption(und, K=0.0, start_date=REF_DATE,
                         maturity_date=in_days(365), option_type=OptionType.CALL)
    heston = _heston_disc(*PARAMS, disc)
    mc = MCEngine(N_PATHS, seed=SEED).price([opt], heston)[0]
    assert mc == pytest.approx(spot, rel=0.01)
