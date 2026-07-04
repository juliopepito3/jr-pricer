"""Correctness : Monte Carlo (seedé) vs prix analytiques.

Ces tests sont robustes au refactor : ils comparent à une référence analytique,
pas à des valeurs figées. Le seed fixe rend les écarts MC déterministes.
"""
from __future__ import annotations

import pytest

from JR_PRICER.instruments.derivatives.equity.base import OptionType
from JR_PRICER.instruments.derivatives.equity.european_option import EuropeanOption
from JR_PRICER.instruments.derivatives.equity.digital_option import DigitalOption
from JR_PRICER.instruments.derivatives.equity.asian_option import AsianOption, AveragingType
from JR_PRICER.pricing.model.blackscholes import BlackScholesModel
from JR_PRICER.pricing.model.local_vol import LocalVolModel
from JR_PRICER.pricing.engine.analytical import AnalyticalEngine
from JR_PRICER.pricing.engine.monte_carlo import MCEngine
from JR_PRICER.utils.frequency import Frequency

from tests.helpers import (
    REF_DATE, in_days, equity_underlying, flat_vol_surface_underlying,
)

N_PATHS = 100_000
SEED = 20240115


@pytest.mark.parametrize("opt_type", [OptionType.CALL, OptionType.PUT])
def test_european_mc_matches_analytic(opt_type):
    und, disc = equity_underlying(spot=100.0, sigma=0.2, r=0.03, q=0.01)
    opt = EuropeanOption(und, K=100.0, start_date=REF_DATE,
                         maturity_date=in_days(365), option_type=opt_type)
    model = BlackScholesModel(disc)
    analytic = AnalyticalEngine().price([opt], model)[0]
    mc = MCEngine(N_PATHS, seed=SEED).price([opt], model)[0]
    assert mc == pytest.approx(analytic, rel=0.01)


def test_digital_mc_matches_analytic():
    und, disc = equity_underlying(spot=100.0, sigma=0.25, r=0.03, q=0.0)
    opt = DigitalOption(und, K=105.0, start_date=REF_DATE,
                        maturity_date=in_days(365), option_type=OptionType.CALL,
                        digital_type="cash")
    model = BlackScholesModel(disc)
    analytic = AnalyticalEngine().price([opt], model)[0]
    mc = MCEngine(N_PATHS, seed=SEED).price([opt], model)[0]
    assert mc == pytest.approx(analytic, rel=0.02)


def test_asian_geometric_mc_matches_analytic():
    und, disc = equity_underlying(spot=100.0, sigma=0.2, r=0.03, q=0.0)
    opt = AsianOption(
        und, K=100.0, start_date=REF_DATE, averaging_start=in_days(30),
        maturity_date=in_days(365), frequency=Frequency.MONTHLY,
        option_type=OptionType.CALL, averaging_type=AveragingType.GEOMETRIC,
    )
    model = BlackScholesModel(disc)
    analytic = AnalyticalEngine().price([opt], model)[0]
    mc = MCEngine(N_PATHS, seed=SEED).price([opt], model)[0]
    assert mc == pytest.approx(analytic, rel=0.02)


def test_local_vol_flat_surface_matches_black_scholes():
    """Vol locale sur surface plate = Black-Scholes (Dupire dégénère vers σ plate).

    À seed et σ identiques, les trajectoires LV et BS sont identiques au bruit de
    différences finies près → on compare directement les deux MC (rapide, robuste),
    et on vérifie que le MC BS converge vers l'analytique.
    """
    # moins de chemins ici : la boucle scalaire de la vol locale est lente avant
    # la vectorisation (Phase 3) ; la comparaison LV vs BS est exacte à tout N.
    n_paths = 30_000
    und_lv, disc = flat_vol_surface_underlying(spot=100.0, sigma=0.2, r=0.03, q=0.0)
    und_bs, _ = equity_underlying(spot=100.0, sigma=0.2, r=0.03, q=0.0)

    opt_lv = EuropeanOption(und_lv, K=100.0, start_date=REF_DATE,
                            maturity_date=in_days(365), option_type=OptionType.CALL)
    opt_bs = EuropeanOption(und_bs, K=100.0, start_date=REF_DATE,
                            maturity_date=in_days(365), option_type=OptionType.CALL)

    bs_analytic = AnalyticalEngine().price([opt_bs], BlackScholesModel(disc))[0]
    bs_mc = MCEngine(n_paths, seed=SEED).price([opt_bs], BlackScholesModel(disc))[0]
    lv_mc = MCEngine(n_paths, seed=SEED).price([opt_lv], LocalVolModel(disc))[0]

    assert lv_mc == pytest.approx(bs_mc, rel=1e-6)       # trajectoires identiques
    assert bs_mc == pytest.approx(bs_analytic, rel=0.02)  # MC converge vers l'analytique
