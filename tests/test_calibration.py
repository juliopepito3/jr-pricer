"""Non-régression du calcul d'implied vol (pièce refactorée en Phase 4)."""
from __future__ import annotations

import numpy as np
import pytest

from JR_PRICER.market_data.quote import Quote
from JR_PRICER.market_data.underlying import Underlying
from JR_PRICER.curves.forward.analytic_forward import AnalyticForwardCurve
from JR_PRICER.surfaces.vol_surface.volsurface import FlatVol
from JR_PRICER.instruments.derivatives.equity.base import OptionType
from JR_PRICER.instruments.derivatives.equity.european_option import EuropeanOption
from JR_PRICER.pricing.model.blackscholes import BlackScholesModel
from JR_PRICER.pricing.engine.analytical import AnalyticalEngine
from JR_PRICER.pricing.implied_vol_calculator import (
    ImpliedVolCalculator,
)

from tests.helpers import REF_DATE, in_days, flat_discount


def _options(disc, sigma, strikes):
    s = Quote(100.0)
    fwd = AnalyticForwardCurve(s, disc, dividend_yield=0.0)
    und = Underlying("EQ", s, fwd, FlatVol(sigma))
    return [
        EuropeanOption(und, K=K, start_date=REF_DATE, maturity_date=in_days(365),
                       option_type=OptionType.CALL)
        for K in strikes
    ]


def test_implied_vol_recovers_flat_vol():
    disc = flat_discount(0.03)
    sigma = 0.27
    strikes = [80.0, 100.0, 120.0]
    opts = _options(disc, sigma, strikes)

    prices = AnalyticalEngine().price(opts, BlackScholesModel(disc))
    ivs = ImpliedVolCalculator(disc).calculate_implied_vol(opts, prices)

    assert np.asarray(ivs) == pytest.approx([sigma] * len(strikes), rel=1e-4)


def test_implied_vol_monotone_in_price():
    """Un prix plus élevé (toutes choses égales) implique une vol implicite plus élevée."""
    disc = flat_discount(0.03)
    opts = _options(disc, 0.2, [100.0])
    base_price = AnalyticalEngine().price(opts, BlackScholesModel(disc))[0]
    calc = ImpliedVolCalculator(disc)
    iv_low = calc.calculate_implied_vol(opts, [base_price * 0.9])[0]
    iv_high = calc.calculate_implied_vol(opts, [base_price * 1.1])[0]
    assert iv_low < iv_high


def test_implied_vol_batch_puts_recovers_vol():
    disc = flat_discount(0.02)
    sigma = 0.31
    s = Quote(100.0)
    fwd = AnalyticForwardCurve(s, disc, dividend_yield=0.0)
    und = Underlying("EQ", s, fwd, FlatVol(sigma))
    opts = [
        EuropeanOption(und, K=K, start_date=REF_DATE, maturity_date=in_days(d),
                       option_type=OptionType.PUT)
        for K, d in [(80.0, 180), (100.0, 365), (130.0, 730)]
    ]
    prices = AnalyticalEngine().price(opts, BlackScholesModel(disc))
    ivs = ImpliedVolCalculator(disc).calculate_implied_vol(opts, prices)
    assert np.asarray(ivs) == pytest.approx([sigma, sigma, sigma], rel=1e-5)
