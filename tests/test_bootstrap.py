"""Non-régression du bootstrap de courbe (dépôts + swaps OIS au pair)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from JR_PRICER.instruments.deposit import Deposit
from JR_PRICER.instruments.swap import OISSwap
from JR_PRICER.instruments.leg import FixedLeg, FloatingLeg
from JR_PRICER.market_data.quote import Quote
from JR_PRICER.curves.bootstrap import BootstrapBuilder
from JR_PRICER.curves.interpolators_1D.log_linear import LogLinearInterpolator1D
from JR_PRICER.utils.day_count import DayCounter
from JR_PRICER.utils.frequency import Frequency
from JR_PRICER.utils.calendar import TARGET
from JR_PRICER.utils.business_day_convention import BusinessDayConvention

REF = date(2024, 1, 15)
DC = DayCounter("act/360")
CAL = TARGET()
BDC = BusinessDayConvention.MODIFIED_FOLLOWING


def _make_ois(rate, maturity):
    q = Quote(rate)
    return OISSwap(
        start_date=REF, maturity_date=maturity, notional=1_000_000,
        fixed_leg=FixedLeg(DC, Frequency.ANNUAL, q),
        floating_leg=FloatingLeg(DC, Frequency.ANNUAL, q),
        calendar=CAL, convention=BDC,
    )


@pytest.fixture
def market():
    deposits = [
        Deposit(Quote(0.03904), date(2024, 1, 16), DC, REF),
        Deposit(Quote(0.03900), date(2024, 1, 22), DC, REF),
        Deposit(Quote(0.03880), date(2024, 2, 15), DC, REF),
        Deposit(Quote(0.03850), date(2024, 4, 15), DC, REF),
        Deposit(Quote(0.03780), date(2024, 7, 15), DC, REF),
    ]
    swaps = [
        _make_ois(0.03600, date(2025, 1, 15)),
        _make_ois(0.03300, date(2026, 1, 15)),
        _make_ois(0.03050, date(2027, 1, 15)),
        _make_ois(0.02850, date(2029, 1, 15)),
        _make_ois(0.02750, date(2031, 1, 15)),
        _make_ois(0.02700, date(2034, 1, 15)),
    ]
    return deposits, swaps


@pytest.fixture
def curve(market):
    deposits, swaps = market
    return BootstrapBuilder(
        instruments=deposits + swaps,
        interpolator=LogLinearInterpolator1D(),
        day_count_convention=DC, reference_date=REF,
    ).bootstrap()


def test_anchor_and_monotonic(curve):
    assert curve.discount(0.0) == pytest.approx(1.0, abs=1e-12)
    ts = np.linspace(0.05, 10.0, 50)
    dfs = np.array([curve.discount(t) for t in ts])
    assert np.all(np.diff(dfs) < 0)  # facteurs d'actualisation strictement décroissants
    assert np.all(dfs > 0)


def test_par_swaps_reprice(market, curve):
    # Invariant exact résolu par le bootstrap : NPV(swap) = 0 sur la courbe finale.
    _, swaps = market
    for swap in swaps:
        assert swap.npv_given_curve(curve) == pytest.approx(0.0, abs=1e-6)
