"""Sentinelles de non-régression : prix MC seedés figés.

Le RNG seedé rend ces prix déterministes. La vectorisation (Phases 1–3) doit
préserver l'ordre des tirages et les valeurs de payoff → ces prix ne doivent pas
bouger (au-delà de la tolérance flottante).
"""
from __future__ import annotations

import pytest

from JR_PRICER.instruments.derivatives.equity.base import OptionType
from JR_PRICER.instruments.derivatives.equity.european_option import EuropeanOption
from JR_PRICER.instruments.derivatives.equity.digital_option import DigitalOption
from JR_PRICER.instruments.derivatives.equity.asian_option import AsianOption, AveragingType
from JR_PRICER.instruments.derivatives.equity.barrier_option import (
    BarrierOption, BarrierType, BarrierDirection,
)
from JR_PRICER.pricing.model.blackscholes import BlackScholesModel
from JR_PRICER.pricing.model.local_vol import LocalVolModel
from JR_PRICER.pricing.engine.monte_carlo import MCEngine
from JR_PRICER.utils.frequency import Frequency

from tests.helpers import REF_DATE, in_days, equity_underlying, flat_vol_surface_underlying

SEED = 123456
N = 50_000

# Valeurs capturées sur le code de référence (Phase 0).
GOLDEN_BS = {
    "euro_call": 8.858657096529068,
    "euro_put": 6.82883259038246,
    "digital": 0.3913806836801133,
    "asian_geo": 5.130716851214782,
    "asian_arith": 5.317361483032409,
    "barrier_uo": 3.9206632452636154,
}
GOLDEN_LV_EURO_CALL = 9.44648553912459  # N = 20_000


def _bs_instruments(und):
    return {
        "euro_call": EuropeanOption(und, 100.0, REF_DATE, in_days(365), OptionType.CALL),
        "euro_put": EuropeanOption(und, 100.0, REF_DATE, in_days(365), OptionType.PUT),
        "digital": DigitalOption(und, 105.0, REF_DATE, in_days(365), OptionType.CALL, digital_type="cash"),
        "asian_geo": AsianOption(und, 100.0, REF_DATE, in_days(30), in_days(365),
                                 Frequency.MONTHLY, OptionType.CALL, AveragingType.GEOMETRIC),
        "asian_arith": AsianOption(und, 100.0, REF_DATE, in_days(30), in_days(365),
                                   Frequency.MONTHLY, OptionType.CALL, AveragingType.ARITHMETIC),
        "barrier_uo": BarrierOption(und, 100.0, REF_DATE, in_days(365), Frequency.MONTHLY,
                                    OptionType.CALL, BarrierType.OUT, BarrierDirection.UP, 130.0),
    }


@pytest.mark.parametrize("name", list(GOLDEN_BS))
def test_black_scholes_mc_golden(name):
    und, disc = equity_underlying(spot=100.0, sigma=0.2, r=0.03, q=0.01)
    inst = _bs_instruments(und)[name]
    price = MCEngine(N, seed=SEED).price([inst], BlackScholesModel(disc))[0]
    assert price == pytest.approx(GOLDEN_BS[name], rel=1e-10)


def test_local_vol_mc_golden():
    und, disc = flat_vol_surface_underlying(spot=100.0, sigma=0.2, r=0.03, q=0.0)
    opt = EuropeanOption(und, 100.0, REF_DATE, in_days(365), OptionType.CALL)
    price = MCEngine(20_000, seed=SEED).price([opt], LocalVolModel(disc))[0]
    assert price == pytest.approx(GOLDEN_LV_EURO_CALL, rel=1e-6)
