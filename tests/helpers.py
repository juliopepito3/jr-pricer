"""Constructeurs partagés pour la suite de tests (filet de non-régression).

Les fixtures s'appuient sur des courbes plates et des surfaces simples afin que
les invariants testés (prix MC vs analytique, vol locale d'une surface plate,
implied vol) soient analytiquement vérifiables et robustes au refactor NumPy.
"""
from __future__ import annotations

from datetime import date, timedelta

from JR_PRICER.utils.day_count import DayCounter
from JR_PRICER.utils.frequency import Frequency
from JR_PRICER.utils.calendar import TARGET
from JR_PRICER.utils.business_day_convention import BusinessDayConvention

from JR_PRICER.market_data.quote import Quote
from JR_PRICER.market_data.underlying import Underlying

from JR_PRICER.curves.temporal.discount import FlatDiscountCurve
from JR_PRICER.curves.forward.analytic_forward import AnalyticForwardCurve
from JR_PRICER.curves.vol_smile.volsmile import VolSmile
from JR_PRICER.curves.vol_smile.moneyness_convention import AbsoluteStrike
from JR_PRICER.curves.interpolators_1D.log_linear import LogLinearInterpolator1D

from JR_PRICER.surfaces.vol_surface.volsurface import FlatVol, VolSurface
from JR_PRICER.surfaces.vol_surface.interpolators_2D.bi_cubic import BiCubicInterpolator

REF_DATE = date(2024, 1, 15)
DC = DayCounter("act/365")
CAL = TARGET()
BDC = BusinessDayConvention.MODIFIED_FOLLOWING


def in_days(n: int) -> date:
    """Date à n jours calendaires de la date de référence (Act/365 → n/365 ans)."""
    return REF_DATE + timedelta(days=n)


def flat_discount(r: float = 0.03) -> FlatDiscountCurve:
    return FlatDiscountCurve(r, DC, REF_DATE)


def equity_underlying(spot: float = 100.0, sigma: float = 0.2,
                      r: float = 0.03, q: float = 0.0, name: str = "EQ"):
    """Underlying actions avec FlatVol (pour BlackScholesModel). Retourne (underlying, discount)."""
    s = Quote(spot, name)
    disc = flat_discount(r)
    fwd = AnalyticForwardCurve(s, disc, dividend_yield=q)
    und = Underlying(name, s, fwd, FlatVol(sigma))
    return und, disc


def flat_vol_surface_underlying(spot: float = 100.0, sigma: float = 0.2,
                                r: float = 0.03, q: float = 0.0, name: str = "EQ"):
    """Underlying avec une VolSurface PLATE (σ constant en K et T), convention AbsoluteStrike.

    Dupire sur une surface plate redonne σ_loc = σ : la vol locale doit coïncider
    avec Black-Scholes. Retourne (underlying, discount).
    """
    s = Quote(spot, name)
    disc = flat_discount(r)
    fwd = AnalyticForwardCurve(s, disc, dividend_yield=q)

    strikes = [60.0, 80.0, 100.0, 120.0, 140.0]
    maturity_dates = [in_days(d) for d in (91, 182, 365, 730)]
    smiles = [
        VolSmile(strikes, [sigma] * len(strikes), LogLinearInterpolator1D(),
                 m, AbsoluteStrike())
        for m in maturity_dates
    ]
    surface = VolSurface(smiles, REF_DATE, DC, BiCubicInterpolator(), forward_curve=fwd)
    surface.calibrate_interpolator()

    und = Underlying(name, s, fwd, surface)
    return und, disc
