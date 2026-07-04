"""Swap OIS : instrument de calibration de la partie longue de la courbe."""
from __future__ import annotations

from datetime import date

from JR_PRICER.instruments.base import Instrument
from JR_PRICER.instruments.leg import FixedLeg, FloatingLeg
from JR_PRICER.utils.calendar import Calendar
from JR_PRICER.utils.business_day_convention import BusinessDayConvention
from JR_PRICER.utils.schedule import Schedule
from JR_PRICER.curves.temporal.base import TemporalCurve


class OISSwap(Instrument):
    """Swap OIS (discounting OIS) utilisé pour bootstrapper la courbe de discount.

    Expose `implied_discount_factor` (formule fermée) et `npv_given_curve` (NPV
    exacte, pour le raffinement par recherche de racine).
    """

    def __init__(self, start_date: date, maturity_date: date, notional: float,
                 fixed_leg: FixedLeg, floating_leg: FloatingLeg,
                 calendar: Calendar, convention: BusinessDayConvention) -> None:
        super().__init__(maturity_date)
        self.start_date = start_date
        self.notional = notional
        self.calendar = calendar
        self.convention = convention
        self.fixed_leg = fixed_leg
        self.floating_leg = floating_leg
        # start_date ajustée au calendrier, cohérente avec les fins de période
        # (que le Schedule ajuste). Sert de borne de la 1re période d'accrual.
        self.adjusted_start_date = calendar.adjust(start_date, convention)
        self.fixed_leg_schedule = Schedule(start_date, maturity_date, fixed_leg.frequency, calendar, convention)
        self.fixed_leg_dates = self.fixed_leg_schedule.dates() # calcul dès la construction pour éviter de le faire à chaque fois dans implied_discount_factor

    def implied_discount_factor(self, curve_so_far: TemporalCurve) -> float:

        """
        Calcule le facteur d'actualisation implicite à partir du taux du swap, en utilisant les facteurs d'actualisation déjà 
        bootstrappés dans curve_so_far pour les périodes précédentes du swap.
        """

        r = self.fixed_leg.fixed_rate.value()
        
        all_dates = [self.adjusted_start_date] + self.fixed_leg_dates
        periods = list(zip(all_dates[:-1], all_dates[1:]))

        cumulative_pv = 0

        for (date_start, date_end) in periods[:-1]:  # on ne prend pas la dernière période, qui est celle que l'on bootstrappe

            tau_i = self.fixed_leg.day_count_convention.year_fraction(date_start, date_end)

            T_i   = curve_so_far.day_count_convention.year_fraction(curve_so_far.reference_date, date_end)
            P_i   = curve_so_far.evaluate(T_i)  # facteur déjà bootstrappé

            cumulative_pv += r * tau_i * P_i

        last_start, last_end = periods[-1]
        tau_n = self.fixed_leg.day_count_convention.year_fraction(last_start, last_end)

        return (1 - cumulative_pv) / (1 + r * tau_n)

    def npv_given_curve(self, curve_so_far: TemporalCurve) -> float:
        """NPV du swap (jambe fixe − jambe flottante OIS) pour une courbe donnée.

        Sert au bootstrap par recherche de racine : exacte même quand des coupons
        tombent au-delà du dernier pilier (la formule fermée extrapole alors).
        Jambe flottante OIS (OIS-discounting) : P(0,T_start) − P(0,T_end).
        Jambe fixe : r · Σ τᵢ · P(0,Tᵢ).
        """
        r = self.fixed_leg.fixed_rate.value()
        dc = curve_so_far.day_count_convention
        ref = curve_so_far.reference_date

        all_dates = [self.adjusted_start_date] + self.fixed_leg_dates
        periods = list(zip(all_dates[:-1], all_dates[1:]))

        fixed_pv = sum(
            r * self.fixed_leg.day_count_convention.year_fraction(d_s, d_e)
            * curve_so_far.evaluate(dc.year_fraction(ref, d_e))
            for d_s, d_e in periods
        )

        T_start = dc.year_fraction(ref, self.start_date)
        T_end = dc.year_fraction(ref, all_dates[-1])
        float_pv = curve_so_far.evaluate(T_start) - curve_so_far.evaluate(T_end)

        return fixed_pv - float_pv

    def __repr__(self) -> str:
        return (f"OISSwap(start={self.start_date}, maturity={self.maturity_date}, "
                f"rate={self.fixed_leg.fixed_rate.value()})")
