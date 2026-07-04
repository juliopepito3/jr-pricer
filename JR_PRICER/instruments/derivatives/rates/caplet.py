"""Caplet / floorlet : option sur un taux forward d'une période d'accumulation."""
from __future__ import annotations

from datetime import date

from JR_PRICER.instruments.derivatives.rates.base import RateDerivative
from JR_PRICER.utils.day_count import DayCounter


class Caplet(RateDerivative):
    """Option sur le taux forward d'une seule période [start_date, end_date].

    `option_type` vaut 'cap' (caplet) ou 'floor' (floorlet).
    """

    def __init__(self, start_date: date, end_date: date,
                 strike: float, notional: float,
                 day_count_convention: DayCounter,
                 option_type: str = 'cap') -> None:
        if option_type not in ('cap', 'floor'):
            raise ValueError("option_type doit être 'cap' ou 'floor'")
        super().__init__(start_date, end_date, strike, notional, day_count_convention)
        self.end_date = end_date
        self.option_type = option_type

    @property
    def accrual(self) -> float:
        """Fraction d'année de la période (τ), selon la convention de day count."""
        return self.day_count_convention.year_fraction(self.start_date, self.end_date)

    def __repr__(self) -> str:
        return (f"Caplet(start={self.start_date}, end={self.end_date}, "
                f"strike={self.strike}, type={self.option_type})")
