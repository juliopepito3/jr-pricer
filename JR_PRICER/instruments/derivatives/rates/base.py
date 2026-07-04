"""Base des dérivés de taux (caplets, caps/floors)."""
from __future__ import annotations

from abc import ABC
from datetime import date

from JR_PRICER.instruments.base import Instrument
from JR_PRICER.utils.day_count import DayCounter


class RateDerivative(Instrument, ABC):
    """Dérivé de taux : période [start_date, maturity_date], strike et notionnel."""

    def __init__(self, start_date: date, maturity_date: date,
                 strike: float, notional: float,
                 day_count_convention: DayCounter) -> None:
        super().__init__(maturity_date)
        self.start_date = start_date
        self.strike = strike
        self.notional = notional
        self.day_count_convention = day_count_convention
