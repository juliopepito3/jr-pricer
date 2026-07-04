"""Base des dérivés sur actions (equity) et type d'option."""
from __future__ import annotations

from datetime import date
from enum import Enum

from JR_PRICER.instruments.base import Instrument
from JR_PRICER.market_data.underlying import Underlying


class OptionType(Enum):
    """Sens d'une option vanille."""
    CALL = 'call'
    PUT = 'put'


class EquityDerivative(Instrument):
    """Dérivé sur un sous-jacent action : maturité, date de départ, notionnel, underlying."""

    def __init__(self, maturity_date: date, start_date: date, notional: float,
                 underlying: Underlying) -> None:
        super().__init__(maturity_date)
        self.start_date = start_date
        self.notional = notional
        self.underlying = underlying
