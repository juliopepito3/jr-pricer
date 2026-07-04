"""Jambes d'un swap : jambe fixe et jambe flottante."""
from __future__ import annotations

from JR_PRICER.market_data.quote import Quote
from JR_PRICER.utils.day_count import DayCounter
from JR_PRICER.utils.frequency import Frequency


class Leg:
    """Jambe générique d'un swap : convention de day count et fréquence de paiement."""

    def __init__(self, day_count_convention: DayCounter, frequency: Frequency) -> None:
        self.day_count_convention = day_count_convention
        self.frequency = frequency

    def __repr__(self) -> str:
        return f"{type(self).__name__}(frequency={self.frequency.name})"


class FixedLeg(Leg):
    """Jambe fixe : verse un taux fixe `fixed_rate`."""

    def __init__(self, day_count_convention: DayCounter, frequency: Frequency,
                 fixed_rate: Quote) -> None:
        super().__init__(day_count_convention, frequency)
        self.fixed_rate = fixed_rate

    def __repr__(self) -> str:
        return f"FixedLeg(rate={self.fixed_rate.value()}, frequency={self.frequency.name})"


class FloatingLeg(Leg):
    """Jambe flottante : indexée sur un taux de marché `index_quote`."""

    def __init__(self, day_count_convention: DayCounter, frequency: Frequency,
                 index_quote: Quote) -> None:
        super().__init__(day_count_convention, frequency)
        self.index_quote = index_quote

    def __repr__(self) -> str:
        return f"FloatingLeg(index={self.index_quote.value()}, frequency={self.frequency.name})"
