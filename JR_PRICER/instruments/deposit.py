"""Dépôt monétaire : instrument de calibration de la partie courte de la courbe."""
from __future__ import annotations

from datetime import date

from JR_PRICER.instruments.base import Instrument
from JR_PRICER.market_data.quote import Quote
from JR_PRICER.utils.day_count import DayCounter
from JR_PRICER.curves.temporal.base import TemporalCurve


class Deposit(Instrument):
    """Dépôt à taux simple sur [settlement_date, maturity_date].

    Sert au bootstrap : `implied_discount_factor` donne P(0, maturity) en forme fermée.
    """

    def __init__(self, quote: Quote, maturity_date: date, day_count_convention: DayCounter,
                 settlement_date: date, notional: float = 1.0) -> None:
        super().__init__(maturity_date)
        self.quote = quote
        self.day_count_convention = day_count_convention
        self.notional = notional
        self.settlement_date = settlement_date

    def implied_discount_factor(self, curve_so_far: TemporalCurve | None = None) -> float:
        """Discount factor P(0, maturity) impliqué par le taux de dépôt.

        Si le dépôt est forward (settlement > reference_date), on enchaîne le stub
        déjà bootstrappé : P(0, mat) = P(0, settle) · P(settle, mat).
        """
        t_deposit = self.day_count_convention.year_fraction(self.settlement_date, self.maturity_date)
        df_settle_to_mat = 1 / (1 + self.quote.value() * t_deposit)

        if curve_so_far is None or self.settlement_date == curve_so_far.reference_date:
            return df_settle_to_mat

        # P(0, maturity) = P(0, settlement) × P(settlement, maturity)
        t_settle = curve_so_far.day_count_convention.year_fraction(
            curve_so_far.reference_date, self.settlement_date
        )
        df_stub = curve_so_far.evaluate(t_settle)
        return df_stub * df_settle_to_mat

    def __repr__(self) -> str:
        return f"Deposit(rate={self.quote.value()}, maturity={self.maturity_date})"
    



