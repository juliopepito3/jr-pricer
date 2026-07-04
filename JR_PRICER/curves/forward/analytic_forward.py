"""Courbe forward analytique : F(T) = S·exp(-q·T) / df(T)."""
from __future__ import annotations

import numpy as np

from JR_PRICER.curves.forward.base import ForwardCurve
from JR_PRICER.curves.temporal.discount import DiscountCurve
from JR_PRICER.market_data.quote import Quote


class AnalyticForwardCurve(ForwardCurve):
    """Courbe forward calculée depuis le spot : F(T) = S·exp(-q·T) / df(T).

    Le spot est un `Quote` — toute mise à jour via `spot.update()` se propage
    immédiatement sans recréer la courbe. Le dividende continu q est constant ;
    la `DiscountCurve` peut être plate ou à terme structure complète.
    """

    def __init__(self, spot: Quote, discount_curve: DiscountCurve,
                 dividend_yield: float = 0.0) -> None:
        if not isinstance(spot, Quote):
            raise TypeError("spot must be a Quote instance")
        if dividend_yield < 0:
            raise ValueError("dividend_yield must be non-negative")

        self._spot = spot
        self.discount_curve = discount_curve
        self.dividend_yield = dividend_yield

    @property
    def spot(self) -> float:
        return self._spot.value()

    def forward(self, T: float) -> float:
        """Prix forward à la maturité T (year fraction) ; renvoie le spot en T=0."""
        if T < 0:
            raise ValueError("T must be non-negative")
        if T == 0:
            return self._spot.value()
        return self._spot.value() * np.exp(-self.dividend_yield * T) / self.discount_curve.discount(T)

    def __repr__(self) -> str:
        return f"AnalyticForwardCurve(spot={self.spot}, q={self.dividend_yield})"
