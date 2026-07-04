"""Courbe forward interpolée à partir de prix de futures cotés."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from JR_PRICER.curves.forward.base import ForwardCurve
from JR_PRICER.curves.base import Curve
from JR_PRICER.curves.interpolators_1D.base import Interpolator1D
from JR_PRICER.market_data.quote import Quote


class MarketForwardCurve(ForwardCurve, Curve):
    """Courbe forward construite depuis des prix de futures observés.

    Les forwards F(T_i) sont interpolés entre les maturités cotées. Le spot
    (`Quote`) sert de valeur en T=0 et peut être mis à jour en live.
    """

    def __init__(self, spot: Quote, times: Sequence[float] | np.ndarray,
                 forwards: Sequence[float] | np.ndarray, interpolator: Interpolator1D) -> None:
        if not isinstance(spot, Quote):
            raise TypeError("spot must be a Quote instance")
        if len(times) != len(forwards):
            raise ValueError("times and forwards must have the same length")
        if np.any(np.asarray(forwards, dtype=float) <= 0):
            raise ValueError("all forward prices must be positive")

        self._spot = spot
        Curve.__init__(self, times, forwards, interpolator)

    @property
    def spot(self) -> float:
        return self._spot.value()

    def forward(self, T: float) -> float:
        """Prix forward à la maturité T (year fraction) ; renvoie le spot en T=0."""
        if T == 0:
            return self._spot.value()
        return self.evaluate(T)

    def __repr__(self) -> str:
        return f"MarketForwardCurve(spot={self.spot}, n_pillars={len(self.x)})"



