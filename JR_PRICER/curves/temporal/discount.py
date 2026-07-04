"""Courbes de discount factors (à terme structure complète ou plate)."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np

from JR_PRICER.curves.temporal.base import TemporalCurve
from JR_PRICER.curves.interpolators_1D.base import Interpolator1D
from JR_PRICER.utils.day_count import DayCounter


class DiscountCurve(TemporalCurve):
    """Courbe de discount factors P(0, t) bootstrappée puis interpolée.

    `discount`, `zero_rate` et `forward_rate` acceptent un temps scalaire ou un
    np.ndarray (l'interpolation sous-jacente est vectorisée).
    """

    def __init__(self, times: Sequence[float] | np.ndarray,
                 discount_factors: Sequence[float] | np.ndarray, interpolator: Interpolator1D,
                 day_count_convention: DayCounter, reference_date: date) -> None:
        super().__init__(times, discount_factors, interpolator, day_count_convention, reference_date)

    def discount(self, t: float | np.ndarray) -> float | np.ndarray:
        """Facteur d'actualisation P(0, t)."""
        return self.evaluate(t)

    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """Taux zéro continu z(t) = -ln P(0, t) / t (indéfini en t <= 0)."""
        if np.any(np.asarray(t) <= 0):
            raise ValueError("zero_rate est indéfini pour t <= 0 (pas d'accrual)")
        return -(1 / t) * np.log(self.evaluate(t))

    def forward_rate(self, t1: float, t2: float) -> float:
        """Taux forward continu entre t1 et t2 : -ln(P(t2)/P(t1)) / (t2 - t1)."""
        return -np.log(self.evaluate(t2) / self.evaluate(t1)) / (t2 - t1)


class FlatDiscountCurve(DiscountCurve):
    """Courbe de discount à taux continu constant : P(0, t) = exp(-r·t).

    Sans piliers ni interpolateur : tout est calculé en forme fermée.
    """

    def __init__(self, flat_rate: float, day_count_convention: DayCounter,
                 reference_date: date) -> None:
        super().__init__([], [], None, day_count_convention, reference_date)
        self.flat_rate = flat_rate

    def discount(self, t: float | np.ndarray) -> float | np.ndarray:
        return np.exp(-self.flat_rate * t)

    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        return self.flat_rate

    def forward_rate(self, t1: float, t2: float) -> float:
        return self.flat_rate

    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        return self.discount(x)

    def __repr__(self) -> str:
        return f"FlatDiscountCurve(rate={self.flat_rate}, ref={self.reference_date})"
