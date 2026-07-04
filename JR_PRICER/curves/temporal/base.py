"""Courbe temporelle : courbe 1D indexée par le temps (year fraction)."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np

from JR_PRICER.curves.base import Curve
from JR_PRICER.curves.interpolators_1D.base import Interpolator1D
from JR_PRICER.utils.day_count import DayCounter


class TemporalCurve(Curve):
    """Courbe 1D dont l'abscisse est un temps en year fraction depuis `reference_date`.

    Ajoute la convention de day count et la date de référence à la `Curve` de base,
    et expose les alias lisibles `times` / `values`.
    """

    def __init__(self, times: Sequence[float] | np.ndarray, values: Sequence[float] | np.ndarray,
                 interpolator: Interpolator1D, day_count_convention: DayCounter,
                 reference_date: date) -> None:
        super().__init__(times, values, interpolator)
        self.day_count_convention = day_count_convention
        self.reference_date = reference_date

    @property
    def times(self) -> np.ndarray:
        """Maturités des piliers (year fraction). Alias en lecture seule de `x`."""
        return self.x

    @property
    def values(self) -> np.ndarray:
        """Valeurs des piliers (ex. discount factors). Alias en lecture seule de `y`."""
        return self.y

    def date_to_time(self, d: date) -> float:
        """Convertit une date en year fraction depuis `reference_date`."""
        return self.day_count_convention.year_fraction(self.reference_date, d)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_pillars={len(self.x)}, ref={self.reference_date})"
