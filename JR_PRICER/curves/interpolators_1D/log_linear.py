"""Interpolateur 1D log-linéaire (linéaire sur ln(y))."""
from __future__ import annotations

from collections.abc import Sequence

from JR_PRICER.curves.interpolators_1D.base import Interpolator1D
import numpy as np


class LogLinearInterpolator1D(Interpolator1D):
    """Interpolateur log-linéaire : interpolation affine de ln(y) en x.

    Adapté aux discount factors (y > 0) : monotone et stable, avec extrapolation
    plate du taux forward de bord. Requiert des ordonnées strictement positives.
    """

    def __init__(self) -> None:
        super().__init__()

    def _fit(self, x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> None:
        """Mémorise les piliers (aucun paramètre à ajuster en log-linéaire)."""
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def interpolate(self, t: float | np.ndarray) -> float | np.ndarray:
        """Interpolation log-linéaire, scalaire OU vectorisée (np.ndarray).

        Hors du domaine calibré [x[0], x[-1]], l'extrapolation prolonge le segment
        de bord en log-linéaire : cela revient à maintenir constant le taux forward
        du dernier (resp. premier) segment. Pour une courbe de discount factors,
        c'est un comportement stable et monotone (contrairement aux splines)."""

        if len(self.x) == 0:
            raise ValueError("La courbe est vide : impossible d'interpoler.")

        scalar = np.ndim(t) == 0
        t_arr = np.asarray(t, dtype=float)

        if len(self.x) == 1:
            res = np.full(t_arr.shape, self.y[0], dtype=float)
            return float(res) if scalar else res

        # Indices de bracket [x[idx], x[idx+1]] avec clamp pour l'extrapolation plate (en log).
        idx = np.clip(np.searchsorted(self.x, t_arr, side="right") - 1, 0, len(self.x) - 2)

        x0, x1 = self.x[idx], self.x[idx + 1]
        y0, y1 = self.y[idx], self.y[idx + 1]

        if np.any(y0 <= 0) or np.any(y1 <= 0):
            raise ValueError("Les valeurs d'ordonnée doivent être strictement positives pour une interpolation log-linéaire.")

        log_y0, log_y1 = np.log(y0), np.log(y1)
        log_y = log_y0 + (log_y1 - log_y0) * (t_arr - x0) / (x1 - x0)
        res = np.exp(log_y)

        return float(res) if scalar else res
