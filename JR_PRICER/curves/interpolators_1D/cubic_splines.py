"""Interpolateur 1D par splines cubiques (wrapper scipy.interpolate.CubicSpline)."""
from __future__ import annotations

from collections.abc import Sequence

from JR_PRICER.curves.interpolators_1D.base import Interpolator1D
from scipy.interpolate import CubicSpline
import numpy as np


class ScipyCubicSplinesInterpolator1D(Interpolator1D):
    """Interpolateur par splines cubiques naturelles (via scipy CubicSpline)."""

    def __init__(self) -> None:
        super().__init__()

    def _fit(self, x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> None:
        """Construit la spline cubique naturelle sur les piliers (x, y)."""
        self.x = x
        self.y = y
        self.cs = CubicSpline(self.x, self.y, bc_type='natural')

    def interpolate(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluation de la spline cubique au point t.

        Avertissement : la spline naturelle interpole directement les discount
        factors. Hors du domaine calibré, l'extrapolation polynomiale peut diverger
        et, même dans le domaine, les taux forward instantanés (dérivés) peuvent
        osciller. Pour une courbe robuste, préférer LogLinearInterpolator1D ;
        un interpolateur spline-sur-log(DF) pourra être ajouté ultérieurement.

        Accepte un scalaire ou un np.ndarray (CubicSpline est vecteur-capable).
        """
        res = self.cs(t)
        return float(res) if np.ndim(t) == 0 else res