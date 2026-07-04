"""Interpolateur 2D global par fonctions de base radiales (wrapper scipy RBFInterpolator)."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator

from JR_PRICER.surfaces.vol_surface.interpolators_2D.base import GlobalInterpolator2D


class RBFInterpolator2D(GlobalInterpolator2D):
    """Interpolation RBF de la variance totale w(K, T), avec clamp hors domaine."""

    def __init__(self, kernel: str = 'multiquadric', epsilon: float = 1.0) -> None:
        super().__init__()
        self.kernel = kernel
        self.epsilon = epsilon

    def _fit(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        KK, TT = np.meshgrid(x, y, indexing='ij')
        points = np.column_stack([KK.ravel(), TT.ravel()])
        values = z.ravel()
        self.rbf = RBFInterpolator(points, values, kernel=self.kernel, epsilon=self.epsilon)
        # Bornes pour clamper les requêtes (extrapolation plate) : hors domaine,
        # une RBF peut osciller et produire une variance totale négative.
        self._x_min, self._x_max = float(x[0]), float(x[-1])
        self._y_min, self._y_max = float(y[0]), float(y[-1])

    def interpolate(self, x: float | np.ndarray, y: float) -> float | np.ndarray:
        """σ au strike x (scalaire OU np.ndarray) et maturité y (scalaire).

        La grille fittée est en variance totale w = σ²·T ; on retourne σ."""
        self._check_fitted()
        scalar = np.ndim(x) == 0
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        if y <= 0:
            res = np.zeros_like(x_arr)
        else:
            xc = np.clip(x_arr, self._x_min, self._x_max)
            yc = min(max(float(y), self._y_min), self._y_max)
            pts = np.column_stack([xc, np.full_like(xc, yc)])
            w = self.rbf(pts)
            res = np.where(w > 0, np.sqrt(np.where(w > 0, w, 0.0) / yc), 0.0)
        return float(res[0]) if scalar else res