"""Interpolateur 2D bicubique sur grille (wrapper scipy RectBivariateSpline)."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline

from JR_PRICER.surfaces.vol_surface.interpolators_2D.base import GridInterpolator2D


class BiCubicInterpolator(GridInterpolator2D):
    """Spline bicubique de la variance totale w(K, T), avec clamp hors domaine."""

    def __init__(self) -> None:
        super().__init__()

    def _fit(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        self._spline = RectBivariateSpline(x, y, z, kx=3, ky=3)
        # Bornes de la grille calibrée : hors de ce domaine, la spline cubique
        # extrapole un polynôme divergent (w pouvant devenir négatif). On clampe
        # donc les requêtes au domaine — extrapolation plate, standard et robuste.
        self._x_min, self._x_max = float(x[0]), float(x[-1])
        self._y_min, self._y_max = float(y[0]), float(y[-1])

    def interpolate(self, x: float | np.ndarray, y: float) -> float | np.ndarray:
        """σ au strike x (scalaire OU np.ndarray) et maturité y (scalaire).

        σ = sqrt(w_clamped / y) : w évaluée à la maturité clampée, annualisée par
        la maturité réelle (extrapolation plate en vol)."""
        self._check_fitted()
        scalar = np.ndim(x) == 0
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        if y <= 0:
            res = np.zeros_like(x_arr)
        else:
            xc = np.clip(x_arr, self._x_min, self._x_max)
            yc = min(max(float(y), self._y_min), self._y_max)
            w = self._spline.ev(xc, np.full_like(xc, yc))  # évaluation point à point
            res = np.where(w > 0, np.sqrt(np.where(w > 0, w, 0.0) / yc), 0.0)
        return float(res[0]) if scalar else res