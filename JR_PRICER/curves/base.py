"""Courbe 1D générique (piliers x, y) interpolée par un Interpolator1D."""
from __future__ import annotations

from collections.abc import Sequence

from JR_PRICER.curves.interpolators_1D.base import Interpolator1D
from JR_PRICER.utils.numerics import index_of_close
import numpy as np
import copy


class Curve:
    """Courbe 1D : abscisses `x` et ordonnées `y` (ndarray), interpolées à la demande.

    L'interpolateur est ajusté paresseusement (lazy) à la première évaluation et
    ré-ajusté après tout `add_point` (utilisé par le bootstrap).
    """

    def __init__(self, x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray,
                 interpolator: Interpolator1D) -> None:
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self._interpolator = copy.deepcopy(interpolator)
        self._fitted = False

    def add_point(self, x: float, y: float) -> None:
        """Ajoute un pilier (x, y) et invalide l'ajustement de l'interpolateur."""
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self._fitted = False

    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        """Évalue la courbe en x — scalaire OU np.ndarray (interpolation vectorisée).

        Pour un scalaire pilier, on renvoie la valeur exacte (évite une erreur
        flottante d'1 ulp et reste valide sur une courbe à un seul point, avant que
        l'interpolateur — ex. spline cubique — ne soit fittable).
        """
        if np.ndim(x) == 0:
            i = index_of_close(x, self.x)
            if i is not None:
                return float(self.y[i])
        if not self._fitted:
            self._interpolator._fit(self.x, self.y)
            self._fitted = True
        return self._interpolator.interpolate(x)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_points={len(self.x)})"
