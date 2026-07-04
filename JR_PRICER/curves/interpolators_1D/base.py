"""Interpolateurs 1D : interface de base."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class Interpolator1D:
    """Interface de base des interpolateurs 1D.

    Mémorise les piliers (x, y) et expose `interpolate`. Les sous-classes
    implémentent `_fit` (préparation) et `interpolate` (évaluation, scalaire ou
    vectorisée sur un np.ndarray).
    """

    def __init__(self) -> None:
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def _fit(self, x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> None:
        """Prépare l'interpolateur à partir des piliers (x, y). À implémenter."""
        raise NotImplementedError("_fit() doit être implémentée dans les classes dérivées.")

    def interpolate(self, t: float | np.ndarray) -> float | np.ndarray:
        """Interpole en t (scalaire ou ndarray). À implémenter par les sous-classes."""
        raise NotImplementedError("interpolate() doit être implémentée dans les classes dérivées.")

    def __repr__(self) -> str:
        n = 0 if self.x is None else len(self.x)
        return f"{type(self).__name__}(n_points={n})"
