"""Optimiseur moindres carrés (wrapper scipy.optimize.least_squares)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from JR_PRICER.calibration.cost_function.base import CostFunction
from JR_PRICER.calibration.optimizer.base import Optimizer


def _to_least_squares_bounds(bounds):
    """Convertit le format canonique `list[(lo, hi)]` → `(lb_array, ub_array)`.

    `None` côté borne devient `-inf` (lo) ou `+inf` (hi), conformément à ce
    qu'attend `scipy.optimize.least_squares`."""
    lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds], dtype=float)
    hi = np.array([np.inf if b[1] is None else b[1] for b in bounds], dtype=float)
    return lo, hi


class SciPyOLSOptimizer(Optimizer):
    """Optimiseur par moindres carrés non linéaires (Levenberg-Marquardt / TRF).

    Exploite directement le vecteur de résidus de la `CostFunction`. Gère les
    bornes, mais **pas** les contraintes non linéaires (limite de `least_squares`)
    → utiliser `SciPyMinimizeOptimizer` pour une calibration contrainte.
    """

    def __init__(self, options=None) -> None:
        super().__init__()
        self.options = options

    def optimize(self, cost_function: CostFunction, theta_0: np.ndarray,
                 bounds=None, constraints=None) -> np.ndarray:
        if constraints:
            raise ValueError(
                "SciPyOLSOptimizer ne gère pas de contraintes non linéaires ; "
                "utilisez SciPyMinimizeOptimizer."
            )
        ls_bounds = (-np.inf, np.inf) if bounds is None else _to_least_squares_bounds(bounds)
        result = least_squares(
            cost_function,
            x0=theta_0,
            bounds=ls_bounds,
            **(self.options if self.options is not None else {})
        )
        return result.x
