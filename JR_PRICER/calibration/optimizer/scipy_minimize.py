"""Optimiseur scalaire contraint (wrapper scipy.optimize.minimize)."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize

from JR_PRICER.calibration.cost_function.base import CostFunction
from JR_PRICER.calibration.optimizer.base import Optimizer


class SciPyMinimizeOptimizer(Optimizer):
    """Minimisation scalaire sous bornes + contraintes (SLSQP par défaut).

    Replie le vecteur de résidus de la `CostFunction` en somme des carrés
    `0.5·‖r‖²` (objectif strictement équivalent à un moindres carrés), ce qui
    permet d'honorer des contraintes non linéaires que `least_squares` ne sait
    pas traiter (ex. condition de non-arbitrage papillon de SVI).

    Le format canonique des bornes `list[(lo, hi)]` est exactement celui attendu
    par `minimize` → aucune conversion.

    `options` est passé tel quel à `scipy.optimize.minimize(options=...)` (options
    du solveur, ex. `{'ftol': 1e-12, 'maxiter': 1000}`). Important pour les coûts en
    variance totale (résidus de faible magnitude) : le `ftol` SLSQP par défaut (1e-6)
    est relatif à la SSE et peut stopper la calibration très tôt.
    """

    def __init__(self, method: str = 'SLSQP', options=None) -> None:
        super().__init__()
        self.method = method
        self.options = options

    def optimize(self, cost_function: CostFunction, theta_0: np.ndarray,
                 bounds=None, constraints=None) -> np.ndarray:
        def scalar_objective(theta: np.ndarray) -> float:
            r = np.asarray(cost_function(theta), dtype=float)
            return 0.5 * float(np.dot(r, r))

        result = minimize(
            scalar_objective,
            x0=np.asarray(theta_0, dtype=float),
            method=self.method,
            bounds=bounds,
            constraints=constraints if constraints is not None else (),
            options=self.options,  # options solveur (ex. {'ftol': 1e-12, 'maxiter': 1000})
        )
        if not result.success:
            label = f" [{cost_function.label}]" if cost_function.label else ""
            warnings.warn(f"Calibration failed{label}: {result.message}")
        return result.x
