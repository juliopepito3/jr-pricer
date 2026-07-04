"""Point d'entrée générique de calibration : optimiseur + fonction de coût → objet calibré."""
from __future__ import annotations

import numpy as np

from JR_PRICER.calibration.cost_function.base import CostFunction
from JR_PRICER.calibration.optimizer.base import Optimizer


def calibrate(optimizer: Optimizer, cost_function: CostFunction,
              theta_0: np.ndarray, bounds=None, constraints=None):
    """Calibre : minimise la fonction de coût puis reconstruit l'objet optimal.

    `theta_0` est le point de départ. `bounds`/`constraints` peuvent être passés
    explicitement, sinon ceux portés par `cost_function` sont utilisés. Retourne
    `cost_function.build(theta_opt)` — paramètres bruts par défaut, ou objet
    reconstruit selon la sous-classe (ex. `Model` pour la calibration de modèle).
    """
    bounds = bounds if bounds is not None else cost_function.bounds
    constraints = constraints if constraints is not None else cost_function.constraints
    theta_opt = optimizer.optimize(cost_function, theta_0, bounds, constraints)
    return cost_function.build(theta_opt)
