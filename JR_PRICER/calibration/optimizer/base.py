"""Interface de base, générique, des optimiseurs de calibration."""
from __future__ import annotations

import numpy as np

from JR_PRICER.calibration.cost_function.base import CostFunction


class Optimizer:
    """Optimiseur : minimise une fonction de coût et renvoie les paramètres optimaux.

    `optimize(cost_function, theta_0, bounds, constraints)` part de `theta_0`,
    respecte les bornes/contraintes éventuelles et retourne le vecteur theta
    optimal. À implémenter par les sous-classes.

    Conventions des arguments (format canonique, indépendant du backend) :
    - `bounds` : `list[(lo, hi)]` (une paire par paramètre) ou `None`.
    - `constraints` : tuple de dicts au format `scipy.optimize.minimize`, ou `None`.
    """

    def __init__(self) -> None:
        pass

    def optimize(self, cost_function: CostFunction, theta_0: np.ndarray,
                 bounds=None, constraints=None) -> np.ndarray:
        """Minimise `cost_function` à partir de `theta_0`. À implémenter."""
        raise NotImplementedError("optimize() doit être implémentée dans les classes dérivées.")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
