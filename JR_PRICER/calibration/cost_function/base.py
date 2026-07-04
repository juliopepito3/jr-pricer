"""Interface de base, générique, des fonctions de coût de calibration.

Cœur de calibration agnostique du domaine : aucune dépendance à `pricing/` ni
`surfaces/`. Une `CostFunction` décrit *quoi* minimiser (résidus, bornes,
contraintes) et *comment* reconstruire l'objet calibré (`build`) ; le *comment
minimiser* est délégué à un `Optimizer`, l'orchestration à `calibrate`.
"""
from __future__ import annotations

import numpy as np


class CostFunction:
    """Fonction de coût générique pour la calibration de paramètres.

    Contrat :
    - `__call__(theta)` renvoie le **vecteur de résidus** à annuler (et non un
      scalaire) : c'est la forme exploitable par les deux familles d'optimiseurs
      (moindres carrés vectoriels, ou minimisation scalaire après repliage SSE).
    - `bounds` : bornes au format canonique `list[(lo, hi)]` (une paire par
      paramètre), ou `None`. `None`/`±np.inf` dénotent un côté non borné.
    - `constraints` : contraintes au format `scipy.optimize.minimize`
      (tuple de dicts `{'type': 'ineq'|'eq', 'fun': ...}`), ou `None`.
    - `label` : étiquette courte pour les diagnostics/warnings.
    - `build(theta)` : reconstruit l'objet calibré à partir des paramètres
      optimaux. Par défaut renvoie `theta` brut ; les sous-classes spécialisées
      (ex. modèle de pricing) le surchargent.
    """

    def __init__(self, bounds=None, constraints=None, label: str = "") -> None:
        self.bounds = bounds
        self.constraints = constraints
        self.label = label

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        """Vecteur de résidus pour les paramètres `theta`. À implémenter."""
        raise NotImplementedError("Subclasses must implement __call__.")

    def build(self, theta: np.ndarray):
        """Objet calibré à partir des paramètres optimaux (par défaut : `theta`)."""
        return theta

    def __repr__(self) -> str:
        return f"{type(self).__name__}(label={self.label!r})"
