"""Base des fonctions de coût d'interpolateurs 2D paramétriques (SVI, SSVI, ...)."""
from __future__ import annotations

import numpy as np

from JR_PRICER.calibration.cost_function.base import CostFunction


class Interpolator2DCostFunction(CostFunction):
    """Calibration d'une forme paramétrique de surface, en variance totale.

    Agnostique du domaine : ne manipule que des tableaux numpy — le
    log-moneyness `k = ln(K/F)` et la variance totale de marché `w_market = σ²·T`.
    L'extraction depuis un `VolSmile`/`ForwardCurve` reste côté interpolateur, ce
    qui évite toute dépendance `calibration → surfaces` (et donc tout cycle).

    Contrat des sous-classes : implémenter `w_model(theta, k)` (variance totale du
    modèle). Le résidu commun `w_market − w_model` est défini ici. `build` renvoie
    les paramètres bruts par défaut (cf. `CostFunction`).
    """

    def __init__(self, k: np.ndarray, w_market: np.ndarray,
                 bounds=None, constraints=None, label: str = "") -> None:
        super().__init__(bounds=bounds, constraints=constraints, label=label)
        self.k = np.asarray(k, dtype=float)
        self.w_market = np.asarray(w_market, dtype=float)

    def w_model(self, theta: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Variance totale du modèle au log-moneyness `k`. À implémenter."""
        raise NotImplementedError("Subclasses must implement w_model.")

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        return self.w_market - self.w_model(theta, self.k)
