"""Fonction de coût de calibration globale SSVI (Gatheral-Jacquier, φ power-law)."""
from __future__ import annotations

import numpy as np

from JR_PRICER.calibration.cost_function.interpolator2d.base import Interpolator2DCostFunction


class SSVICostFunction(Interpolator2DCostFunction):
    """Calibration globale d'une surface SSVI en variance totale (power-law φ).

    Paramètres `params = (ρ, η, γ)`, log-moneyness `k`, variance totale ATM `θ` par
    point (alignée sur `k`/`w_market`) :

        φ(θ) = η·θ^(−γ)
        w_model(k) = (θ/2)·{ 1 + ρ·φ·k + √[ (φ·k + ρ)² + (1 − ρ²) ] }

    - `bounds` : **dictionnaire par défaut** `DEFAULT_BOUNDS` fusionné avec celui
      fourni (l'utilisateur peut n'overrider qu'une clé), résolu dans l'ordre
      canonique `PARAM_NAMES`.
    - `constraints` : condition papillon de Gatheral-Jacquier **pré-calculée** pour
      la power-law, `η·(1+|ρ|) ≤ 2` (passée à SLSQP par `calibrate`).

    ⚠️ `params` = vecteur de l'optimiseur (convention de la base) ; `theta_atm` = θ_T
    (variance totale ATM), à ne pas confondre.
    """

    PARAM_NAMES = ('rho', 'eta', 'gamma')
    DEFAULT_BOUNDS = {'rho': (-0.999, 0.999), 'eta': (1e-6, 5.0), 'gamma': (1e-6, 0.99)}

    def __init__(self, k: np.ndarray, w_market: np.ndarray, theta_atm: np.ndarray,
                 bounds: dict | None = None, label: str = 'SSVI') -> None:
        merged = {**self.DEFAULT_BOUNDS, **(bounds or {})}
        ordered = [merged[name] for name in self.PARAM_NAMES]
        super().__init__(
            k, w_market,
            bounds=ordered,
            # Papillon GJ pré-calculé (power-law) : η·(1+|ρ|) ≤ 2  ⇔  2 − η(1+|ρ|) ≥ 0.
            constraints=({'type': 'ineq', 'fun': lambda p: 2.0 - p[1] * (1.0 + abs(p[0]))},),
            label=label,
        )
        self.theta_atm = np.asarray(theta_atm, dtype=float)

    def w_model(self, params: np.ndarray, k: np.ndarray) -> np.ndarray:
        rho, eta, gamma = params
        th = self.theta_atm
        x = (eta * th ** (-gamma)) * k          # φ(θ)·k
        return 0.5 * th * (1.0 + rho * x + np.sqrt((x + rho) ** 2 + (1.0 - rho ** 2)))
