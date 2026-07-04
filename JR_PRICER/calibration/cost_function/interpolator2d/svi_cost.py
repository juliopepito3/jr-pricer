"""Fonction de coût de calibration d'une slice SVI (raw SVI de Gatheral) à T fixe."""
from __future__ import annotations

import numpy as np

from JR_PRICER.calibration.cost_function.interpolator2d.base import Interpolator2DCostFunction


class SVISliceCost(Interpolator2DCostFunction):
    """Calibration d'une slice SVI à maturité T fixe, en variance totale.

    Paramètres `theta = (a, b, rho, m, sigma)`, log-moneyness `k = ln(K/F(T))` :
        w_model(k) = a + b·(ρ·(k-m) + sqrt((k-m)² + σ²)).

    `bounds` et la contrainte de non-arbitrage papillon de Gatheral
    (`b·(1+|ρ|) ≤ 4/T`, dépendante de T) sont portées ici ; `calibrate` les passe
    à l'optimiseur (SLSQP requis pour honorer la contrainte non linéaire).
    """

    def __init__(self, k: np.ndarray, w_market: np.ndarray, T: float) -> None:
        super().__init__(
            k, w_market,
            bounds=[(-np.inf, np.inf), (0.0, np.inf), (-1.0, 1.0),
                    (-np.inf, np.inf), (1e-6, np.inf)],
            constraints=({'type': 'ineq', 'fun': lambda p: 4.0 / T - p[1] * (1 + abs(p[2]))},),
            label=f"SVI T={T:.2f}",
        )

    def w_model(self, theta: np.ndarray, k: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = theta
        km = k - m
        return a + b * (rho * km + np.sqrt(km * km + sigma * sigma))
