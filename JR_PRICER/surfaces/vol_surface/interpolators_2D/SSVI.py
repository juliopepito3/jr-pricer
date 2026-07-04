"""Interpolateur de surface SSVI (Gatheral-Jacquier), calibration globale, φ power-law."""
from __future__ import annotations

import numpy as np

from JR_PRICER.surfaces.vol_surface.interpolators_2D.base import LayeredInterpolator2D
from JR_PRICER.calibration.calibrate import calibrate
from JR_PRICER.calibration.optimizer.scipy_minimize import SciPyMinimizeOptimizer
from JR_PRICER.calibration.cost_function.interpolator2d.ssvi_cost import SSVICostFunction
from JR_PRICER.curves.forward.base import ForwardCurve
from JR_PRICER.surfaces.vol_surface.theta_curve import ThetaCurve
from JR_PRICER.curves.vol_smile.volsmile import VolSmile


class SSVIInterpolator(LayeredInterpolator2D):
    """Surface SSVI de Gatheral-Jacquier, calibrée globalement (power-law φ).

    Paramétrage en variance totale, log-moneyness k = ln(K/F), θ_T = variance ATM :
        φ(θ) = η·θ^(−γ)
        w(k, θ) = (θ/2)·{ 1 + ρ·φ·k + √[ (φ·k + ρ)² + (1 − ρ²) ] }

    - Trois paramètres globaux `(ρ, η, γ)` calibrés sur toute la surface (cf.
      `SSVICostFunction`), sous la contrainte papillon pré-calculée `η·(1+|ρ|) ≤ 2`.
    - `θ_T` est fournie via une `ThetaCurve` strictement croissante (no-arbitrage
      calendaire garanti par construction).
    - `interpolate(K, T)` retourne σ (cf. contrat Interpolator2D). Dérivées de la
      variance totale `w` analytiques (`dw_dK`, `d2w_dK2` closed-form ; `dw_dT` via
      la chaîne SSVI closed-form + deux pentes de courbe 1D par différence finie).
    """

    # Options SLSQP : ftol serré car les résidus en variance totale sont de faible
    # magnitude (SSE ~ 1e-5) — le ftol par défaut (1e-6) stoppe la calibration trop tôt.
    _SLSQP_OPTIONS = {'ftol': 1e-12, 'maxiter': 1000}

    def __init__(self, forward_curve: ForwardCurve, theta_curve: ThetaCurve,
                 bounds: dict | None = None,
                 theta_0: tuple = (-0.3, 1.0, 0.3)) -> None:
        super().__init__()
        self._forward_curve = forward_curve
        self._theta_curve = theta_curve
        self._bounds = bounds
        self._theta_0 = theta_0
        self._params: np.ndarray | None = None  # (rho, eta, gamma)

    def __repr__(self) -> str:
        return f"SSVIInterpolator(params={self._params}, fitted={self._fitted})"

    # ---- calibration ---------------------------------------------------------

    def fit(self, maturities: list[float], smiles: list[VolSmile]) -> None:
        """Calibre `(ρ, η, γ)` globalement sur tous les smiles via `calibrate`.

        Conventions identiques à SVI (smiles en strike absolu). θ_T provient
        entièrement de la `ThetaCurve` (non extrait/stocké ici)."""
        k_all, w_all, theta_all = [], [], []
        for T, smile in zip(maturities, smiles):
            F = self._forward_curve.forward(T)
            k = np.log(smile.x / F)
            w = smile.y ** 2 * T
            theta_atm = float(self._theta_curve.evaluate(T))
            k_all.append(k)
            w_all.append(w)
            theta_all.append(np.full_like(k, theta_atm))

        cost = SSVICostFunction(np.concatenate(k_all), np.concatenate(w_all),
                                np.concatenate(theta_all), bounds=self._bounds)
        optimizer = SciPyMinimizeOptimizer(method='SLSQP', options=self._SLSQP_OPTIONS)
        self._params = calibrate(optimizer, cost, theta_0=self._theta_0)
        self._fitted = True

    # ---- volatilité ----------------------------------------------------------

    def interpolate(self, K: float | np.ndarray, T: float) -> float | np.ndarray:
        """σ au strike K (scalaire ou ndarray) et maturité T (cf. contrat Interpolator2D)."""
        self._check_fitted()
        if T <= 0:
            return 0.0 if np.ndim(K) == 0 else np.zeros(np.shape(K))
        theta = float(self._theta_curve.evaluate(T))
        rho, eta, gamma = self._params
        phi = eta * theta ** (-gamma)
        k = np.log(K / self._forward_curve.forward(T))
        x = phi * k
        w = 0.5 * theta * (1.0 + rho * x + np.sqrt((x + rho) ** 2 + (1.0 - rho ** 2)))
        return np.sqrt(np.maximum(w, 0.0) / T)

    # ---- dérivées analytiques de la variance totale w = σ²·T -----------------

    def w_and_derivs(self, K: float | np.ndarray, T: float):
        """(w, ∂w/∂K, ∂²w/∂K², ∂w/∂T) en variance totale, en une seule traversée.

        Mutualise θ, φ, k, D (cf. `LocalVolModel.sigma_loc`, chemin chaud du MC).
        - ∂w/∂K, ∂²w/∂K² : closed-form (même chaîne strike que SVI).
        - ∂w/∂T à K fixé : chaîne SSVI closed-form (∂w/∂θ, ∂w/∂k) + deux pentes de
          courbe 1D (dθ/dT, dlnF/dT) par différence finie centrée (ni `Curve` ni
          `ForwardCurve` n'exposent de dérivée analytique)."""
        self._check_fitted()
        theta = float(self._theta_curve.evaluate(T))
        rho, eta, gamma = self._params
        phi = eta * theta ** (-gamma)
        k = np.log(K / self._forward_curve.forward(T))

        u = phi * k + rho
        D = np.sqrt(u * u + (1.0 - rho * rho))
        w = 0.5 * theta * (1.0 + rho * phi * k + D)
        dw_dk = 0.5 * theta * phi * (rho + u / D)
        d2w_dk2 = 0.5 * theta * phi * phi * (1.0 - rho * rho) / D ** 3

        invK = 1.0 / K
        dw_dK = dw_dk * invK
        d2w_dK2 = (d2w_dk2 - dw_dk) * invK * invK

        # ∂w/∂T à K fixé : chaîne SSVI analytique + drift des deux courbes 1D (FF).
        dw_dtheta = (w - gamma * k * dw_dk) / theta
        dw_dT = dw_dtheta * self._dtheta_dT(T) - dw_dk * self._dlnF_dT(T)
        return w, dw_dK, d2w_dK2, dw_dT

    def dw_dK(self, K: float | np.ndarray, T: float) -> float | np.ndarray:
        """∂w/∂K (variance totale) — analytique."""
        return self.w_and_derivs(K, T)[1]

    def d2w_dK2(self, K: float | np.ndarray, T: float) -> float | np.ndarray:
        """∂²w/∂K² (variance totale) — analytique."""
        return self.w_and_derivs(K, T)[2]

    def dw_dT(self, K: float | np.ndarray, T: float) -> float | np.ndarray:
        """∂w/∂T (variance totale) à K fixé — chaîne analytique + pentes de courbe (FF)."""
        return self.w_and_derivs(K, T)[3]

    # ---- pentes des courbes 1D (différence finie centrée, pas relatif) --------

    def _dtheta_dT(self, T: float, h: float = 1e-5) -> float:
        hT = h * T if T > 1e-10 else h
        tp = float(self._theta_curve.evaluate(T + hT))
        tm = float(self._theta_curve.evaluate(T - hT))
        return (tp - tm) / (2.0 * hT)

    def _dlnF_dT(self, T: float, h: float = 1e-5) -> float:
        hT = h * T if T > 1e-10 else h
        Fp = self._forward_curve.forward(T + hT)
        Fm = self._forward_curve.forward(T - hT)
        return (np.log(Fp) - np.log(Fm)) / (2.0 * hT)
