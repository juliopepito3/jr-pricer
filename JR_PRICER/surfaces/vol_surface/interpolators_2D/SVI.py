"""Interpolateur de surface SVI (raw SVI de Gatheral), calibré par maturité."""
from __future__ import annotations

import numpy as np

from JR_PRICER.surfaces.vol_surface.interpolators_2D.base import LayeredInterpolator2D
from JR_PRICER.calibration.calibrate import calibrate
from JR_PRICER.calibration.optimizer.scipy_minimize import SciPyMinimizeOptimizer
from JR_PRICER.calibration.cost_function.interpolator2d.svi_cost import SVISliceCost
from JR_PRICER.curves.forward.base import ForwardCurve
from JR_PRICER.curves.vol_smile.volsmile import VolSmile
from JR_PRICER.utils.numerics import index_of_close


class SVIInterpolator(LayeredInterpolator2D):
    """Surface SVI calibrée maturité par maturité (raw SVI de Gatheral).

    Paramétrage en variance totale :
        w(k) = a + b·(ρ·(k-m) + sqrt((k-m)² + σ²)),   k = ln(K/F).

    - `interpolate(K, T)` retourne σ (cf. contrat Interpolator2D).
    - Entre maturités calibrées : interpolation linéaire de la variance totale w
      (préserve l'absence d'arbitrage calendaire).
    - Hors du domaine calibré : extrapolation plate en volatilité
      (σ(K,T) = σ(K, T_pillar le plus proche)), i.e. w ∝ T.
    """

    # Options SLSQP : ftol serré car les résidus en variance totale sont de faible
    # magnitude (SSE ~ 1e-5) — le ftol par défaut (1e-6) stoppe la calibration trop tôt.
    _SLSQP_OPTIONS = {'ftol': 1e-12, 'maxiter': 1000}

    def __init__(self, forward_curve: ForwardCurve) -> None:
        super().__init__()
        self._forward_curve = forward_curve
        self._params: dict[float, tuple] = {}  # {T: (a, b, rho, m, sigma)}
        self._sorted_maturities: list[float] = []  # cache trié (évite sorted() à chaque appel)

    def __repr__(self) -> str:
        return f"SVIInterpolator(n_pillars={len(self._params)}, fitted={self._fitted})"

    def fit(self, maturities: list[float], smiles: list[VolSmile]) -> None:
        for T, smile in zip(maturities, smiles):
            self._params[T] = self._fit_svi(smile, T)
        self._sorted_maturities = sorted(self._params)
        self._fitted = True

    def _fit_svi(self, smile: VolSmile, T: float) -> np.ndarray:
        """Calibre les paramètres SVI de la slice T via le cœur générique `calibrate`.

        L'extraction marché (log-moneyness k, variance totale w) se fait ici ;
        `SVISliceCost` ne reçoit que des tableaux (aucune dépendance au domaine).
        Bornes + contrainte papillon de Gatheral sont portées par la cost function ;
        SLSQP est requis pour honorer la contrainte non linéaire.
        """
        F = self._forward_curve.forward(T)
        k = np.log(smile.x / F)               # smile.x est déjà un ndarray (cf. Curve)
        w_market = smile.y ** 2 * T
        return calibrate(
            SciPyMinimizeOptimizer(method='SLSQP', options=self._SLSQP_OPTIONS),
            SVISliceCost(k, w_market, T),
            theta_0=(0.1, 0.1, 0.0, 0.0, 0.1),
        )

    def _svi_total_variance(self, k: float | np.ndarray, a, b, rho, m, sigma) -> float | np.ndarray:
        """Variance totale SVI brute en log-moneyness k = ln(K/F)."""
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    # ---- résolution de maturité ---------------------------------------------

    def _lookup_T(self, T: float) -> float | None:
        """Clé du pillar calibré proche de T (tolérance flottante), sinon None."""
        maturities = self._sorted_maturities
        i = index_of_close(T, maturities)
        return maturities[i] if i is not None else None

    def _clamp_T(self, T: float) -> float | None:
        """Pillar à utiliser hors du domaine calibré (extrapolation plate en vol) :
        T_min si T < T_min, T_max si T > T_max, sinon None (T est à l'intérieur)."""
        maturities = self._sorted_maturities
        if T < maturities[0]:
            return maturities[0]
        if T > maturities[-1]:
            return maturities[-1]
        return None

    def _bracket_T(self, T: float) -> tuple[float, float]:
        """Pillars (T1, T2) encadrant T à l'intérieur du domaine calibré."""
        maturities = self._sorted_maturities
        for i in range(len(maturities) - 1):
            if maturities[i] <= T <= maturities[i + 1]:
                return maturities[i], maturities[i + 1]
        raise RuntimeError(f"SVI: maturité T={T:.4f} non encadrable dans le domaine calibré.")

    def _w_at(self, K: float | np.ndarray, T_pillar: float) -> float | np.ndarray:
        """Variance totale du pillar calibré T_pillar au strike K (scalaire ou ndarray)."""
        F = self._forward_curve.forward(T_pillar)
        k = np.log(K / F)
        return self._svi_total_variance(k, *self._params[T_pillar])

    def _core(self, K: float | np.ndarray, T_pillar: float):
        """(w, ∂w/∂k, ∂²w/∂k²) au pillar calibré, en UNE passe (un seul log, un seul sqrt).

        Mutualise les intermédiaires partagés par `_w_at`, `_dw_dK_calibrated` et
        `_d2w_dK2_calibrated` (évite de recalculer log(K/F) et sqrt((k-m)²+σ²))."""
        a, b, rho, m, sigma = self._params[T_pillar]
        F = self._forward_curve.forward(T_pillar)
        km = np.log(K / F) - m
        root = np.sqrt(km * km + sigma * sigma)
        w = a + b * (rho * km + root)
        dw_dk = b * (rho + km / root)
        d2w_dk2 = b * sigma * sigma / root ** 3
        return w, dw_dk, d2w_dk2

    def w_and_derivs(self, K: float | np.ndarray, T: float):
        """(w, ∂w/∂K, ∂²w/∂K², ∂w/∂T) en variance totale, en une seule traversée.

        - `w` égale `interpolate(K, T)**2 * T` (variance totale consistante avec σ),
        - les trois dérivées équivalent bit-à-bit à `dw_dK`, `d2w_dK2`, `dw_dT`,

        mais tout partage le bracketing en T, `forward()`, log et sqrt (cf. `_core`).
        Utilisé par `LocalVolModel.sigma_loc` (chemin chaud du Monte-Carlo) pour
        éviter ~4 appels redondants et ~6 `np.log` par pas."""
        self._check_fitted()
        invK = 1.0 / K
        invK2 = invK * invK

        def strike_chain(dw_dk, d2w_dk2):
            # k = ln K : dk/dK = 1/K, d²k/dK² = -1/K²
            return dw_dk * invK, d2w_dk2 * invK2 - dw_dk * invK2

        T_exact = self._lookup_T(T)
        if T_exact is not None:
            w0, dwk, d2wk = self._core(K, T_exact)
            dw_dK, d2w_dK2 = strike_chain(dwk, d2wk)
            T1, T2 = self._bracket_T(T)               # même règle que dw_dT (pas de branche exacte)
            dw_dT = (self._core(K, T2)[0] - self._core(K, T1)[0]) / (T2 - T1)
            return w0 * (T / T_exact), dw_dK, d2w_dK2, dw_dT

        T_clamp = self._clamp_T(T)
        if T_clamp is not None:
            w0, dwk, d2wk = self._core(K, T_clamp)
            scale = T / T_clamp
            dw_dK, d2w_dK2 = strike_chain(dwk, d2wk)
            return w0 * scale, dw_dK * scale, d2w_dK2 * scale, w0 / T_clamp

        T1, T2 = self._bracket_T(T)
        alpha = (T - T1) / (T2 - T1)
        w1, dwk1, d2wk1 = self._core(K, T1)
        w2, dwk2, d2wk2 = self._core(K, T2)
        dwk = (1 - alpha) * dwk1 + alpha * dwk2
        d2wk = (1 - alpha) * d2wk1 + alpha * d2wk2
        dw_dK, d2w_dK2 = strike_chain(dwk, d2wk)
        return w1 + (w2 - w1) * alpha, dw_dK, d2w_dK2, (w2 - w1) / (T2 - T1)

    # ---- volatilité ----------------------------------------------------------

    def interpolate(self, K: float | np.ndarray, T: float) -> float | np.ndarray:
        """σ au strike K (scalaire ou ndarray) et maturité T (cf. contrat Interpolator2D)."""
        self._check_fitted()

        T_exact = self._lookup_T(T)
        if T_exact is not None:
            return np.sqrt(self._w_at(K, T_exact) / T_exact)

        T_clamp = self._clamp_T(T)
        if T_clamp is not None:
            # Extrapolation plate en vol : σ(K,T) = σ(K, T_pillar)
            return np.sqrt(self._w_at(K, T_clamp) / T_clamp)

        # Interpolation linéaire en variance totale entre pillars encadrants
        T1, T2 = self._bracket_T(T)
        w1, w2 = self._w_at(K, T1), self._w_at(K, T2)
        w_interp = w1 + (w2 - w1) * (T - T1) / (T2 - T1)
        return np.sqrt(w_interp / T)

    # ---- dérivées analytiques de la variance totale w = σ²·T -----------------

    def dw_dK(self, K: float | np.ndarray, T: float) -> float | np.ndarray:
        """∂w/∂K (variance totale) — analytique, scalaire ou ndarray sur K."""
        self._check_fitted()
        T_exact = self._lookup_T(T)
        if T_exact is not None:
            return self._dw_dK_calibrated(K, T_exact)
        T_clamp = self._clamp_T(T)
        if T_clamp is not None:
            # w(K,T) = w(K,T_pillar)·(T/T_pillar) → ∂w/∂K mis à l'échelle
            return self._dw_dK_calibrated(K, T_clamp) * (T / T_clamp)
        T1, T2 = self._bracket_T(T)
        alpha = (T - T1) / (T2 - T1)
        return (1 - alpha) * self._dw_dK_calibrated(K, T1) + alpha * self._dw_dK_calibrated(K, T2)

    def _dw_dK_calibrated(self, K: float | np.ndarray, T: float) -> float | np.ndarray:
        a, b, rho, m, sigma = self._params[T]
        F = self._forward_curve.forward(T)
        k = np.log(K / F)
        dw_dk = b * (rho + (k - m) / np.sqrt((k - m) ** 2 + sigma ** 2))
        dk_dK = 1 / K
        return dw_dk * dk_dK

    def d2w_dK2(self, K: float | np.ndarray, T: float) -> float | np.ndarray:
        """∂²w/∂K² (variance totale) — analytique, scalaire ou ndarray sur K."""
        self._check_fitted()
        T_exact = self._lookup_T(T)
        if T_exact is not None:
            return self._d2w_dK2_calibrated(K, T_exact)
        T_clamp = self._clamp_T(T)
        if T_clamp is not None:
            return self._d2w_dK2_calibrated(K, T_clamp) * (T / T_clamp)
        T1, T2 = self._bracket_T(T)
        alpha = (T - T1) / (T2 - T1)
        return (1 - alpha) * self._d2w_dK2_calibrated(K, T1) + alpha * self._d2w_dK2_calibrated(K, T2)

    def _d2w_dK2_calibrated(self, K: float | np.ndarray, T: float) -> float | np.ndarray:
        a, b, rho, m, sigma = self._params[T]
        F = self._forward_curve.forward(T)
        k = np.log(K / F)
        dw_dk = b * (rho + (k - m) / np.sqrt((k - m) ** 2 + sigma ** 2))
        d2w_dk2 = b * sigma ** 2 / ((k - m) ** 2 + sigma ** 2) ** (3 / 2)
        dk_dK = 1 / K
        d2k_dK2 = -1 / K ** 2
        return d2w_dk2 * dk_dK ** 2 + dw_dk * d2k_dK2

    def dw_dT(self, K: float | np.ndarray, T: float) -> float | np.ndarray:
        """∂w/∂T (variance totale) — analytique, scalaire ou ndarray sur K."""
        self._check_fitted()
        T_clamp = self._clamp_T(T)
        if T_clamp is not None:
            # Extrapolation plate en vol : w = σ̂²·T → ∂w/∂T = σ̂² = w(K,T_pillar)/T_pillar
            return self._w_at(K, T_clamp) / T_clamp
        # Intérieur : pente de variance totale entre pillars encadrants
        T1, T2 = self._bracket_T(T)
        return (self._w_at(K, T2) - self._w_at(K, T1)) / (T2 - T1)
