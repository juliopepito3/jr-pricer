"""Régressions affines pondérées et robustes (pure numpy, sans dépendance domaine).

Brique statistique du builder de parité put-call : on y régresse (C - P) contre K,
une relation exactement affine en théorie mais polluée en pratique par le bruit
bid-ask et par quelques quotes aberrantes (stale, croisées, fat fingers). D'où
trois stratégies :

- `WeightedLeastSquares` : moindres carrés pondérés classiques (la base).
- `MADRejectionRegression` : fit / rejet des gros résidus / refit — le geste desk
  standard, avec un masque d'inliers inspectable.
- `HuberIRLSRegression` : alternative lisse (pas de seuil dur), les outliers sont
  progressivement dépondérés plutôt qu'exclus.

Module volontairement agnostique : il ne connaît ni strikes ni options, seulement
des points (x, y) et des poids.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# Facteur de cohérence de la MAD avec l'écart-type sous hypothèse gaussienne :
# scale = 1.4826 * MAD est un estimateur robuste de sigma.
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class LinearFitResult:
    """Résultat d'un fit affine y = intercept + slope * x.

    Les résidus sont donnés sur TOUS les points d'entrée (rejetés inclus), dans
    l'ordre d'origine ; `inlier_mask` indique les points effectivement utilisés
    par le fit final. Les statistiques (r², rmse, erreurs standard) sont
    calculées sur les inliers, avec les poids fournis.
    """
    intercept: float
    slope: float
    inlier_mask: np.ndarray
    residuals: np.ndarray
    r_squared: float
    rmse: float
    stderr_intercept: float
    stderr_slope: float
    n_iterations: int

    @property
    def n_inliers(self) -> int:
        return int(np.sum(self.inlier_mask))

    def predict(self, x: float | np.ndarray) -> float | np.ndarray:
        """Valeur prédite par la droite ajustée."""
        return self.intercept + self.slope * np.asarray(x, dtype=float)


def _validate_inputs(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
    if x.ndim != 1 or x.shape != y.shape or x.shape != weights.shape:
        raise ValueError("x, y et weights doivent être des vecteurs 1D de même taille.")
    if len(x) < 2:
        raise ValueError("Au moins 2 points sont nécessaires pour un fit affine.")
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError("Les poids doivent être strictement positifs et finis.")


def _solve_wls(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Résout le WLS min Σ w_i (y_i - a - b x_i)² ; retourne (intercept, slope).

    On multiplie la design matrix et y par √w (et non w : pondérer les équations
    par w reviendrait à minimiser Σ w² r², soit des poids au carré).
    """
    design = np.column_stack([np.ones_like(x), x])
    sqrt_w = np.sqrt(weights)
    (intercept, slope), _, _, _ = np.linalg.lstsq(
        sqrt_w[:, None] * design, sqrt_w * y, rcond=None
    )
    return float(intercept), float(slope)


def _build_result(x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                  intercept: float, slope: float, inlier_mask: np.ndarray,
                  n_iterations: int) -> LinearFitResult:
    """Assemble le résultat : résidus sur tous les points, stats sur les inliers."""
    residuals = y - (intercept + slope * x)

    x_in = x[inlier_mask]
    r_in = residuals[inlier_mask]
    w_in = weights[inlier_mask]
    n_in = len(x_in)

    sum_w = np.sum(w_in)
    ss_res = float(np.sum(w_in * r_in ** 2))
    y_bar = float(np.sum(w_in * y[inlier_mask]) / sum_w)
    ss_tot = float(np.sum(w_in * (y[inlier_mask] - y_bar) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / sum_w))

    # Erreurs standard depuis cov(θ) = σ̂² (XᵀWX)⁻¹ avec σ̂² = ss_res / (n - 2).
    if n_in > 2:
        sigma2_hat = ss_res / (n_in - 2)
        design_in = np.column_stack([np.ones_like(x_in), x_in])
        xtwx = design_in.T @ (w_in[:, None] * design_in)
        try:
            cov = sigma2_hat * np.linalg.inv(xtwx)
            stderr_intercept = float(np.sqrt(cov[0, 0]))
            stderr_slope = float(np.sqrt(cov[1, 1]))
        except np.linalg.LinAlgError:
            stderr_intercept = stderr_slope = float("nan")
    else:
        stderr_intercept = stderr_slope = float("nan")

    return LinearFitResult(
        intercept=intercept,
        slope=slope,
        inlier_mask=inlier_mask,
        residuals=residuals,
        r_squared=r_squared,
        rmse=rmse,
        stderr_intercept=stderr_intercept,
        stderr_slope=stderr_slope,
        n_iterations=n_iterations,
    )


def _robust_scale(residuals: np.ndarray) -> tuple[float, float]:
    """Échelle robuste des résidus : (médiane, 1.4826 * MAD autour de la médiane)."""
    med = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - med)))
    return med, _MAD_TO_SIGMA * mad


class LinearRegression(ABC):
    """Interface commune des régressions affines : fit(x, y, weights) → résultat."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> LinearFitResult:
        """Ajuste y ≈ intercept + slope * x avec les poids fournis."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class WeightedLeastSquares(LinearRegression):
    """Moindres carrés pondérés : tous les points sont des inliers."""

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> LinearFitResult:
        x, y, weights = (np.asarray(v, dtype=float) for v in (x, y, weights))
        _validate_inputs(x, y, weights)
        intercept, slope = _solve_wls(x, y, weights)
        mask = np.ones(len(x), dtype=bool)
        return _build_result(x, y, weights, intercept, slope, mask, n_iterations=1)


class MADRejectionRegression(LinearRegression):
    """WLS itéré avec rejet des résidus |r - med| > k * (1.4826 * MAD), puis refit.

    C'est le geste desk classique : on fit, on écarte les quotes manifestement
    fausses (stale, croisées), on refit. Le seuil est relatif à l'échelle robuste
    des résidus, donc insensible aux outliers eux-mêmes. Le rejet s'arrête si une
    passe ne rejette plus rien, si `max_passes` est atteint, ou si la passe
    suivante laisserait moins de `min_inliers` points.
    """

    def __init__(self, k_mad: float = 3.0, max_passes: int = 2, min_inliers: int = 4) -> None:
        if k_mad <= 0:
            raise ValueError("k_mad doit être strictement positif.")
        if min_inliers < 2:
            raise ValueError("min_inliers doit être au moins 2 (fit affine).")
        self.k_mad = k_mad
        self.max_passes = max_passes
        self.min_inliers = min_inliers

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> LinearFitResult:
        x, y, weights = (np.asarray(v, dtype=float) for v in (x, y, weights))
        _validate_inputs(x, y, weights)

        mask = np.ones(len(x), dtype=bool)
        intercept, slope = _solve_wls(x, y, weights)
        n_iterations = 1

        for _ in range(self.max_passes):
            residuals = y - (intercept + slope * x)
            med, scale = _robust_scale(residuals[mask])
            if scale == 0.0:
                # Fit quasi exact sur les inliers : plus rien à rejeter de façon
                # relative. On garde le masque courant.
                break
            candidate = mask & (np.abs(residuals - med) <= self.k_mad * scale)
            if candidate.sum() == mask.sum():
                break  # aucune nouvelle quote rejetée : stable
            if candidate.sum() < self.min_inliers:
                break  # rejeter davantage rendrait le fit dégénéré
            mask = candidate
            intercept, slope = _solve_wls(x[mask], y[mask], weights[mask])
            n_iterations += 1

        return _build_result(x, y, weights, intercept, slope, mask, n_iterations)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(k_mad={self.k_mad}, max_passes={self.max_passes}, "
                f"min_inliers={self.min_inliers})")


class HuberIRLSRegression(LinearRegression):
    """IRLS avec poids de Huber : dépondération progressive des gros résidus.

    À chaque itération, l'échelle des résidus est ré-estimée par MAD, puis chaque
    point reçoit un poids multiplicatif min(1, delta / |r/scale|). Contrairement
    au rejet dur, aucun point n'est exclu (inlier_mask reste plein) : les
    outliers pèsent simplement de moins en moins. delta=1.345 donne ~95%
    d'efficacité sous bruit gaussien (valeur canonique de Huber).
    """

    def __init__(self, delta: float = 1.345, max_iterations: int = 50,
                 tol: float = 1e-10) -> None:
        if delta <= 0:
            raise ValueError("delta doit être strictement positif.")
        self.delta = delta
        self.max_iterations = max_iterations
        self.tol = tol

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> LinearFitResult:
        x, y, weights = (np.asarray(v, dtype=float) for v in (x, y, weights))
        _validate_inputs(x, y, weights)

        intercept, slope = _solve_wls(x, y, weights)
        n_iterations = 1

        for _ in range(self.max_iterations):
            residuals = y - (intercept + slope * x)
            _, scale = _robust_scale(residuals)
            if scale == 0.0:
                break  # fit quasi exact : les poids de Huber n'apportent plus rien

            u = np.abs(residuals) / scale
            huber_w = np.minimum(1.0, self.delta / np.maximum(u, 1e-300))
            new_intercept, new_slope = _solve_wls(x, y, weights * huber_w)
            n_iterations += 1

            if (abs(new_intercept - intercept) < self.tol
                    and abs(new_slope - slope) < self.tol):
                intercept, slope = new_intercept, new_slope
                break
            intercept, slope = new_intercept, new_slope

        mask = np.ones(len(x), dtype=bool)
        return _build_result(x, y, weights, intercept, slope, mask, n_iterations)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(delta={self.delta}, "
                f"max_iterations={self.max_iterations})")
