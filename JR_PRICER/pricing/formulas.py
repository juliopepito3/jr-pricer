"""Formules Black-Scholes/Black-76 vectorisées (prix forward, vega, implied vol).

Toutes les fonctions acceptent des scalaires ou des np.ndarray et opèrent en
forward (non actualisé, sans notionnel). L'actualisation et le notionnel sont
appliqués par l'appelant.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black_forward_price(F: float | np.ndarray, K: float | np.ndarray, T: float | np.ndarray,
                        sigma: float | np.ndarray, is_call: bool | np.ndarray = True) -> np.ndarray:
    """Prix forward (non actualisé) d'une option européenne en modèle de Black.

    F·N(d1) − K·N(d2) (call), K·N(−d2) − F·N(−d1) (put). Vectorisé.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    sqrtT = np.sqrt(T)
    vol = sigma * sqrtT
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * vol**2) / vol
        d2 = d1 - vol
    call = F * norm.cdf(d1) - K * norm.cdf(d2)
    put = K * norm.cdf(-d2) - F * norm.cdf(-d1)
    # σ→0 : valeur intrinsèque
    intrinsic_call = np.maximum(F - K, 0.0)
    intrinsic_put = np.maximum(K - F, 0.0)
    call = np.where(vol > 0, call, intrinsic_call)
    put = np.where(vol > 0, put, intrinsic_put)
    return np.where(is_call, call, put)


def black_forward_vega(F: float | np.ndarray, K: float | np.ndarray, T: float | np.ndarray,
                       sigma: float | np.ndarray) -> np.ndarray:
    """Vega forward (∂prix_forward/∂σ) — identique call/put. Vectorisé."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    sqrtT = np.sqrt(T)
    vol = sigma * sqrtT
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * vol**2) / vol
    return np.where(vol > 0, F * norm.pdf(d1) * sqrtT, 0.0)


def implied_vol_newton(target_forward_price: float | np.ndarray, F: float | np.ndarray,
                       K: float | np.ndarray, T: float | np.ndarray,
                       is_call: bool | np.ndarray = True, tol: float = 1e-10,
                       max_iter: int = 100, sigma_min: float = 1e-8,
                       sigma_max: float = 10.0) -> np.ndarray:
    """Vol implicite (Black) par Newton vectorisé sur la vega analytique.

    `target_forward_price` : prix forward cible (marché actualisé / (df·notionnel)).
    Tous les arguments-vecteurs sont alignés. Retourne un np.ndarray de vols.

    Initialisation de Brenner-Subrahmanyam (ATM) puis Newton ; clamp aux bornes.
    Pour les rares entrées non convergées (vega ~ 0), repli sur une bissection.
    """
    target = np.asarray(target_forward_price, dtype=float)
    F = np.broadcast_to(np.asarray(F, dtype=float), target.shape).astype(float)
    K = np.broadcast_to(np.asarray(K, dtype=float), target.shape).astype(float)
    T = np.broadcast_to(np.asarray(T, dtype=float), target.shape).astype(float)
    is_call = np.broadcast_to(np.asarray(is_call), target.shape)

    # Brenner-Subrahmanyam : σ ≈ sqrt(2π/T)·prix/F (exact à l'ATM, bon point de départ)
    sigma = np.sqrt(2 * np.pi / T) * np.clip(target / F, 1e-6, None)
    sigma = np.clip(sigma, sigma_min, sigma_max)

    for _ in range(max_iter):
        price = black_forward_price(F, K, T, sigma, is_call)
        vega = black_forward_vega(F, K, T, sigma)
        diff = price - target
        # division sûre : pas de pas là où la vega est ~0 (évite inf/nan)
        step = np.divide(diff, vega, out=np.zeros_like(diff), where=vega > 1e-12)
        sigma = np.clip(sigma - step, sigma_min, sigma_max)
        if np.all(np.abs(step) < tol):
            break

    # Repli bissection pour les entrées non convergées (vega trop faible).
    resid = np.abs(black_forward_price(F, K, T, sigma, is_call) - target)
    bad = resid > 1e-6 * np.maximum(F, 1.0)
    if np.any(bad):
        lo = np.full(sigma.shape, sigma_min)
        hi = np.full(sigma.shape, sigma_max)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            pm = black_forward_price(F, K, T, mid, is_call)
            up = pm < target
            lo = np.where(bad & up, mid, lo)
            hi = np.where(bad & ~up, mid, hi)
        sigma = np.where(bad, 0.5 * (lo + hi), sigma)

    return sigma
