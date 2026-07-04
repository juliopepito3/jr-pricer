"""Petits utilitaires numériques : correspondance de piliers, recherche de racine, log continu."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq


def index_of_close(x: float, xs: Sequence[float] | np.ndarray,
                   rel_tol: float = 1e-12, abs_tol: float = 1e-12) -> int | None:
    """Indice du premier élément de xs proche de x à tolérance près, sinon None.

    Évite les comparaisons d'égalité flottante exacte sur des grilles courtes
    (piliers de courbe, maturités calibrées).
    """
    for i, xi in enumerate(xs):
        if math.isclose(x, xi, rel_tol=rel_tol, abs_tol=abs_tol):
            return i
    return None


def find_root(f, x0: float, max_expand: int = 60, xtol: float = 1e-14) -> float:
    """Racine de f proche de x0, par bracketing géométrique puis Brent.

    f(x0) doit être fini. Élargit symétriquement un intervalle autour de x0
    jusqu'à un changement de signe, puis applique brentq. Adapté au bootstrap
    où x0 (formule fermée) est déjà une excellente approximation.
    """
    f0 = f(x0)
    if f0 == 0.0:
        return x0
    step = max(abs(x0) * 0.05, 1e-3)
    lo, hi = x0, x0
    for _ in range(max_expand):
        lo, hi = lo - step, hi + step
        f_lo, f_hi = f(lo), f(hi)
        if f_lo * f0 < 0:
            return brentq(f, lo, x0, xtol=xtol)
        if f_hi * f0 < 0:
            return brentq(f, x0, hi, xtol=xtol)
        step *= 1.6
    raise RuntimeError(f"find_root: aucun changement de signe autour de x0={x0}")


def continuous_log(z_prev, z_curr, log_prev):
    """
    Calcule ln(z_curr) de façon continue par rapport à log_prev = ln(z_prev). 
    Permet d'éviter les sauts de branche de la fonction logarithme complexe.
    """
    log_curr = np.log(z_curr)
    # Correction de branche : on ajuste si l'argument a sauté
    delta = np.imag(log_curr) - np.imag(log_prev)
    if delta > np.pi:
        log_curr -= 2j * np.pi
    elif delta < -np.pi:
        log_curr += 2j * np.pi
    return log_curr