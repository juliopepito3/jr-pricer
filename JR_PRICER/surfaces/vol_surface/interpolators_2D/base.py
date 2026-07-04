"""Interpolateurs 2D de surfaces de volatilité : interface et familles de base."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Interpolator2D(ABC):
    """Interpolateur 2D pour surfaces de volatilité.

    Contrat :
    - `interpolate(K, T)` retourne la volatilité implicite σ (annualisée), quelle
      que soit la représentation interne (la grille fittée est en variance totale
      w = σ²·T pour préserver l'arbitrage calendaire, mais la sortie est σ).
    - `dw_dT`, `dw_dK`, `d2w_dK2` retournent les dérivées de la variance totale
      w = σ²·T (et non de σ), telles qu'utilisées par la formule de Dupire/Gatheral.

    Les méthodes acceptent un strike scalaire ou un np.ndarray (T reste scalaire).
    """

    def __init__(self) -> None:
        self._fitted = False

    @abstractmethod
    def interpolate(self, x: float | np.ndarray, y: float) -> float | np.ndarray: ...

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Interpolator must be fitted before evaluation.")

    def dz_dx(self, x: float | np.ndarray, y: float, h: float = 1e-4) -> float | np.ndarray:
        """∂σ/∂K par différence finie centrée (pas relatif au strike)."""
        # x peut être un scalaire ou un np.ndarray (y reste scalaire) → np.where
        # remplace le test booléen scalaire pour rester vectorisé.
        self._check_fitted()
        ax = np.abs(x)
        hx = np.where(ax > 1e-10, h * ax, h)
        return (self.interpolate(x + hx, y) - self.interpolate(x - hx, y)) / (2 * hx)

    def dz_dy(self, x: float | np.ndarray, y: float, h: float = 1e-4) -> float | np.ndarray:
        """∂σ/∂T par différence finie centrée (pas relatif à la maturité)."""
        self._check_fitted()
        ay = np.abs(y)
        hy = h * ay if ay > 1e-10 else h
        return (self.interpolate(x, y + hy) - self.interpolate(x, y - hy)) / (2 * hy)

    def d2z_dx2(self, x: float | np.ndarray, y: float, h: float = 1e-4) -> float | np.ndarray:
        """∂²σ/∂K² par différence finie centrée d'ordre 2."""
        self._check_fitted()
        ax = np.abs(x)
        hx = np.where(ax > 1e-10, h * ax, h)
        return (self.interpolate(x + hx, y) - 2 * self.interpolate(x, y) + self.interpolate(x - hx, y)) / hx**2

    def dw_dT(self, x: float | np.ndarray, y: float) -> float | np.ndarray:
        """∂w/∂T avec w = σ²·T : σ² + 2·σ·T·∂σ/∂T (chaîne sur la variance totale)."""
        self._check_fitted()
        sigma = self.interpolate(x, y)
        dsigma_dT = self.dz_dy(x, y)
        return sigma**2 + 2 * sigma * y * dsigma_dT

    def dw_dK(self, x: float | np.ndarray, y: float) -> float | np.ndarray:
        """∂w/∂K avec w = σ²·T : 2·σ·T·∂σ/∂K."""
        self._check_fitted()
        sigma = self.interpolate(x, y)
        dsigma_dK = self.dz_dx(x, y)
        return 2 * sigma * y * dsigma_dK

    def d2w_dK2(self, x: float | np.ndarray, y: float) -> float | np.ndarray:
        """∂²w/∂K² avec w = σ²·T : 2·T·((∂σ/∂K)² + σ·∂²σ/∂K²)."""
        self._check_fitted()
        sigma = self.interpolate(x, y)
        dsigma_dK = self.dz_dx(x, y)
        d2sigma_dK2 = self.d2z_dx2(x, y)
        return 2 * y * (dsigma_dK**2 + sigma * d2sigma_dK2)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(fitted={self._fitted})"


class GridInterpolator2D(Interpolator2D):
    """Famille d'interpolateurs fittés sur une grille (strikes, maturités, variance)."""

    def __init__(self) -> None:
        super().__init__()

    def fit(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        """Ajuste l'interpolateur sur la grille de variance totale z(x, y)."""
        self._fit(x, y, z)
        self._fitted = True


class LayeredInterpolator2D(Interpolator2D):
    """Famille d'interpolateurs calibrés à partir des smiles : par maturité (ex. SVI)
    ou globalement sur toute la surface (ex. SSVI). Reçoit `(maturités, smiles)`."""

    def __init__(self) -> None:
        super().__init__()

    def fit(self, maturities: list[float], smiles: list) -> None:
        """Calibre un jeu de paramètres par maturité à partir des smiles."""
        self._fit(maturities, smiles)
        self._fitted = True


class GlobalInterpolator2D(Interpolator2D):
    """Famille d'interpolateurs globaux sur l'ensemble de la grille (ex. RBF)."""

    def __init__(self) -> None:
        super().__init__()

    def fit(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        """Ajuste l'interpolateur global sur la grille de variance totale z(x, y)."""
        self._fit(x, y, z)
        self._fitted = True
