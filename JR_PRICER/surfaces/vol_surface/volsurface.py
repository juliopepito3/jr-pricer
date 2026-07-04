"""Fournisseurs de volatilité : vol plate et surface de vol (smiles + interpolation 2D)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import numpy as np

from JR_PRICER.curves.vol_smile.volsmile import VolSmile
from JR_PRICER.curves.vol_smile.moneyness_convention import AbsoluteStrike
from JR_PRICER.curves.forward.base import ForwardCurve
from JR_PRICER.utils.day_count import DayCounter
from JR_PRICER.surfaces.vol_surface.interpolators_2D.base import (
    Interpolator2D, GridInterpolator2D, LayeredInterpolator2D, GlobalInterpolator2D,
)


class VolProvider(ABC):
    """Interface fournissant une volatilité implicite σ(K, T)."""

    @abstractmethod
    def sigma(self, K: float | np.ndarray, T: float) -> float | np.ndarray:
        """Volatilité implicite au strike K (scalaire ou ndarray) et maturité T."""
        raise NotImplementedError("sigma must be implemented by subclasses")


class FlatVol(VolProvider):
    """Volatilité constante (indépendante du strike et de la maturité)."""

    def __init__(self, sigma: float) -> None:
        self.sigma_value = sigma

    def sigma(self, K: float | np.ndarray, T: float) -> float:
        return self.sigma_value

    def __repr__(self) -> str:
        return f"FlatVol(sigma={self.sigma_value})"


class VolSurface(VolProvider):
    """Surface de volatilité : un jeu de smiles par maturité + interpolateur 2D.

    Les smiles doivent partager la même convention de moneyness. `vol(K, T)`
    convertit K en coordonnée puis délègue à l'interpolateur (qui renvoie déjà σ).
    """

    def __init__(self, smiles: list[VolSmile], reference_date: date,
                 day_count_convention: DayCounter, interpolator: Interpolator2D,
                 forward_curve: ForwardCurve | None = None) -> None:

        # Tous les smiles doivent partager la même convention de moneyness.
        # On compare les TYPES : deux instances AbsoluteStrike() distinctes sont
        # équivalentes (les conventions n'ont pas d'__eq__).
        convention_types = {type(smile.moneyness_convention) for smile in smiles}
        if len(convention_types) != 1:
            raise ValueError("All smiles must have the same moneyness convention.")
        self.moneyness_convention = smiles[0].moneyness_convention

        self.smiles = sorted(smiles, key=lambda s: s.maturity)
        self.reference_date = reference_date
        self.day_count_convention = day_count_convention
        self.interpolator = interpolator
        # Requis seulement si la convention de moneyness dépend de S ou F.
        self._forward_curve = forward_curve

        # Maturités en year fraction (axe T de la surface).
        self.maturities = [self.day_count_convention.year_fraction(self.reference_date, smile.maturity)
                           for smile in smiles]

    def total_variance_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Grille (strikes, maturités, variance totale w = σ²·T) pour fitter l'interpolateur."""
        x = np.unique(np.concatenate([smile.x for smile in self.smiles]))  # strikes
        y = np.array(self.maturities)                                       # maturités
        z = np.zeros((len(x), len(y)))
        for j, smile in enumerate(self.smiles):
            # smile.evaluate est vectorisé : une passe sur toute la grille de strikes
            z[:, j] = np.asarray(smile.evaluate(x)) ** 2 * y[j]
        return x, y, z

    def calibrate_interpolator(self) -> None:
        """Ajuste l'interpolateur 2D selon sa famille (grille, par couches, ou global)."""
        if isinstance(self.interpolator, GridInterpolator2D):
            x, y, z = self.total_variance_grid()
            self.interpolator.fit(x, y, z)
        elif isinstance(self.interpolator, LayeredInterpolator2D):
            # Calibration par maturité (ex. SVI) : reçoit directement les smiles.
            self.interpolator.fit(self.maturities, self.smiles)
        elif isinstance(self.interpolator, GlobalInterpolator2D):
            x, y, z = self.total_variance_grid()
            self.interpolator.fit(x, y, z)

    def vol(self, K: float | np.ndarray, T: float) -> float | np.ndarray:
        """Volatilité implicite σ au strike absolu K (scalaire OU ndarray) et maturité T.

        Convertit K dans la coordonnée de l'interpolateur via la convention de
        moneyness, puis interpole. Contrat : interpolate() retourne déjà σ
        (cf. Interpolator2D) — pas de reconversion depuis la variance totale.
        """
        if T <= 0:
            return 0.0 if np.ndim(K) == 0 else np.zeros(np.shape(K))
        if isinstance(self.moneyness_convention, AbsoluteStrike):
            x = K
        else:
            if self._forward_curve is None:
                raise ValueError(
                    "Cette convention de moneyness nécessite une forward_curve sur la VolSurface."
                )
            S = self._forward_curve.spot
            F = self._forward_curve.forward(T)
            x = self.moneyness_convention.to_moneyness(K, S, F, T)
        return self.interpolator.interpolate(x, T)

    def sigma(self, K: float | np.ndarray, T: float) -> float | np.ndarray:
        # Interface VolProvider : strike absolu. Délègue à vol().
        return self.vol(K, T)

    def __repr__(self) -> str:
        return f"VolSurface(n_smiles={len(self.smiles)}, interp={type(self.interpolator).__name__})"
