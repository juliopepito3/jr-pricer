"""Structure par terme de variance totale ATM θ_T (= σ_ATM(T)²·T) pour SSVI."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from JR_PRICER.curves.base import Curve
from JR_PRICER.curves.interpolators_1D.base import Interpolator1D


class ThetaCurve(Curve):
    """Structure par terme de variance totale ATM `θ_T = σ_ATM(T)²·T`.

    Wrapper minimal autour d'une interpolation 1D (`Curve`) qui **valide la stricte
    monotonie à la construction** : c'est le cœur de la garantie d'absence
    d'arbitrage calendaire de SSVI (`∂_T θ_T > 0`). Avec un interpolateur
    préservant la monotonie (ex. `LogLinearInterpolator1D`, qui interpole `ln(θ)`
    linéairement et extrapole à plat le forward de bord), des piliers strictement
    croissants ⇒ `θ_T` strictement croissante sur tout le domaine.
    """

    def __init__(self, maturities: Sequence[float] | np.ndarray,
                 thetas: Sequence[float] | np.ndarray, interpolator: Interpolator1D) -> None:
        x = np.asarray(maturities, dtype=float)
        y = np.asarray(thetas, dtype=float)
        if x.ndim != 1 or len(x) < 2:
            raise ValueError("ThetaCurve : au moins deux piliers (maturités) requis.")
        if np.any(np.diff(x) <= 0):
            raise ValueError("ThetaCurve : maturités non strictement croissantes.")
        if np.any(y <= 0):
            raise ValueError("ThetaCurve : la variance totale ATM θ_T doit être strictement positive.")
        if np.any(np.diff(y) <= 0):
            raise ValueError(
                "ThetaCurve : θ_T non strictement croissante (arbitrage calendaire dans les ATM)."
            )
        super().__init__(x, y, interpolator)

    @classmethod
    def from_smiles(cls, maturities: Sequence[float], smiles: Sequence,
                    forward_curve, interpolator: Interpolator1D) -> "ThetaCurve":
        """Construit la structure depuis les ATM du marché : `θ_i = σ_ATM(T_i)²·T_i`.

        `σ_ATM` est la vol du smile au forward `F(T_i)` (smiles en strike absolu,
        convention partagée avec SVI/Dupire)."""
        thetas = [float(smile.evaluate(forward_curve.forward(T))) ** 2 * T
                  for T, smile in zip(maturities, smiles)]
        return cls(maturities, thetas, interpolator)
