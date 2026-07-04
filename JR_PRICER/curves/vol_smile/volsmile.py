"""Smile de volatilité implicite à maturité fixe."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np

from JR_PRICER.curves.base import Curve
from JR_PRICER.curves.interpolators_1D.base import Interpolator1D
from JR_PRICER.curves.vol_smile.moneyness_convention import MoneynessConvention, AbsoluteStrike


class VolSmile(Curve):
    """Smile de volatilité implicite à maturité fixe.

    L'axe interne `self.x` est exprimé dans la coordonnée de `moneyness_convention`
    (strike absolu pour AbsoluteStrike, log-moneyness pour LogMoneyness*, ...).
    `vol(K, ...)` convertit un strike absolu K dans cette coordonnée puis interpole.
    """

    def __init__(self, strikes: Sequence[float] | np.ndarray,
                 volatilities: Sequence[float] | np.ndarray, interpolator: Interpolator1D,
                 maturity: date, moneyness_convention: MoneynessConvention) -> None:

        if len(strikes) != len(volatilities):
            raise ValueError("Strikes and volatilities lists must have the same length.")

        # tri des piliers par strike croissant (l'interpolateur attend un x ordonné)
        strikes = np.asarray(strikes, dtype=float)
        volatilities = np.asarray(volatilities, dtype=float)
        order = np.argsort(strikes)

        super().__init__(strikes[order], volatilities[order], interpolator)
        self.maturity = maturity
        self.moneyness_convention = moneyness_convention

    def vol(self, K: float | np.ndarray, S: float = None, F: float = None,
            T: float = None) -> float | np.ndarray:
        """Volatilité implicite au strike absolu K (scalaire ou ndarray).

        Convertit K dans la coordonnée du smile via la convention de moneyness,
        puis interpole. S, F, T ne sont requis que pour les conventions qui en
        dépendent (LogMoneyness*, SimpleMoneyness) ; AbsoluteStrike les ignore.
        """
        if not isinstance(self.moneyness_convention, AbsoluteStrike) and F is None and S is None:
            raise ValueError(
                "Cette convention de moneyness nécessite le spot S et/ou le forward F."
            )
        x = self.moneyness_convention.to_moneyness(K, S, F, T)
        return self.evaluate(x)

    def __repr__(self) -> str:
        return f"VolSmile(maturity={self.maturity}, n_strikes={len(self.x)})"
