"""Fonction de coût en erreur de volatilité implicite."""
from __future__ import annotations

from typing import TYPE_CHECKING, Type

import numpy as np

from JR_PRICER.calibration.cost_function.model.base import ModelCostFunction
from JR_PRICER.pricing.implied_vol_calculator import ImpliedVolCalculator

if TYPE_CHECKING:
    from JR_PRICER.pricing.model.base import Model
    from JR_PRICER.instruments.derivatives.equity.european_option import EuropeanOption
    from JR_PRICER.pricing.engine.base import Engine
    from JR_PRICER.curves.temporal.discount import DiscountCurve


class ImpliedVolErrorCost(ModelCostFunction):
    """Résidus = vols implicites du modèle − vols implicites de marché."""

    def __init__(self, instruments: list[EuropeanOption], market_implied_vols: np.ndarray,
                 engine: Engine, discount_curve: DiscountCurve, model_class: Type[Model]) -> None:
        super().__init__(instruments, market_implied_vols, engine, discount_curve, model_class)
        self.implied_vol_calculator = ImpliedVolCalculator(discount_curve)

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        model = self.model_class(*theta, discount_curve=self.discount_curve)

        model_prices = self.engine.price(self.instruments, model)
        # Inversion vectorisée de tous les instruments en un seul appel.
        model_implied_vols = self.implied_vol_calculator.calculate_implied_vol(
            instruments=self.instruments, market_prices=model_prices)

        return np.asarray(model_implied_vols) - np.asarray(self.objective)
