"""Fonction de coût en erreur de prix."""
from __future__ import annotations

from typing import TYPE_CHECKING, Type

import numpy as np

from JR_PRICER.calibration.cost_function.model.base import ModelCostFunction

if TYPE_CHECKING:
    from JR_PRICER.pricing.model.base import Model
    from JR_PRICER.instruments.derivatives.equity.european_option import EuropeanOption
    from JR_PRICER.pricing.engine.base import Engine
    from JR_PRICER.curves.temporal.discount import DiscountCurve


class PriceErrorCost(ModelCostFunction):
    """Résidus = prix du modèle − prix de marché."""

    def __init__(self, instruments: list[EuropeanOption], market_prices: np.ndarray,
                 engine: Engine, discount_curve: DiscountCurve, model_class: Type[Model]) -> None:
        super().__init__(instruments, market_prices, engine, discount_curve, model_class)

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        model = self.model_class(*theta, discount_curve=self.discount_curve)
        model_prices = np.array(self.engine.price(self.instruments, model))
        return model_prices - self.objective
