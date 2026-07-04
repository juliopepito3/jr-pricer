"""Base des fonctions de coût « marché vs modèle » (branche calibration de modèle)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Type

import numpy as np

from JR_PRICER.calibration.cost_function.base import CostFunction

if TYPE_CHECKING:
    # Imports d'annotation uniquement : au runtime, `instruments`, `engine`,
    # `discount_curve` et `model_class` sont injectés (jamais construits par leur
    # nom ici). Les garder sous TYPE_CHECKING évite de tirer `pricing` à l'import
    # de ce module et préserve l'acyclicité calibration → pricing.
    from JR_PRICER.instruments.derivatives.equity.european_option import EuropeanOption
    from JR_PRICER.pricing.engine.base import Engine
    from JR_PRICER.curves.temporal.discount import DiscountCurve
    from JR_PRICER.pricing.model.base import Model


class ModelCostFunction(CostFunction):
    """Fonction de coût pour la calibration d'un modèle de pricing.

    Spécialise la `CostFunction` générique au cas « marché vs modèle » : on
    reconstruit un `Model` à partir de chaque jeu de paramètres `theta`, on price
    des `instruments` via un `engine`, et on compare à un `objective` de marché
    (prix ou vols). Les sous-classes implémentent `__call__(theta)` (résidus).

    `build(theta)` reconstruit le modèle calibré — c'est ce que renvoie
    `calibrate` pour ce type de coût.
    """

    def __init__(self, instruments: list[EuropeanOption], objective: np.ndarray, engine: Engine,
                 discount_curve: DiscountCurve, model_class: Type[Model],
                 bounds=None, constraints=None, label: str = "") -> None:
        super().__init__(bounds=bounds, constraints=constraints, label=label)
        self.instruments = instruments
        self.objective = objective
        self.engine = engine
        self.discount_curve = discount_curve
        self.model_class = model_class

    def build(self, theta: np.ndarray) -> "Model":
        """Modèle reconstruit aux paramètres optimaux."""
        return self.model_class(*theta, discount_curve=self.discount_curve)

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement __call__.")

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(n_instruments={len(self.instruments)}, "
                f"model={getattr(self.model_class, '__name__', self.model_class)})")
