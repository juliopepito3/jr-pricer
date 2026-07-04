"""Interface de base des moteurs de pricing."""
from __future__ import annotations

from abc import ABC, abstractmethod

from JR_PRICER.pricing.model.base import Model
from JR_PRICER.instruments.base import Instrument


class Engine(ABC):
    """Moteur de pricing : associe une méthode numérique à un modèle."""

    @abstractmethod
    def price(self, instruments: list[Instrument], model: Model) -> list[float]:
        """Prix d'une liste d'instruments pour un modèle donné (un prix par instrument)."""
        raise NotImplementedError("price method must be implemented by subclasses")

