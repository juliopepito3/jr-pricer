"""Interface de base des courbes forward (prix à terme de l'actif sous-jacent)."""
from __future__ import annotations

from abc import ABC, abstractmethod


class ForwardCurve(ABC):
    """Contrat des courbes forward : spot (T=0) et prix forward F(T)."""

    @property
    @abstractmethod
    def spot(self) -> float:
        """Prix spot de l'actif sous-jacent (F à T=0)."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, T: float) -> float:
        """Prix forward de l'actif sous-jacent à la maturité T (en year fraction)."""
        raise NotImplementedError
