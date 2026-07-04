"""Conventions de moneyness : conversions strike <-> coordonnée normalisée du smile."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy import log, exp


class MoneynessConvention(ABC):
    """Stratégie de conversion entre un strike absolu K et la coordonnée du smile.

    S (spot), F (forward) et T (maturité) ne sont utilisés que par les conventions
    qui en dépendent ; les autres les ignorent (peuvent valoir None).
    """

    @abstractmethod
    def to_moneyness(self, K: float | np.ndarray, S: float | None, F: float | None,
                     T: float | None) -> float | np.ndarray:
        """Convertit un strike (scalaire ou ndarray) en coordonnée normalisée."""
        raise NotImplementedError("to_moneyness method must be implemented by subclasses")

    @abstractmethod
    def to_strike(self, x: float | np.ndarray, S: float | None, F: float | None,
                  T: float | None) -> float | np.ndarray:
        """Convertit une coordonnée normalisée (scalaire ou ndarray) en strike."""
        raise NotImplementedError("to_strike method must be implemented by subclasses")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class AbsoluteStrike(MoneynessConvention):
    """Coordonnée = strike absolu (aucune transformation)."""

    def to_moneyness(self, K, S, F, T):
        return K

    def to_strike(self, x, S, F, T):
        return x


class LogMoneynessForward(MoneynessConvention):
    """Coordonnée = log-moneyness forward : ln(K / F)."""

    def to_moneyness(self, K, S, F, T):
        return log(K / F)

    def to_strike(self, x, S, F, T):
        return F * exp(x)


class LogMoneynessSpot(MoneynessConvention):
    """Coordonnée = log-moneyness spot : ln(K / S)."""

    def to_moneyness(self, K, S, F, T):
        return log(K / S)

    def to_strike(self, x, S, F, T):
        return S * exp(x)


class SimpleMoneyness(MoneynessConvention):
    """Coordonnée = moneyness simple : K / S."""

    def to_moneyness(self, K, S, F, T):
        return K / S

    def to_strike(self, x, S, F, T):
        return x * S
