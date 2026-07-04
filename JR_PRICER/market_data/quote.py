"""Valeur de marché observable et mutable (spot, taux, vol...)."""
from __future__ import annotations


class Quote:
    """Encapsule une valeur de marché observable et mutable.

    On n'utilise pas un simple float : la valeur peut être mise à jour en un seul
    endroit (`update`) et tous les objets qui la référencent voient le changement.
    """

    def __init__(self, value: float, name: str = "") -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("La valeur doit être un nombre")
        self._value = float(value)
        self.name = name  # optionnel, utile pour déboguer

    def value(self) -> float:
        """Valeur courante."""
        return self._value

    def update(self, new_value: float) -> None:
        """Met à jour la valeur (propagée à tous les objets référençant ce Quote)."""
        self._value = float(new_value)

    def __repr__(self) -> str:
        return f"Quote(name='{self.name}', value={self._value})"