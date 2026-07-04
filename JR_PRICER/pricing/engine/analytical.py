"""Moteur de pricing analytique (formules fermées du modèle)."""
from __future__ import annotations

from JR_PRICER.pricing.engine.base import Engine
from JR_PRICER.pricing.model.base import Model
from JR_PRICER.instruments.base import Instrument


class AnalyticalEngine(Engine):
    """Moteur déléguant à `model.analytic_price` (pricing en forme fermée)."""

    def price(self, instruments: list[Instrument], model: Model) -> list[float]:
        return [model.analytic_price(instr) for instr in instruments]

    def __repr__(self) -> str:
        return "AnalyticalEngine()"