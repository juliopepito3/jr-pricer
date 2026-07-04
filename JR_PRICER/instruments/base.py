"""Interface de base des instruments financiers."""
from __future__ import annotations

from abc import ABC
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from JR_PRICER.curves.temporal.base import TemporalCurve
    from JR_PRICER.utils.day_count import DayCounter


class Instrument(ABC):
    """Classe de base d'un instrument : porte au minimum sa date de maturité.

    Les méthodes par défaut lèvent NotImplementedError ; chaque famille
    d'instruments n'implémente que celles qui la concernent (calibration de
    courbe via `implied_discount_factor`, simulation via `simulation_times` /
    `payoff`).
    """

    def __init__(self, maturity_date: date) -> None:
        self.maturity_date = maturity_date

    def implied_discount_factor(self, curve_so_far: TemporalCurve) -> float:
        """Discount factor implicite (instruments de calibration de courbe)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} n'est pas un instrument de calibration"
        )

    def simulation_times(self, reference_date: date, day_count_convention: DayCounter) -> list[float]:
        """Dates d'observation à simuler (year fraction) — produits simulables."""
        raise NotImplementedError(
            f"{self.__class__.__name__} n'est pas un instrument de simulation"
        )

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Payoff (non actualisé) sur un ensemble de trajectoires simulées.

        `paths` : np.ndarray de shape (n_paths, n_steps+1). Retourne un
        np.ndarray de shape (n_paths,) — un payoff par trajectoire (API vectorisée).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} n'est pas un instrument de simulation"
        )
