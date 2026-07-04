"""Option asiatique : payoff sur la moyenne (arithmétique ou géométrique) du sous-jacent."""
from __future__ import annotations

from datetime import date
from enum import Enum

import numpy as np

from JR_PRICER.instruments.derivatives.equity.base import EquityDerivative, OptionType
from JR_PRICER.market_data.underlying import Underlying
from JR_PRICER.utils.day_count import DayCounter
from JR_PRICER.utils.schedule import Schedule
from JR_PRICER.utils.calendar import Calendar, TARGET
from JR_PRICER.utils.business_day_convention import BusinessDayConvention
from JR_PRICER.utils.frequency import Frequency


def _geometric_mean(x: np.ndarray, axis: int | None = None) -> float | np.ndarray:
    """Moyenne géométrique exp(moyenne(ln x)), éventuellement le long d'un axe."""
    return np.exp(np.mean(np.log(x), axis=axis))


class AveragingType(Enum):
    """Type de moyenne d'une option asiatique."""
    ARITHMETIC = 'arithmetic'
    GEOMETRIC = 'geometric'


class AsianOption(EquityDerivative):
    """Option asiatique de payoff max(±(moyenne − K), 0)·notionnel.

    La moyenne est observée sur un calendrier (averaging_start → maturity).
    """

    def __init__(self, underlying: Underlying,
                 K: float,
                 start_date: date,
                 averaging_start: date,
                 maturity_date: date,
                 frequency: Frequency,
                 option_type: OptionType,
                 averaging_type: AveragingType = AveragingType.ARITHMETIC,
                 notional: float = 1.0,
                 business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
                 calendar: Calendar | None = None) -> None:

        super().__init__(maturity_date, start_date, notional, underlying)
        self.K = K
        self.averaging_start = averaging_start
        self.frequency = frequency
        self.option_type = option_type
        self.averaging_type = averaging_type
        self.business_day_convention = business_day_convention
        self.calendar = calendar if calendar is not None else TARGET()

        # include_start_date=True : averaging_start est une date d'observation à
        # part entière (la moyenne court de averaging_start à maturity).
        self.schedule = Schedule(self.averaging_start, self.maturity_date, self.frequency,
                                 self.calendar, self.business_day_convention,
                                 include_start_date=True)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Payoff vectorisé : paths (n_paths, n_steps+1) → (n_paths,).

        La moyenne porte sur les colonnes correspondant aux dates d'observation
        (les n dernières, cf. simulation_times)."""
        n = len(self.schedule.dates())
        window = paths[:, -n:]  # n dernières colonnes = dates d'observation
        if self.averaging_type == AveragingType.ARITHMETIC:
            average = window.mean(axis=1)
        else:
            average = _geometric_mean(window, axis=1)

        if self.option_type == OptionType.CALL:
            return np.maximum(average - self.K, 0.0) * self.notional
        else:
            return np.maximum(self.K - average, 0.0) * self.notional

    def simulation_times(self, reference_date: date, day_count_convention: DayCounter) -> list[float]:
        """Dates d'observation de la moyenne (year fraction)."""
        return [day_count_convention.year_fraction(reference_date, d) for d in self.schedule.dates()]

    def __repr__(self) -> str:
        return (f"AsianOption(K={self.K}, maturity={self.maturity_date}, "
                f"type={self.option_type.name}, averaging={self.averaging_type.name})")
