"""Option à barrière (knock-in / knock-out, up / down) à monitoring discret."""
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


class BarrierType(Enum):
    """Activation (IN) ou désactivation (OUT) au franchissement de la barrière."""
    IN = 1
    OUT = 2


class BarrierDirection(Enum):
    """Sens de franchissement de la barrière."""
    UP = 1
    DOWN = 2


class BarrierOption(EquityDerivative):
    """Option vanille conditionnée au franchissement d'une barrière (monitoring discret)."""

    def __init__(self, underlying: Underlying,
                 K: float,
                 start_date: date,
                 maturity_date: date,
                 frequency: Frequency,
                 option_type: OptionType,
                 barrier_type: BarrierType,
                 barrier_direction: BarrierDirection,
                 barrier_level: float,
                 notional: float = 1.0,
                 business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
                 calendar: Calendar | None = None) -> None:

        if not isinstance(option_type, OptionType):
            raise TypeError(f"option_type must be an OptionType, got {type(option_type)}")

        super().__init__(maturity_date, start_date, notional, underlying)
        self.K = K
        self.frequency = frequency
        self.option_type = option_type
        self.barrier_type = barrier_type
        self.barrier_direction = barrier_direction
        self.barrier_level = barrier_level
        self.business_day_convention = business_day_convention
        self.calendar = calendar if calendar is not None else TARGET()

        self.schedule = Schedule(self.start_date, self.maturity_date, self.frequency,
                                 self.calendar, self.business_day_convention)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Payoff vectorisé : paths (n_paths, n_steps+1) → (n_paths,).

        Monitoring discret sur les colonnes 1: (on exclut le spot initial)."""
        monitoring = paths[:, 1:]
        if self.barrier_direction == BarrierDirection.UP:
            breached = np.any(monitoring > self.barrier_level, axis=1)
        else:
            breached = np.any(monitoring < self.barrier_level, axis=1)

        S_T = paths[:, -1]
        if self.option_type == OptionType.CALL:
            vanilla = np.maximum(S_T - self.K, 0.0) * self.notional
        else:
            vanilla = np.maximum(self.K - S_T, 0.0) * self.notional

        # knock-in : payoff si barrière touchée ; knock-out : payoff sinon.
        alive = breached if self.barrier_type == BarrierType.IN else ~breached
        return np.where(alive, vanilla, 0.0)

    def simulation_times(self, reference_date: date, day_count_convention: DayCounter) -> list[float]:
        """Dates de monitoring de la barrière (year fraction)."""
        return [day_count_convention.year_fraction(reference_date, d) for d in self.schedule.dates()]

    def __repr__(self) -> str:
        return (f"BarrierOption(K={self.K}, barrier={self.barrier_level}, "
                f"{self.barrier_direction.name}-{self.barrier_type.name}, "
                f"type={self.option_type.name}, maturity={self.maturity_date})")
