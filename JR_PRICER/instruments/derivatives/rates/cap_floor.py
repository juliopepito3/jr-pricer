"""Cap / Floor : strip de caplets (resp. floorlets)."""
from __future__ import annotations

from datetime import date

from JR_PRICER.instruments.derivatives.rates.base import RateDerivative
from JR_PRICER.instruments.derivatives.rates.caplet import Caplet
from JR_PRICER.utils.day_count import DayCounter
from JR_PRICER.utils.schedule import Schedule
from JR_PRICER.utils.frequency import Frequency
from JR_PRICER.utils.calendar import Calendar
from JR_PRICER.utils.business_day_convention import BusinessDayConvention


class Cap(RateDerivative):
    """Strip de caplets couvrant [start_date, end_date] à la fréquence donnée."""

    def __init__(self, start_date: date, end_date: date,
                 frequency: Frequency,
                 strike: float, notional: float,
                 day_count_convention: DayCounter,
                 calendar: Calendar,
                 business_day_convention: BusinessDayConvention,
                 option_type: str = 'cap',
                 include_first_caplet: bool = False) -> None:
        if option_type not in ('cap', 'floor'):
            raise ValueError("option_type doit être 'cap' ou 'floor'")
        super().__init__(start_date, end_date, strike, notional, day_count_convention)
        self.end_date = end_date
        self.frequency = frequency
        self.calendar = calendar
        self.business_day_convention = business_day_convention
        self.option_type = option_type
        # Convention de marché : un cap spot-start EXCLUT le premier caplet, dont
        # le fixing est déjà connu à l'inception (aucune optionalité). Mettre True
        # pour un cap forward-start dont toutes les périodes sont encore optionnelles.
        self.include_first_caplet = include_first_caplet

        self._caplets: list[Caplet] | None = None

    def get_caplets(self) -> list[Caplet]:
        """Décompose le cap/floor en caplets/floorlets (mémoïsé)."""
        if self._caplets is not None:
            return self._caplets

        schedule = Schedule(
            self.start_date, self.end_date,
            self.frequency, self.calendar,
            self.business_day_convention,
            generate_backwards=True,
        )
        period_end_dates = schedule.dates()
        all_dates = [self.start_date] + period_end_dates

        caplets = [
            Caplet(
                start_date=all_dates[i],
                end_date=all_dates[i + 1],
                strike=self.strike,
                notional=self.notional,
                day_count_convention=self.day_count_convention,
                option_type=self.option_type,
            )
            for i in range(len(all_dates) - 1)
        ]
        if not self.include_first_caplet:
            caplets = caplets[1:]

        self._caplets = caplets
        return self._caplets

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(start={self.start_date}, end={self.maturity_date}, "
                f"frequency={self.frequency.name}, strike={self.strike})")


class Floor(Cap):
    """Strip de floorlets (Cap avec option_type='floor')."""

    def __init__(self, start_date: date, end_date: date,
                 frequency: Frequency,
                 strike: float, notional: float,
                 day_count_convention: DayCounter,
                 calendar: Calendar,
                 business_day_convention: BusinessDayConvention,
                 include_first_caplet: bool = False) -> None:
        super().__init__(start_date, end_date, frequency, strike, notional,
                         day_count_convention, calendar, business_day_convention,
                         option_type='floor', include_first_caplet=include_first_caplet)
