"""Génération d'échéanciers de dates (cash flows, fixings) ajustés au calendrier."""
from __future__ import annotations

from datetime import date

from dateutil.relativedelta import relativedelta

from JR_PRICER.utils.calendar import Calendar
from JR_PRICER.utils.business_day_convention import BusinessDayConvention
from JR_PRICER.utils.frequency import Frequency


class Schedule:
    """Échéancier de dates entre start_date et end_date, à la fréquence donnée.

    Génère (en avant ou en arrière) les dates de période, ajustées au calendrier
    et à la convention de jour ouvré. start_date n'est incluse que si demandé.
    """

    def __init__(self, start_date: date, end_date: date, frequency: Frequency,
                 calendar: Calendar, business_day_convention: BusinessDayConvention,
                 generate_backwards: bool = True, include_start_date: bool = False) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.calendar = calendar
        self.business_day_convention = business_day_convention
        self.generate_backwards = generate_backwards
        # Par défaut start_date est exclue (OISSwap/Cap la préfixent eux-mêmes).
        # Mettre True pour les produits qui observent aussi la date de départ
        # (ex. AsianOption : averaging_start est une date d'observation).
        self.include_start_date = include_start_date

    def _period_step(self) -> relativedelta:
        """Pas d'une période selon la fréquence.

        WEEKLY et DAILY doivent se compter en semaines / jours : `12 // 52` (ou
        `12 // 365`) vaut 0, ce qui donnerait un `relativedelta(months=0)` — un pas
        NUL, donc une boucle infinie dans `dates()`. Les autres fréquences (mensuelle
        à annuelle) divisent bien 12 mois.
        """
        f = self.frequency
        if f == Frequency.WEEKLY:
            return relativedelta(weeks=1)
        if f == Frequency.DAILY:
            return relativedelta(days=1)
        return relativedelta(months=12 // f.value)

    def dates(self) -> list[date]:
        """Liste des dates de période ajustées (jusqu'à end_date incluse).

        Génération par pas de `frequency`, en avant ou en arrière selon
        `generate_backwards` ; start_date n'est incluse que si include_start_date=True.
        """
        step = self._period_step()
        if self.generate_backwards:

            dates_list = [self.calendar.adjust(self.end_date, self.business_day_convention)]
            current_date = self.end_date
            while current_date > self.start_date:
                current_date = current_date - step
                if current_date > self.start_date:
                    adjusted_date = self.calendar.adjust(current_date, self.business_day_convention)
                    dates_list.append(adjusted_date)

            if self.include_start_date:
                dates_list.append(self.calendar.adjust(self.start_date, self.business_day_convention))

            return sorted(dates_list)

        else :
            dates_list = [self.calendar.adjust(self.start_date, self.business_day_convention)] if self.include_start_date else []
            current_date = self.start_date
            while current_date < self.end_date:
                current_date = current_date + step
                if current_date <= self.end_date:
                    adjusted_date = self.calendar.adjust(current_date, self.business_day_convention)
                    dates_list.append(adjusted_date)

            return dates_list

    def __repr__(self) -> str:
        return (f"Schedule(start={self.start_date}, end={self.end_date}, "
                f"frequency={self.frequency.name})")