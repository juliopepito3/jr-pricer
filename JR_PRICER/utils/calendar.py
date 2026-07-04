"""Calendriers de jours ouvrés et ajustement de dates."""
from __future__ import annotations

from datetime import date, timedelta

from JR_PRICER.utils.business_day_convention import BusinessDayConvention


class Calendar:
    """Calendrier de base : weekends ouvrés, aucun jour férié, ajustement de dates."""

    def is_weekend(self, d: date) -> bool:
        return d.weekday() >= 5

    def is_business_day(self, d: date) -> bool:
        return not self.is_weekend(d) and not self.is_holiday(d)

    def is_holiday(self, d: date) -> bool:
        # Par défaut, pas de jours fériés
        return False

    def adjust(self, d: date, business_day_convention: BusinessDayConvention) -> date:
        """Ajuste une date non ouvrée selon la convention de jour ouvré."""
        if business_day_convention == BusinessDayConvention.FOLLOWING:
            while not self.is_business_day(d):
                d += timedelta(days=1)
            return d

        elif business_day_convention == BusinessDayConvention.PRECEDING:
            while not self.is_business_day(d):
                d -= timedelta(days=1)
            return d

        elif business_day_convention == BusinessDayConvention.MODIFIED_FOLLOWING:
            # Suivant, sauf si on change de mois → on bascule en précédent.
            adjusted_date = d
            while not self.is_business_day(adjusted_date):
                adjusted_date += timedelta(days=1)
            if adjusted_date.month != d.month:
                adjusted_date = d
                while not self.is_business_day(adjusted_date):
                    adjusted_date -= timedelta(days=1)
            return adjusted_date

        elif business_day_convention == BusinessDayConvention.MODIFIED_PRECEDING:
            # Précédent, sauf si on change de mois → on bascule en suivant.
            adjusted_date = d
            while not self.is_business_day(adjusted_date):
                adjusted_date -= timedelta(days=1)
            if adjusted_date.month != d.month:
                adjusted_date = d
                while not self.is_business_day(adjusted_date):
                    adjusted_date += timedelta(days=1)
            return adjusted_date

        elif business_day_convention == BusinessDayConvention.UNADJUSTED:
            return d

        else:
            raise ValueError(f"Business day convention inconnue : {business_day_convention}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class TARGET(Calendar):
    """Calendrier TARGET (zone euro).

    - Weekends : samedi et dimanche.
    - Jours fériés : 1er janvier, Vendredi saint, Lundi de Pâques, 1er mai, 25 décembre.
    """

    def __init__(self) -> None:
        super().__init__()
        # Jours fériés fixes (mois, jour) ; les fériés mobiles (Pâques) sont calculés.
        self.fixed_holidays = {
            (1, 1),    # 1er janvier
            (5, 1),    # 1er mai
            (12, 25),  # 25 décembre
        }

    @staticmethod
    def easter_date(year: int) -> date:
        """Date de Pâques (algorithme de Meeus/Jones/Butcher)."""
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return date(year, month, day)

    def is_holiday(self, d: date) -> bool:
        # Jours fériés fixes
        if (d.month, d.day) in self.fixed_holidays:
            return True
        # Jours fériés mobiles : Vendredi saint (Pâques − 2j), Lundi de Pâques (Pâques + 1j)
        easter = self.easter_date(d.year)
        if d == easter - timedelta(days=2):
            return True
        if d == easter + timedelta(days=1):
            return True
        return False
