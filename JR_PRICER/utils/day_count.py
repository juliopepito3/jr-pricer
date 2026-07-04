"""Conventions de décompte des jours (year fraction entre deux dates)."""
from __future__ import annotations

from datetime import date, timedelta


class DayCounter:
    """
    Convertit un intervalle entre deux dates en fraction d'année,
    selon une convention de marché.

    Conventions supportées :
        - 'act/365' : jours réels / 365 (actions, options)
        - 'act/360' : jours réels / 360 (marché monétaire)
        - '30/360'  : chaque mois = 30 jours (obligations)
    """

    def __init__(self, convention: str = 'act/365') -> None:
        convention = convention.lower()
        if convention not in ('act/365', 'act/360', '30/360'):
            raise ValueError(f"Convention inconnue : {convention}")
        self.convention = convention

    def __repr__(self) -> str:
        return f"DayCounter(convention='{self.convention}')"

    def year_fraction(self, d1: date, d2: date) -> float:
        """
        Retourne le temps entre d1 et d2 en années.
        Toujours positif — d1 doit être avant d2.
        """
        if d1 > d2:
            raise ValueError("d1 doit être antérieure à d2")

        if self.convention == 'act/365':
            return (d2 - d1).days / 365.0

        elif self.convention == 'act/360':
            return (d2 - d1).days / 360.0

        elif self.convention == '30/360':
            # Chaque mois compte pour 30 jours
            # Formule standard ISDA 30/360
            d1y, d1m, d1d = d1.year, d1.month, d1.day
            d2y, d2m, d2d = d2.year, d2.month, d2.day

            # Ajustements des jours selon la convention
            if d1d == 31:
                d1d = 30
            if d2d == 31 and d1d == 30:
                d2d = 30

            days = (d2y - d1y) * 360 + (d2m - d1m) * 30 + (d2d - d1d)
            return days / 360.0
        
    def to_date(self,reference_date: date, year_fraction: float)-> date:
        """
        Renvoie la date correspondant à la date de référence + une fraction d'années
        """
        if self.convention == 'act/365':
            return reference_date + timedelta(int(year_fraction * 365))
        elif self.convention == 'act/360':
            return reference_date + timedelta(int(year_fraction * 360))
        elif self.convention == '30/360':
            raise NotImplementedError()