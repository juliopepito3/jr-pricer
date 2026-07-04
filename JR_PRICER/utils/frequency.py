"""Fréquences de paiement / d'observation (valeur = nombre de périodes par an)."""
from __future__ import annotations

from enum import Enum


class Frequency(Enum):
    """Fréquence annuelle ; la valeur est le nombre de périodes par an."""

    DAILY = 365
    WEEKLY = 52
    MONTHLY = 12
    QUARTERLY = 4
    SEMIANNUAL = 2
    ANNUAL = 1