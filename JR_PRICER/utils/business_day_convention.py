"""Conventions de jour ouvré pour l'ajustement des dates."""
from __future__ import annotations

from enum import Enum


class BusinessDayConvention(Enum):
    """Règle d'ajustement d'une date tombant un jour non ouvré."""
    FOLLOWING = "FOLLOWING"
    MODIFIED_FOLLOWING = "MODIFIED_FOLLOWING"
    PRECEDING = "PRECEDING"
    MODIFIED_PRECEDING = "MODIFIED_PRECEDING"
    UNADJUSTED = "UNADJUSTED"
