"""Option européenne vanille (call/put)."""
from __future__ import annotations

from datetime import date

import numpy as np

from JR_PRICER.instruments.derivatives.equity.base import EquityDerivative, OptionType
from JR_PRICER.market_data.underlying import Underlying
from JR_PRICER.utils.day_count import DayCounter


class EuropeanOption(EquityDerivative):
    """Call/put européen de payoff terminal max(±(S_T − K), 0)·notionnel."""

    def __init__(self, underlying: Underlying, K: float,
                 start_date: date, maturity_date: date,
                 option_type: OptionType, notional: float = 1.0) -> None:

        if not isinstance(option_type, OptionType):
            raise TypeError(f"option_type must be an OptionType, got {type(option_type)}")

        super().__init__(maturity_date, start_date, notional, underlying)
        self.K = K
        self.option_type = option_type

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Payoff vectorisé : paths (n_paths, n_steps+1) → (n_paths,)."""
        S_T = paths[:, -1]  # valeur terminale de chaque trajectoire
        if self.option_type == OptionType.CALL:
            return np.maximum(S_T - self.K, 0.0) * self.notional
        else:
            return np.maximum(self.K - S_T, 0.0) * self.notional

    def simulation_times(self, reference_date: date, day_count_convention: DayCounter) -> list[float]:
        """Une seule date d'observation : la maturité (year fraction)."""
        return [day_count_convention.year_fraction(reference_date, self.maturity_date)]

    def __repr__(self) -> str:
        return (f"EuropeanOption(K={self.K}, maturity={self.maturity_date}, "
                f"type={self.option_type.name}, notional={self.notional})")
