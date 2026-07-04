"""Option digitale (binaire) : paiement cash-or-nothing ou asset-or-nothing."""
from __future__ import annotations

from datetime import date

import numpy as np

from JR_PRICER.instruments.derivatives.equity.base import EquityDerivative, OptionType
from JR_PRICER.market_data.underlying import Underlying
from JR_PRICER.utils.day_count import DayCounter


class DigitalOption(EquityDerivative):
    """Option binaire : verse un montant fixe ('cash') ou S_T ('asset') si dans la monnaie."""

    def __init__(self, underlying: Underlying, K: float,
                 start_date: date, maturity_date: date,
                 option_type: OptionType, notional: float = 1.0,
                 digital_type: str = 'cash') -> None:

        if not isinstance(option_type, OptionType):
            raise TypeError(f"option_type must be an OptionType, got {type(option_type)}")
        if digital_type not in ['cash', 'asset']:
            raise ValueError("digital_type must be 'cash' or 'asset'")

        super().__init__(maturity_date, start_date, notional, underlying)
        self.K = K
        self.option_type = option_type
        self.digital_type = digital_type

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Payoff vectorisé : paths (n_paths, n_steps+1) → (n_paths,)."""
        S_T = paths[:, -1]
        in_money = S_T > self.K if self.option_type == OptionType.CALL else S_T < self.K
        # cash : montant fixe ; asset : on livre le sous-jacent (S_T)
        amount = self.notional if self.digital_type == 'cash' else self.notional * S_T
        return np.where(in_money, amount, 0.0)

    def simulation_times(self, reference_date: date, day_count_convention: DayCounter) -> list[float]:
        """Une seule date d'observation : la maturité (year fraction)."""
        return [day_count_convention.year_fraction(reference_date, self.maturity_date)]

    def __repr__(self) -> str:
        return (f"DigitalOption(K={self.K}, maturity={self.maturity_date}, "
                f"type={self.option_type.name}, digital={self.digital_type})")
