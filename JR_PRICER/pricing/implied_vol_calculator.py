"""Calcul vectorisé de la volatilité implicite Black-Scholes d'options européennes."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from JR_PRICER.curves.temporal.discount import DiscountCurve
from JR_PRICER.instruments.derivatives.equity.european_option import EuropeanOption
from JR_PRICER.instruments.derivatives.equity.base import OptionType
from JR_PRICER.pricing.formulas import implied_vol_newton


class ImpliedVolCalculator:
    """Inverse les prix d'options européennes en vols implicites Black-Scholes."""

    def __init__(self, discount_curve: DiscountCurve) -> None:
        self.discount_curve = discount_curve

    def calculate_implied_vol(self, instruments: list[EuropeanOption],
                              market_prices: Sequence[float] | np.ndarray) -> np.ndarray:
        """Volatilités implicites Black-Scholes pour une liste d'options européennes.

        Inversion vectorisée (Newton sur la vega analytique, cf. pricing.formulas)
        de tous les instruments en parallèle — remplace l'ancienne boucle brentq.
        """
        dc = self.discount_curve.day_count_convention
        ref = self.discount_curve.reference_date

        T = np.array([dc.year_fraction(ref, instr.maturity_date) for instr in instruments])
        df = np.array([self.discount_curve.discount(t) for t in T])
        F = np.array([instr.underlying.forward_curve.forward(t)
                      for instr, t in zip(instruments, T)])
        K = np.array([instr.K for instr in instruments])
        notional = np.array([instr.notional for instr in instruments])
        is_call = np.array([instr.option_type == OptionType.CALL for instr in instruments])

        # Le prix marché inclut actualisation et notionnel → on revient au prix forward.
        target_forward = np.asarray(market_prices, dtype=float) / (df * notional)

        return implied_vol_newton(target_forward, F, K, T, is_call)

    def __repr__(self) -> str:
        return f"ImpliedVolCalculator(discount={self.discount_curve!r})"
