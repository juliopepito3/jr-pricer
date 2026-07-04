"""Swaption européenne sur un swap OIS."""
from __future__ import annotations

from datetime import date

from JR_PRICER.instruments.base import Instrument
from JR_PRICER.instruments.swap import OISSwap


class Swaption(Instrument):
    """Swaption européenne sur un `OISSwap`.

    `expiry_date` est la date d'exercice (= maturité de la swaption ; doit être
    antérieure ou égale au start du swap sous-jacent). `strike` est le taux fixe K.
    `swaption_type` vaut 'payer' (droit de payer K, recevoir le flottant) ou
    'receiver' (droit de recevoir K).
    """

    def __init__(self, underlying_swap: OISSwap, expiry_date: date,
                 strike: float, notional: float,
                 swaption_type: str) -> None:
        if swaption_type not in ('payer', 'receiver'):
            raise ValueError("swaption_type doit être 'payer' ou 'receiver'")
        if expiry_date > underlying_swap.start_date:
            raise ValueError("expiry_date doit être <= underlying_swap.start_date")

        super().__init__(expiry_date)
        self.underlying_swap = underlying_swap
        self.expiry_date = expiry_date
        self.strike = strike
        self.notional = notional
        self.swaption_type = swaption_type

    def __repr__(self) -> str:
        return (f"Swaption(expiry={self.expiry_date}, strike={self.strike}, "
                f"type={self.swaption_type})")
