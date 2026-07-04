from __future__ import annotations

import numpy as np
from scipy.stats import norm

from JR_PRICER.pricing.model.base import Model
from JR_PRICER.surfaces.vol_surface.volsurface import FlatVol
from JR_PRICER.curves.temporal.discount import DiscountCurve
from JR_PRICER.instruments.derivatives.rates.base import RateDerivative

from JR_PRICER.instruments.derivatives.rates.caplet import Caplet
from JR_PRICER.instruments.derivatives.rates.cap_floor import Cap
from JR_PRICER.instruments.derivatives.rates.swaption import Swaption


class Black76Model(Model):
    """Modèle Black-76 (taux lognormaux) — pricing MC et analytique des caps/floors et swaptions."""

    def __init__(self, discount_curve: DiscountCurve, vol_provider: FlatVol) -> None:
        super().__init__(discount_curve)
        self.vol_provider = vol_provider

    def drift(self, _S, _t, _forward_curve):
        raise NotImplementedError("Black76Model ne supporte pas le pricing PDE")

    def diffusion(self, _S, _t, _vol_provider):
        raise NotImplementedError("Black76Model ne supporte pas le pricing PDE")

    def discount(self, T: float) -> float:
        return self.discount_curve.discount(T)

    def __repr__(self) -> str:
        return f"Black76Model(vol={self.vol_provider!r}, discount={self.discount_curve!r})"

    # Monte Carlo --------------------------------------------------------------

    def simulate(self, instrument, n_paths: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Retourne (n_paths,) payoffs actualisés pour l'instrument donné."""
        if rng is None:
            rng = np.random.default_rng()
        if not isinstance(instrument, (RateDerivative, Swaption)):
            raise TypeError(
                f"Black76Model ne peut pas pricer {type(instrument).__name__}. "
                f"Utilisez un instrument RateDerivative ou Swaption."
            )
        if isinstance(instrument, Cap):
            total = np.zeros(n_paths)
            for c in instrument.get_caplets():
                total += self._mc_caplet(c, n_paths, rng)
            return total
        if isinstance(instrument, Caplet):
            return self._mc_caplet(instrument, n_paths, rng)
        if isinstance(instrument, Swaption):
            return self._mc_swaption(instrument, n_paths, rng)
        raise NotImplementedError(
            f"simulate_for non implémenté pour {type(instrument).__name__}"
        )

    def _mc_caplet(self, caplet: Caplet, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        dc, ref = self.day_count_convention, self.reference_date
        T_s = dc.year_fraction(ref, caplet.start_date)
        T_e = dc.year_fraction(ref, caplet.end_date)
        tau = caplet.accrual
        df_s = self.discount_curve.discount(T_s)
        df_e = self.discount_curve.discount(T_e)
        F0 = (df_s / df_e - 1.0) / tau
        K = caplet.strike

        if T_s <= 0:
            intrinsic = max(F0 - K, 0) if caplet.option_type == 'cap' else max(K - F0, 0)
            return np.full(n_paths, intrinsic * tau * caplet.notional * df_e)

        sigma = self.vol_provider.sigma(K, T_s)
        Z = rng.standard_normal(n_paths)
        F_T = F0 * np.exp(-0.5 * sigma**2 * T_s + sigma * np.sqrt(T_s) * Z)

        if caplet.option_type == 'cap':
            payoffs = np.maximum(F_T - K, 0)
        else:
            payoffs = np.maximum(K - F_T, 0)
        return payoffs * tau * caplet.notional * df_e

    def _mc_swaption(self, swaption: Swaption, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        dc, ref = self.day_count_convention, self.reference_date
        swap = swaption.underlying_swap
        T_exp = dc.year_fraction(ref, swaption.expiry_date)

        all_dates = [swap.adjusted_start_date] + swap.fixed_leg_dates
        periods = list(zip(all_dates[:-1], all_dates[1:]))
        annuity = sum(
            swap.fixed_leg.day_count_convention.year_fraction(d_s, d_e)
            * self.discount_curve.discount(dc.year_fraction(ref, d_e))
            for d_s, d_e in periods
        )
        df_start = self.discount_curve.discount(dc.year_fraction(ref, swap.start_date))
        df_end   = self.discount_curve.discount(dc.year_fraction(ref, swap.maturity_date))
        S0 = (df_start - df_end) / annuity
        K = swaption.strike
        sigma = self.vol_provider.sigma(K, T_exp)

        Z = rng.standard_normal(n_paths)
        S_T = S0 * np.exp(-0.5 * sigma**2 * T_exp + sigma * np.sqrt(T_exp) * Z)

        if swaption.swaption_type == 'payer':
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        return payoffs * annuity * swaption.notional

    # Pricing analytique -------------------------------------------------------

    def analytic_price(self, instrument) -> float:
        if not isinstance(instrument, (RateDerivative, Swaption)):
            raise TypeError(
                f"Black76Model ne peut pas pricer {type(instrument).__name__}. "
                f"Utilisez un instrument RateDerivative ou Swaption."
            )
        if isinstance(instrument, Cap):
            return self._price_cap(instrument)
        if isinstance(instrument, Caplet):
            return self._price_caplet(instrument)
        if isinstance(instrument, Swaption):
            return self._price_swaption(instrument)
        raise NotImplementedError(
            f"Black76Model: pricing non implémenté pour {type(instrument).__name__}"
        )

    def _price_caplet(self, caplet: Caplet) -> float:
        """Black 76 pour un caplet ou floorlet individuel."""
        dc = self.day_count_convention
        ref = self.reference_date

        T_s = dc.year_fraction(ref, caplet.start_date)
        T_e = dc.year_fraction(ref, caplet.end_date)
        tau = caplet.accrual

        df_s = self.discount_curve.discount(T_s)
        df_e = self.discount_curve.discount(T_e)

        F = (df_s / df_e - 1.0) / tau
        K = caplet.strike

        # Caplet déjà fixé : on retourne la valeur intrinsèque actualisée
        if T_s <= 0:
            if caplet.option_type == 'cap':
                return df_e * tau * max(F - K, 0.0) * caplet.notional
            else:
                return df_e * tau * max(K - F, 0.0) * caplet.notional

        sigma = self.vol_provider.sigma(K, T_s)

        d1 = (np.log(F / K) + 0.5 * sigma**2 * T_s) / (sigma * np.sqrt(T_s))
        d2 = d1 - sigma * np.sqrt(T_s)

        if caplet.option_type == 'cap':
            price = df_e * tau * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            price = df_e * tau * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

        return price * caplet.notional

    def _price_cap(self, cap: Cap) -> float:
        """Prix d'un cap/floor = somme des caplets/floorlets."""
        return sum(self._price_caplet(c) for c in cap.get_caplets())

    def _price_swaption(self, swaption: Swaption) -> float:
        """Black 76 pour une swaption européenne payer ou receiver."""
        dc = self.day_count_convention
        ref = self.reference_date
        swap = swaption.underlying_swap

        T_exp = dc.year_fraction(ref, swaption.expiry_date)

        # Annuité A = Σ τ_i · df(T_i) sur les périodes du fixed leg
        all_dates = [swap.adjusted_start_date] + swap.fixed_leg_dates
        periods = list(zip(all_dates[:-1], all_dates[1:]))
        annuity = sum(
            swap.fixed_leg.day_count_convention.year_fraction(d_s, d_e)
            * self.discount_curve.discount(dc.year_fraction(ref, d_e))
            for d_s, d_e in periods
        )

        T_swap_start = dc.year_fraction(ref, swap.start_date)
        T_swap_end = dc.year_fraction(ref, swap.maturity_date)
        df_start = self.discount_curve.discount(T_swap_start)
        df_end = self.discount_curve.discount(T_swap_end)

        S = (df_start - df_end) / annuity
        K = swaption.strike
        sigma = self.vol_provider.sigma(K, T_exp)

        d1 = (np.log(S / K) + 0.5 * sigma**2 * T_exp) / (sigma * np.sqrt(T_exp))
        d2 = d1 - sigma * np.sqrt(T_exp)

        if swaption.swaption_type == 'payer':
            price = annuity * (S * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            price = annuity * (K * norm.cdf(-d2) - S * norm.cdf(-d1))

        return price * swaption.notional
