from __future__ import annotations

from JR_PRICER.pricing.model.base import Model
from JR_PRICER.pricing.engine.monte_carlo import MCPaths
from JR_PRICER.surfaces.vol_surface.volsurface import FlatVol, VolSurface, VolProvider
from JR_PRICER.curves.temporal.discount import DiscountCurve
from JR_PRICER.curves.forward.base import ForwardCurve
from JR_PRICER.instruments.derivatives.equity.base import EquityDerivative, OptionType

from JR_PRICER.instruments.derivatives.equity.european_option import EuropeanOption
from JR_PRICER.instruments.derivatives.equity.digital_option import DigitalOption
from JR_PRICER.instruments.derivatives.equity.asian_option import AsianOption
from JR_PRICER.instruments.derivatives.equity.asian_option import AveragingType

import numpy as np
from scipy.stats import norm


class BlackScholesModel(Model):
    """Modèle Black-Scholes lognormal (vol plate) : MC et pricing analytique equity.

    Requiert un `vol_provider` de type `FlatVol` ; le drift suit la courbe forward.
    """

    def __init__(self, discount_curve: DiscountCurve) -> None:
        super().__init__(discount_curve)

    def discount(self, T: float) -> float:
        return self.discount_curve.discount(T)

    # Méthodes pour Monte Carlo ---------------------------------------------------

    def simulate(self, instrument: EquityDerivative, n_paths: int,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        """Simule (n_paths,) payoffs actualisés ; vol plate, drift = ratio de forwards."""
        return self.simulate_paths(instrument, n_paths, rng).discounted_payoff

    def simulate_paths(self, instrument: EquityDerivative, n_paths: int,
                       rng: np.random.Generator | None = None) -> MCPaths:
        """Simule les trajectoires et renvoie un `MCPaths` (vol plate, drift = ratio de forwards)."""
        if rng is None:
            rng = np.random.default_rng()
        if not isinstance(instrument, EquityDerivative):
            raise TypeError(
                f"BlackScholesModel ne peut pas pricer {type(instrument).__name__}. "
                f"Utilisez un instrument EquityDerivative."
            )
        times = instrument.simulation_times(self.reference_date, self.day_count_convention)
        forward_curve = instrument.underlying.forward_curve
        vol_provider = instrument.underlying.vol_provider

        if not isinstance(vol_provider,FlatVol):
            raise TypeError(
                f"BlackScholesModel nécessite un vol_provider de type FlatVol. "
                f"Le vol_provider pour le produit {type(instrument).__name__}  est de type {type(vol_provider).__name__}."
            )

        times_full = [0.0] + times
        dt = np.diff(times_full)
        n_steps = len(times_full) - 1

        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = forward_curve.spot

        for i in range(n_steps):
            t      = times_full[i]
            t_next = times_full[i + 1]

            F_t      = forward_curve.forward(t)      if t > 0 else forward_curve.spot
            F_t_next = forward_curve.forward(t_next)
            growth   = F_t_next / F_t

            sigma = vol_provider.sigma(K=None, T=t)
            Z = rng.standard_normal(n_paths)

            S[:, i + 1] = S[:, i] * growth * np.exp(-0.5 * sigma**2 * dt[i] + sigma * np.sqrt(dt[i]) * Z)

        df = self.discount(times[-1])
        return MCPaths(
            times=np.asarray(times_full),
            spot_paths=S,
            discounted_payoff=df * instrument.payoff(S),
            discount_factor=df,
            model_name=type(self).__name__,
            instrument_label=type(instrument).__name__,
        )

    # Méthodes pour PDE -----------------------------------------------------------

    def drift(self, S: float, t: float, forward_curve: ForwardCurve) -> float:
        """Dérive μ·S avec un taux effectif lu sur la pente locale de la courbe forward."""
        eps = 1e-6
        F_t     = forward_curve.forward(t)
        F_t_eps = forward_curve.forward(t + eps)
        r_eff = (F_t_eps - F_t) / (eps * F_t)  # taux instantané ≈ ∂ln F/∂t
        return r_eff * S

    def diffusion(self, S: float, t: float, vol_provider: FlatVol) -> float:
        """Diffusion σ·S (vol plate)."""
        return vol_provider.sigma(K=None, T=t) * S

    # Pricing analytique ----------------------------------------------------------

    def analytic_price(self, instrument: EquityDerivative) -> float:
        """Prix fermé Black-Scholes (European, Digital, Asian géométrique)."""
        if not isinstance(instrument, EquityDerivative):
            raise TypeError(
                f"BlackScholesModel ne peut pas pricer {type(instrument).__name__}. "
                f"Utilisez un instrument EquityDerivative."
            )

        fwd = instrument.underlying.forward_curve
        vol = instrument.underlying.vol_provider

        if not isinstance(vol, FlatVol):
            raise TypeError(
                f"BlackScholesModel nécessite un vol_provider de type FlatVol pour le pricing analytique. "
                f"Le vol_provider pour le produit {type(instrument).__name__}  est de type {type(vol).__name__}."
            )

        if isinstance(instrument, EuropeanOption):

            T     = self.day_count_convention.year_fraction(self.reference_date, instrument.maturity_date)
            df    = self.discount(T)
            F     = fwd.forward(T)
            sigma = vol.sigma(instrument.K, T)

            d1 = (np.log(F / instrument.K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if instrument.option_type == OptionType.CALL:
                price = df * (F * norm.cdf(d1) - instrument.K * norm.cdf(d2))
            else:
                price = df * (instrument.K * norm.cdf(-d2) - F * norm.cdf(-d1))

            return price * instrument.notional

        elif isinstance(instrument, DigitalOption):

            T     = self.day_count_convention.year_fraction(self.reference_date, instrument.maturity_date)
            df    = self.discount(T)
            F     = fwd.forward(T)
            sigma = vol.sigma(instrument.K, T)

            d1 = (np.log(F / instrument.K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if instrument.option_type == OptionType.CALL:
                if instrument.digital_type == 'cash':
                    price = df * norm.cdf(d2)
                elif instrument.digital_type == 'asset':
                    price = df * F * norm.cdf(d1)
                else:
                    raise ValueError("digital_type must be 'cash' or 'asset'")
            elif instrument.option_type == OptionType.PUT:
                if instrument.digital_type == 'cash':
                    price = df * norm.cdf(-d2)
                elif instrument.digital_type == 'asset':
                    price = df * F * norm.cdf(-d1)
                else:
                    raise ValueError("digital_type must be 'cash' or 'asset'")

            return price * instrument.notional

        elif isinstance(instrument, AsianOption):
            if instrument.averaging_type == AveragingType.ARITHMETIC:
                raise NotImplementedError("Analytic pricing for arithmetic Asian options is not implemented")

            fixing_times = instrument.simulation_times(self.reference_date, self.day_count_convention)
            n     = len(fixing_times)
            T     = fixing_times[-1]
            t_bar = np.mean(fixing_times)
            sigma = vol.sigma(instrument.K, T)

            # Variance de la moyenne géométrique : (σ²/n²)·Σ_ij min(t_i, t_j)
            # (covariance des log-rendements browniens aux dates de fixing).
            sigma_G_sq = (sigma**2 / n**2) * sum(
                min(fixing_times[i], fixing_times[j])
                for i in range(n)
                for j in range(n)
            )
            sigma_G = np.sqrt(sigma_G_sq)

            log_forwards = [np.log(fwd.forward(t)) for t in fixing_times]
            mu_G = np.mean(log_forwards) - 0.5 * sigma**2 * t_bar

            d1 = (mu_G + sigma_G_sq - np.log(instrument.K)) / sigma_G
            d2 = d1 - sigma_G

            df = self.discount_curve.discount(T)

            if instrument.option_type == OptionType.CALL:
                price = df * (np.exp(mu_G + 0.5 * sigma_G_sq) * norm.cdf(d1) - instrument.K * norm.cdf(d2))
            else:
                price = df * (instrument.K * norm.cdf(-d2) - np.exp(mu_G + 0.5 * sigma_G_sq) * norm.cdf(-d1))

            return price * instrument.notional

        else:
            raise NotImplementedError(
                f"Analytic pricing not implemented for {type(instrument).__name__}"
            )

    def __repr__(self) -> str:
        return f"BlackScholesModel(discount={self.discount_curve!r})"
