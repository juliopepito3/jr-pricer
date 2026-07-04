"""Modèle de Heston (volatilité stochastique) — fonction caractéristique pour Fourier."""
from __future__ import annotations

import numpy as np

from JR_PRICER.pricing.model.base import Model
from JR_PRICER.pricing.engine.monte_carlo import MCPaths
from JR_PRICER.pricing.model.discretization.CIR.base import CIRDiscretizationScheme
from JR_PRICER.curves.temporal.discount import DiscountCurve
from JR_PRICER.instruments.derivatives.equity.base import EquityDerivative


class HestonModel(Model):
    """Modèle de Heston — fonction caractéristique pour le pricing par Fourier.

    Les paramètres-modèle viennent en premier (puis discount_curve) pour respecter
    le contrat de calibration générique : model_class(*theta, discount_curve=...).
    Pricing par transformée de Fourier uniquement (pas de Monte-Carlo).
    """

    def __init__(self, kappa: float, theta: float, sigma_v: float, rho: float, v0: float,
                discount_curve: DiscountCurve) -> None:
        super().__init__(discount_curve)
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.v0 = v0

        self.discretization_scheme = None
        self.max_dt = None

    def __repr__(self) -> str:
        return (f"HestonModel(kappa={self.kappa}, theta={self.theta}, sigma_v={self.sigma_v}, "
                f"rho={self.rho}, v0={self.v0})")

    def discount(self, T: float | np.ndarray) -> float | np.ndarray:
        return self.discount_curve.discount(T)
    
    def discretize(self, scheme: CIRDiscretizationScheme, max_dt: float) -> "HestonModel":
        new_model = HestonModel(self.kappa, self.theta, self.sigma_v, self.rho, self.v0, self.discount_curve)
        new_model.discretization_scheme = scheme
        new_model.max_dt = max_dt
        return new_model

    def _densify_times(self,structural_times: list[float], max_dt: float) -> list[float]:
        """Insère des points intermédiaires pour respecter un pas <= max_dt,
        sans jamais perdre les dates structurelles (fixings, maturité)."""
        full_times = [0.0]
        prev_t = 0.0
        for t in structural_times:
            interval = t - prev_t
            n_sub_steps = max(1, int(np.ceil(interval / max_dt)))
            sub_times = np.linspace(prev_t, t, n_sub_steps + 1)[1:]  # exclut prev_t, déjà dans full_times
            full_times.extend(sub_times.tolist())
            prev_t = t
        return full_times

    def simulate(self, instrument: EquityDerivative, n_paths: int,
                rng: np.random.Generator | None = None) -> np.ndarray:
        """Simule (n_paths,) payoffs actualisés sous Heston, schéma self.discretization_scheme."""
        return self.simulate_paths(instrument, n_paths, rng).discounted_payoff

    def simulate_paths(self, instrument: EquityDerivative, n_paths: int,
                       rng: np.random.Generator | None = None) -> MCPaths:
        """Simule sous-jacent ET variance sous Heston, et renvoie un `MCPaths`."""
        if rng is None:
            rng = np.random.default_rng()

        if not isinstance(instrument, EquityDerivative):
            raise TypeError(
                f"HestonModel ne peut pas pricer {type(instrument).__name__}. "
                f"Utilisez un instrument EquityDerivative."
            )

        if self.discretization_scheme is None:
            raise ValueError(
                "Aucun discretization_scheme attaché. Appeler .discretize(scheme) avant simulate()."
            )

        structural_times = instrument.simulation_times(self.reference_date, self.day_count_convention)
        times_full = self._densify_times(structural_times, self.max_dt)
        forward_curve = instrument.underlying.forward_curve

        dt = np.diff(times_full)
        n_steps = len(times_full) - 1

        # --- Bruit corrélé, généré une fois pour tout l'horizon ---
        Z1 = rng.standard_normal(size=(n_steps, n_paths))
        Z2 = rng.standard_normal(size=(n_steps, n_paths))
        Zv = Z1
        Zs = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

        # --- Trajectoires ---
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = forward_curve.spot
        v[:, 0] = self.v0

        for i in range(n_steps):
            t, t_next = times_full[i], times_full[i + 1]
            F_t = forward_curve.forward(t) if t > 0 else forward_curve.spot
            F_t_next = forward_curve.forward(t_next)
            growth = F_t_next / F_t

            v_next, v_floored = self.discretization_scheme.step(
                v[:, i], dt[i], self.kappa, self.theta, self.sigma_v, Zv[i]
            )

            S[:, i + 1] = S[:, i] * growth * np.exp(
                -0.5 * v_floored * dt[i] + np.sqrt(v_floored * dt[i]) * Zs[i]
            )
            v[:, i + 1] = v_next

        df = self.discount(times_full[-1])
        return MCPaths(
            times=np.asarray(times_full),
            spot_paths=S,
            discounted_payoff=df * instrument.payoff(S),
            discount_factor=df,
            variance_paths=v,
            model_name=type(self).__name__,
            instrument_label=type(instrument).__name__,
        )

    def characteristic_function(self, u_grid: np.ndarray, T: float) -> np.ndarray:
            
        """
        Fonction caractéristique du modèle de Heston pour une grille de valeurs u et une maturité T, sous la mesure forward.
        Le logarithme complexe est déroulé (np.unwrap) pour éviter les sauts de branche, de façon entièrement vectorisée.
        """

        # Etape 1 : tout vectorisé
        d = np.sqrt((self.kappa - 1j * self.rho * self.sigma_v * u_grid)**2 
                    + self.sigma_v**2 * (1j * u_grid + u_grid**2))
        g = (self.kappa - 1j * self.rho * self.sigma_v * u_grid - d) / \
            (self.kappa - 1j * self.rho * self.sigma_v * u_grid + d)
        
        z = (1 - g * np.exp(-d * T)) / (1 - g)  # argument du ln, vectorisé

        # Etape 2 : ln continu, entièrement vectorisé.
        # La correction de branche séquentielle équivaut à un déroulement de phase
        # (np.unwrap) sur la partie imaginaire du log principal : sur une grille
        # ordonnée et assez fine, les deux sont identiques (sauts < 2π).
        log_z = np.log(z)
        log_z = log_z.real + 1j * np.unwrap(log_z.imag)

        # Etape 3 : assemblage vectorisé
        A = (self.kappa * self.theta / self.sigma_v**2) * \
            ((self.kappa - 1j * self.rho * self.sigma_v * u_grid - d) * T - 2 * log_z)
        B = (self.v0 / self.sigma_v**2) * \
            (self.kappa - 1j * self.rho * self.sigma_v * u_grid - d) * \
            (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    
        return np.exp(A + B)
    





