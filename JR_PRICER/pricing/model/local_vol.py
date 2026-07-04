"""Modèle à volatilité locale de Dupire (vol locale dérivée d'une surface de vol implicite)."""
from __future__ import annotations

import numpy as np

from JR_PRICER.pricing.model.base import Model
from JR_PRICER.pricing.engine.monte_carlo import MCPaths
from JR_PRICER.curves.temporal.discount import DiscountCurve
from JR_PRICER.surfaces.vol_surface.volsurface import VolSurface
from JR_PRICER.curves.forward.base import ForwardCurve
from JR_PRICER.curves.vol_smile.moneyness_convention import AbsoluteStrike
from JR_PRICER.instruments.derivatives.equity.base import EquityDerivative


class LocalVolModel(Model):
    """Modèle à vol locale de Dupire, simulé par Euler sur une grille de dates.

    La vol locale σ_loc(K, t) est extraite d'une `VolSurface` (en AbsoluteStrike)
    via la formule de Gatheral. Pricing Monte-Carlo uniquement.
    """

    # Plancher temporel (~1h en années) : la formule de Gatheral dégénère en t=0
    # car w = σ²·t → 0 (termes 1/w). lim_{t→0} σ_loc(K,t) = σ_impl(K,0⁺), donc
    # flooorer t introduit une erreur O(T_FLOOR) négligeable devant le pas de MC.
    T_FLOOR = 1e-4

    def __init__(self, discount_curve: DiscountCurve) -> None:
        # Model.__init__ initialise discount_curve, reference_date et day_count_convention.
        super().__init__(discount_curve)

    def __repr__(self) -> str:
        return f"LocalVolModel(discount={self.discount_curve!r})"

    def sigma_loc(self, S: float | np.ndarray, t: float, vol_surface: VolSurface,
                  forward_curve: ForwardCurve) -> float | np.ndarray:
        """Volatilité locale de Dupire/Gatheral en (K=S, t).

        Vectorisé : S peut être un scalaire ou un np.ndarray (vecteur de spots à
        un même pas de temps t scalaire). Retourne la même forme que S.
        """
        if not isinstance(vol_surface.moneyness_convention, AbsoluteStrike):
            raise ValueError("LocalVolModel requiert une VolSurface en AbsoluteStrike.")

        t = max(t, self.T_FLOOR)

        # La formule de Gatheral est en log-moneyness y = ln(K/F). Les interpolateurs
        # exposent les dérivées de w par rapport au strike K ; on convertit via
        # y = ln K - ln F :  ∂w/∂y = K·∂w/∂K,  ∂²w/∂y² = K²·∂²w/∂K² + K·∂w/∂K.
        # Chemin chaud du MC : si l'interpolateur expose `w_and_derivs` (ex. SVI),
        # on récupère w ET ses trois dérivées en une seule passe (log/sqrt/bracketing
        # mutualisés) ; sinon on retombe sur les appels séparés.
        interp = vol_surface.interpolator
        fused = getattr(interp, "w_and_derivs", None)
        if fused is not None:
            w, dw_dK, d2w_dK2, dw_dT = fused(S, t)
        else:
            w = vol_surface.sigma(S, t) ** 2 * t
            dw_dK   = interp.dw_dK(S, t)
            d2w_dK2 = interp.d2w_dK2(S, t)
            dw_dT   = interp.dw_dT(S, t)

        k = np.log(S / forward_curve.forward(t))

        dw_dy   = S * dw_dK
        d2w_dy2 = S ** 2 * d2w_dK2 + S * dw_dK

        # Le numérateur de Gatheral est ∂_T w à log-moneyness y FIXE. Les interpolateurs
        # fournissent ∂_T w à STRIKE K fixe (dw_dT) ; on corrige via
        #   ∂_T w|_y = ∂_T w|_K + μ·∂_y w,   μ = d ln F/dt  (= r − q pour un forward continu),
        # car à y fixe le strike suit le forward. Sans ce terme la vol locale est biaisée dès
        # que taux/dividendes ≠ 0 (biais ∝ |r−q|·skew ; nul sur surface plate).
        mu = self._forward_log_drift(t, forward_curve)
        numerator = dw_dT + mu * dw_dy
        denominator = (1
                    - k / w * dw_dy
                    + 0.25 * (-0.25 - 1/w + k**2/w**2) * dw_dy**2
                    + 0.5 * d2w_dy2)

        if np.any(denominator <= 0):
            idx = int(np.argmin(np.atleast_1d(denominator)))
            K_bad = float(np.atleast_1d(S)[idx])
            raise ValueError(f"Arbitrage détecté en (K={K_bad:.2f}, T={t:.2f}) : dénominateur négatif.")

        return np.sqrt(numerator / denominator)

    @staticmethod
    def _forward_log_drift(t: float, forward_curve: ForwardCurve, h: float = 1e-5) -> float:
        """Dérive log-forward μ(t) = d ln F/dt (= r − q), par différence finie centrée.

        La `ForwardCurve` n'expose que F(t) — d'où l'estimation par 2 évaluations (t est
        déjà planchéré à `T_FLOOR`, donc t ± h·t reste strictement positif)."""
        ht = h * t if t > 1e-10 else h
        return (np.log(forward_curve.forward(t + ht)) - np.log(forward_curve.forward(t - ht))) / (2 * ht)

    def simulate(self, instrument : EquityDerivative , n_paths: int, rng: np.random.Generator | None = None) -> np.ndarray :
        return self.simulate_paths(instrument, n_paths, rng).discounted_payoff

    def simulate_paths(self, instrument: EquityDerivative, n_paths: int,
                       rng: np.random.Generator | None = None) -> MCPaths:
        """Simule les trajectoires (Euler sur σ_loc de Dupire) et renvoie un `MCPaths`."""
        if rng is None:
            rng = np.random.default_rng()

        vol_surface = instrument.underlying.vol_provider
        forward_curve = instrument.underlying.forward_curve

        times = instrument.simulation_times(self.reference_date, self.day_count_convention)
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

            # σ_loc évaluée au début du pas (schéma d'Euler non-anticipatif) ;
            # sigma_loc applique un plancher temporel pour le premier pas (t=0).
            # Vectorisé sur les n_paths spots S[:, i] en un seul appel.
            sigma = self.sigma_loc(S[:, i], t, vol_surface, forward_curve)
            Z = rng.standard_normal(n_paths)

            S[:, i + 1] = S[:, i] * growth * np.exp(-0.5 * sigma**2 * dt[i] + sigma * np.sqrt(dt[i]) * Z)

        df = self.discount_curve.discount(times[-1])
        return MCPaths(
            times=np.asarray(times_full),
            spot_paths=S,
            discounted_payoff=df * instrument.payoff(S),
            discount_factor=df,
            model_name=type(self).__name__,
            instrument_label=type(instrument).__name__,
        )
