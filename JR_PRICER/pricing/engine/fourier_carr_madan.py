"""Moteur de pricing par transformée de Fourier (méthode de Carr-Madan, FFT)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from JR_PRICER.pricing.engine.base import Engine
from JR_PRICER.instruments.derivatives.equity.european_option import EuropeanOption
from JR_PRICER.pricing.model.base import Model
from JR_PRICER.curves.interpolators_1D.base import Interpolator1D
from JR_PRICER.curves.interpolators_1D.cubic_splines import ScipyCubicSplinesInterpolator1D
from JR_PRICER.curves.base import Curve


class FourierCarrMadanEngine(Engine):
    """Pricing d'options européennes par FFT de Carr-Madan.

    Requiert un modèle exposant `characteristic_function` (ex. Heston). Les
    paramètres `alpha` (amortissement), `n` (taille FFT N=2ⁿ) et `eta` (pas en
    fréquence) contrôlent la précision de la quadrature.

    `alpha` fixe le **plancher numérique** des prix profondément OTM : plus il est
    grand, plus le plancher est bas (α=1.5 → ~2e-7 ; α=2.0 → ~4e-10 ; α=2.5 → ~1e-12),
    ce qui évite les prix négatifs parasites (et donc les vols implicites aberrantes)
    dans les ailes / à courte maturité. `n` n'a **aucun** effet sur ce plancher (il
    n'améliore que la précision ATM). ⚠️ Un `alpha` trop grand sort de la bande de
    stabilité de la fonction caractéristique (**explosion de moment** de Heston pour
    ρ, σ_v, T élevés) et **corrompt silencieusement** les prix : réduire `alpha` sur
    ces paramètres extrêmes. Le défaut α=2.0 est un compromis robuste.
    """

    def __init__(self, alpha: float = 2.0, n: int = 5, eta: float = 0.25,
                 interpolator: Interpolator1D | None = None) -> None:
        self.alpha = alpha
        self.n = n
        self.eta = eta
        if interpolator is None:
            self.interpolator = ScipyCubicSplinesInterpolator1D()
        else:
            self.interpolator = interpolator
        self.N = 2 ** self.n
        self.delta_k = 2 * np.pi / (self.N * self.eta)

    def price(self, instruments: list[EuropeanOption], model: Model) -> list[float]:
        """Prix par FFT, regroupé par (sous-jacent, maturité) puis interpolé en strike."""


        #1) Construire la grille des strikes et des fréquences pour la transformée de Fourier

        u_grid = np.arange(self.N) * self.eta
        k_grid = -self.N / 2 * self.delta_k + np.arange(self.N) * self.delta_k

        # Pour chaque couple (underlying, maturity), on va calculer les prix des options sur une grille de strikes et ensuite interpoler pour obtenir les prix aux strikes demandés.

        couple_instrument_lists = defaultdict(list)

        for instr in instruments:
            couple = (instr.underlying, instr.maturity_date)
            couple_instrument_lists[couple].append(instr)

        price_by_instruments = {}

        for couple in couple_instrument_lists.keys():

            underlying = couple[0]
            T = model.day_count_convention.year_fraction(model.reference_date, couple[1])  # Maturité en années
            
            F_T = underlying.forward_curve.forward(T)   # Prix forward de l'actif sous-jacent à la maturité T
            P_0T = model.discount(T)                    # Discount factor à la maturité T

            # 2) Calculer la fonction caractéristique pour la maturité T et la grille de fréquences u_grid

            phi_T_evaluated = model.characteristic_function(u_grid - (self.alpha + 1) * 1j, T)

            phi_u_evaluated = P_0T * phi_T_evaluated / (self.alpha ** 2 + self.alpha - u_grid ** 2 + 1j * (2 * self.alpha + 1) * u_grid)

            # Calcul de la transformée inverse de Fourier pour obtenir les prix des options sur la grille de strikes k_grid

            # Poids de Simpson (sans dimension : le pas η n'est appliqué qu'une
            # seule fois, dans z ci-dessous — auparavant η était compté deux fois).

            simpson_weights_grid = np.ones(self.N) / 3
            simpson_weights_grid[1:-2:2] = 4 / 3
            simpson_weights_grid[2:-1:2] = 2 / 3

            x_j = np.exp(-1j * u_grid * k_grid[0]) * phi_u_evaluated * simpson_weights_grid

            # 3) Transformée de Fourier inverse via FFT numpy (∫ ≈ η · Σ poids·f)

            z = self.eta / np.pi * np.fft.fft(x_j).real

            #4) Désamortissement du vecteur des prix pour obtenir les prix des options sur la grille de strikes k_grid

            z_desamort = np.exp(-self.alpha * k_grid) * z

            # Curve accepte directement les ndarray (cf. refactor NumPy).
            z_desamort_curve = Curve(k_grid, z_desamort, self.interpolator)

            #5) Interpolation et évaluation
            # La fonction caractéristique est sous la mesure forward (CF de
            # log(S_T/F)) : le prix normalisé est mis à l'échelle par le forward F_T.

            for instr in couple_instrument_lists[couple]:
                price_by_instruments[instr] = F_T * z_desamort_curve.evaluate(np.log(instr.K / F_T))

        return [price_by_instruments[instr] for instr in instruments]

    def __repr__(self) -> str:
        return f"FourierCarrMadanEngine(alpha={self.alpha}, n={self.n}, eta={self.eta})"