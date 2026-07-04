from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from JR_PRICER.instruments.base import Instrument as DerivativeInstrument
from JR_PRICER.curves.forward.base import ForwardCurve
from JR_PRICER.curves.temporal.discount import DiscountCurve

import numpy as np

if TYPE_CHECKING:
    # Import seulement pour le typage : à l'exécution, l'annotation `-> MCPaths`
    # reste une string (cf. `from __future__ import annotations`). Évite le cycle
    # model.base <-> engine.monte_carlo (qui importe model.base pour `Model`).
    from JR_PRICER.pricing.engine.monte_carlo import MCPaths


class Model(ABC):
    """Modèle de diffusion/pricing partageant une courbe de discount.

    Selon le modèle, seul un sous-ensemble des moteurs est supporté : Monte-Carlo
    (`simulate`), analytique (`analytic_price`), Fourier (`characteristic_function`)
    ou PDE (`drift`/`diffusion`). Les méthodes non supportées lèvent NotImplementedError.
    """

    def __init__(self, discount_curve: DiscountCurve) -> None:
        self.discount_curve = discount_curve
        self.reference_date = discount_curve.reference_date
        self.day_count_convention = discount_curve.day_count_convention

    # Monte Carlo — interface unifiée equity + taux
    @abstractmethod
    def simulate(self, instrument: DerivativeInstrument, n_paths: int,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        """Retourne un vecteur de shape (n_paths,) de payoffs actualisés pour l'instrument.

        `rng` : Generator NumPy pour la reproductibilité. None → Generator par défaut.
        """
        raise NotImplementedError

    def simulate_paths(self, instrument: DerivativeInstrument, n_paths: int,
                       rng: np.random.Generator | None = None) -> MCPaths:
        """Comme `simulate`, mais renvoie un objet `MCPaths` (trajectoires + payoff).

        Hook optionnel : seuls les modèles diffusifs sur grille temporelle (equity)
        l'implémentent. Les modèles sans trajectoire (ex. tirage terminal one-shot)
        ne le surchargent pas.
        """
        raise NotImplementedError("Ce modèle n'enregistre pas les trajectoires MC.")

    # Hooks optionnels pour le pricing PDE (non abstraits : tous les modèles ne les fournissent pas)
    def drift(self, S: float, t: float, forward_curve: ForwardCurve) -> float:
        """Dérive du processus à (S, t) pour la ForwardCurve donnée."""
        raise NotImplementedError("drift must be implemented by subclasses")

    def diffusion(self, S: float, t: float) -> float:
        """Diffusion du processus à (S, t) pour le VolProvider donné."""
        raise NotImplementedError("diffusion must be implemented by subclasses")

    # Actualisation — partagée par tous les moteurs
    def discount(self, T: float) -> float:
        """Facteur d'actualisation P(0, T) (délègue en général à discount_curve)."""
        raise NotImplementedError("discount must be implemented by subclasses")

    # Pricing analytique
    def analytic_price(self, instrument: DerivativeInstrument) -> float:
        raise NotImplementedError("analytic_price must be implemented by subclasses")
    
    # Pricing par transformée de Fourier
    def characteristic_function(self, u_grid: np.ndarray, T: float) -> np.ndarray:
        """Fonction caractéristique du modèle pour une grille de valeurs u et une maturité T."""
        raise NotImplementedError("characteristic_function must be implemented by subclasses")
