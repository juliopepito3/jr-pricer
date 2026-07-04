"""Moteur de pricing Monte-Carlo (moyenne des payoffs actualisés simulés).

Héberge aussi `MCPaths`, l'objet résultat porteur des trajectoires simulées (pour
la visualisation), afin de garder toute la logique du moteur dans un seul fichier.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from JR_PRICER.pricing.engine.base import Engine
from JR_PRICER.pricing.model.base import Model
from JR_PRICER.instruments.base import Instrument


@dataclass
class MCPaths:
    """Résultat détaillé d'une simulation Monte-Carlo (un instrument).

    Conteneur passif : il ne copie rien, il garde des références aux arrays déjà
    alloués par la simulation (aucun surcoût mémoire/CPU côté vectorisation).
    Pour la visualisation, préférer un `n_paths` modeste (~1–5k) : les trajectoires
    complètes sont conservées tant que l'objet est vivant.
    """

    times: np.ndarray                       # (n_steps+1,) grille temporelle (year fraction), inclut 0
    spot_paths: np.ndarray                  # (n_paths, n_steps+1) trajectoires du sous-jacent
    discounted_payoff: np.ndarray           # (n_paths,) payoff actualisé par trajectoire
    discount_factor: float                  # P(0, T)
    variance_paths: np.ndarray | None = None  # (n_paths, n_steps+1) variance (Heston) ou None
    model_name: str = ""
    instrument_label: str = ""
    extra: dict[str, np.ndarray] = field(default_factory=dict)  # extensibilité (ex. vol locale)

    @property
    def price(self) -> float:
        """Prix Monte-Carlo = moyenne des payoffs actualisés."""
        return float(np.mean(self.discounted_payoff))

    @property
    def terminal_spot(self) -> np.ndarray:
        """Sous-jacent à la dernière date simulée — shape (n_paths,)."""
        return self.spot_paths[:, -1]

    @property
    def undiscounted_payoff(self) -> np.ndarray:
        """Payoff non actualisé (payoff actualisé / facteur d'actualisation)."""
        return self.discounted_payoff / self.discount_factor

    @property
    def n_paths(self) -> int:
        return self.spot_paths.shape[0]

    @property
    def n_steps(self) -> int:
        return self.spot_paths.shape[1] - 1


class MCEngine(Engine):
    """Moteur Monte-Carlo : prix = moyenne des payoffs actualisés simulés.

    `seed` rend les tirages reproductibles (un `Generator` par appel `price`).
    """

    def __init__(self, n_paths: int, seed: int | None = None) -> None:
        self.n_paths = n_paths
        self.seed = seed

    def price(self, instruments: list[Instrument], model: Model) -> list[float]:
        # Un seul Generator par appel : résultats reproductibles à seed fixé
        # (et indépendants de l'état global np.random). À seed=None, comportement
        # non déterministe classique.
        rng = np.random.default_rng(self.seed)
        return [float(np.mean(model.simulate(instr, self.n_paths, rng=rng))) for instr in instruments]

    def simulate_paths(self, instrument: Instrument, model: Model) -> MCPaths:
        """Simule un instrument et renvoie l'objet `MCPaths` complet (trajectoires + prix).

        Même graine reproductible que `price`. Le prix est accessible via `.price`.
        Lève NotImplementedError si le modèle n'enregistre pas les trajectoires.
        """
        rng = np.random.default_rng(self.seed)
        return model.simulate_paths(instrument, self.n_paths, rng=rng)

    def __repr__(self) -> str:
        return f"MCEngine(n_paths={self.n_paths}, seed={self.seed})"
