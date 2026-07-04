"""Sous-jacent : regroupe les données de marché d'un actif (spot, forward, vol)."""
from __future__ import annotations

from JR_PRICER.market_data.quote import Quote
from JR_PRICER.curves.forward.base import ForwardCurve
from JR_PRICER.surfaces.vol_surface.volsurface import VolProvider


class Underlying:
    """Regroupe toutes les données de marché propres à un actif sous-jacent.

    Le spot est un Quote (live) — une mise à jour via spot.update() se propage
    automatiquement à la ForwardCurve et à tous les instruments qui référencent
    cet Underlying.

    Architecture :
    - vol_surface (ImpliedVolSurface) : données brutes de vol implicite — ce que
      cote le marché. C'est l'entrée de la calibration (Dupire, Heston, …).
    - vol_provider (VolProvider)      : interface modèle sigma(K,T). Conservé pour
      la compatibilité avec BlackScholesModel. Pour les nouveaux modèles, le
      vol_provider vit dans le modèle, pas dans l'Underlying.
    """

    def __init__(self, name: str, spot: Quote,
                 forward_curve: ForwardCurve,
                 vol_provider: VolProvider | None = None) -> None:
        self.name = name
        self.spot = spot
        self.forward_curve = forward_curve
        self.vol_provider = vol_provider

    def __repr__(self) -> str:
        return f"Underlying(name='{self.name}', spot={self.spot.value()})"
