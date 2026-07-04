"""Visualisation 3D des surfaces de volatilité implicite.

Le tracé se fait **toujours dans la coordonnée de moneyness native** des smiles
(`smile.x`), quelle que soit la convention de la surface (strike absolu,
log-moneyness forward/spot, moneyness simple). C'est la seule coordonnée toujours
disponible — `smile.x` et `interpolator.interpolate(m, T)` y sont déjà exprimés —
donc aucune courbe forward n'est requise. Le libellé de l'axe des abscisses est
déduit automatiquement de la convention.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from JR_PRICER.surfaces.vol_surface.volsurface import VolSurface
from JR_PRICER.curves.vol_smile.moneyness_convention import (
    AbsoluteStrike, LogMoneynessForward, LogMoneynessSpot, SimpleMoneyness,
)

# Libellé d'axe lisible par type de convention de moneyness.
_MONEYNESS_LABELS = {
    AbsoluteStrike: "Strike K",
    LogMoneynessForward: "Log-moneyness  ln(K/F)",
    LogMoneynessSpot: "Log-moneyness  ln(K/S)",
    SimpleMoneyness: "Moneyness  K/S",
}


def _moneyness_label(convention) -> str:
    """Libellé de l'axe des abscisses pour la coordonnée native du smile."""
    return _MONEYNESS_LABELS.get(type(convention), "Moneyness")


def _ensure_3d_axes(ax):
    """Retourne un Axes3D : crée une figure si `ax` est None, sinon valide l'axe."""
    if ax is None:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(projection="3d")
    elif getattr(ax, "name", None) != "3d":
        raise ValueError("plot_surface nécessite un axe 3D (créé avec projection='3d').")
    return ax


def plot_surface(surface: VolSurface, ax=None, with_interpolation: bool = True,
                 n_grid: int = 60, cmap: str = "viridis", show_pillars: bool = True):
    """Trace en 3D une surface de volatilité implicite, en moneyness native.

    L'axe des abscisses est toujours la coordonnée de moneyness de la surface
    (`surface.moneyness_convention`) ; son libellé est déduit automatiquement.

    Parameters
    ----------
    surface : VolSurface
        Surface à tracer. Si ``with_interpolation=True``, elle doit avoir été
        calibrée au préalable (``surface.calibrate_interpolator()``).
    ax : Axes3D, optionnel
        Axe 3D existant. Si None, une figure et un axe 3D sont créés.
    with_interpolation : bool, défaut True
        True  -> surface continue interpolée (``plot_surface`` matplotlib).
        False -> nuage de points des piliers de marché uniquement.
    n_grid : int, défaut 60
        Résolution de la grille interpolée (nombre de points par axe).
    cmap : str, défaut "viridis"
        Colormap matplotlib.
    show_pillars : bool, défaut True
        Superpose en rouge les piliers de marché (les vols cotées ayant servi à
        construire la surface) au tracé interpolé. Contrôle visuel du fit ; sans
        effet quand ``with_interpolation=False`` (le tracé EST déjà les piliers).

    Returns
    -------
    Axes3D
        L'axe sur lequel la surface a été tracée.
    """
    if not surface.smiles:
        raise ValueError("La surface ne contient aucun smile à tracer.")

    ax = _ensure_3d_axes(ax)
    xlabel = _moneyness_label(surface.moneyness_convention)

    # Piliers de marché : (coordonnée native du smile, maturité, vol cotée).
    pillar_x, pillar_t, pillar_z = [], [], []
    for smile, T in zip(surface.smiles, surface.maturities):
        pillar_x.append(np.asarray(smile.x, dtype=float))
        pillar_t.append(np.full(len(smile.x), T))
        pillar_z.append(np.asarray(smile.y, dtype=float))
    pillar_x = np.concatenate(pillar_x)
    pillar_t = np.concatenate(pillar_t)
    pillar_z = np.concatenate(pillar_z)

    if with_interpolation:
        if not surface.interpolator._fitted:
            raise RuntimeError(
                "L'interpolateur n'est pas calibré : appelez "
                "surface.calibrate_interpolator() avant de tracer la surface interpolée."
            )

        # Grille en moneyness (couvrant tous les piliers) x maturités.
        m_grid = np.linspace(pillar_x.min(), pillar_x.max(), n_grid)
        t_grid = np.linspace(min(surface.maturities), max(surface.maturities), n_grid)
        M, T_mesh = np.meshgrid(m_grid, t_grid)  # shape (n_grid, n_grid)

        # Z calculé via l'interpolateur, ligne par ligne (maturité scalaire attendue).
        Z = np.empty_like(M)
        for i, T in enumerate(t_grid):
            Z[i, :] = surface.interpolator.interpolate(m_grid, T)

        surf = ax.plot_surface(M, T_mesh, Z, cmap=cmap, alpha=0.85,
                               linewidth=0, antialiased=True)
        ax.figure.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.1,
                           label="Implied vol  σ")

        if show_pillars:
            ax.scatter(pillar_x, pillar_t, pillar_z, color="red", s=12,
                       depthshade=False, label="Piliers de marché")
            ax.legend(loc="upper left")
    else:
        # Nuage de points discret : piliers de marché uniquement.
        sc = ax.scatter(pillar_x, pillar_t, pillar_z, c=pillar_z, cmap=cmap, s=18)
        ax.figure.colorbar(sc, ax=ax, shrink=0.5, aspect=12, pad=0.1,
                           label="Implied vol  σ")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Maturité T (year fraction)")
    ax.set_zlabel("Implied vol  σ")
    ax.set_title("Surface de volatilité implicite")
    return ax


def plot_smiles(surface: VolSurface, maturities, ax=None, n_points: int = 200,
                cmap: str = "viridis", show_pillars: bool = False):
    """Trace, sur un même graphe 2D, les smiles interpolés à plusieurs maturités.

    Chaque smile est évalué via l'interpolateur **2D de la surface**
    (``surface.interpolator``), et non via les interpolateurs 1D des ``VolSmile`` :
    les maturités demandées peuvent donc être quelconques, y compris hors des
    piliers. L'axe des abscisses est la coordonnée de moneyness native de la
    surface (libellé déduit de la convention).

    Parameters
    ----------
    surface : VolSurface
        Surface calibrée (``surface.calibrate_interpolator()`` au préalable).
    maturities : float | Sequence[float]
        Maturité(s) en year fraction pour lesquelles tracer le smile.
    ax : Axes, optionnel
        Axe 2D existant. Si None, une figure et un axe sont créés.
    n_points : int, défaut 200
        Nombre de points de la grille de moneyness (résolution de chaque smile).
    cmap : str, défaut "viridis"
        Colormap utilisée pour colorer les smiles par maturité croissante.
    show_pillars : bool, défaut False
        Superpose les piliers de marché d'un smile dont la maturité coïncide avec
        une maturité demandée (contrôle visuel du fit aux maturités piliers).

    Returns
    -------
    Axes
        L'axe sur lequel les smiles ont été tracés.
    """
    if not surface.smiles:
        raise ValueError("La surface ne contient aucun smile à tracer.")
    if not surface.interpolator._fitted:
        raise RuntimeError(
            "L'interpolateur n'est pas calibré : appelez "
            "surface.calibrate_interpolator() avant de tracer les smiles."
        )

    maturities = np.atleast_1d(np.asarray(maturities, dtype=float))
    if np.any(maturities <= 0):
        raise ValueError("Les maturités doivent être strictement positives.")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    xlabel = _moneyness_label(surface.moneyness_convention)

    # Grille de moneyness couvrant tous les piliers de la surface.
    all_moneyness = np.concatenate([smile.x for smile in surface.smiles])
    m_grid = np.linspace(all_moneyness.min(), all_moneyness.max(), n_points)

    # Couleur en fonction de la maturité (dégradé croissant via la colormap).
    cmap_obj = plt.get_cmap(cmap)
    t_min, t_max = float(maturities.min()), float(maturities.max())

    def _color(T: float):
        return cmap_obj(0.5 if t_max == t_min else (T - t_min) / (t_max - t_min))

    for T in maturities:
        color = _color(T)
        sigma = surface.interpolator.interpolate(m_grid, T)
        ax.plot(m_grid, sigma, color=color, label=f"T = {T:.2f}")
        if show_pillars:
            # Piliers du smile dont la maturité coïncide avec T (le cas échéant).
            for smile, T_smile in zip(surface.smiles, surface.maturities):
                if np.isclose(T_smile, T):
                    ax.scatter(smile.x, smile.y, color=color, s=15, zorder=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Implied vol  σ")
    ax.set_title("Smiles de volatilité implicite")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Maturité (yf)")
    return ax
