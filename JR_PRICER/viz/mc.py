"""Visualisation des simulations Monte-Carlo à partir d'un objet `MCPaths`.

Briques granulaires (pas de figure composite) : l'utilisateur compose lui-même sa
figure, par ex.

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    plot_paths(res, ax1, quantity="spot")
    plot_paths(res, ax2, quantity="variance")   # Heston uniquement

Mêmes conventions que `viz/surfaces.py` : `ax=None` crée la figure, backend matplotlib,
libellés FR.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from JR_PRICER.pricing.engine.monte_carlo import MCPaths


def plot_paths(result: MCPaths, ax=None, quantity: str = "spot", n_paths_shown: int = 100,
               show_mean: bool = True, quantiles: tuple[float, float] | None = None,
               seed: int | None = None, color=None, label: str | None = None):
    """Trace un échantillon de trajectoires simulées (sous-jacent ou variance) dans le temps.

    Parameters
    ----------
    result : MCPaths
        Résultat d'une simulation (`MCEngine.simulate_paths`).
    ax : Axes, optionnel
        Axe 2D existant. Si None, une figure et un axe sont créés. Passer un `ax`
        partagé permet de superposer plusieurs simulations sur le même graphe.
    quantity : {"spot", "variance"}, défaut "spot"
        "spot" trace `spot_paths` (sous-jacent) ; "variance" trace `variance_paths`
        (Heston). Lève une erreur si "variance" et que le résultat n'a pas de variance.
    n_paths_shown : int, défaut 100
        Nombre de trajectoires individuelles affichées (échantillon aléatoire). Mettre
        0 pour n'afficher que la moyenne (utile pour comparer plusieurs simulations).
    show_mean : bool, défaut True
        Superpose la trajectoire moyenne (calculée sur toutes les trajectoires).
    quantiles : (float, float) | None, défaut None
        Si fourni (ex. (0.05, 0.95)), ombre la bande inter-quantiles (fan chart),
        calculée à chaque pas sur **toutes** les trajectoires. None = aucune bande.
    seed : int | None
        Graine du sous-échantillonnage des trajectoires affichées (reproductible).
    color : couleur matplotlib | None, défaut None
        Couleur unique de la série (trajectoires + bande + moyenne). None → défaut
        (trajectoires bleues, moyenne noire). Utile pour distinguer les séries en
        superposition.
    label : str | None, défaut None
        Étiquette de légende de la série, portée par un seul artefact (moyenne si
        `show_mean`, sinon la bande de quantiles, sinon les trajectoires). Utile pour
        comparer plusieurs séries sur le même axe. None → comportement par défaut
        (moyenne libellée « Moyenne »).

    Returns
    -------
    Axes
        L'axe sur lequel les trajectoires ont été tracées.
    """
    if quantity == "spot":
        paths = result.spot_paths
        ylabel = "Sous-jacent S"
    elif quantity == "variance":
        if result.variance_paths is None:
            raise ValueError(
                "Ce résultat ne contient pas de variance_paths "
                "(modèle sans variance stochastique — quantity='variance' invalide)."
            )
        paths = result.variance_paths
        ylabel = "Variance v"
    else:
        raise ValueError("quantity doit valoir 'spot' ou 'variance'.")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    t = result.times
    n_total = paths.shape[0]
    n_show = min(n_paths_shown, n_total)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_show, replace=False)

    paths_color = color if color is not None else "steelblue"
    mean_color = color if color is not None else "black"
    legacy = label is None and color is None  # mode mono-série par défaut

    # L'étiquette de série n'est portée que par UN artefact (priorité moyenne > bande
    # > trajectoires), pour une légende propre même en superposition de plusieurs séries.
    carrier = "mean" if show_mean else ("band" if quantiles is not None else "paths")

    # Échantillon de trajectoires (faible alpha pour la lisibilité).
    if len(idx):
        path_lines = ax.plot(t, paths[idx].T, color=paths_color, alpha=0.3, linewidth=0.7)
        if label is not None and carrier == "paths":
            path_lines[0].set_label(label)

    if quantiles is not None:
        lo, hi = np.quantile(paths, quantiles, axis=0)
        if label is not None:
            band_label = label if carrier == "band" else None
        else:
            band_label = f"Quantiles {quantiles[0]:.0%}–{quantiles[1]:.0%}" if legacy else None
        ax.fill_between(t, lo, hi, color=paths_color, alpha=0.2, label=band_label)

    if show_mean:
        mean_label = label if label is not None else ("Moyenne" if legacy else None)
        ax.plot(t, paths.mean(axis=0), color=mean_color, linewidth=2, label=mean_label)

    ax.set_xlabel("Temps t (year fraction)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Trajectoires Monte-Carlo — {quantity} ({result.model_name})")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="best")
    return ax


def plot_distribution(result: MCPaths, steps, ax=None, bins: int = 50):
    """Trace la distribution du sous-jacent à un ou plusieurs pas de temps.

    Parameters
    ----------
    result : MCPaths
        Résultat d'une simulation (`MCEngine.simulate_paths`).
    steps : int | Sequence[int]
        Indice(s) de pas dans l'axe temps (colonnes de `spot_paths` / `result.times`).
        Indices négatifs autorisés : `[-1]` = distribution terminale, `[-1, -2]` = deux
        derniers pas, `[0]` = initial. **Obligatoire.**
    ax : Axes, optionnel
        Axe 2D existant. Si None, une figure et un axe sont créés.
    bins : int, défaut 50
        Nombre de classes de l'histogramme.

    Returns
    -------
    Axes
        L'axe sur lequel la (les) distribution(s) ont été tracées.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    for step in np.atleast_1d(steps):
        step = int(step)
        sample = result.spot_paths[:, step]
        T = float(result.times[step])
        ax.hist(sample, bins=bins, density=True, histtype="step", label=f"t = {T:.2f}")

    ax.set_xlabel("Sous-jacent S")
    ax.set_ylabel("Densité")
    ax.set_title(f"Distribution du sous-jacent ({result.model_name})")
    ax.legend(title="Pas (yf)")
    return ax
