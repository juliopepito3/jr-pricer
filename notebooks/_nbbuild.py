"""Outil interne (échafaudage) de construction des notebooks pédagogiques.

Assemble une liste de cellules (markdown / code) en un notebook nbformat, l'exécute
avec un kernel Python (outputs embarqués), puis l'écrit sur disque.

Ce fichier est un utilitaire de génération, pas une dépendance des notebooks
eux-mêmes (chaque notebook est auto-suffisant). Il peut être supprimé une fois
les .ipynb produits.
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbclient import NotebookClient


def make(path: str, cells: list[tuple[str, str]], execute: bool = True,
         timeout: int = 1200) -> None:
    """Construit (et exécute) un notebook à partir d'une liste (kind, source).

    kind ∈ {"md", "code"}. `path` est le chemin .ipynb de sortie.
    """
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(src) if kind == "md" else new_code_cell(src)
        for kind, src in cells
    ]
    nb.metadata["kernelspec"] = {
        "name": "python3", "display_name": "Python 3", "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

    out = Path(path)
    if execute:
        client = NotebookClient(
            nb, timeout=timeout, kernel_name="python3",
            resources={"metadata": {"path": str(out.parent)}},
        )
        client.execute()

    nbformat.write(nb, str(out))
    print(f"écrit : {out}  ({len(nb.cells)} cellules, exécuté={execute})")
