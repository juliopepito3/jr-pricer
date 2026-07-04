"""Chargement d'un snapshot de chaîne d'options et appariement call/put.

Fait le pont entre un fichier plat (CSV téléchargé une fois, par exemple depuis
yfinance dans un notebook dédié) et les `PutCallQuote` consommés par le builder
de parité. Le téléchargement lui-même ne vit PAS ici : ce module ne dépend
d'aucune source de données, seulement du schéma de fichier ci-dessous.

Schéma attendu (une ligne par option) :

    quote_date      date du snapshot (YYYY-MM-DD), répétée sur chaque ligne
    spot            spot du sous-jacent au snapshot, répété sur chaque ligne
    expiry          date d'expiration (YYYY-MM-DD)
    option_type     'C' ou 'P'
    strike          strike absolu
    bid, ask, last  prix de marché ('last' conservé pour inspection, jamais
                    utilisé pour les mids : potentiellement périmé)
    volume          volume du jour
    open_interest   open interest
    iv_yf           vol implicite fournie par la source (indicative)

Note : `pandas` est une dépendance de ce sous-package `loaders` uniquement,
le reste de JR_PRICER n'en a pas besoin.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from JR_PRICER.curves.builder.parity_builder import PutCallQuote
from JR_PRICER.market_data.quote import Quote

_REQUIRED_COLUMNS = frozenset({
    "quote_date", "spot", "expiry", "option_type", "strike",
    "bid", "ask", "volume", "open_interest",
})


@dataclass(frozen=True)
class ChainQualityFilter:
    """Filtres d'hygiène appliqués leg par leg sur la chaîne brute.

    Ils écartent les quotes inutilisables avant tout appariement : options
    jamais traitées (open interest nul), quotes à un seul côté ou croisées,
    prix au tick (pur bruit), spreads si larges que le mid ne veut rien dire.
    Les filtres économiques (fenêtre de moneyness) relèvent du builder de
    parité, pas d'ici.
    """
    min_bid: float = 0.05
    max_relative_spread: float = 0.50
    min_volume: int = 0
    min_open_interest: int = 1
    require_two_sided: bool = True

    def apply(self, chain: pd.DataFrame) -> pd.DataFrame:
        """Retourne la sous-chaîne qui passe tous les filtres."""
        mask = pd.Series(True, index=chain.index)
        if self.require_two_sided:
            mask &= (chain["bid"] > 0) & (chain["ask"] > chain["bid"])
        mask &= chain["bid"] >= self.min_bid
        mid = (chain["bid"] + chain["ask"]) / 2.0
        relative_spread = (chain["ask"] - chain["bid"]) / mid.where(mid > 0)
        mask &= relative_spread.le(self.max_relative_spread).fillna(False)
        mask &= chain["volume"].fillna(0) >= self.min_volume
        mask &= chain["open_interest"].fillna(0) >= self.min_open_interest
        return chain[mask]


@dataclass(frozen=True)
class OptionChainSnapshot:
    """Snapshot d'une chaîne d'options : date, spot et chaîne brute.

    `chain` conserve toutes les lignes du fichier (aucun filtrage à la
    lecture) : les filtres sont appliqués explicitement, et de façon
    inspectable, par `to_put_call_quotes`.
    """
    quote_date: date
    spot: Quote
    chain: pd.DataFrame

    @classmethod
    def from_csv(cls, path: "str | Path") -> "OptionChainSnapshot":
        """Charge un snapshot depuis un CSV au schéma du module."""
        frame = pd.read_csv(path, parse_dates=["quote_date", "expiry"])
        missing = _REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes dans {path} : {sorted(missing)}")

        quote_dates = frame["quote_date"].dt.date.unique()
        spots = frame["spot"].unique()
        if len(quote_dates) != 1 or len(spots) != 1:
            raise ValueError(
                "Le snapshot doit porter une seule quote_date et un seul spot "
                f"(trouvé {len(quote_dates)} dates, {len(spots)} spots)."
            )

        chain = frame.drop(columns=["quote_date", "spot"]).copy()
        chain["expiry"] = chain["expiry"].dt.date
        return cls(
            quote_date=quote_dates[0],
            spot=Quote(float(spots[0]), name="snapshot spot"),
            chain=chain,
        )

    @property
    def expiries(self) -> list[date]:
        """Expirations présentes dans la chaîne, triées."""
        return sorted(self.chain["expiry"].unique())

    def to_put_call_quotes(self,
                           quality: Optional[ChainQualityFilter] = None,
                           expiries: Optional[Sequence[date]] = None) -> list[PutCallQuote]:
        """Filtre la chaîne, apparie calls et puts par (expiry, strike).

        L'appariement est un inner join : un strike qui n'a pas les deux legs
        (après filtres) disparaît. Les mids sont (bid + ask) / 2 ; `last` n'est
        jamais utilisé.
        """
        quality = quality if quality is not None else ChainQualityFilter()
        chain = self.chain
        if expiries is not None:
            chain = chain[chain["expiry"].isin(set(expiries))]
        chain = quality.apply(chain)

        calls = chain[chain["option_type"] == "C"]
        puts = chain[chain["option_type"] == "P"]
        paired = calls.merge(
            puts, on=["expiry", "strike"], suffixes=("_call", "_put"), how="inner",
        ).sort_values(["expiry", "strike"])

        return [
            PutCallQuote(
                strike=float(row.strike),
                call_mid=float((row.bid_call + row.ask_call) / 2.0),
                put_mid=float((row.bid_put + row.ask_put) / 2.0),
                call_bid=float(row.bid_call),
                call_ask=float(row.ask_call),
                put_bid=float(row.bid_put),
                put_ask=float(row.ask_put),
                maturity_date=row.expiry,
            )
            for row in paired.itertuples()
        ]

    def __repr__(self) -> str:
        return (f"OptionChainSnapshot(quote_date={self.quote_date}, "
                f"spot={self.spot.value()}, n_rows={len(self.chain)}, "
                f"n_expiries={len(self.expiries)})")
