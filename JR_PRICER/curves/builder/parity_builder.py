"""Extraction jointe discount curve + forward curve par parité put-call.

Relation exploitée, par maturité T :

    C(K, T) - P(K, T) = D(0, T) * F(0, T) - D(0, T) * K

(C - P) est donc exactement affine en K : la pente donne -D(0, T), l'intercept
donne D(0, T) * F(0, T). Une régression par maturité extrait les deux courbes
simultanément, avec la même origine (donc une cohérence mutuelle garantie : ce
ne sont pas deux estimations indépendantes assemblées après coup).

Sur données réelles, la relation est polluée par le bruit bid-ask, les quotes
aberrantes et, pour les options américaines, la prime d'exercice anticipé (EEP).
Le builder expose donc les choix qui font la qualité du fit :

- pondération des quotes (`ParityWeighting`) : uniforme ou inverse du spread,
- régression (`LinearRegression`) : WLS, rejet MAD (défaut) ou Huber,
- fenêtre de strikes en deux passes (`StrikeWindow`) : première estimation du
  forward sur tous les strikes, puis refit restreint autour de F̂.

Biais américain (single names, ETF) : la parité stricte ne tient que pour des
options européennes. Pour des américaines, C - P est décalé par l'EEP, maximale
pour les puts très ITM (K >> F, exercice pour toucher les intérêts) et pour les
calls ITM avant détachement de dividende. Près de l'ATM, l'EEP est de second
ordre : la fenêtre de strikes par défaut (0.85 F à 1.15 F) et la pondération
1/spread en font une mitigation pratique raisonnable, sans dé-américanisation.
Conséquence assumée : le "taux" implicite extrait est un taux effectif
(funding - borrow - biais EEP), pas l'OIS, et peut être non monotone. C'est la
mesure de pricing auto-cohérente : les vanilles se repricent exactement dans
ces courbes, ce qui est le but pour calibrer des modèles dessus.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from JR_PRICER.curves.builder.regression import (
    LinearFitResult,
    LinearRegression,
    MADRejectionRegression,
)
from JR_PRICER.curves.temporal.discount import DiscountCurve
from JR_PRICER.curves.forward.market_forward import MarketForwardCurve
from JR_PRICER.curves.interpolators_1D.base import Interpolator1D
from JR_PRICER.market_data.quote import Quote
from JR_PRICER.utils.day_count import DayCounter


@dataclass(frozen=True)
class PutCallQuote:
    """Paire call/put appariée : même strike, même maturité.

    L'appariement depuis une chaîne d'options brute est une responsabilité
    amont (voir `market_data.loaders.option_chain`), pas de ce module.
    """
    strike: float

    call_mid: float
    put_mid: float

    call_bid: float
    call_ask: float

    put_bid: float
    put_ask: float

    maturity_date: date


class ParityWeighting(ABC):
    """Stratégie de pondération des quotes dans la régression de parité."""

    @abstractmethod
    def weights(self, quotes: Sequence[PutCallQuote]) -> np.ndarray:
        """Poids (strictement positifs) alignés sur les quotes."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class UniformWeights(ParityWeighting):
    """Toutes les quotes pèsent pareil."""

    def weights(self, quotes: Sequence[PutCallQuote]) -> np.ndarray:
        return np.ones(len(quotes))


class InverseSpreadWeights(ParityWeighting):
    """Poids w = 1 / (spread_call + spread_put)^power.

    Le spread bid-ask est le proxy de liquidité le plus direct : les quotes
    serrées (ATM, liquides) dominent la régression, les ailes larges comptent
    peu. power=1 est un compromis doux ; power=2 approxime une pondération en
    1/variance si l'on assimile le spread à l'écart-type du bruit de mid.
    """

    def __init__(self, power: float = 1.0) -> None:
        if power <= 0:
            raise ValueError("power doit être strictement positif.")
        self.power = power

    def weights(self, quotes: Sequence[PutCallQuote]) -> np.ndarray:
        spreads = np.array([
            (q.call_ask - q.call_bid) + (q.put_ask - q.put_bid) for q in quotes
        ])
        if np.any(spreads <= 0):
            raise ValueError(
                "Spread bid-ask non positif rencontré : filtrer les quotes croisées "
                "en amont ou utiliser UniformWeights."
            )
        return 1.0 / spreads ** self.power

    def __repr__(self) -> str:
        return f"{type(self).__name__}(power={self.power})"


@dataclass(frozen=True)
class StrikeWindow:
    """Fenêtre multiplicative de strikes autour du forward estimé (passe 2).

    Après une première régression sur tous les strikes, on ne garde que
    K dans [lower * F̂, upper * F̂] et on refit. Près du forward, les deux legs
    sont OTM ou peu ITM : quotes les plus liquides et prime d'exercice anticipé
    minimale (cas américain).
    """
    lower: float = 0.85
    upper: float = 1.15

    def __post_init__(self) -> None:
        if not (0.0 < self.lower < 1.0 < self.upper):
            raise ValueError("StrikeWindow attend lower < 1 < upper (multiplicateurs de F).")


@dataclass(frozen=True)
class ForwardParityFitDiagnostics:
    """Diagnostics de la régression de parité pour une maturité donnée.

    `strikes`, `residuals` et `inlier_mask` portent sur les quotes de la passe
    finale (après fenêtre de strikes éventuelle), dans l'ordre croissant des
    strikes ; `inlier_mask` marque celles conservées par la régression robuste.
    """
    maturity: float
    maturity_date: date
    n_quotes_input: int
    n_quotes_used: int
    discount_factor: float
    forward: float
    zero_rate: float
    implied_carry: float
    r_squared: float
    rmse: float
    stderr_discount_factor: float
    stderr_forward: float
    strikes: np.ndarray
    residuals: np.ndarray
    inlier_mask: np.ndarray
    window_applied: bool


@dataclass(frozen=True)
class MarketImpliedCurves:
    """Discount et forward curves extraites simultanément de la parité put-call."""
    discount_curve: DiscountCurve
    forward_curve: MarketForwardCurve
    diagnostics: dict[float, ForwardParityFitDiagnostics]
    warnings: tuple[str, ...] = field(default=())

    @property
    def has_monotonic_discount_factors(self) -> bool:
        """True si les discount factors des piliers sont strictement décroissants."""
        return bool(np.all(np.diff(self.discount_curve.values) < 0))

    def diagnostics_table(self) -> list[dict]:
        """Diagnostics par maturité sous forme de liste de dicts (une ligne par pilier).

        Pensé pour `pd.DataFrame(curves.diagnostics_table())` dans un notebook,
        sans imposer pandas au package.
        """
        rows = []
        for maturity in sorted(self.diagnostics):
            d = self.diagnostics[maturity]
            rows.append({
                "maturity": d.maturity,
                "maturity_date": d.maturity_date,
                "n_quotes_input": d.n_quotes_input,
                "n_quotes_used": d.n_quotes_used,
                "discount_factor": d.discount_factor,
                "forward": d.forward,
                "zero_rate": d.zero_rate,
                "implied_carry": d.implied_carry,
                "r_squared": d.r_squared,
                "rmse": d.rmse,
                "stderr_forward": d.stderr_forward,
                "window_applied": d.window_applied,
            })
        return rows


class PutCallParityCurveBuilder:
    """Construit discount curve et forward curve depuis des quotes call/put réelles.

    La configuration (pondération, régression, fenêtre de strikes, seuils) est
    portée par le builder ; `build` ne prend que les données de marché. Voir la
    docstring du module pour la mécanique et la gestion du biais américain.

    Parameters
    ----------
    day_count_convention : DayCounter
        Conversion des maturity_date en year fractions.
    discount_interpolator : Interpolator1D
        Interpolateur de la DiscountCurve (log-linéaire en pratique : taux zéro
        constant par morceaux).
    forward_interpolator : Interpolator1D
        Interpolateur de la MarketForwardCurve (convexité différente de log D,
        rien n'impose le même choix).
    weighting : ParityWeighting
        Pondération des quotes. Défaut : InverseSpreadWeights().
    regression : LinearRegression
        Régression affine. Défaut : MADRejectionRegression() (fit, rejet des
        résidus aberrants, refit).
    strike_window : StrikeWindow | None
        Fenêtre de strikes en deux passes autour du forward estimé. None pour
        désactiver (une seule passe sur tous les strikes).
    min_strikes_per_maturity : int
        En dessous, la maturité est ignorée (régression trop sensible au bruit).
    min_r_squared : float | None
        Si fourni, les maturités dont le fit final a un R² inférieur sont
        écartées (avec warning). Par défaut on garde tout : la sélection fine
        reste manuelle, diagnostics à l'appui.
    """

    # Garde-fous économiques sur les sorties de régression : un DF hors de
    # (0, MAX_DISCOUNT_FACTOR] ou un forward négatif signale une slice cassée
    # (quotes croisées massives, maturité illiquide), pas un marché exotique.
    MAX_DISCOUNT_FACTOR = 1.2

    def __init__(self,
                 day_count_convention: DayCounter,
                 discount_interpolator: Interpolator1D,
                 forward_interpolator: Interpolator1D,
                 weighting: ParityWeighting | None = None,
                 regression: LinearRegression | None = None,
                 strike_window: StrikeWindow | None = StrikeWindow(),
                 min_strikes_per_maturity: int = 6,
                 min_r_squared: float | None = None) -> None:
        if min_strikes_per_maturity < 3:
            raise ValueError("min_strikes_per_maturity doit être au moins 3.")
        self.day_count_convention = day_count_convention
        self.discount_interpolator = discount_interpolator
        self.forward_interpolator = forward_interpolator
        self.weighting = weighting if weighting is not None else InverseSpreadWeights()
        self.regression = regression if regression is not None else MADRejectionRegression()
        self.strike_window = strike_window
        self.min_strikes_per_maturity = min_strikes_per_maturity
        self.min_r_squared = min_r_squared

    def build(self, quotes: list[PutCallQuote], spot: Quote,
              reference_date: date) -> MarketImpliedCurves:
        """Extrait les deux courbes depuis les quotes ; lève si aucune maturité ne survit."""
        quotes_by_maturity: dict[float, list[PutCallQuote]] = defaultdict(list)
        for quote in quotes:
            maturity = self.day_count_convention.year_fraction(
                reference_date, quote.maturity_date)
            if maturity > 0:
                quotes_by_maturity[maturity].append(quote)

        forwards_by_maturity: dict[float, float] = {}
        dfs_by_maturity: dict[float, float] = {}
        diagnostics: dict[float, ForwardParityFitDiagnostics] = {}
        warnings: list[str] = []

        for maturity in sorted(quotes_by_maturity):
            maturity_quotes = sorted(quotes_by_maturity[maturity], key=lambda q: q.strike)
            if len(maturity_quotes) < self.min_strikes_per_maturity:
                warnings.append(
                    f"T={maturity:.4f} : {len(maturity_quotes)} strikes < "
                    f"{self.min_strikes_per_maturity}, maturité ignorée."
                )
                continue

            result = self._fit_maturity(maturity, maturity_quotes, spot.value(), warnings)
            if result is None:
                continue
            diag = result
            forwards_by_maturity[maturity] = diag.forward
            dfs_by_maturity[maturity] = diag.discount_factor
            diagnostics[maturity] = diag

        if not forwards_by_maturity:
            raise ValueError(
                "Aucune maturité exploitable : trop peu de strikes ou fits tous rejetés. "
                f"Warnings : {warnings}"
            )

        times = np.array(sorted(forwards_by_maturity))
        forwards = np.array([forwards_by_maturity[t] for t in times])
        dfs = np.array([dfs_by_maturity[t] for t in times])

        if not np.all(np.diff(dfs) < 0):
            warnings.append(
                "Discount factors non strictement décroissants entre piliers : "
                "attendu sur données bruitées/américaines, inspecter les diagnostics."
            )

        forward_curve = MarketForwardCurve(
            spot=spot, times=times, forwards=forwards,
            interpolator=self.forward_interpolator,
        )
        discount_curve = DiscountCurve(
            times=times, discount_factors=dfs,
            interpolator=self.discount_interpolator,
            day_count_convention=self.day_count_convention,
            reference_date=reference_date,
        )

        return MarketImpliedCurves(
            discount_curve=discount_curve,
            forward_curve=forward_curve,
            diagnostics=diagnostics,
            warnings=tuple(warnings),
        )

    # ------------------------------------------------------------------ #
    # Une maturité : régression en (au plus) deux passes + garde-fous
    # ------------------------------------------------------------------ #

    def _fit_maturity(self, maturity: float, maturity_quotes: list[PutCallQuote],
                      spot_value: float,
                      warnings: list[str]) -> ForwardParityFitDiagnostics | None:
        n_input = len(maturity_quotes)

        # Passe 1 : tous les strikes.
        fit = self._regress(maturity_quotes)
        extracted = self._extract(fit)
        if extracted is None:
            warnings.append(
                f"T={maturity:.4f} : régression dégénérée en passe 1 "
                f"(DF ou forward hors bornes), maturité écartée."
            )
            return None
        _, forward_hat = extracted

        # Passe 2 : refit sur la fenêtre de strikes autour du forward estimé.
        window_applied = False
        final_quotes = maturity_quotes
        if self.strike_window is not None:
            lo = self.strike_window.lower * forward_hat
            hi = self.strike_window.upper * forward_hat
            windowed = [q for q in maturity_quotes if lo <= q.strike <= hi]
            if len(windowed) >= self.min_strikes_per_maturity:
                fit_w = self._regress(windowed)
                extracted_w = self._extract(fit_w)
                if extracted_w is not None:
                    fit, extracted = fit_w, extracted_w
                    final_quotes = windowed
                    window_applied = True
                else:
                    warnings.append(
                        f"T={maturity:.4f} : refit fenêtré dégénéré, "
                        f"fit passe 1 conservé."
                    )
            else:
                warnings.append(
                    f"T={maturity:.4f} : {len(windowed)} strikes dans la fenêtre "
                    f"[{lo:.2f}, {hi:.2f}] < {self.min_strikes_per_maturity}, "
                    f"fit passe 1 conservé."
                )

        discount_factor, forward = extracted

        if self.min_r_squared is not None and not (fit.r_squared >= self.min_r_squared):
            warnings.append(
                f"T={maturity:.4f} : R²={fit.r_squared:.4f} < {self.min_r_squared}, "
                f"maturité écartée."
            )
            return None

        # Propagation d'incertitude : F = intercept / D avec D = -slope, donc
        # dF = dI / D et dF = (I / D²) dD ; combinaison en quadrature (les deux
        # estimateurs sont corrélés, c'est un ordre de grandeur, pas un IC exact).
        stderr_forward = float(np.hypot(
            fit.stderr_intercept / discount_factor,
            forward * fit.stderr_slope / discount_factor,
        ))

        strikes = np.array([q.strike for q in final_quotes])
        zero_rate = float(-np.log(discount_factor) / maturity)

        return ForwardParityFitDiagnostics(
            maturity=maturity,
            maturity_date=final_quotes[0].maturity_date,
            n_quotes_input=n_input,
            n_quotes_used=fit.n_inliers,
            discount_factor=discount_factor,
            forward=forward,
            zero_rate=zero_rate,
            # Carry q tel que F = S e^{(r - q) T} : borrow + rendement de
            # dividende implicites (et biais EEP résiduel pour des américaines).
            implied_carry=float(zero_rate - np.log(forward / spot_value) / maturity),
            r_squared=fit.r_squared,
            rmse=fit.rmse,
            stderr_discount_factor=fit.stderr_slope,
            stderr_forward=stderr_forward,
            strikes=strikes,
            residuals=fit.residuals,
            inlier_mask=fit.inlier_mask,
            window_applied=window_applied,
        )

    def _regress(self, quotes: list[PutCallQuote]) -> LinearFitResult:
        strikes = np.array([q.strike for q in quotes])
        price_diffs = np.array([q.call_mid - q.put_mid for q in quotes])
        weights = self.weighting.weights(quotes)
        return self.regression.fit(strikes, price_diffs, weights)

    def _extract(self, fit: LinearFitResult) -> tuple[float, float] | None:
        """(discount_factor, forward) depuis (intercept, slope), ou None si dégénéré."""
        discount_factor = -fit.slope
        if not (0.0 < discount_factor <= self.MAX_DISCOUNT_FACTOR):
            return None
        forward = fit.intercept / discount_factor
        if forward <= 0.0:
            return None
        return discount_factor, forward

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(weighting={self.weighting!r}, "
                f"regression={self.regression!r}, strike_window={self.strike_window}, "
                f"min_strikes={self.min_strikes_per_maturity})")
