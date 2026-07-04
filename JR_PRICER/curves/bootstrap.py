"""Bootstrap d'une courbe de discount à partir d'instruments de marché (dépôts, swaps)."""
from __future__ import annotations

from datetime import date

from JR_PRICER.curves.interpolators_1D.base import Interpolator1D
from JR_PRICER.curves.temporal.discount import DiscountCurve
from JR_PRICER.instruments.base import Instrument
from JR_PRICER.utils.day_count import DayCounter
from JR_PRICER.utils.numerics import find_root


class BootstrapBuilder:
    """Construit séquentiellement une `DiscountCurve` qui reprice les instruments.

    Les instruments sont triés par maturité ; chaque pilier est posé via la
    formule fermée (`implied_discount_factor`) puis raffiné par recherche de racine
    sur la NPV pour les produits à flux intermédiaires (swaps).
    """

    def __init__(self, instruments: list[Instrument], interpolator: Interpolator1D,
                 day_count_convention: DayCounter, reference_date: date) -> None:
        self.instruments = instruments
        self.interpolator = interpolator
        self.day_count_convention = day_count_convention
        self.reference_date = reference_date

    def bootstrap(self) -> DiscountCurve:
        """Bootstrappe et retourne la `DiscountCurve` calibrée sur les instruments."""
        
        # 1) Trier les instruments par maturité

        sorted_instruments = sorted(self.instruments, key=lambda x: x.maturity_date)

        # 2) Utiliser les méthodes implied_discount_factor de chaque instrument pour construire la courbe

        # P(0,0) = 1 par définition — ancre la courbe et garantit 2 points dès le 1er instrument
        curve = DiscountCurve([0.0], [1.0], self.interpolator, self.day_count_convention, self.reference_date)

        for instrument in sorted_instruments:
            T = self.day_count_convention.year_fraction(self.reference_date, instrument.maturity_date)

            # Guess via formule fermée (exacte pour les dépôts et les swaps dont
            # tous les coupons tombent sur des piliers déjà bootstrappés).
            df_guess = instrument.implied_discount_factor(curve)
            curve.add_point(T, df_guess)

            # Raffinement : pour les instruments à flux intermédiaires (swaps), la
            # formule fermée extrapole les DF des coupons situés au-delà du dernier
            # pilier. On résout alors DF(T) pour annuler la NPV exacte.
            if hasattr(instrument, "npv_given_curve"):
                def npv_error(df: float) -> float:
                    curve.y[-1] = df
                    curve._fitted = False
                    return instrument.npv_given_curve(curve)

                df_solved = find_root(npv_error, df_guess)
                curve.y[-1] = df_solved
                curve._fitted = False

        return curve

    def __repr__(self) -> str:
        return f"BootstrapBuilder(n_instruments={len(self.instruments)}, ref={self.reference_date})"



