"""Tests du builder de courbes par parité put-call (curves/builder/parity_builder.py).

Les quotes sont fabriquées depuis des courbes plates (FlatDiscountCurve +
AnalyticForwardCurve) et des prix Black : la parité C - P = D·(F - K) tient
alors exactement, ce qui rend les cibles de recovery analytiques.
"""
from datetime import date

import numpy as np
import pytest

from JR_PRICER.curves.builder.parity_builder import (
    InverseSpreadWeights,
    PutCallParityCurveBuilder,
    PutCallQuote,
    StrikeWindow,
    UniformWeights,
)
from JR_PRICER.curves.builder.regression import (
    MADRejectionRegression,
    WeightedLeastSquares,
)
from JR_PRICER.curves.forward.analytic_forward import AnalyticForwardCurve
from JR_PRICER.curves.interpolators_1D.log_linear import LogLinearInterpolator1D
from JR_PRICER.curves.temporal.discount import FlatDiscountCurve
from JR_PRICER.market_data.quote import Quote
from JR_PRICER.pricing.formulas import black_forward_price

from tests.helpers import DC, REF_DATE, in_days

SPOT = 100.0


def make_quotes(maturity_days, strikes, r=0.03, q=0.01, sigma=0.25,
                spread=0.4, mid_noise=0.0, rng=None):
    """Quotes call/put cohérentes avec des courbes plates (r, q) et un vol Black.

    `spread` est le spread bid-ask absolu de chaque leg ; `mid_noise` un bruit
    gaussien optionnel ajouté aux mids (même tirage sur bid et ask).
    """
    disc = FlatDiscountCurve(r, DC, REF_DATE)
    fwd = AnalyticForwardCurve(Quote(SPOT), disc, dividend_yield=q)

    quotes = []
    for days in maturity_days:
        maturity_date = in_days(days)
        T = DC.year_fraction(REF_DATE, maturity_date)
        D = disc.discount(T)
        F = fwd.forward(T)
        for K in strikes:
            call = float(D * black_forward_price(F, K, T, sigma, True))
            put = float(D * black_forward_price(F, K, T, sigma, False))
            if mid_noise > 0:
                call += float(rng.normal(0, mid_noise))
                put += float(rng.normal(0, mid_noise))
            quotes.append(PutCallQuote(
                strike=float(K),
                call_mid=call, put_mid=put,
                call_bid=call - spread / 2, call_ask=call + spread / 2,
                put_bid=put - spread / 2, put_ask=put + spread / 2,
                maturity_date=maturity_date,
            ))
    return quotes


def default_builder(**overrides):
    kwargs = dict(
        day_count_convention=DC,
        discount_interpolator=LogLinearInterpolator1D(),
        forward_interpolator=LogLinearInterpolator1D(),
        weighting=UniformWeights(),
        regression=WeightedLeastSquares(),
        strike_window=None,
        min_strikes_per_maturity=6,
    )
    kwargs.update(overrides)
    return PutCallParityCurveBuilder(**kwargs)


STRIKES = np.arange(70.0, 131.0, 5.0)
MATURITY_DAYS = (91, 182, 365)


# --------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------- #

def test_exact_recovery_without_noise():
    r, q = 0.03, 0.01
    quotes = make_quotes(MATURITY_DAYS, STRIKES, r=r, q=q)
    curves = default_builder().build(quotes, Quote(SPOT), REF_DATE)

    for days in MATURITY_DAYS:
        T = DC.year_fraction(REF_DATE, in_days(days))
        assert curves.discount_curve.discount(T) == pytest.approx(np.exp(-r * T), abs=1e-10)
        assert curves.forward_curve.forward(T) == pytest.approx(
            SPOT * np.exp((r - q) * T), abs=1e-8)

        diag = curves.diagnostics[T]
        assert diag.zero_rate == pytest.approx(r, abs=1e-10)
        assert diag.implied_carry == pytest.approx(q, abs=1e-10)
        assert diag.r_squared == pytest.approx(1.0, abs=1e-12)
        assert diag.n_quotes_used == len(STRIKES)

    assert curves.has_monotonic_discount_factors
    assert curves.warnings == ()


def test_recovery_under_bid_ask_noise():
    rng = np.random.default_rng(7)
    quotes = make_quotes(MATURITY_DAYS, STRIKES, mid_noise=0.05, rng=rng)
    curves = default_builder().build(quotes, Quote(SPOT), REF_DATE)

    for days in MATURITY_DAYS:
        T = DC.year_fraction(REF_DATE, in_days(days))
        diag = curves.diagnostics[T]
        assert diag.forward == pytest.approx(SPOT * np.exp((0.03 - 0.01) * T), rel=2e-3)
        assert diag.discount_factor == pytest.approx(np.exp(-0.03 * T), abs=5e-3)
        assert diag.r_squared > 0.999


def test_inverse_spread_weights_beat_uniform_under_heteroscedastic_noise():
    """Bruit proportionnel au spread : pondérer en 1/spread² doit réduire l'erreur sur F."""
    T = DC.year_fraction(REF_DATE, in_days(182))
    true_forward = SPOT * np.exp((0.03 - 0.01) * T)

    err_uniform, err_weighted = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        # Spread étroit près de l'ATM, large sur les ailes ; bruit ~ spread.
        quotes = []
        for K in STRIKES:
            base = make_quotes((182,), [K])[0]
            spread = 0.1 + 0.05 * abs(K - SPOT)
            noise_c = float(rng.normal(0, spread / 4))
            noise_p = float(rng.normal(0, spread / 4))
            quotes.append(PutCallQuote(
                strike=base.strike,
                call_mid=base.call_mid + noise_c, put_mid=base.put_mid + noise_p,
                call_bid=base.call_mid + noise_c - spread / 2,
                call_ask=base.call_mid + noise_c + spread / 2,
                put_bid=base.put_mid + noise_p - spread / 2,
                put_ask=base.put_mid + noise_p + spread / 2,
                maturity_date=base.maturity_date,
            ))
        f_u = default_builder(weighting=UniformWeights()).build(
            quotes, Quote(SPOT), REF_DATE).diagnostics[T].forward
        f_w = default_builder(weighting=InverseSpreadWeights(power=2.0)).build(
            quotes, Quote(SPOT), REF_DATE).diagnostics[T].forward
        err_uniform.append(abs(f_u - true_forward))
        err_weighted.append(abs(f_w - true_forward))

    assert np.mean(err_weighted) < np.mean(err_uniform)


def test_mad_rejection_recovers_despite_outliers():
    rng = np.random.default_rng(11)
    quotes = make_quotes((182,), STRIKES, mid_noise=0.01, rng=rng)
    # Deux quotes polluées (stale) : call_mid décalé de plusieurs dollars.
    polluted = []
    for i, quote in enumerate(quotes):
        if i in (2, 9):
            polluted.append(PutCallQuote(
                strike=quote.strike,
                call_mid=quote.call_mid + 4.0, put_mid=quote.put_mid,
                call_bid=quote.call_bid + 4.0, call_ask=quote.call_ask + 4.0,
                put_bid=quote.put_bid, put_ask=quote.put_ask,
                maturity_date=quote.maturity_date,
            ))
        else:
            polluted.append(quote)

    T = DC.year_fraction(REF_DATE, in_days(182))
    true_forward = SPOT * np.exp((0.03 - 0.01) * T)

    diag_mad = default_builder(regression=MADRejectionRegression()).build(
        polluted, Quote(SPOT), REF_DATE).diagnostics[T]
    diag_wls = default_builder(regression=WeightedLeastSquares()).build(
        polluted, Quote(SPOT), REF_DATE).diagnostics[T]

    assert diag_mad.n_quotes_used == len(STRIKES) - 2
    assert not diag_mad.inlier_mask[2]
    assert not diag_mad.inlier_mask[9]
    assert abs(diag_mad.forward - true_forward) < abs(diag_wls.forward - true_forward)
    assert diag_mad.forward == pytest.approx(true_forward, rel=1e-3)


def test_strike_window_mitigates_wing_bias():
    """Biais type EEP sur les legs ITM profonds : la fenêtre ATM doit le neutraliser."""
    quotes = []
    T = DC.year_fraction(REF_DATE, in_days(365))
    true_forward = SPOT * np.exp((0.03 - 0.01) * T)
    for quote in make_quotes((365,), STRIKES):
        put_bias = 1.5 if quote.strike > 1.15 * true_forward else 0.0   # puts ITM profonds
        call_bias = 1.0 if quote.strike < 0.85 * true_forward else 0.0  # calls ITM profonds
        quotes.append(PutCallQuote(
            strike=quote.strike,
            call_mid=quote.call_mid + call_bias, put_mid=quote.put_mid + put_bias,
            call_bid=quote.call_bid + call_bias, call_ask=quote.call_ask + call_bias,
            put_bid=quote.put_bid + put_bias, put_ask=quote.put_ask + put_bias,
            maturity_date=quote.maturity_date,
        ))

    diag_naked = default_builder(strike_window=None).build(
        quotes, Quote(SPOT), REF_DATE).diagnostics[T]
    diag_window = default_builder(strike_window=StrikeWindow(0.85, 1.15)).build(
        quotes, Quote(SPOT), REF_DATE).diagnostics[T]

    assert diag_window.window_applied
    assert not diag_naked.window_applied
    assert diag_window.forward == pytest.approx(true_forward, abs=1e-8)
    assert abs(diag_naked.forward - true_forward) > 0.1


def test_strike_window_falls_back_when_too_few_strikes():
    # 6 strikes très écartés : la fenêtre [0.95F, 1.05F] n'en contient pas assez.
    strikes = [50.0, 70.0, 90.0, 110.0, 130.0, 150.0]
    quotes = make_quotes((182,), strikes)
    curves = default_builder(strike_window=StrikeWindow(0.95, 1.05)).build(
        quotes, Quote(SPOT), REF_DATE)

    T = DC.year_fraction(REF_DATE, in_days(182))
    assert not curves.diagnostics[T].window_applied
    assert any("fenêtre" in w for w in curves.warnings)
    # Le fit passe 1 (quotes exactes) reste correct.
    assert curves.diagnostics[T].forward == pytest.approx(
        SPOT * np.exp((0.03 - 0.01) * T), abs=1e-8)


# --------------------------------------------------------------------------- #
# Filtres, warnings et cas limites
# --------------------------------------------------------------------------- #

def test_maturity_skipped_below_min_strikes():
    quotes = make_quotes((91,), STRIKES) + make_quotes((182,), STRIKES[:3])
    curves = default_builder(min_strikes_per_maturity=6).build(
        quotes, Quote(SPOT), REF_DATE)

    assert len(curves.diagnostics) == 1
    assert any("ignorée" in w for w in curves.warnings)


def test_raises_when_no_maturity_survives():
    quotes = make_quotes((91,), STRIKES[:3])
    with pytest.raises(ValueError):
        default_builder(min_strikes_per_maturity=6).build(quotes, Quote(SPOT), REF_DATE)


def test_min_r_squared_drops_pure_noise_slice():
    rng = np.random.default_rng(3)
    good = make_quotes((91,), STRIKES)
    noisy = []
    for quote in make_quotes((365,), STRIKES):
        shift = float(rng.normal(0, 15.0))  # C - P détruit : plus rien d'affine
        noisy.append(PutCallQuote(
            strike=quote.strike,
            call_mid=quote.call_mid + shift, put_mid=quote.put_mid,
            call_bid=quote.call_bid + shift, call_ask=quote.call_ask + shift,
            put_bid=quote.put_bid, put_ask=quote.put_ask,
            maturity_date=quote.maturity_date,
        ))

    curves = default_builder(min_r_squared=0.99).build(
        good + noisy, Quote(SPOT), REF_DATE)

    assert len(curves.diagnostics) == 1
    assert any("R²" in w for w in curves.warnings)


def test_non_monotonic_discount_factors_flagged():
    # Deux maturités avec des DF inversés (D long > D court) : quotes forgées
    # directement sur la droite C - P = D (F - K).
    quotes = []
    for days, D, F in ((91, 0.97, 101.0), (182, 0.99, 102.0)):
        for K in STRIKES:
            diff = D * (F - K)
            quotes.append(PutCallQuote(
                strike=float(K),
                call_mid=max(diff, 0.0) + 1.0, put_mid=max(-diff, 0.0) + 1.0,
                call_bid=max(diff, 0.0) + 0.9, call_ask=max(diff, 0.0) + 1.1,
                put_bid=max(-diff, 0.0) + 0.9, put_ask=max(-diff, 0.0) + 1.1,
                maturity_date=in_days(days),
            ))

    curves = default_builder().build(quotes, Quote(SPOT), REF_DATE)
    assert not curves.has_monotonic_discount_factors
    assert any("décroissants" in w for w in curves.warnings)


def test_structural_coherence_and_diagnostics_table():
    quotes = make_quotes(MATURITY_DAYS, STRIKES)
    curves = default_builder().build(quotes, Quote(SPOT), REF_DATE)

    np.testing.assert_array_equal(curves.discount_curve.times, curves.forward_curve.x)
    assert curves.discount_curve.reference_date == REF_DATE
    assert curves.discount_curve.day_count_convention is DC
    assert curves.forward_curve.forward(0.0) == SPOT

    table = curves.diagnostics_table()
    assert len(table) == len(MATURITY_DAYS)
    assert [row["maturity"] for row in table] == sorted(curves.diagnostics)
    assert {"forward", "discount_factor", "zero_rate", "implied_carry",
            "r_squared", "window_applied"} <= set(table[0].keys())


def test_default_configuration_end_to_end():
    """Défauts du builder (InverseSpread + MADRejection + fenêtre) sur données propres."""
    builder = PutCallParityCurveBuilder(
        day_count_convention=DC,
        discount_interpolator=LogLinearInterpolator1D(),
        forward_interpolator=LogLinearInterpolator1D(),
    )
    quotes = make_quotes(MATURITY_DAYS, STRIKES)
    curves = builder.build(quotes, Quote(SPOT), REF_DATE)

    for days in MATURITY_DAYS:
        T = DC.year_fraction(REF_DATE, in_days(days))
        assert curves.diagnostics[T].forward == pytest.approx(
            SPOT * np.exp((0.03 - 0.01) * T), rel=1e-6)
        assert curves.diagnostics[T].window_applied
