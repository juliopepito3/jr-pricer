"""Tests du loader de chaîne d'options (market_data/loaders/option_chain.py)."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from JR_PRICER.curves.builder.parity_builder import PutCallParityCurveBuilder
from JR_PRICER.curves.forward.analytic_forward import AnalyticForwardCurve
from JR_PRICER.curves.interpolators_1D.log_linear import LogLinearInterpolator1D
from JR_PRICER.curves.temporal.discount import FlatDiscountCurve
from JR_PRICER.market_data.loaders.option_chain import (
    ChainQualityFilter,
    OptionChainSnapshot,
)
from JR_PRICER.market_data.quote import Quote
from JR_PRICER.pricing.formulas import black_forward_price

from tests.helpers import DC, REF_DATE, in_days

SPOT = 100.0


def make_chain_rows(expiry, strikes, option_type, bid, ask,
                    volume=100, open_interest=500):
    """Lignes de chaîne au schéma CSV, prix constants (contrôle des filtres)."""
    return [{
        "quote_date": REF_DATE.isoformat(), "spot": SPOT,
        "expiry": expiry.isoformat(), "option_type": option_type,
        "strike": float(K), "bid": bid, "ask": ask, "last": (bid + ask) / 2,
        "volume": volume, "open_interest": open_interest, "iv_yf": 0.25,
    } for K in strikes]


def write_csv(tmp_path, rows):
    path = tmp_path / "chain.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def make_black_chain(maturity_days=(91, 182), strikes=np.arange(70.0, 131.0, 5.0),
                     r=0.03, q=0.01, sigma=0.25, spread=0.4):
    """Chaîne complète cohérente avec des courbes plates (prix Black actualisés)."""
    disc = FlatDiscountCurve(r, DC, REF_DATE)
    fwd = AnalyticForwardCurve(Quote(SPOT), disc, dividend_yield=q)
    rows = []
    for days in maturity_days:
        expiry = in_days(days)
        T = DC.year_fraction(REF_DATE, expiry)
        D, F = disc.discount(T), fwd.forward(T)
        for K in strikes:
            for opt_type, is_call in (("C", True), ("P", False)):
                mid = float(D * black_forward_price(F, K, T, sigma, is_call))
                rows.append({
                    "quote_date": REF_DATE.isoformat(), "spot": SPOT,
                    "expiry": expiry.isoformat(), "option_type": opt_type,
                    "strike": float(K),
                    "bid": mid - spread / 2, "ask": mid + spread / 2, "last": mid,
                    "volume": 100, "open_interest": 500, "iv_yf": sigma,
                })
    return rows


# --------------------------------------------------------------------------- #
# from_csv
# --------------------------------------------------------------------------- #

def test_from_csv_parses_snapshot(tmp_path):
    path = write_csv(tmp_path, make_black_chain())
    snapshot = OptionChainSnapshot.from_csv(path)

    assert snapshot.quote_date == REF_DATE
    assert snapshot.spot.value() == SPOT
    assert snapshot.expiries == [in_days(91), in_days(182)]
    assert isinstance(snapshot.chain["expiry"].iloc[0], date)


def test_from_csv_rejects_missing_columns(tmp_path):
    rows = make_black_chain()
    frame = pd.DataFrame(rows).drop(columns=["open_interest"])
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="open_interest"):
        OptionChainSnapshot.from_csv(path)


def test_from_csv_rejects_inconsistent_snapshot(tmp_path):
    rows = make_black_chain()
    rows[0]["spot"] = SPOT + 1.0
    path = write_csv(tmp_path, rows)
    with pytest.raises(ValueError, match="spot"):
        OptionChainSnapshot.from_csv(path)


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #

def test_pairing_inner_join_and_mids(tmp_path):
    expiry = in_days(91)
    rows = (make_chain_rows(expiry, [90, 100, 110], "C", bid=5.0, ask=5.4)
            + make_chain_rows(expiry, [100, 110, 120], "P", bid=3.0, ask=3.2))
    snapshot = OptionChainSnapshot.from_csv(write_csv(tmp_path, rows))
    quotes = snapshot.to_put_call_quotes()

    # Strikes 90 (call seul) et 120 (put seul) disparaissent.
    assert [q.strike for q in quotes] == [100.0, 110.0]
    assert quotes[0].call_mid == pytest.approx(5.2)
    assert quotes[0].put_mid == pytest.approx(3.1)
    assert quotes[0].maturity_date == expiry


def test_expiry_selection(tmp_path):
    path = write_csv(tmp_path, make_black_chain(maturity_days=(91, 182)))
    snapshot = OptionChainSnapshot.from_csv(path)
    quotes = snapshot.to_put_call_quotes(expiries=[in_days(91)])
    assert {q.maturity_date for q in quotes} == {in_days(91)}


# --------------------------------------------------------------------------- #
# ChainQualityFilter (chaque filtre isolément)
# --------------------------------------------------------------------------- #

def _one_pair(expiry, strike=100.0, **overrides):
    call = make_chain_rows(expiry, [strike], "C", bid=5.0, ask=5.4)[0]
    put = make_chain_rows(expiry, [strike], "P", bid=3.0, ask=3.2)[0]
    call.update({k: v for k, v in overrides.items() if not k.startswith("put_")})
    put.update({k[4:]: v for k, v in overrides.items() if k.startswith("put_")})
    return [call, put]


@pytest.mark.parametrize("overrides,filter_kwargs", [
    ({"bid": 0.02, "ask": 0.06}, {}),                       # min_bid
    ({"bid": 1.0, "ask": 2.5}, {}),                         # spread relatif > 50%
    ({"volume": 0}, {"min_volume": 10}),                    # min_volume
    ({"open_interest": 0}, {}),                             # min_open_interest
    ({"bid": 0.0, "ask": 5.4}, {}),                         # one-sided
    ({"bid": 5.0, "ask": 4.8}, {}),                         # marché croisé
])
def test_quality_filters_drop_bad_call_leg(tmp_path, overrides, filter_kwargs):
    expiry = in_days(91)
    rows = _one_pair(expiry, **overrides)
    snapshot = OptionChainSnapshot.from_csv(write_csv(tmp_path, rows))
    quotes = snapshot.to_put_call_quotes(ChainQualityFilter(**filter_kwargs))
    assert quotes == []  # le leg call filtré casse la paire


def test_quality_filter_keeps_clean_pair(tmp_path):
    rows = _one_pair(in_days(91))
    snapshot = OptionChainSnapshot.from_csv(write_csv(tmp_path, rows))
    assert len(snapshot.to_put_call_quotes(ChainQualityFilter())) == 1


# --------------------------------------------------------------------------- #
# Round-trip : chaîne synthétique → loader → builder → recovery
# --------------------------------------------------------------------------- #

def test_round_trip_chain_to_curves(tmp_path):
    r, q = 0.03, 0.01
    path = write_csv(tmp_path, make_black_chain(r=r, q=q))
    snapshot = OptionChainSnapshot.from_csv(path)
    quotes = snapshot.to_put_call_quotes(ChainQualityFilter())

    builder = PutCallParityCurveBuilder(
        day_count_convention=DC,
        discount_interpolator=LogLinearInterpolator1D(),
        forward_interpolator=LogLinearInterpolator1D(),
    )
    curves = builder.build(quotes, snapshot.spot, snapshot.quote_date)

    for days in (91, 182):
        T = DC.year_fraction(REF_DATE, in_days(days))
        assert curves.diagnostics[T].forward == pytest.approx(
            SPOT * np.exp((r - q) * T), rel=1e-6)
        assert curves.diagnostics[T].zero_rate == pytest.approx(r, abs=1e-4)
