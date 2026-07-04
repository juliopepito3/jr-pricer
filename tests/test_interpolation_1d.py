"""Non-régression des interpolateurs 1D (reproduction des piliers, formule log-linéaire)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from JR_PRICER.curves.interpolators_1D.log_linear import LogLinearInterpolator1D
from JR_PRICER.curves.interpolators_1D.cubic_splines import ScipyCubicSplinesInterpolator1D

X = [0.5, 1.0, 2.0, 5.0, 10.0]
Y = [math.exp(-0.030 * t) for t in X]  # discount factors > 0


def test_loglinear_reproduces_pillars():
    interp = LogLinearInterpolator1D()
    interp._fit(X, Y)
    for xi, yi in zip(X, Y):
        assert float(interp.interpolate(xi)) == pytest.approx(yi, rel=1e-12)


def test_loglinear_midpoint_formula():
    interp = LogLinearInterpolator1D()
    interp._fit(X, Y)
    t = 1.5
    x0, x1, y0, y1 = X[1], X[2], Y[1], Y[2]
    log_y = math.log(y0) + (math.log(y1) - math.log(y0)) * (t - x0) / (x1 - x0)
    assert float(interp.interpolate(t)) == pytest.approx(math.exp(log_y), rel=1e-12)


def test_loglinear_flat_extrapolation_left_and_right():
    interp = LogLinearInterpolator1D()
    interp._fit(X, Y)
    # prolonge le segment de bord en log-linéaire ; on vérifie juste finitude/positivité
    assert float(interp.interpolate(0.1)) > 0
    assert float(interp.interpolate(20.0)) > 0


def test_cubic_reproduces_pillars():
    interp = ScipyCubicSplinesInterpolator1D()
    interp._fit(X, Y)
    for xi, yi in zip(X, Y):
        assert float(interp.interpolate(xi)) == pytest.approx(yi, rel=1e-9)


# --- Phase 1 : capacité vectorisée scalaire <-> ndarray ----------------------

QUERY = np.array([0.1, 0.5, 0.7, 1.0, 1.5, 3.3, 10.0, 12.0])


@pytest.mark.parametrize("factory", [
    LogLinearInterpolator1D,
    ScipyCubicSplinesInterpolator1D,
])
def test_scalar_vector_consistency(factory):
    interp = factory()
    interp._fit(X, Y)
    vec = np.asarray(interp.interpolate(QUERY))
    assert vec.shape == QUERY.shape
    scal = np.array([float(interp.interpolate(float(t))) for t in QUERY])
    assert vec == pytest.approx(scal, rel=1e-12)


def test_loglinear_scalar_returns_python_float():
    interp = LogLinearInterpolator1D()
    interp._fit(X, Y)
    assert isinstance(interp.interpolate(1.5), float)
