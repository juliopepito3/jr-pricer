"""Tests des régressions affines pondérées et robustes (curves/builder/regression.py)."""
import numpy as np
import pytest

from JR_PRICER.curves.builder.regression import (
    HuberIRLSRegression,
    MADRejectionRegression,
    WeightedLeastSquares,
)

RNG_SEED = 1234


def _line(x, intercept, slope):
    return intercept + slope * x


# --------------------------------------------------------------------------- #
# WeightedLeastSquares
# --------------------------------------------------------------------------- #

def test_wls_exact_on_perfect_line():
    x = np.linspace(80.0, 120.0, 15)
    y = _line(x, 5.0, -0.97)
    fit = WeightedLeastSquares().fit(x, y, np.ones_like(x))

    assert fit.intercept == pytest.approx(5.0, abs=1e-10)
    assert fit.slope == pytest.approx(-0.97, abs=1e-12)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    assert fit.rmse == pytest.approx(0.0, abs=1e-10)
    assert fit.inlier_mask.all()
    np.testing.assert_allclose(fit.residuals, 0.0, atol=1e-10)


def test_wls_invariant_to_weight_scale():
    rng = np.random.default_rng(RNG_SEED)
    x = np.linspace(0.0, 10.0, 30)
    y = _line(x, 1.0, 2.0) + rng.normal(0, 0.1, x.size)
    w = rng.uniform(0.5, 2.0, x.size)

    fit_1 = WeightedLeastSquares().fit(x, y, w)
    fit_k = WeightedLeastSquares().fit(x, y, 1000.0 * w)

    assert fit_1.intercept == pytest.approx(fit_k.intercept, abs=1e-10)
    assert fit_1.slope == pytest.approx(fit_k.slope, abs=1e-12)


def test_wls_weights_help_under_heteroscedastic_noise():
    """Deux populations de bruit : pondérer en 1/sigma² doit battre l'uniforme."""
    rng = np.random.default_rng(RNG_SEED)
    x = np.linspace(50.0, 150.0, 200)
    sigma = np.where(x < 100.0, 0.05, 2.0)  # moitié précise, moitié bruyante
    true_slope = -0.95

    slope_err_uniform = []
    slope_err_weighted = []
    for _ in range(20):
        y = _line(x, 10.0, true_slope) + rng.normal(0, sigma)
        fit_u = WeightedLeastSquares().fit(x, y, np.ones_like(x))
        fit_w = WeightedLeastSquares().fit(x, y, 1.0 / sigma ** 2)
        slope_err_uniform.append(abs(fit_u.slope - true_slope))
        slope_err_weighted.append(abs(fit_w.slope - true_slope))

    assert np.mean(slope_err_weighted) < np.mean(slope_err_uniform)


def test_wls_predict():
    x = np.array([0.0, 1.0, 2.0])
    y = _line(x, 1.0, 3.0)
    fit = WeightedLeastSquares().fit(x, y, np.ones_like(x))
    assert fit.predict(10.0) == pytest.approx(31.0, abs=1e-10)


# --------------------------------------------------------------------------- #
# MADRejectionRegression
# --------------------------------------------------------------------------- #

def test_mad_rejection_isolates_outliers():
    rng = np.random.default_rng(RNG_SEED)
    x = np.linspace(80.0, 120.0, 25)
    y = _line(x, 5.0, -0.97) + rng.normal(0, 0.02, x.size)
    y[3] += 8.0    # quote stale
    y[17] -= 5.0   # quote croisée

    fit = MADRejectionRegression(k_mad=3.0, max_passes=2).fit(x, y, np.ones_like(x))

    assert not fit.inlier_mask[3]
    assert not fit.inlier_mask[17]
    assert fit.n_inliers == x.size - 2
    assert fit.slope == pytest.approx(-0.97, abs=5e-3)

    # Contrôle négatif : le WLS nu est décalé par les outliers.
    fit_wls = WeightedLeastSquares().fit(x, y, np.ones_like(x))
    assert abs(fit_wls.slope + 0.97) > abs(fit.slope + 0.97)
    assert abs(fit_wls.intercept - 5.0) > abs(fit.intercept - 5.0)


def test_mad_rejection_no_outliers_keeps_everything():
    rng = np.random.default_rng(RNG_SEED)
    x = np.linspace(0.0, 10.0, 40)
    y = _line(x, 2.0, 1.5) + rng.normal(0, 0.1, x.size)

    fit = MADRejectionRegression().fit(x, y, np.ones_like(x))
    fit_wls = WeightedLeastSquares().fit(x, y, np.ones_like(x))

    # Sous bruit gaussien propre, quelques rejets de queue sont possibles mais
    # le fit doit rester quasi identique au WLS.
    assert fit.slope == pytest.approx(fit_wls.slope, abs=5e-3)
    assert fit.n_inliers >= int(0.9 * x.size)


def test_mad_rejection_respects_min_inliers():
    # 6 points dont 3 aberrants : avec min_inliers=5, on refuse de descendre à 3.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = _line(x, 0.0, 1.0)
    y[[1, 3, 5]] += np.array([10.0, -12.0, 15.0])

    fit = MADRejectionRegression(k_mad=2.0, max_passes=3, min_inliers=5).fit(
        x, y, np.ones_like(x))
    assert fit.n_inliers >= 5


def test_mad_rejection_perfect_line_no_crash():
    x = np.linspace(0.0, 5.0, 10)
    y = _line(x, 1.0, 1.0)
    fit = MADRejectionRegression().fit(x, y, np.ones_like(x))
    assert fit.inlier_mask.all()
    assert fit.slope == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# HuberIRLSRegression
# --------------------------------------------------------------------------- #

def test_huber_matches_wls_without_outliers():
    rng = np.random.default_rng(RNG_SEED)
    x = np.linspace(0.0, 10.0, 50)
    y = _line(x, 1.0, 2.0) + rng.normal(0, 0.05, x.size)

    fit_h = HuberIRLSRegression().fit(x, y, np.ones_like(x))
    fit_w = WeightedLeastSquares().fit(x, y, np.ones_like(x))

    assert fit_h.slope == pytest.approx(fit_w.slope, abs=2e-3)
    assert fit_h.intercept == pytest.approx(fit_w.intercept, abs=1e-2)


def test_huber_downweights_outliers_and_converges():
    rng = np.random.default_rng(RNG_SEED)
    x = np.linspace(80.0, 120.0, 25)
    y = _line(x, 5.0, -0.97) + rng.normal(0, 0.02, x.size)
    y[5] += 6.0
    y[20] -= 9.0

    reg = HuberIRLSRegression(max_iterations=50)
    fit = reg.fit(x, y, np.ones_like(x))

    assert fit.slope == pytest.approx(-0.97, abs=5e-3)
    assert fit.n_iterations < 50  # convergence avant la borne
    assert fit.inlier_mask.all()  # pas de rejet dur : tout le monde reste

    fit_wls = WeightedLeastSquares().fit(x, y, np.ones_like(x))
    assert abs(fit_wls.slope + 0.97) > abs(fit.slope + 0.97)


# --------------------------------------------------------------------------- #
# Cas dégénérés et validation
# --------------------------------------------------------------------------- #

def test_constant_y_gives_nan_r_squared():
    x = np.linspace(0.0, 5.0, 10)
    y = np.full_like(x, 3.0)
    fit = WeightedLeastSquares().fit(x, y, np.ones_like(x))
    assert np.isnan(fit.r_squared)
    assert fit.slope == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("regression", [
    WeightedLeastSquares(),
    MADRejectionRegression(),
    HuberIRLSRegression(),
])
def test_input_validation(regression):
    with pytest.raises(ValueError):
        regression.fit(np.array([1.0]), np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        regression.fit(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        regression.fit(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, -1.0]))


def test_constructor_validation():
    with pytest.raises(ValueError):
        MADRejectionRegression(k_mad=0.0)
    with pytest.raises(ValueError):
        MADRejectionRegression(min_inliers=1)
    with pytest.raises(ValueError):
        HuberIRLSRegression(delta=-1.0)
